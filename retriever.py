import logging
import re
import sqlite3
from difflib import get_close_matches

from config import PEOPLE, PLACES, TOP_K, BROAD_TOP_K, SQLITE_DB_PATH
from embedder import embed
from vector_store import get_people_collection, get_places_collection, query_collection

logger = logging.getLogger(__name__)

_PERSON_KEYWORDS = {
    "who", "person", "born", "died", "discovered", "invented", "wrote",
    "painted", "scientist", "author", "artist", "musician", "athlete",
    "philosopher", "politician", "leader", "president", "queen", "king",
}
_PLACE_KEYWORDS = {
    "where", "located", "built", "country", "city", "monument", "landmark",
    "visit", "travel", "height", "tall", "size", "area", "tower", "wall",
    "temple", "mountain", "canyon", "waterfall", "forest", "desert", "island",
    "ruins", "palace", "statue", "pyramid", "wonder",
    "place",  # "which place is in X?" → place signal
}

_PEOPLE_LOWER = {p.lower(): p for p in PEOPLE}
_PLACES_LOWER = {p.lower(): p for p in PLACES}
_ALL_PEOPLE_NAMES = list(_PEOPLE_LOWER.keys())
_ALL_PLACES_NAMES = list(_PLACES_LOWER.keys())

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "is", "are", "was", "were", "be", "been", "do", "did", "has",
    "have", "had", "it", "its", "this", "that", "with", "by", "from",
    "as", "who", "what", "where", "when", "why", "how", "which", "who",
    "compare", "tell", "me", "about", "give", "list", "describe",
}
_MIN_PROBE_LEN = 4  # skip fuzzy match on probes shorter than this

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    return re.findall(r"[a-z']+", text.lower())


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _extract_entities(query_lower: str, candidates: dict[str, str], cutoff: float = 0.75) -> list[str]:
    """
    Return canonical entity names (original casing) that appear in the query.
    Checks exact substrings first, then fuzzy n-gram matches.
    """
    found: list[str] = []
    tokens = _tokenize(query_lower)

    # Build all token n-grams (1-, 2-, 3-word) to probe against candidate names
    probes: list[str] = list(tokens)
    for i in range(len(tokens) - 1):
        probes.append(f"{tokens[i]} {tokens[i+1]}")
    for i in range(len(tokens) - 2):
        probes.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    for probe in probes:
        # Skip stop words and very short probes to avoid false positives
        if probe in _STOP_WORDS or len(probe) < _MIN_PROBE_LEN:
            continue
        # Exact match
        if probe in candidates:
            canonical = candidates[probe]
            if canonical not in found:
                found.append(canonical)
            continue
        # Fuzzy match
        matches = get_close_matches(probe, candidates.keys(), n=1, cutoff=cutoff)
        if matches:
            canonical = candidates[matches[0]]
            if canonical not in found:
                found.append(canonical)

    return found


def _classify_query(query: str) -> tuple[str, list[str], list[str]]:
    """
    Returns (category, matched_people, matched_places).
    category is 'person', 'place', or 'both'.
    """
    q_lower = query.lower()
    tokens = set(_tokenize(q_lower))

    matched_people = _extract_entities(q_lower, _PEOPLE_LOWER)
    matched_places = _extract_entities(q_lower, _PLACES_LOWER)

    person_signal = bool(tokens & _PERSON_KEYWORDS)
    place_signal  = bool(tokens & _PLACE_KEYWORDS)

    if matched_people and matched_places:
        category = "both"
    elif matched_people:
        category = "person"
    elif matched_places:
        category = "place"
    elif person_signal and not place_signal:
        category = "person"
    elif place_signal and not person_signal:
        category = "place"
    else:
        category = "both"

    logger.debug(
        "classify | query=%r | people=%s places=%s p_sig=%s pl_sig=%s → %s",
        query, matched_people, matched_places, person_signal, place_signal, category,
    )
    return category, matched_people, matched_places


# ---------------------------------------------------------------------------
# Collection querying helpers
# ---------------------------------------------------------------------------

def _query_filtered(collection, query_embedding: list[float], entity_names: list[str], top_k: int) -> list[dict]:
    """
    Query a ChromaDB collection filtered to specific entity titles.

    Single entity  → pure cosine ranking (precise mode).
    Multiple entities → round-robin interleaving: guarantees every keyword-
                        matched entity gets at least one chunk in the result,
                        regardless of cross-entity cosine distances.
    No entities    → unfiltered cosine search across the whole collection.
    """
    if not entity_names:
        return query_collection(collection, query_embedding, top_k)

    per_entity = max(2, top_k // max(1, len(entity_names)))

    # Collect ranked chunks per entity
    per_entity_chunks: list[list[dict]] = []
    for name in entity_names:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=per_entity,
            where={"entity_title": name},
            include=["documents", "metadatas", "distances"],
        )
        entity_chunks = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        per_entity_chunks.append(entity_chunks)

    if len(entity_names) == 1:
        # Single entity: pure distance sort
        return per_entity_chunks[0][:top_k]

    # Multiple entities: round-robin to guarantee each entity is represented,
    # then fill remaining slots with best remaining chunks by distance.
    slots: list[dict] = []
    # First pass: one best chunk per entity
    remainder: list[dict] = []
    for chunks in per_entity_chunks:
        if chunks:
            slots.append(chunks[0])
            remainder.extend(chunks[1:])

    # Fill up to top_k with leftover chunks sorted by distance
    remainder.sort(key=lambda c: c["distance"])
    slots.extend(remainder)
    return slots[:top_k]


# ---------------------------------------------------------------------------
# SQLite keyword fallback for broad/geographic queries
# ---------------------------------------------------------------------------

def _keyword_search_sqlite(query_lower: str, entity_type: str) -> list[str]:
    """
    Search Wikipedia full-text in SQLite for entities whose text contains
    specific (rare) words from the query.

    Words like 'famous', 'place', 'located' appear in nearly every article
    and are useless as filters. Words like 'turkey', 'egypt', 'radioactivity'
    appear in only a few articles — those are the ones that matter.

    Strategy: discard any word that matches more than half of the entities
    in the store (it's too generic). Use only the rare words for filtering.
    """
    words = [
        w for w in _tokenize(query_lower)
        if w not in _STOP_WORDS and len(w) >= 4
    ]
    if not words:
        return []

    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        total = conn.execute(
            "SELECT COUNT(*) FROM wikipedia_pages WHERE entity_type = ?", (entity_type,)
        ).fetchone()[0]
        if total == 0:
            conn.close()
            return []

        # Map each word → list of matching entity titles
        word_hits: dict[str, list[str]] = {}
        for word in words:
            rows = conn.execute(
                "SELECT title FROM wikipedia_pages "
                "WHERE entity_type = ? AND lower(full_text) LIKE ?",
                (entity_type, f"%{word}%"),
            ).fetchall()
            word_hits[word] = [r[0] for r in rows]
        conn.close()
    except Exception as exc:
        logger.warning("SQLite keyword search failed: %s", exc)
        return []

    # Use the rarest word as the filter: "turkey" (1 match) beats "france"
    # (5 matches) beats "place" (20 matches). This picks the most specific
    # signal in the query regardless of any fixed threshold.
    # If the rarest word still matches >50% of entities it's too generic — return
    # [] so the caller falls back to pure vector search.
    non_empty = {w: titles for w, titles in word_hits.items() if titles}
    if not non_empty:
        return []

    rarest_word = min(non_empty, key=lambda w: len(non_empty[w]))
    rarest_titles = non_empty[rarest_word]

    if len(rarest_titles) / total >= 0.75:
        logger.debug("keyword_search | all words too common — pure vector fallback")
        return []

    logger.debug(
        "keyword_search | type=%s rarest_word=%r matches=%d → %s",
        entity_type, rarest_word, len(rarest_titles), rarest_titles,
    )
    return rarest_titles


# ---------------------------------------------------------------------------
# Public retrieve()
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> tuple[list[dict], str]:
    """
    Returns (chunks, category).
    Strategy:
    - Specific entity named → metadata-filtered retrieval at TOP_K (precise).
    - No entity named → pure vector similarity across the whole store at
      BROAD_TOP_K (exploratory, e.g. "which place is in Turkey?").
    """
    category, matched_people, matched_places = _classify_query(query)
    logger.info("category=%s people=%s places=%s | %r", category, matched_people, matched_places, query)

    query_embedding = embed(query)
    chunks: list[dict] = []

    if category == "person":
        col = get_people_collection()
        if col.count() == 0:
            logger.warning("people_store is empty")
            return [], category
        entities = matched_people
        if not entities:
            entities = _keyword_search_sqlite(query.lower(), "person")
            logger.info("keyword fallback people: %s", entities[:5])
        effective_k = top_k if matched_people else BROAD_TOP_K
        chunks = _query_filtered(col, query_embedding, entities, effective_k)
        logger.info("Retrieved %d chunks from people_store", len(chunks))
        return chunks, category

    if category == "place":
        col = get_places_collection()
        if col.count() == 0:
            logger.warning("places_store is empty")
            return [], category
        entities = matched_places
        if not entities:
            entities = _keyword_search_sqlite(query.lower(), "place")
            logger.info("keyword fallback places: %s", entities[:5])
        # Broad mode: no exact entity name found — fetch more chunks so that
        # all keyword-matched entities get adequate representation.
        effective_k = top_k if matched_places else BROAD_TOP_K
        chunks = _query_filtered(col, query_embedding, entities, effective_k)
        logger.info("Retrieved %d chunks from places_store", len(chunks))
        return chunks, category

    # both — search each store with its respective entity filter
    people_col = get_people_collection()
    places_col  = get_places_collection()
    broad = not matched_people and not matched_places
    effective_k = BROAD_TOP_K if broad else top_k
    per_store = max(1, effective_k // 2)

    p_entities = matched_people or _keyword_search_sqlite(query.lower(), "person")
    pl_entities = matched_places or _keyword_search_sqlite(query.lower(), "place")

    if people_col.count() > 0:
        chunks.extend(_query_filtered(people_col, query_embedding, p_entities, per_store))
    if places_col.count() > 0:
        chunks.extend(_query_filtered(places_col, query_embedding, pl_entities, per_store))

    chunks.sort(key=lambda c: c["distance"])
    logger.info("Retrieved %d chunks from both stores (broad=%s)", len(chunks), broad)
    return chunks[:effective_k], "both"
