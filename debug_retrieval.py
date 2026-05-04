"""
Diagnostic script to inspect ChromaDB collection contents and test raw similarity search.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from config import CHROMA_DB_PATH, PEOPLE_COLLECTION, PLACES_COLLECTION
from embedder import embed

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# ── 1. Collection counts ──────────────────────────────────────────────────────
all_collections = [c.name for c in client.list_collections()]
print("=== Collections in ChromaDB ===")
print(f"Found: {all_collections}\n")

def get_col(name):
    try:
        return client.get_collection(name)
    except Exception as e:
        print(f"  [ERROR] Could not open '{name}': {e}")
        return None

people_col = get_col(PEOPLE_COLLECTION)
places_col = get_col(PLACES_COLLECTION)

people_count = people_col.count() if people_col else 0
places_count = places_col.count() if places_col else 0

print(f"people_store  : {people_count} documents")
print(f"places_store  : {places_count} documents\n")

if places_count == 0:
    print("places_store is EMPTY — data was never indexed or was stored in the wrong collection.")
    print("Run: python build_index.py")
    sys.exit(1)

# ── 2. Sample a few stored titles from places_store ──────────────────────────
print("=== Sample metadata from places_store (first 5 docs) ===")
sample = places_col.get(limit=5, include=["metadatas"])
for meta in sample["metadatas"]:
    print(f"  title={meta.get('entity_title')!r}  type={meta.get('entity_type')!r}  chunk={meta.get('chunk_index')}")
print()

# ── 3. Raw similarity search helper ─────────────────────────────────────────
def raw_search(collection, query: str, top_k: int = 3):
    print(f"=== Raw similarity search: {query!r} in '{collection.name}' ===")
    emb = embed(query)
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        print("  No results returned.\n")
        return

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        print(f"  [{i}] distance={dist:.4f}")
        print(f"       title={meta.get('entity_title')!r}  chunk={meta.get('chunk_index')}")
        print(f"       url={meta.get('source_url')}")
        print(f"       text preview: {doc[:200].replace(chr(10), ' ')!r}")
        print()

raw_search(places_col, "Eiffel Tower")
raw_search(places_col, "Colosseum")

# ── 4. Sanity-check: also search people_store for "Eiffel Tower" ─────────────
if people_col and people_count > 0:
    print("=== Sanity check: searching people_store for 'Eiffel Tower' ===")
    emb = embed("Eiffel Tower")
    results = people_col.query(
        query_embeddings=[emb],
        n_results=3,
        include=["metadatas", "distances"],
    )
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        print(f"  distance={dist:.4f}  title={meta.get('entity_title')!r}")
    print()
