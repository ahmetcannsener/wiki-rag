# Local Wikipedia RAG Assistant

**GitHub Repository:** https://github.com/ahmetcannsener/wiki-rag

**Video Link:** https://www.youtube.com/watch?v=FSOX0U-_Zg4

A fully local, ChatGPT-style question-answering system that retrieves information from Wikipedia pages about famous people and places, and generates grounded answers using a local language model. No external API calls at runtime.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Models pulled:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

Run the following steps in order:

**Step 1 — Ingest Wikipedia pages into SQLite**
```bash
python ingest.py
```
Fetches 20 people and 20 places from Wikipedia and stores raw text in `wikipedia.db`. Skips any entity already in the database, so it is safe to re-run.

**Step 2 — Chunk, embed, and index into ChromaDB**
```bash
python build_index.py
```
Reads all rows from `wikipedia.db`, splits each article into 500-word overlapping chunks, embeds each chunk with `nomic-embed-text`, and upserts into the vector store.

**Step 3 — Start the chat interface**
```bash
streamlit run app.py
```
Opens the Streamlit UI at `http://localhost:8501`.

> If a page fails during ingestion (rate limit, disambiguation), use `add_missing.py` to re-fetch individual entities.

---

## Example Queries

### Person queries
- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Why is Nikola Tesla famous?"
- "What is Frida Kahlo known for?"

### Place queries
- "Where is the Eiffel Tower located?"
- "What was the Colosseum used for?"
- "Where is Mount Everest?"
- "Which famous place is located in Turkey?"

### Mixed / comparison queries
- "Compare Albert Einstein and Nikola Tesla"
- "Compare the Eiffel Tower and the Statue of Liberty"
- "Which place is in Egypt?"

### Failure cases (model returns "I don't know")
- "Who is the president of Mars?"
- "Tell me about John Doe"

---

## Project Structure

```
wiki-rag/
├── app.py              # Streamlit chat UI with sidebar status and message history
├── ingest.py           # Fetches Wikipedia pages → stores in SQLite (wikipedia.db)
├── build_index.py      # Reads SQLite → chunks → embeds → upserts into ChromaDB
├── add_missing.py      # One-off script to re-fetch a single failed entity
├── debug_retrieval.py  # Diagnostic script for inspecting ChromaDB contents
├── chunker.py          # Fixed-size word-count chunking with sentence-boundary overlap
├── embedder.py         # Calls Ollama nomic-embed-text to produce embedding vectors
├── vector_store.py     # ChromaDB read/write: upsert chunks, query by embedding
├── retriever.py        # Query classifier + entity extraction + chunk retrieval
├── generator.py        # Builds prompt, calls Ollama llama3.2, returns answer string
├── config.py           # Shared constants: model names, paths, chunk sizes, entity lists
├── wikipedia.db        # SQLite database (auto-created by ingest.py)
├── chroma_db/          # ChromaDB persistent storage (auto-created by build_index.py)
└── requirements.txt    # Python dependencies
```

---

## Design Decisions

### Two separate ChromaDB collections (Option A)

The system uses two collections — `people_store` and `places_store` — rather than a single unified collection.

**Rationale:**
- People and places are semantically distinct domains. A query about a scientist and a query about a landmark rarely share relevant vocabulary, so keeping them separate avoids cross-domain noise.
- When a query is clearly about a person, searching only `people_store` improves precision and reduces irrelevant results.
- Routing logic is straightforward: classify the query, pick the right collection, no metadata filter syntax needed.

**Tradeoff:** slightly more code to maintain two stores, but the retrieval quality improvement justifies it.

### Metadata-filtered retrieval + SQLite keyword fallback

Pure vector similarity with `nomic-embed-text` (via Ollama) cannot reliably distinguish individual landmarks — all famous-place introductions cluster together in embedding space. The system uses two complementary strategies:

1. **Exact/fuzzy entity name matching** — if the query names a specific entity (e.g. "Eiffel Tower"), ChromaDB is filtered to that entity's chunks before cosine ranking.
2. **SQLite keyword search** — for broad geographic queries ("Which place is in Turkey?"), the system searches the raw Wikipedia text for rare, specific words from the query and uses the matching entities as the filter.

This hybrid approach delivers reliable retrieval without requiring a more capable embedding model.

### Chunking strategy

Chunks are ~500 words with 50-word overlap, split on sentence boundaries where possible. Each chunk is stored with metadata (`entity_title`, `entity_type`, `chunk_index`, `source_url`) to enable both filtering and source attribution in the UI.
