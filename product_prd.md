# Product Requirements Document
## Local Wikipedia RAG Assistant (BLG483E – Project 3)

---

## 1. Overview

Build a fully local, ChatGPT-style question-answering system that retrieves information from
Wikipedia pages about famous people and places, and generates grounded answers using a
local language model. The system must run entirely on localhost — no external APIs allowed.

---

## 2. Goals

- Ingest Wikipedia data for at least 20 people and 20 places
- Chunk, embed, and store data in local vector stores
- Retrieve relevant context based on user queries
- Generate accurate, grounded answers using a local LLM
- Provide a simple Streamlit-based chat interface

---

## 3. Tech Stack

| Component       | Technology                        |
|----------------|-----------------------------------|
| Language        | Python 3.10+                      |
| LLM             | Ollama — llama3.2 (3B)            |
| Embeddings      | Ollama — nomic-embed-text         |
| Vector Store    | ChromaDB                          |
| Relational DB   | SQLite                            |
| UI              | Streamlit                         |

---

## 4. Data Requirements

### 4.1 Required People (minimum)
- Albert Einstein
- Marie Curie
- Leonardo da Vinci
- William Shakespeare
- Ada Lovelace
- Nikola Tesla
- Lionel Messi
- Cristiano Ronaldo
- Taylor Swift
- Frida Kahlo

### 4.2 Required Places (minimum)
- Eiffel Tower
- Great Wall of China
- Taj Mahal
- Grand Canyon
- Machu Picchu
- Colosseum
- Hagia Sophia
- Statue of Liberty
- Pyramids of Giza
- Mount Everest

### 4.3 Additional Entities
The system will ingest 10 more people and 10 more places beyond the required set to meet the
minimum of 20 each. Suggested additions:

Additional People:
- Isaac Newton
- Charles Darwin
- Cleopatra
- Napoleon Bonaparte
- Mahatma Gandhi
- Nelson Mandela
- Aristotle
- Vincent van Gogh
- Wolfgang Amadeus Mozart
- Stephen Hawking

Additional Places:
- Stonehenge
- Angkor Wat
- Petra
- Acropolis of Athens
- Chichen Itza
- Niagara Falls
- Victoria Falls
- Amazon Rainforest
- Sahara Desert
- Venice

---

## 5. System Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Classifier   │  ← Keyword/rule-based: person / place / both
└─────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           ChromaDB Vector Stores        │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │  People Store│  │  Places Store    │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Retrieved Chunks   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Ollama llama3.2    │  ← Generate grounded answer
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Streamlit UI       │  ← Display answer + optional source chunks
└─────────────────────┘
```

---

## 6. Module Breakdown

### 6.1 `ingest.py` — Data Ingestion
- Use the `wikipedia` Python library to fetch pages by title
- For each entity, extract: title, summary, full text, URL
- Save raw data to SQLite (`wikipedia.db`) with fields:
  - `id`, `title`, `entity_type` (person/place), `summary`, `full_text`, `url`, `ingested_at`
- Handle errors gracefully: if a page is not found or ambiguous, log and skip
- Do not re-ingest entities that already exist in the DB

### 6.2 `chunker.py` — Text Chunking
- Strategy: **fixed-size chunks with overlap**
  - Chunk size: 500 tokens (approx. 400 words)
  - Overlap: 50 tokens (approx. 40 words)
- Split on sentence boundaries where possible to avoid mid-sentence cuts
- Each chunk carries metadata: `entity_title`, `entity_type`, `chunk_index`, `source_url`
- Designed to handle large documents gracefully

### 6.3 `embedder.py` — Embedding Generation
- Use Ollama's `nomic-embed-text` model to generate embeddings locally
- Embed each chunk as a string
- Return embedding vectors as lists of floats
- No external API calls allowed

### 6.4 `vector_store.py` — Vector Store Management

**Design Choice: Option A — Two Separate ChromaDB Collections**

Rationale:
- People and places are semantically distinct domains. Keeping them separate avoids
  cross-domain noise in retrieval results.
- When the query is clearly about a person, searching only the people collection improves
  precision and reduces irrelevant results.
- Simpler query logic: no need for metadata filtering syntax, just select the right collection.
- Tradeoff: slightly more code to maintain two stores vs. one, but the retrieval quality
  benefit justifies this.

Implementation:
- Collection 1: `people_store`
- Collection 2: `places_store`
- Each document stored with metadata: `entity_title`, `entity_type`, `chunk_index`, `source_url`
- Persist ChromaDB to disk at `./chroma_db/`

### 6.5 `retriever.py` — Query Classification + Retrieval
- **Query Classifier** (rule-based):
  - Maintain keyword lists for known people names and place names
  - If query contains a known person name → search `people_store`
  - If query contains a known place name → search `places_store`
  - If query contains both, or is ambiguous → search both stores and merge results
  - Additional keywords: "who", "person", "born", "discovered" → likely person
  - Additional keywords: "where", "located", "built", "country" → likely place
- Retrieve top-5 most relevant chunks from the selected store(s)
- Return chunks with their metadata for optional display

### 6.6 `generator.py` — Answer Generation
- Build a prompt template:
  ```
  You are a knowledgeable assistant. Answer the question using ONLY the context below.
  If the answer is not in the context, say "I don't know based on available information."

  Context:
  {retrieved_chunks}

  Question: {user_query}
  Answer:
  ```
- Call Ollama API locally (http://localhost:11434) with `llama3.2`
- Return the generated answer as a string
- Do not hallucinate — if context is empty or irrelevant, return "I don't know"

### 6.7 `app.py` — Streamlit Chat Interface
- Chat-style UI with message history
- Input box for user questions
- Display assistant answer
- Expandable section: "View source chunks" (optional, toggleable)
- Sidebar with:
  - "Clear conversation" button
  - System status indicators (Ollama running, DB loaded)
- Handle loading states with spinners

---

## 7. File Structure

```
wiki-rag/
├── app.py                  # Streamlit chat UI
├── ingest.py               # Wikipedia ingestion + SQLite storage
├── chunker.py              # Text chunking logic
├── embedder.py             # Ollama nomic-embed-text embeddings
├── vector_store.py         # ChromaDB read/write operations
├── retriever.py            # Query classifier + chunk retrieval
├── generator.py            # Ollama llama3.2 answer generation
├── config.py               # Shared constants (chunk size, model names, paths)
├── wikipedia.db            # SQLite database (auto-created)
├── chroma_db/              # ChromaDB persistent storage (auto-created)
├── requirements.txt        # Python dependencies
├── README.md
├── product_prd.md
└── recommendation.md
```

---

## 8. Configuration (`config.py`)

```python
# Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 500        # tokens (approximate word count)
CHUNK_OVERLAP = 50      # tokens

# Retrieval
TOP_K = 5               # number of chunks to retrieve

# Storage
SQLITE_DB_PATH = "./wikipedia.db"
CHROMA_DB_PATH = "./chroma_db"
PEOPLE_COLLECTION = "people_store"
PLACES_COLLECTION = "places_store"
```

---

## 9. Example Queries the System Must Handle

### Person queries
- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Why is Nikola Tesla famous?"
- "What is Frida Kahlo known for?"

### Place queries
- "Where is the Eiffel Tower located?"
- "What was the Colosseum used for?"
- "Where is Mount Everest?"

### Mixed / comparison queries
- "Which famous place is located in Turkey?"
- "Compare Albert Einstein and Nikola Tesla"
- "Compare the Eiffel Tower and the Statue of Liberty"

### Failure cases (must return "I don't know")
- "Who is the president of Mars?"
- "Tell me about John Doe"

---

## 10. Non-Functional Requirements

- The system must run fully offline after initial model download
- Ingestion of 40 entities should complete in under 10 minutes
- Query-to-answer latency target: under 30 seconds on a standard laptop
- No external LLM or embedding API calls at runtime

---

## 11. Out of Scope

- Multi-language support
- User authentication
- Cloud deployment
- Real-time Wikipedia updates
