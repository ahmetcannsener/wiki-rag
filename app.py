import sqlite3
import requests
import streamlit as st

from config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, SQLITE_DB_PATH
from retriever import retrieve
from generator import generate


# ── helpers ──────────────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _db_loaded() -> bool:
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM wikipedia_pages").fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def _status_icon(ok: bool) -> str:
    return "🟢" if ok else "🔴"


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local Wikipedia RAG Assistant",
    page_icon="📚",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    },
)

st.markdown("""
    <style>
        .stDeployButton {display: none;}
    </style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Wiki RAG")
    st.markdown("---")

    ollama_ok = _ollama_running()
    db_ok = _db_loaded()

    st.subheader("System Status")
    st.markdown(f"{_status_icon(ollama_ok)} Ollama ({LLM_MODEL} / {EMBEDDING_MODEL})")
    st.markdown(f"{_status_icon(db_ok)} Wikipedia DB")

    st.markdown("---")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Runs fully locally — no external API calls.")

# ── chat state ────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── title ─────────────────────────────────────────────────────────────────────

st.title("📚 Local Wikipedia RAG Assistant")
st.caption("Ask me anything about famous people and places.")

# ── render history ────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander("View source chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    meta = chunk["metadata"]
                    st.markdown(
                        f"**[{i}] {meta.get('entity_title', '?')}** "
                        f"(chunk {meta.get('chunk_index', '?')}) — "
                        f"[source]({meta.get('source_url', '#')})"
                    )
                    st.text(chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else ""))
                    st.markdown("---")

# ── input ─────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not ollama_ok:
            answer = "Ollama is not running. Please start it with `ollama serve`."
            chunks = []
            st.warning(answer)
        elif not db_ok:
            answer = "The Wikipedia database is empty. Run `python ingest.py` then `python build_index.py` first."
            chunks = []
            st.warning(answer)
        else:
            with st.spinner("Retrieving context…"):
                chunks, category = retrieve(prompt)

            with st.spinner(f"Generating answer using {LLM_MODEL}…"):
                answer = generate(chunks, prompt)

            st.markdown(answer)

            if chunks:
                with st.expander("View source chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        meta = chunk["metadata"]
                        st.markdown(
                            f"**[{i}] {meta.get('entity_title', '?')}** "
                            f"(chunk {meta.get('chunk_index', '?')}) — "
                            f"[source]({meta.get('source_url', '#')})"
                        )
                        st.text(chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else ""))
                        st.markdown("---")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
