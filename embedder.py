import requests

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def _embed_raw(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


# The Ollama build of nomic-embed-text does NOT support the "search_query: /
# search_document: " task prefixes — passing them causes the model to embed
# the prefix only and return the same vector for every distinct input.
# Use raw text for both documents and queries.

def embed_document(text: str) -> list[float]:
    return _embed_raw(text)


def embed_query(text: str) -> list[float]:
    return _embed_raw(text)


def embed(text: str) -> list[float]:
    return _embed_raw(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [_embed_raw(t) for t in texts]
