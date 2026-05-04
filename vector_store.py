import chromadb

from config import CHROMA_DB_PATH, PEOPLE_COLLECTION, PLACES_COLLECTION
from chunker import Chunk
from embedder import embed_document


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_collection(client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def collection_for_type(client: chromadb.ClientAPI, entity_type: str) -> chromadb.Collection:
    if entity_type == "person":
        return _get_collection(client, PEOPLE_COLLECTION)
    return _get_collection(client, PLACES_COLLECTION)


def upsert_chunks(chunks: list[Chunk], entity_type: str) -> None:
    if not chunks:
        return

    client = _get_client()
    collection = collection_for_type(client, entity_type)

    texts = [c.text for c in chunks]
    # Prepend the entity title to anchor the embedding to the specific entity.
    # Without this, long 500-word chunks of different famous places cluster
    # together and short queries can't distinguish them.
    embed_inputs = [f"{c.entity_title}\n{c.text}" for c in chunks]
    embeddings = [embed_document(t) for t in embed_inputs]

    ids = [
        f"{c.entity_title}__{c.chunk_index}".replace(" ", "_")
        for c in chunks
    ]
    metadatas = [
        {
            "entity_title": c.entity_title,
            "entity_type": c.entity_type,
            "chunk_index": c.chunk_index,
            "source_url": c.source_url,
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int,
) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def get_people_collection() -> chromadb.Collection:
    client = _get_client()
    return _get_collection(client, PEOPLE_COLLECTION)


def get_places_collection() -> chromadb.Collection:
    client = _get_client()
    return _get_collection(client, PLACES_COLLECTION)
