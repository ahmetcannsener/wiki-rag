"""
Run after ingest.py.
Reads all rows from SQLite, chunks them, embeds each chunk, and upserts
into the appropriate ChromaDB collection.
"""
import sqlite3
import logging

from config import SQLITE_DB_PATH
from chunker import chunk_text
from vector_store import upsert_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_index() -> None:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    rows = conn.execute(
        "SELECT title, entity_type, full_text, url FROM wikipedia_pages"
    ).fetchall()
    conn.close()

    logger.info("Building index for %d entities…", len(rows))

    for title, entity_type, full_text, url in rows:
        logger.info("Chunking + embedding '%s'…", title)
        chunks = chunk_text(
            text=full_text,
            entity_title=title,
            entity_type=entity_type,
            source_url=url,
        )
        logger.info("  %d chunks → upserting…", len(chunks))
        upsert_chunks(chunks, entity_type)

    logger.info("Index build complete.")


if __name__ == "__main__":
    build_index()
