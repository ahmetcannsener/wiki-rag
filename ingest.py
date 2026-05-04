import sqlite3
import logging
import time
from datetime import datetime

import wikipedia

from config import SQLITE_DB_PATH, PEOPLE, PLACES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FETCH_DELAY = 2       # seconds between each successful fetch
RETRY_DELAY = 5       # seconds to wait before a retry
MAX_RETRIES = 3


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wikipedia_pages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT    NOT NULL UNIQUE,
            entity_type  TEXT    NOT NULL,
            summary      TEXT,
            full_text    TEXT,
            url          TEXT,
            ingested_at  TEXT    NOT NULL
        )
    """)
    conn.commit()


def already_ingested(conn: sqlite3.Connection, title: str) -> bool:
    row = conn.execute(
        "SELECT id FROM wikipedia_pages WHERE title = ?", (title,)
    ).fetchone()
    return row is not None


def _fetch_once(title: str) -> dict | None:
    """Single attempt to fetch a Wikipedia page by title."""
    # Confirm the page exists via search before fetching
    results = wikipedia.search(title, results=3)
    if not results:
        logger.warning("[%s] search() returned no results — skipping", title)
        return None

    # Pick the best-matching search result
    best = results[0]
    if title.lower() not in best.lower() and best.lower() not in title.lower():
        logger.info("[%s] search() best match: '%s'", title, best)

    try:
        page = wikipedia.page(best, auto_suggest=False)
        return {
            "title": page.title,
            "summary": page.summary,
            "full_text": page.content,
            "url": page.url,
        }
    except wikipedia.exceptions.DisambiguationError as e:
        logger.warning(
            "[%s] DisambiguationError — trying first option '%s'", title, e.options[0]
        )
        page = wikipedia.page(e.options[0], auto_suggest=False)
        return {
            "title": page.title,
            "summary": page.summary,
            "full_text": page.content,
            "url": page.url,
        }


def fetch_page(title: str) -> dict | None:
    """Fetch with retry logic. Returns None only after all retries fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = _fetch_once(title)
            return data
        except wikipedia.exceptions.PageError:
            logger.error(
                "[%s] PageError on attempt %d/%d — page not found",
                title, attempt, MAX_RETRIES,
            )
            return None  # no point retrying a missing page
        except Exception as exc:
            logger.error(
                "[%s] %s on attempt %d/%d: %s",
                title, type(exc).__name__, attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                logger.info("[%s] Retrying in %ds…", title, RETRY_DELAY)
                time.sleep(RETRY_DELAY)
            else:
                logger.error("[%s] All %d attempts failed — skipping", title, MAX_RETRIES)
                return None

    return None


def ingest_entity(conn: sqlite3.Connection, title: str, entity_type: str) -> None:
    if already_ingested(conn, title):
        logger.info("Skipping '%s' — already in DB", title)
        return

    logger.info("Fetching '%s' (%s)…", title, entity_type)
    data = fetch_page(title)
    if data is None:
        return

    conn.execute(
        """
        INSERT INTO wikipedia_pages (title, entity_type, summary, full_text, url, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            data["title"],
            entity_type,
            data["summary"],
            data["full_text"],
            data["url"],
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    logger.info("Stored '%s'", data["title"])

    time.sleep(FETCH_DELAY)


def run_ingestion() -> None:
    conn = sqlite3.connect(SQLITE_DB_PATH)
    init_db(conn)

    for name in PEOPLE:
        ingest_entity(conn, name, "person")

    for place in PLACES:
        ingest_entity(conn, place, "place")

    conn.close()
    logger.info("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
