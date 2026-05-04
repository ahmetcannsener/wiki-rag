"""
One-off script to fetch and store a single missing entity.
Usage: python add_missing.py
"""
import sqlite3
import logging

from config import SQLITE_DB_PATH
from ingest import init_db, ingest_entity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MISSING = [
    ("Nelson Mandela", "person"),
]

conn = sqlite3.connect(SQLITE_DB_PATH)
init_db(conn)

for title, entity_type in MISSING:
    ingest_entity(conn, title, entity_type)

conn.close()
