"""SQLite connection factory shared by all persistent store classes."""

import sqlite3
from pathlib import Path


def open_db(path: Path | str) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection in autocommit mode."""
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
