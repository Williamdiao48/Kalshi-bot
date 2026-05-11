"""Persistent deduplication state backed by SQLite.

Tracks which document IDs have already been processed so reruns and
polling loops don't re-surface the same news events.
"""

import sqlite3
import logging
from pathlib import Path

from .db import open_db, STATE_DB

_DEFAULT_DB_PATH = STATE_DB


class SeenDocuments:
    """Thread-safe SQLite store for tracking processed document IDs.

    Usage:
        seen = SeenDocuments()
        new_docs = [d for d in docs if not seen.contains(d["document_number"])]
        # process new_docs ...
        seen.mark_many(d["document_number"] for d in new_docs)
    """

    def __init__(self, db_path: Path | str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn = open_db(self._db_path)
        self._init_schema()
        logging.debug("SeenDocuments store opened at %s", self._db_path)

    def _init_schema(self) -> None:
        # Connection is autocommit (isolation_level=None); no commit() needed.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_documents (
                doc_id   TEXT PRIMARY KEY,
                source   TEXT NOT NULL DEFAULT '',
                seen_at  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    def contains(self, doc_id: str) -> bool:
        """Return True if this document ID has already been processed."""
        row = self._conn.execute(
            "SELECT 1 FROM seen_documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def mark(self, doc_id: str, source: str = "") -> None:
        """Record a single document ID as processed."""
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_documents (doc_id, source) VALUES (?, ?)",
            (doc_id, source),
        )

    def mark_many(self, doc_ids: list[str], source: str = "") -> None:
        """Record multiple document IDs as processed."""
        self._conn.executemany(
            "INSERT OR IGNORE INTO seen_documents (doc_id, source) VALUES (?, ?)",
            [(doc_id, source) for doc_id in doc_ids],
        )

    def filter_new(
        self, docs: list[dict], id_field: str = "document_number"
    ) -> list[dict]:
        """Return only documents whose ID has not been seen before.

        Uses a single batched SQL query instead of N individual lookups.
        Documents missing the id_field are treated as new and logged as a warning.

        Args:
            docs:     List of document dicts.
            id_field: Key in each dict that holds the unique document ID.

        Returns:
            Subset of docs that are new (unseen).
        """
        if not docs:
            return []
        valid: list[tuple[dict, str]] = []
        for d in docs:
            raw = d.get(id_field)
            doc_id = str(raw) if raw is not None else ""
            if not doc_id:
                logging.warning(
                    "filter_new: document missing '%s' field — skipping dedup check, treating as new",
                    id_field,
                )
                valid.append((d, ""))
            else:
                valid.append((d, doc_id))

        ids_to_check = [doc_id for _, doc_id in valid if doc_id]
        seen_ids: set[str] = set()
        if ids_to_check:
            placeholders = ",".join("?" * len(ids_to_check))
            seen_ids = {
                row[0]
                for row in self._conn.execute(
                    f"SELECT doc_id FROM seen_documents WHERE doc_id IN ({placeholders})",
                    ids_to_check,
                ).fetchall()
            }

        return [d for d, doc_id in valid if doc_id not in seen_ids]

    def close(self) -> None:
        self._conn.close()
