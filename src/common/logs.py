"""SQLite logging utilities for the Kaabil RAG service."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


_DB_PATH = Path("data/logs.db")


def init_db(db_path: str = "data/logs.db") -> None:
    """Create the logging database and schema if missing."""

    global _DB_PATH

    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(_DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS turns(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              ts TEXT,
              session_id TEXT,
              ip TEXT,
              user_agent TEXT,
              query TEXT,
              answer TEXT,
              used_evidence INTEGER,
              citations TEXT,
              latency_ms INTEGER,
              provider TEXT,
              model TEXT,
              topk INTEGER,
              threshold REAL
            );
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts);
            """
        )
        connection.commit()


def log_turn(**kwargs: Any) -> None:
    """Insert a conversation turn into the logging database."""

    columns = list(kwargs.keys())
    if not columns:
        return

    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    values = [kwargs[column] for column in columns]

    with sqlite3.connect(_DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute(
            f"INSERT INTO turns ({column_sql}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
