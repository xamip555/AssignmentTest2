from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "rag.db"


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                upload_time TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                vector_id INTEGER NOT NULL UNIQUE,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )


def create_document(name: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    upload_time = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO documents (name, upload_time) VALUES (?, ?)",
            (name, upload_time),
        )
        return int(cursor.lastrowid)


def add_chunks(
    doc_id: int,
    chunk_records: Iterable[tuple[int, str, int]],
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO chunks (doc_id, chunk_index, text, vector_id)
            VALUES (?, ?, ?, ?)
            """,
            [(doc_id, chunk_index, text, vector_id) for chunk_index, text, vector_id in chunk_records],
        )


def list_documents(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, upload_time FROM documents ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def document_exists(doc_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE id = ?",
            (doc_id,),
        ).fetchone()
    return row is not None


def get_vector_ids_for_document(doc_id: int, db_path: Path = DEFAULT_DB_PATH) -> list[int]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT vector_id FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        ).fetchall()
    return [int(row["vector_id"]) for row in rows]


def delete_document(doc_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def fetch_chunks_by_vector_ids(
    vector_ids: list[int], db_path: Path = DEFAULT_DB_PATH
) -> list[dict]:
    if not vector_ids:
        return []

    placeholders = ",".join("?" for _ in vector_ids)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                c.id,
                c.doc_id,
                c.chunk_index,
                c.text,
                c.vector_id,
                d.name AS doc_name
            FROM chunks c
            JOIN documents d ON d.id = c.doc_id
            WHERE c.vector_id IN ({placeholders})
            """,
            vector_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def get_max_vector_id(db_path: Path = DEFAULT_DB_PATH) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COALESCE(MAX(vector_id), 0) AS max_id FROM chunks").fetchone()
    return int(row["max_id"])
