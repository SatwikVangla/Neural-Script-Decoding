import json
import os
import sqlite3
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    original_file_name TEXT,
    selected_engine TEXT,
    raw_text TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    pdf_file_name TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_db(database_path):
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(SCHEMA)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(ocr_runs)").fetchall()]
        if "original_file_name" not in columns:
            connection.execute("ALTER TABLE ocr_runs ADD COLUMN original_file_name TEXT")
        connection.commit()


def save_run(database_path, payload, pdf_path=None):
    created_at = datetime.now(timezone.utc).isoformat()
    pdf_file_name = os.path.basename(pdf_path) if pdf_path else None
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO ocr_runs (file_name, original_file_name, selected_engine, raw_text, corrected_text, pdf_file_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("file"),
                payload.get("original_file"),
                payload.get("selected_engine"),
                payload.get("raw_text", ""),
                payload.get("corrected_text", ""),
                pdf_file_name,
                json.dumps(payload, ensure_ascii=False),
                created_at,
            ),
        )
        connection.commit()
        return cursor.lastrowid


def list_runs(database_path, limit=50):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, file_name, original_file_name, selected_engine, corrected_text, pdf_file_name, created_at
            FROM ocr_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_run(database_path, run_id):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, file_name, original_file_name, selected_engine, raw_text, corrected_text, pdf_file_name, payload_json, created_at
            FROM ocr_runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result["payload_json"])
        return result
