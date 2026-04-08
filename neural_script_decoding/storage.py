import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    original_file_name TEXT,
    selected_engine TEXT,
    raw_text TEXT NOT NULL,
    corrected_text TEXT NOT NULL,
    pdf_file_name TEXT,
    preview_file_name TEXT,
    overlay_file_name TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_db(database_path):
    db_path = Path(database_path)
    if db_path.parent and str(db_path.parent) != ".":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(SCHEMA)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(ocr_runs)").fetchall()]
        if "original_file_name" not in columns:
            connection.execute("ALTER TABLE ocr_runs ADD COLUMN original_file_name TEXT")
        if "preview_file_name" not in columns:
            connection.execute("ALTER TABLE ocr_runs ADD COLUMN preview_file_name TEXT")
        if "overlay_file_name" not in columns:
            connection.execute("ALTER TABLE ocr_runs ADD COLUMN overlay_file_name TEXT")
        connection.commit()


def save_run(database_path, payload, pdf_path=None, preview_path=None, overlay_path=None):
    created_at = datetime.now(timezone.utc).isoformat()
    pdf_file_name = os.path.basename(pdf_path) if pdf_path else None
    preview_file_name = os.path.basename(preview_path) if preview_path else None
    overlay_file_name = os.path.basename(overlay_path) if overlay_path else None
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO ocr_runs (file_name, original_file_name, selected_engine, raw_text, corrected_text, pdf_file_name, preview_file_name, overlay_file_name, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("file"),
                payload.get("original_file"),
                payload.get("selected_engine"),
                payload.get("raw_text", ""),
                payload.get("corrected_text", ""),
                pdf_file_name,
                preview_file_name,
                overlay_file_name,
                json.dumps(payload, ensure_ascii=False),
                created_at,
            ),
        )
        connection.commit()
        return cursor.lastrowid


def list_runs(database_path, limit=50, query=None, engine=None):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        sql = """
            SELECT id, file_name, original_file_name, selected_engine, corrected_text, pdf_file_name, preview_file_name, overlay_file_name, created_at
            FROM ocr_runs
            WHERE 1=1
        """
        params = []

        if query:
            sql += """
                AND (
                    file_name LIKE ?
                    OR original_file_name LIKE ?
                    OR corrected_text LIKE ?
                    OR raw_text LIKE ?
                )
            """
            like = f"%{query}%"
            params.extend([like, like, like, like])

        if engine:
            sql += " AND selected_engine = ?"
            params.append(engine)

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]


def get_run(database_path, run_id):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, file_name, original_file_name, selected_engine, raw_text, corrected_text, pdf_file_name, preview_file_name, overlay_file_name, payload_json, created_at
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


def delete_run(database_path, run_id):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT id, pdf_file_name, preview_file_name, overlay_file_name FROM ocr_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        connection.execute("DELETE FROM ocr_runs WHERE id = ?", (run_id,))
        connection.commit()
        return dict(row)


def prune_old_runs(database_path, keep_limit):
    if keep_limit <= 0:
        return []

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, pdf_file_name, preview_file_name, overlay_file_name
            FROM ocr_runs
            WHERE id NOT IN (
                SELECT id
                FROM ocr_runs
                ORDER BY id DESC
                LIMIT ?
            )
            """,
            (keep_limit,),
        ).fetchall()
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        connection.executemany("DELETE FROM ocr_runs WHERE id = ?", [(run_id,) for run_id in ids])
        connection.commit()
        return [dict(row) for row in rows]
