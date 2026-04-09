import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .services import normalize_result_payload


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

JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr_jobs (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    original_file_name TEXT,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    queue_job_id TEXT,
    error_message TEXT,
    result_payload_json TEXT,
    run_id INTEGER,
    pdf_file_name TEXT,
    preview_file_name TEXT,
    overlay_file_name TEXT,
    paths_json TEXT NOT NULL,
    system_status_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    canceled_at TEXT
);
"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_ocr_runs_created_at ON ocr_runs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_runs_selected_engine ON ocr_runs(selected_engine)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_runs_file_name ON ocr_runs(file_name)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_runs_original_file_name ON ocr_runs(original_file_name)",
)

JOB_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_ocr_jobs_created_at ON ocr_jobs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr_jobs(status)",
)


def init_db(database_path):
    db_path = Path(database_path)
    if db_path.parent and str(db_path.parent) != ".":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(SCHEMA)
        connection.execute(JOB_SCHEMA)
        for statement in INDEXES:
            connection.execute(statement)
        for statement in JOB_INDEXES:
            connection.execute(statement)
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
        result["payload"] = normalize_result_payload(json.loads(result["payload_json"]), created_at=result["created_at"])
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


def create_job(database_path, *, job_id, paths, system_status, backend, queue_job_id=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO ocr_jobs (
                id, file_name, original_file_name, backend, status, attempt_count, queue_job_id,
                paths_json, system_status_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                paths.get("filename"),
                paths.get("original_name"),
                backend,
                "queued",
                1,
                queue_job_id,
                json.dumps(paths, ensure_ascii=False),
                json.dumps(system_status, ensure_ascii=False),
                created_at,
            ),
        )
        connection.commit()
    return get_job(database_path, job_id)


def get_job(database_path, job_id):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM ocr_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_job(dict(row))


def list_jobs(database_path, limit=50):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM ocr_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_deserialize_job(dict(row)) for row in rows]


def update_job(database_path, job_id, **fields):
    if not fields:
        return get_job(database_path, job_id)

    prepared = {}
    for key, value in fields.items():
        if key in {"paths", "system_status", "result_payload"} and value is not None:
            mapping = {
                "paths": "paths_json",
                "system_status": "system_status_json",
                "result_payload": "result_payload_json",
            }
            prepared[mapping[key]] = json.dumps(value, ensure_ascii=False)
        else:
            prepared[key] = value

    assignments = ", ".join(f"{field} = ?" for field in prepared)
    params = list(prepared.values()) + [job_id]
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"UPDATE ocr_jobs SET {assignments} WHERE id = ?", params)
        connection.commit()
    return get_job(database_path, job_id)


def summarize_jobs(database_path):
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM ocr_jobs GROUP BY status"
        ).fetchall()
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "canceled": 0}
    for status, count in rows:
        counts[status] = count
    counts["total"] = sum(counts.values())
    return counts


def _deserialize_job(row):
    row["paths"] = json.loads(row["paths_json"])
    row["system_status"] = json.loads(row["system_status_json"])
    if row.get("result_payload_json"):
        row["result_payload"] = normalize_result_payload(
            json.loads(row["result_payload_json"]),
            created_at=row["completed_at"] or row["created_at"],
        )
    else:
        row["result_payload"] = None
    return row
