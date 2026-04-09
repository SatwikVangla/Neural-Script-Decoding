import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
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

USER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT
);
"""

AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor_username TEXT,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT,
    details_json TEXT,
    created_at TEXT NOT NULL
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

USER_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
    "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active)",
)

AUDIT_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_username ON audit_logs(actor_username)",
)


def init_db(database_path):
    db_path = Path(database_path)
    if db_path.parent and str(db_path.parent) != ".":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(SCHEMA)
        connection.execute(JOB_SCHEMA)
        connection.execute(USER_SCHEMA)
        connection.execute(AUDIT_SCHEMA)
        for statement in INDEXES:
            connection.execute(statement)
        for statement in JOB_INDEXES:
            connection.execute(statement)
        for statement in USER_INDEXES:
            connection.execute(statement)
        for statement in AUDIT_INDEXES:
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


def prune_jobs(database_path, *, keep_limit, retention_days):
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        stale_rows = connection.execute(
            """
            SELECT *
            FROM ocr_jobs
            WHERE status IN ('completed', 'failed', 'canceled')
              AND datetime(created_at) < datetime(?)
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        overflow_rows = connection.execute(
            """
            SELECT *
            FROM ocr_jobs
            WHERE status IN ('completed', 'failed', 'canceled')
              AND id NOT IN (
                SELECT id
                FROM ocr_jobs
                ORDER BY created_at DESC
                LIMIT ?
              )
            """,
            (keep_limit,),
        ).fetchall()

        rows_by_id = {row["id"]: dict(row) for row in stale_rows}
        rows_by_id.update({row["id"]: dict(row) for row in overflow_rows})
        if not rows_by_id:
            return []

        connection.executemany("DELETE FROM ocr_jobs WHERE id = ?", [(job_id,) for job_id in rows_by_id])
        connection.commit()
        return [_deserialize_job(row) for row in rows_by_id.values()]


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


def upsert_user(database_path, *, username, password_hash, is_active=True):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        existing = connection.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?, is_active = ?, updated_at = ?
                WHERE username = ?
                """,
                (password_hash, int(is_active), now, username),
            )
        else:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, password_hash, int(is_active), now, now),
            )
        connection.commit()
    return get_user_by_username(database_path, username)


def get_user_by_username(database_path, username):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, username, password_hash, is_active, created_at, updated_at, last_login_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None


def list_users(database_path):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, username, is_active, created_at, updated_at, last_login_at
            FROM users
            ORDER BY username ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def update_user_password(database_path, *, username, password_hash):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?, updated_at = ?
            WHERE username = ?
            """,
            (password_hash, now, username),
        )
        connection.commit()
    return get_user_by_username(database_path, username)


def touch_user_login(database_path, username):
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE users SET last_login_at = ?, updated_at = ? WHERE username = ?",
            (now, now, username),
        )
        connection.commit()
    return get_user_by_username(database_path, username)


def create_audit_log(database_path, *, actor_username, action, target_type, target_id=None, details=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO audit_logs (actor_username, action, target_type, target_id, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                actor_username,
                action,
                target_type,
                target_id,
                json.dumps(details or {}, ensure_ascii=False),
                created_at,
            ),
        )
        connection.commit()


def list_audit_logs(database_path, limit=50):
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, actor_username, action, target_type, target_id, details_json, created_at
            FROM audit_logs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_deserialize_audit_log(dict(row)) for row in rows]


def _deserialize_audit_log(row):
    row["details"] = json.loads(row["details_json"]) if row.get("details_json") else {}
    return row
