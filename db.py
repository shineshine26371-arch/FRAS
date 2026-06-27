"""SQLite database setup and CRUD helpers for FRAS."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("fras.db")
STORAGE_DIR = Path("storage/files")
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a row-factory SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist, and apply additive migrations."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                upload_date TEXT NOT NULL,
                file_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                last_error TEXT,
                file_hash TEXT,
                category TEXT
            );

            CREATE TABLE IF NOT EXISTS extracted_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                summary TEXT,
                key_points TEXT,
                document_type TEXT,
                entities TEXT,
                raw_response TEXT,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                comment TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
            );
            """
        )
        # Additive migrations
        _migrate_add_last_error(conn)
        _migrate_add_file_hash(conn)
        _migrate_add_category(conn)
        _migrate_add_risks(conn)
        _migrate_add_flagged(conn)
        _migrate_add_structured_data(conn)
        _migrate_add_reviewed(conn)


def _migrate_add_last_error(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE files ADD COLUMN last_error TEXT")
    except sqlite3.OperationalError:
        pass


def _migrate_add_file_hash(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE files ADD COLUMN file_hash TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")
    except sqlite3.OperationalError:
        pass


def _migrate_add_category(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ALTER TABLE files ADD COLUMN category TEXT")
    except sqlite3.OperationalError:
        pass


def _migrate_add_risks(conn: sqlite3.Connection) -> None:
    """Add risks column to extracted_data if missing."""
    try:
        conn.execute("ALTER TABLE extracted_data ADD COLUMN risks TEXT")
    except sqlite3.OperationalError:
        pass


def _migrate_add_flagged(conn: sqlite3.Connection) -> None:
    """Add flagged column to files if missing."""
    try:
        conn.execute("ALTER TABLE files ADD COLUMN flagged INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass


def _migrate_add_structured_data(conn: sqlite3.Connection) -> None:
    """Add structured_data column to extracted_data if missing."""
    try:
        conn.execute("ALTER TABLE extracted_data ADD COLUMN structured_data TEXT")
    except sqlite3.OperationalError:
        pass


def _migrate_add_reviewed(conn: sqlite3.Connection) -> None:
    """Add reviewed column to files if missing."""
    try:
        conn.execute("ALTER TABLE files ADD COLUMN reviewed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass


# ---------------------------------------------------------------------------
# Files CRUD
# ---------------------------------------------------------------------------

def insert_file(filename: str, filepath: str, file_type: str) -> int:
    upload_date = datetime.now().isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO files (filename, filepath, upload_date, file_type, status)
            VALUES (?, ?, ?, ?, 'uploaded')
            """,
            (filename, filepath, upload_date, file_type),
        )
        return cursor.lastrowid


def update_file_status(file_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE files SET status = ? WHERE id = ?", (status, file_id))


def get_file(file_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        return dict(row) if row else None


def get_all_files() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM files ORDER BY upload_date DESC").fetchall()
        return [dict(row) for row in rows]


def get_files_by_ids(file_ids: list[int]) -> list[dict[str, Any]]:
    """Fetch multiple files by their IDs."""
    if not file_ids:
        return []
    placeholders = ",".join("?" for _ in file_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM files WHERE id IN ({placeholders}) ORDER BY upload_date DESC",
            file_ids,
        ).fetchall()
        return [dict(row) for row in rows]


def search_files(query: str) -> list[dict[str, Any]]:
    pattern = f"%{query.lower()}%"
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT f.*, e.summary, e.key_points
            FROM files f
            LEFT JOIN extracted_data e ON f.id = e.file_id
            WHERE LOWER(f.filename) LIKE ?
               OR LOWER(e.summary) LIKE ?
               OR LOWER(e.key_points) LIKE ?
            ORDER BY f.upload_date DESC
            """,
            (pattern, pattern, pattern),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_extracted_data(
    file_id: int,
    summary: str | None,
    key_points: str | None,
    document_type: str | None,
    entities: str | None,
    raw_response: str | None,
    risks: str | None = None,
    structured_data: str | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO extracted_data (file_id, summary, key_points, document_type, entities, raw_response, risks, structured_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (file_id, summary, key_points, document_type, entities, raw_response, risks, structured_data),
        )


def update_extracted_risks(file_id: int, risks: str | None) -> None:
    """Update risks for an existing extracted_data row."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE extracted_data SET risks = ? WHERE file_id = ?",
            (risks, file_id),
        )


def update_extracted_structured_data(file_id: int, structured_data: str | None) -> None:
    """Update structured_data for an existing extracted_data row."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE extracted_data SET structured_data = ? WHERE file_id = ?",
            (structured_data, file_id),
        )


def get_extracted_data(file_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM extracted_data WHERE file_id = ?", (file_id,)
        ).fetchone()
        return dict(row) if row else None


def get_files_with_extracted_data() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT f.*, e.summary, e.key_points, e.document_type, e.entities
            FROM files f
            LEFT JOIN extracted_data e ON f.id = e.file_id
            ORDER BY f.upload_date DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def update_file_error(file_id: int, error_message: str | None) -> None:
    truncated = (error_message or "")[:500]
    with get_connection() as conn:
        conn.execute(
            "UPDATE files SET last_error = ? WHERE id = ?",
            (truncated, file_id),
        )


def delete_file(file_id: int) -> None:
    file = get_file(file_id)
    if not file:
        return
    filepath = file["filepath"]
    with get_connection() as conn:
        conn.execute("DELETE FROM comments WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM extracted_data WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    try:
        Path(filepath).unlink(missing_ok=True)
    except Exception:
        pass


def get_file_by_hash(file_hash: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM files WHERE file_hash = ? LIMIT 1",
            (file_hash,),
        ).fetchone()
        return dict(row) if row else None


def update_file_category(file_id: int, category: str | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE files SET category = ? WHERE id = ?",
            (category, file_id),
        )


def update_file_flagged(file_id: int, flagged: bool) -> None:
    """Set the flagged status of a file."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE files SET flagged = ? WHERE id = ?",
            (1 if flagged else 0, file_id),
        )


def update_file_reviewed(file_id: int, reviewed: bool) -> None:
    """Set the reviewed status of a file."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE files SET reviewed = ? WHERE id = ?",
            (1 if reviewed else 0, file_id),
        )


# ---------------------------------------------------------------------------
# Comments CRUD
# ---------------------------------------------------------------------------

def add_comment(file_id: int, comment: str) -> None:
    timestamp = datetime.now().isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO comments (file_id, comment, timestamp) VALUES (?, ?, ?)",
            (file_id, comment, timestamp),
        )


def get_comments(file_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM comments WHERE file_id = ? ORDER BY timestamp ASC",
            (file_id,),
        ).fetchall()
        return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Dashboard Aggregations
# ---------------------------------------------------------------------------

def get_dashboard_stats() -> dict[str, Any]:
    """Return aggregated stats for the dashboard."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()["cnt"]
        with_risks = conn.execute(
            "SELECT COUNT(*) as cnt FROM extracted_data WHERE risks IS NOT NULL AND risks != '[]' AND risks != ''"
        ).fetchone()["cnt"]
        flagged = conn.execute(
            "SELECT COUNT(*) as cnt FROM files WHERE flagged = 1"
        ).fetchone()["cnt"]

        # Documents by type
        type_rows = conn.execute(
            "SELECT file_type, COUNT(*) as cnt FROM files GROUP BY file_type ORDER BY cnt DESC"
        ).fetchall()
        by_type = [{"type": r["file_type"], "count": r["cnt"]} for r in type_rows]

        # Uploads over time (by date)
        date_rows = conn.execute(
            "SELECT DATE(upload_date) as day, COUNT(*) as cnt FROM files GROUP BY day ORDER BY day ASC"
        ).fetchall()
        uploads_over_time = [{"date": r["day"], "count": r["cnt"]} for r in date_rows]

        return {
            "total": total,
            "with_risks": with_risks,
            "flagged": flagged,
            "by_type": by_type,
            "uploads_over_time": uploads_over_time,
        }