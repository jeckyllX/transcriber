"""Persistent SQLite Storage Module with Full-Text Search (FTS5).

Manages thread-safe persistence for transcription jobs, batches, and dialogue turns,
enabling historical job retrieval across server restarts and fast keyword search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from config import settings

logger = logging.getLogger(__name__)


class JobStore:
    """Thread-safe SQLite storage engine for jobs, batches, and full-text search."""

    def __init__(self, db_path: Path | str | None = None, enabled: bool = True) -> None:
        self.db_path = Path(db_path or settings.storage_db_path)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a configured SQLite connection with row access and WAL journal."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode; transactions managed via BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row
        try:
            # Set connection-level pragmas for concurrent access
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA foreign_keys = ON;")
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize database schema, indexes, and FTS5 virtual tables."""
        if not self.enabled:
            return

        with self._lock, self._get_connection() as conn:
            # Persistent database pragma
            conn.execute("PRAGMA journal_mode = WAL;")

            conn.execute("BEGIN IMMEDIATE;")
            try:
                # 1. Jobs Table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        batch_id TEXT,
                        filename TEXT NOT NULL,
                        status TEXT NOT NULL,
                        progress INTEGER NOT NULL DEFAULT 0,
                        message TEXT,
                        duration REAL DEFAULT 0.0,
                        processing_time REAL DEFAULT 0.0,
                        engine_used TEXT,
                        model_used TEXT,
                        language TEXT,
                        num_speakers INTEGER DEFAULT 0,
                        audio_enhanced INTEGER DEFAULT 0,
                        result_json TEXT,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                # Indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs(batch_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);")

                # 2. Batches Table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS batches (
                        batch_id TEXT PRIMARY KEY,
                        total_files INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'queued',
                        overall_progress INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                # 3. Full-Text Search FTS5 Virtual Table
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
                        job_id UNINDEXED,
                        filename,
                        text,
                        summary,
                        tokenize = 'unicode61'
                    );
                    """
                )

                # Recover orphaned jobs from unclean shutdown/restart
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        error = 'Interrupted by server restart',
                        updated_at = ?
                    WHERE status NOT IN ('completed', 'failed');
                    """,
                    (datetime.now(timezone.utc).isoformat(),),
                )
                if cursor.rowcount > 0:
                    logger.warning("Recovered %d orphaned in-flight jobs as failed", cursor.rowcount)

                conn.execute("COMMIT;")
                logger.info("Initialized SQLite JobStore at %s", self.db_path)
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.exception("Failed to initialize SQLite JobStore: %s", e)
                raise

    def save_job(self, job: Any) -> None:
        """Upsert a Job dataclass into the jobs table and synchronize full-text search."""
        if not self.enabled:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        job_created_at = (
            job.created_at.isoformat()
            if hasattr(job, "created_at") and isinstance(job.created_at, datetime)
            else now_iso
        )

        result_dict = getattr(job, "result", None)
        result_json = json.dumps(result_dict) if result_dict else None

        engine_used = result_dict.get("engine_used") if result_dict else None
        model_used = result_dict.get("model_used") if result_dict else None
        language = result_dict.get("language") if result_dict else None
        num_speakers = result_dict.get("num_speakers", 0) if result_dict else 0
        audio_enhanced = 1 if (result_dict and result_dict.get("audio_enhanced")) else 0

        # Extract text and summary for FTS
        transcript_text = result_dict.get("text", "") if result_dict else ""
        summary_text = result_dict.get("summary", "") if result_dict else ""

        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, batch_id, filename, status, progress, message,
                        duration, processing_time, engine_used, model_used,
                        language, num_speakers, audio_enhanced, result_json,
                        error, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        batch_id = excluded.batch_id,
                        filename = excluded.filename,
                        status = excluded.status,
                        progress = excluded.progress,
                        message = excluded.message,
                        duration = excluded.duration,
                        processing_time = excluded.processing_time,
                        engine_used = excluded.engine_used,
                        model_used = excluded.model_used,
                        language = excluded.language,
                        num_speakers = excluded.num_speakers,
                        audio_enhanced = excluded.audio_enhanced,
                        result_json = excluded.result_json,
                        error = excluded.error,
                        updated_at = excluded.updated_at;
                    """,
                    (
                        job.job_id,
                        getattr(job, "batch_id", None),
                        job.filename,
                        job.status,
                        job.progress,
                        job.message,
                        getattr(job, "duration", 0.0),
                        getattr(job, "processing_time", 0.0),
                        engine_used,
                        model_used,
                        language,
                        num_speakers,
                        audio_enhanced,
                        result_json,
                        getattr(job, "error", None),
                        job_created_at,
                        now_iso,
                    ),
                )

                # Synchronize FTS5 virtual table
                conn.execute("DELETE FROM transcript_fts WHERE job_id = ?;", (job.job_id,))
                if transcript_text or summary_text:
                    conn.execute(
                        "INSERT INTO transcript_fts (job_id, filename, text, summary) VALUES (?, ?, ?, ?);",
                        (job.job_id, job.filename, transcript_text, summary_text),
                    )

                conn.execute("COMMIT;")
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error("Failed to save job %s to database: %s", job.job_id, e)

    def update_job_ai_output(
        self,
        job_id: str,
        action: str,
        text: str,
        summary_level: str | None = None,
    ) -> bool:
        """Update a job's stored AI polish or summary output in SQLite and sync FTS5."""
        if not self.enabled:
            return False

        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                row = conn.execute("SELECT filename, result_json FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK;")
                    return False

                result_data: dict[str, Any] = json.loads(row["result_json"]) if row["result_json"] else {}
                if action == "polish":
                    result_data["polished"] = text
                elif action == "summary":
                    level = summary_level or "bullets"
                    if "summaries" not in result_data or not isinstance(result_data["summaries"], dict):
                        result_data["summaries"] = {}
                    result_data["summaries"][level] = text
                    result_data["summary"] = text
                    result_data["summary_level"] = level

                now_iso = datetime.now(timezone.utc).isoformat()
                new_result_json = json.dumps(result_data)

                conn.execute(
                    """
                    UPDATE jobs
                    SET result_json = ?,
                        updated_at = ?
                    WHERE job_id = ?;
                    """,
                    (new_result_json, now_iso, job_id),
                )

                # Resynchronize FTS5 virtual table
                transcript_text = result_data.get("text", "")
                summary_text = result_data.get("summary", "")
                conn.execute("DELETE FROM transcript_fts WHERE job_id = ?;", (job_id,))
                if transcript_text or summary_text:
                    conn.execute(
                        "INSERT INTO transcript_fts (job_id, filename, text, summary) VALUES (?, ?, ?, ?);",
                        (job_id, row["filename"], transcript_text, summary_text),
                    )

                conn.execute("COMMIT;")
                return True
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error("Failed to update AI output for job %s: %s", job_id, e)
                return False

    def get_job(self, job_id: str) -> Any | None:
        """Retrieve a stored job by ID and restore it as a Job dataclass instance."""
        if not self.enabled:
            return None

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?;", (job_id,)).fetchone()
            if not row:
                return None

            from jobs import Job

            res_data = json.loads(row["result_json"]) if row["result_json"] else None
            created_dt = datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc)

            job = Job(
                job_id=row["job_id"],
                filename=row["filename"],
                status=row["status"],
                progress=row["progress"],
                message=row["message"] or "",
                duration=row["duration"] or 0.0,
                processing_time=row["processing_time"] or 0.0,
                result=res_data,
                error=row["error"],
                batch_id=row["batch_id"],
                created_at=created_dt,
            )
            return job

    def list_jobs(
        self,
        page: int = 1,
        page_size: int = 20,
        status: str | None = None,
        batch_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List historical jobs with pagination and optional status/batch filtering."""
        if not self.enabled:
            return [], 0

        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size

        where_clauses: list[str] = []
        params: list[Any] = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if batch_id:
            where_clauses.append("batch_id = ?")
            params.append(batch_id)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._get_connection() as conn:
            # Count total
            count_row = conn.execute(f"SELECT COUNT(*) as total FROM jobs {where_sql};", params).fetchone()
            total = count_row["total"] if count_row else 0

            # Fetch paginated rows
            rows = conn.execute(
                f"""
                SELECT job_id, batch_id, filename, status, progress, message,
                       duration, processing_time, engine_used, model_used,
                       language, num_speakers, audio_enhanced, error,
                       created_at, updated_at,
                       (result_json IS NOT NULL) as has_result,
                       result_json
                FROM jobs
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
                """,
                params + [page_size, offset],
            ).fetchall()

            items = []
            for r in rows:
                preview_text = ""
                has_summary = False
                has_polish = False
                if r["result_json"]:
                    try:
                        res_obj = json.loads(r["result_json"])
                        raw_text = res_obj.get("text", "")
                        preview_text = (raw_text[:180] + "...") if len(raw_text) > 180 else raw_text
                        has_summary = bool(res_obj.get("summary"))
                        has_polish = bool(res_obj.get("polished"))
                    except Exception:
                        pass

                items.append(
                    {
                        "job_id": r["job_id"],
                        "batch_id": r["batch_id"],
                        "filename": r["filename"],
                        "status": r["status"],
                        "progress": r["progress"],
                        "message": r["message"],
                        "duration": r["duration"],
                        "processing_time": r["processing_time"],
                        "engine_used": r["engine_used"],
                        "model_used": r["model_used"],
                        "language": r["language"],
                        "num_speakers": r["num_speakers"],
                        "audio_enhanced": bool(r["audio_enhanced"]),
                        "has_result": bool(r["has_result"]),
                        "has_summary": has_summary,
                        "has_polish": has_polish,
                        "preview_text": preview_text,
                        "error": r["error"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }
                )

            return items, total

    def search_transcripts(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Perform full-text search across transcripts and summaries using FTS5."""
        if not self.enabled or not query or not query.strip():
            return []

        clean_query = query.strip()
        # Escape double quotes and format as prefix search tokens if standard alphanumeric
        tokens = [f'"{token}*"' for token in clean_query.split() if token]
        fts_match = " ".join(tokens) if tokens else f'"{clean_query}"'

        limit = max(1, min(50, limit))

        with self._get_connection() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT 
                        t.job_id,
                        t.filename,
                        j.created_at,
                        j.duration,
                        j.language,
                        j.num_speakers,
                        j.status,
                        snippet(transcript_fts, 2, '<mark>', '</mark>', '...', 20) AS text_snippet,
                        snippet(transcript_fts, 3, '<mark>', '</mark>', '...', 20) AS summary_snippet,
                        rank
                    FROM transcript_fts t
                    JOIN jobs j ON t.job_id = j.job_id
                    WHERE transcript_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?;
                    """,
                    (fts_match, limit),
                ).fetchall()

                results = []
                for r in rows:
                    results.append(
                        {
                            "job_id": r["job_id"],
                            "filename": r["filename"],
                            "created_at": r["created_at"],
                            "duration": r["duration"],
                            "language": r["language"],
                            "num_speakers": r["num_speakers"],
                            "status": r["status"],
                            "text_snippet": r["text_snippet"] or "",
                            "summary_snippet": r["summary_snippet"] or "",
                            "score": round(abs(r["rank"]), 2),
                        }
                    )
                return results

            except sqlite3.OperationalError as e:
                logger.warning("FTS search syntax error for query '%s': %s", query, e)
                # Fallback to simple LIKE search if complex FTS syntax causes error
                fallback_rows = conn.execute(
                    """
                    SELECT job_id, filename, created_at, duration, language, num_speakers, status
                    FROM jobs
                    WHERE filename LIKE ? OR result_json LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (f"%{clean_query}%", f"%{clean_query}%", limit),
                ).fetchall()

                return [
                    {
                        "job_id": r["job_id"],
                        "filename": r["filename"],
                        "created_at": r["created_at"],
                        "duration": r["duration"],
                        "language": r["language"],
                        "num_speakers": r["num_speakers"],
                        "status": r["status"],
                        "text_snippet": f"Matching keyword '{clean_query}'",
                        "summary_snippet": "",
                        "score": 1.0,
                    }
                    for r in fallback_rows
                ]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its FTS search index from the database."""
        if not self.enabled:
            return False

        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = conn.execute("DELETE FROM jobs WHERE job_id = ?;", (job_id,))
                deleted = cur.rowcount > 0
                conn.execute("DELETE FROM transcript_fts WHERE job_id = ?;", (job_id,))
                conn.execute("COMMIT;")
                return deleted
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error("Failed to delete job %s: %s", job_id, e)
                return False

    def save_batch(
        self,
        batch_id: str,
        total_files: int,
        status: str = "queued",
        overall_progress: int = 0,
        progress: int | None = None,
    ) -> None:
        """Upsert batch metadata."""
        if not self.enabled:
            return

        effective_progress = progress if progress is not None else overall_progress
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO batches (batch_id, total_files, status, overall_progress, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    total_files = excluded.total_files,
                    status = excluded.status,
                    overall_progress = excluded.overall_progress,
                    updated_at = excluded.updated_at;
                """,
                (batch_id, total_files, status, effective_progress, now_iso, now_iso),
            )

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        """Retrieve stored batch details and associated jobs."""
        if not self.enabled:
            return None

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM batches WHERE batch_id = ?;", (batch_id,)).fetchone()
            if not row:
                return None

            jobs_rows = conn.execute(
                """
                SELECT job_id, filename, status, progress, message, duration, processing_time, error,
                       (result_json IS NOT NULL) as has_result
                FROM jobs WHERE batch_id = ? ORDER BY created_at ASC;
                """,
                (batch_id,),
            ).fetchall()

            jobs_list = [dict(j) for j in jobs_rows]

            return {
                "batch_id": row["batch_id"],
                "total_files": row["total_files"],
                "status": row["status"],
                "overall_progress": row["overall_progress"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "jobs": jobs_list,
            }

    def delete_batch(self, batch_id: str, delete_jobs: bool = True) -> bool:
        """Delete a batch and optionally delete all constituent jobs."""
        if not self.enabled:
            return False

        with self._lock, self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                if delete_jobs:
                    # Get job IDs to clean up FTS
                    rows = conn.execute("SELECT job_id FROM jobs WHERE batch_id = ?;", (batch_id,)).fetchall()
                    for r in rows:
                        conn.execute("DELETE FROM transcript_fts WHERE job_id = ?;", (r["job_id"],))
                    conn.execute("DELETE FROM jobs WHERE batch_id = ?;", (batch_id,))

                cur = conn.execute("DELETE FROM batches WHERE batch_id = ?;", (batch_id,))
                conn.execute("COMMIT;")
                return cur.rowcount > 0
            except Exception as e:
                conn.execute("ROLLBACK;")
                logger.error("Failed to delete batch %s: %s", batch_id, e)
                return False

    def get_stats(self) -> dict[str, Any]:
        """Aggregate summary statistics across all historical jobs."""
        if not self.enabled:
            return {"enabled": False}

        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_jobs,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_jobs,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_jobs,
                    COALESCE(SUM(duration), 0.0) as total_duration_seconds,
                    COALESCE(SUM(processing_time), 0.0) as total_processing_seconds
                FROM jobs;
                """
            ).fetchone()

            batch_count = conn.execute("SELECT COUNT(*) as total FROM batches;").fetchone()

            total_sec = row["total_duration_seconds"] if row else 0.0
            return {
                "enabled": True,
                "db_path": str(self.db_path),
                "total_jobs": row["total_jobs"] if row else 0,
                "completed_jobs": row["completed_jobs"] if row else 0,
                "failed_jobs": row["failed_jobs"] if row else 0,
                "total_batches": batch_count["total"] if batch_count else 0,
                "total_duration_seconds": round(total_sec, 1),
                "total_duration_hours": round(total_sec / 3600.0, 2),
            }


# Global singleton storage instance
job_store = JobStore()
