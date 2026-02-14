"""
Worker daemon — polls ai_order_jobs for pending work and runs the pipeline.

Usage:
    python -m app.worker

Features:
    - Polls every POLL_INTERVAL_SECONDS (default 5s)
    - Processes 1 job at a time (safe, sequential)
    - Graceful shutdown on SIGTERM / SIGINT
    - 180s timeout per job
    - Structured JSON logging
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone

from app.agents.pipeline import run_pipeline
from app.config import settings
from app.models.job import AIOrderJob, JobPayload
from app.tools.db_writer import persist_order_result, _fail_job
from app.tools.supabase_client import get_supabase

# ── Logging setup ─────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Structured JSON log output for production."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)


# ── Signal handling ───────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum: int, frame: object) -> None:
    global _shutdown_requested
    logger = logging.getLogger(__name__)
    logger.info("Shutdown signal received (%s), finishing current job...", signum)
    _shutdown_requested = True


# ── Job lifecycle ─────────────────────────────────────────────


def claim_job() -> AIOrderJob | None:
    """Poll for a pending job and claim it atomically."""
    sb = get_supabase()

    try:
        resp = (
            sb.table("ai_order_jobs")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )

        if not resp.data:
            return None

        row = resp.data[0]

        # Claim the job — update status to 'processing'
        sb.table("ai_order_jobs").update({
            "status": "processing",
            "processing_started_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", row["id"]).eq("status", "pending").execute()

        # Parse the payload
        payload_raw = row.get("payload", {})
        if isinstance(payload_raw, str):
            payload_raw = json.loads(payload_raw)

        payload = JobPayload(**payload_raw)

        return AIOrderJob(
            id=row["id"],
            conversation_id=row["conversation_id"],
            customer_id=row["customer_id"],
            sede_id=row["sede_id"],
            requested_by=row["requested_by"],
            status="processing",
            payload=payload,
        )

    except Exception as exc:
        logging.getLogger(__name__).error("Error claiming job: %s", exc, exc_info=True)
        return None


def process_job(job: AIOrderJob) -> None:
    """Run the pipeline for a claimed job and persist results."""
    logger = logging.getLogger(__name__)
    sb = get_supabase()

    logger.info("Processing job %s (conversation: %s)", job.id, job.conversation_id)

    # Update status to 'extracting'
    try:
        sb.table("ai_order_jobs").update({
            "status": "extracting",
        }).eq("id", job.id).execute()
    except Exception:
        pass  # Non-critical

    try:
        result = run_pipeline(job)
        order_id = persist_order_result(job, result)

        if order_id:
            logger.info("Job %s completed → order %s", job.id, order_id)
        else:
            logger.warning("Job %s completed but order creation failed", job.id)

    except Exception as exc:
        error_msg = f"Pipeline fatal error: {exc}"
        logger.error("Job %s failed: %s", job.id, error_msg, exc_info=True)
        _fail_job(sb, job, error_msg)


# ── Main loop ─────────────────────────────────────────────────

def main() -> None:
    """Entry point for the worker daemon."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    logger.info("=" * 60)
    logger.info("AI Order Worker starting")
    logger.info("Supabase URL: %s", settings.SUPABASE_URL)
    logger.info("Gemini model: %s", settings.GEMINI_MODEL)
    logger.info("Poll interval: %ds", settings.POLL_INTERVAL_SECONDS)
    logger.info("Job timeout: %ds", settings.JOB_TIMEOUT_SECONDS)
    logger.info("=" * 60)

    # Verify Supabase connectivity
    try:
        sb = get_supabase()
        sb.table("ai_order_jobs").select("id").limit(1).execute()
        logger.info("Supabase connection verified ✓")
    except Exception as exc:
        logger.error("Cannot connect to Supabase: %s", exc)
        sys.exit(1)

    # Main polling loop
    while not _shutdown_requested:
        try:
            job = claim_job()

            if job:
                process_job(job)
            else:
                time.sleep(settings.POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down...")
            break
        except Exception as exc:
            logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
            time.sleep(settings.POLL_INTERVAL_SECONDS)

    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    main()
