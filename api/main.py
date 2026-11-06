"""FastAPI API for health checks, RAG queries, and safe event-index rebuilds."""
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# Keep this module-level setting patchable for tests and deployments.
LAST_REBUILD_FILE = PROJECT_ROOT / "data" / "last_rebuild.json"

# These wrappers preserve the module's existing patch points while ensuring
# importing the API does not load transformer models, FAISS, or the LLM stack.
def generate_response(query: str) -> str:
    """Import the RAG stack only when an /ask request actually needs it."""
    from .mistral_qa import generate_response as _generate_response

    return _generate_response(query)


def fetch_events_main() -> Any:
    from scripts.fetch_events import main

    return main()


def clean_events_main() -> Any:
    from scripts.clean_events import main

    return main()


def vectorize_and_index() -> Any:
    from scripts.vectorize_events import main

    return main()


rebuild_lock = asyncio.Lock()
scheduler = AsyncIOScheduler()
last_rebuild: Optional[datetime] = None


def save_last_rebuild() -> None:
    """Persist the last successful rebuild timestamp, if available."""
    try:
        LAST_REBUILD_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_rebuild": last_rebuild.isoformat() if last_rebuild else None}
        LAST_REBUILD_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("last_rebuild persisted to %s", LAST_REBUILD_FILE)
    except Exception as exc:
        # A persistence failure should be visible in logs but must not hide the
        # successful pipeline result or crash the health/status endpoints.
        logger.warning("Unable to persist last_rebuild: %s", exc)


def load_last_rebuild() -> Optional[datetime]:
    """Load a persisted timestamp; absent, malformed, or invalid data means None."""
    try:
        if LAST_REBUILD_FILE.exists():
            data = json.loads(LAST_REBUILD_FILE.read_text(encoding="utf-8"))
            raw_value = data.get("last_rebuild") if isinstance(data, dict) else None
            if raw_value:
                return datetime.fromisoformat(raw_value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Unable to load last_rebuild: %s", exc)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted state and run the weekly rebuild scheduler lifecycle."""
    global last_rebuild

    last_rebuild = load_last_rebuild()
    logger.info("Loaded last_rebuild: %s", last_rebuild)

    scheduler.add_job(
        scheduled_weekly_rebuild,
        trigger=CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="weekly_rebuild",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Weekly rebuild scheduler started (Monday 03:00)")
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


app = FastAPI(title="Puls Events API", version="1.0.0", lifespan=lifespan)


async def run_full_pipeline_safe() -> Dict[str, Any]:
    """Run the ordered pipeline once, report failures, and persist success."""
    global last_rebuild

    # No await occurs between the check and acquiring an unlocked asyncio.Lock;
    # this rejects overlapping invocations on the application's event loop.
    if rebuild_lock.locked():
        logger.warning("A rebuild is already running; request ignored")
        return {"status": "already_running"}

    async with rebuild_lock:
        try:
            logger.info("Starting automatic/manual rebuild")
            # Callable entry points (not shell commands) are deliberately used:
            # arguments cannot be interpolated into a shell command.
            fetch_events_main()
            clean_events_main()
            vectorize_and_index()

            last_rebuild = datetime.now()
            save_last_rebuild()
            logger.info("Rebuild completed at %s", last_rebuild)
            return {"status": "ok", "completed_at": last_rebuild.isoformat()}
        except Exception as exc:
            logger.exception("Rebuild failed: %s", exc)
            return {"status": "error", "detail": str(exc)}


async def scheduled_weekly_rebuild() -> None:
    """Scheduler entry point."""
    logger.info("Starting scheduled weekly rebuild")
    result = await run_full_pipeline_safe()
    logger.info("Scheduled rebuild result: %s", result)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Expose API and scheduler health information."""
    job = scheduler.get_job("weekly_rebuild")
    next_run = job.next_run_time if job else None
    return {
        "status": "OK",
        "message": "API opérationnelle",
        "last_rebuild": last_rebuild.isoformat() if last_rebuild else None,
        "next_scheduled_rebuild": next_run.isoformat() if next_run else None,
    }


@app.post("/ask")
async def ask(request: QueryRequest) -> Dict[str, Any]:
    """Generate an answer from the event index and LLM."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La requête est vide.")
    try:
        response = await run_in_threadpool(generate_response, request.query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erreur /ask : %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"response": response}


@app.post("/rebuild")
async def rebuild(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Queue a non-blocking rebuild unless one is already running."""
    if rebuild_lock.locked():
        return {"status": "already_running", "message": "Un rebuild est déjà en cours"}
    background_tasks.add_task(run_full_pipeline_safe)
    return {"status": "success", "message": "Pipeline lancé en arrière-plan"}


@app.post("/rebuild/now")
async def rebuild_now() -> Dict[str, Any]:
    """Run the pipeline synchronously from the caller's perspective."""
    return await run_full_pipeline_safe()


@app.get("/rebuild/status")
async def rebuild_status() -> Dict[str, Any]:
    """Report actual in-process rebuild and scheduler state."""
    job = scheduler.get_job("weekly_rebuild")
    return {
        "is_running": rebuild_lock.locked(),
        "last_rebuild": last_rebuild.isoformat() if last_rebuild else None,
        "next_scheduled_rebuild": (
            job.next_run_time.isoformat() if job and job.next_run_time else None
        ),
        "scheduler_running": scheduler.running,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
