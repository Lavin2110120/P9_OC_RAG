"""API FastAPI : /health, /ask, /rebuild."""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Permet `from api.mistral_qa import ...` quel que soit le cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.mistral_qa import generate_response
from scripts.fetch_events import main as fetch_events_main
from scripts.clean_events import main as clean_events_main
from scripts.vectorize_events import main as vectorize_and_index

logger = logging.getLogger(__name__)

app = FastAPI(title="Puls Events API", version="1.0.0")

rebuild_lock = asyncio.Lock()


async def run_full_pipeline_safe() -> Dict[str, Any]:
    if rebuild_lock.locked():
        return {"status": "already_running"}
    async with rebuild_lock:
        try:
            fetch_events_main()
            clean_events_main()
            vectorize_and_index()
            return {"status": "ok"}
        except Exception as exc:
            logger.exception("Échec du rebuild : %s", exc)
            return {"status": "error", "detail": str(exc)}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {"status": "OK", "message": "API opérationnelle"}


@app.post("/ask")
async def ask(request: QueryRequest) -> Dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La requête est vide.")
    try:
        response = generate_response(request.query)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur /ask : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"response": response}


@app.post("/rebuild")
async def rebuild(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if rebuild_lock.locked():
        return {"status": "already_running"}
    background_tasks.add_task(run_full_pipeline_safe)
    return {"status": "success", "message": "Pipeline lancé en arrière-plan"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
