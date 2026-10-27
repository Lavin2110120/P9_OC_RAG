"""API FastAPI : /health, /ask, /rebuild."""
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

from api.mistral_qa import generate_response
from scripts.fetch_events import main as fetch_events_main
from scripts.clean_events import main as clean_events_main
from scripts.vectorize_events import main as vectorize_and_index

logger = logging.getLogger(__name__)

# Fichier de persistance pour last_rebuild
LAST_REBUILD_FILE = PROJECT_ROOT / "data" / "last_rebuild.json"

# État global de l'application
rebuild_lock = asyncio.Lock()
scheduler = AsyncIOScheduler()
last_rebuild: Optional[datetime] = None


def save_last_rebuild() -> None:
    """Persiste last_rebuild dans un fichier JSON."""
    try:
        LAST_REBUILD_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"last_rebuild": last_rebuild.isoformat() if last_rebuild else None}
        LAST_REBUILD_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("last_rebuild persisté dans %s", LAST_REBUILD_FILE)
    except Exception as exc:
        logger.warning("Impossible de persister last_rebuild : %s", exc)


def load_last_rebuild() -> Optional[datetime]:
    """Charge last_rebuild depuis le fichier JSON."""
    try:
        if LAST_REBUILD_FILE.exists():
            data = json.loads(LAST_REBUILD_FILE.read_text(encoding="utf-8"))
            raw_value = data.get("last_rebuild")
            if raw_value:
                return datetime.fromisoformat(raw_value)
    except Exception as exc:
        logger.warning("Impossible de charger last_rebuild : %s", exc)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application (startup/shutdown)."""
    global last_rebuild

    # Startup : charger l'état persisté et démarrer le scheduler
    last_rebuild = load_last_rebuild()
    logger.info("last_rebuild chargé : %s", last_rebuild)

    scheduler.add_job(
        scheduled_weekly_rebuild,
        trigger=CronTrigger(day_of_week="mon", hour=3, minute=0),
        id="weekly_rebuild",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler démarré - rebuild hebdomadaire configuré (lundi 3h)")

    yield

    # Shutdown : arrêter proprement le scheduler
    scheduler.shutdown(wait=False)
    logger.info("Scheduler arrêté")


app = FastAPI(title="Puls Events API", version="1.0.0", lifespan=lifespan)


async def run_full_pipeline_safe() -> Dict[str, Any]:
    """Exécute le pipeline complet avec gestion du verrou et persistance."""
    global last_rebuild

    if rebuild_lock.locked():
        logger.warning("Rebuild déjà en cours, ignoré")
        return {"status": "already_running"}

    async with rebuild_lock:
        try:
            logger.info("Début du rebuild automatique/manuel")
            fetch_events_main()
            clean_events_main()
            vectorize_and_index()

            last_rebuild = datetime.now()
            save_last_rebuild()  # Persistance immédiate

            logger.info("Rebuild terminé avec succès à %s", last_rebuild)
            return {"status": "ok", "completed_at": last_rebuild.isoformat()}

        except Exception as exc:
            logger.exception("Échec du rebuild : %s", exc)
            return {"status": "error", "detail": str(exc)}


async def scheduled_weekly_rebuild() -> None:
    """Tâche appelée automatiquement toutes les semaines."""
    logger.info("Lancement du rebuild hebdomadaire planifié")
    result = await run_full_pipeline_safe()
    logger.info("Résultat du rebuild hebdomadaire : %s", result)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Vérifie l'état de l'API et affiche les infos de planification."""
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
    """Génère une réponse à partir d'une requête utilisateur."""
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
    """Déclenche un rebuild en arrière-plan (non bloquant)."""
    if rebuild_lock.locked():
        return {"status": "already_running", "message": "Un rebuild est déjà en cours"}

    background_tasks.add_task(run_full_pipeline_safe)
    return {"status": "success", "message": "Pipeline lancé en arrière-plan"}


@app.post("/rebuild/now")
async def rebuild_now() -> Dict[str, Any]:
    """Déclenche un rebuild immédiat (bloquant, pour debug/test)."""
    return await run_full_pipeline_safe()


@app.get("/rebuild/status")
async def rebuild_status() -> Dict[str, Any]:
    """Retourne l'état détaillé du système de rebuild."""
    job = scheduler.get_job("weekly_rebuild")

    return {
        "is_running": rebuild_lock.locked(),
        "last_rebuild": last_rebuild.isoformat() if last_rebuild else None,
        "next_scheduled_rebuild": job.next_run_time.isoformat() if job and job.next_run_time else None,
        "scheduler_running": scheduler.running,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)