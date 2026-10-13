"""API FastAPI - Orchestration du pipeline événementiel et FAQ RAG."""
import logging
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

# Permet les imports `config.*` et `scripts.*` depuis la racine du projet,
# quel que soit le répertoire d'exécution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_events import main as fetch_events_main
from scripts.clean_events import main as clean_events_main
from scripts.vectorize_events import main as vectorize_and_index

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Puls Events API",
    version="1.0.0",
    description="API de recommandation d'événements culturels avec RAG Mistral/FAISS.",
)

# 1. On crée un verrou global pour l'API
rebuild_lock = asyncio.Lock()

async def run_full_pipeline_safe():
    """Tâche de fond protégée par un verrou pour la reconstruction."""
    # Si le verrou est déjà pris, on annule silencieusement
    if rebuild_lock.locked():
        logger.warning("⏳ [Pipeline] Une reconstruction est déjà en cours, annulation de la nouvelle demande.")
        return

    async with rebuild_lock:
        logger.info("🔄 [Pipeline] Début de la reconstruction complète de l'index...")
        try:
            # Note : Les données sont traitées via le dossier data/processed/
            fetch_events_main()
            clean_events_main()
            vectorize_and_index()
            logger.info("✅ [Pipeline] Index FAISS reconstruit avec succès !")
        except Exception as exc:
            logger.error(f"❌ [Pipeline] Erreur lors de la reconstruction : {exc}", exc_info=True)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question posée au chatbot")

_generate_response = None

def get_generate_response():
    """Charge le module Mistral/FAISS à la première question.

    Import différé volontairement pour permettre à l'API de démarrer
    (et donc à `/rebuild` de fonctionner) même si l'index FAISS n'existe
    pas encore sur disque.
    """
    global _generate_response
    if _generate_response is None:
        logger.info("Chargement du module Mistral / FAISS...")
        from scripts.mistral_qa import generate_response
        _generate_response = generate_response
        logger.info("Module Mistral / FAISS initialisé.")
    return _generate_response

# ==========================================
# ROUTES DE L'API
# ==========================================

@app.get("/health", summary="Vérifier l'état de l'API")
async def health_check():
    return {"status": "OK", "message": "L'API est opérationnelle."}

@app.post("/ask", summary="Poser une question au chatbot")
async def ask(request: QueryRequest) -> Dict[str, Any]:
    try:
        logger.info(f"Question reçue : {request.query}")
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

        generate_response = get_generate_response()
        response = generate_response(request.query)
        return {"question": request.query, "response": response}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

@app.post("/rebuild", summary="Reconstruire l'index FAISS")
async def rebuild_index(background_tasks: BackgroundTasks):
    if rebuild_lock.locked():
        raise HTTPException(status_code=429, detail="Une reconstruction est déjà en cours.")

    background_tasks.add_task(run_full_pipeline_safe)
    return {
        "status": "success",
        "message": "Reconstruction de l'index lancée en tâche de fond."
    }

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)