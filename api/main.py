from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Puls-Events RAG API",
    description="API de recherche sémantique pour les événements culturels (Open Agenda).",
    version="1.0.0"
)

# CORS (pour le frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- Configuration ---
INDEX_PATH = Path("data/vector_store/faiss_index.bin")
METADATA_PATH = Path("data/vector_store/metadata.parquet")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Charger les ressources au démarrage
try:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {METADATA_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    metadata = pd.read_parquet(METADATA_PATH)
    model = SentenceTransformer(MODEL_NAME)
    log.info("✅ Ressources chargées avec succès.")
except Exception as e:
    log.error(f"❌ Erreur au démarrage : {e}")
    raise

# --- Modèles Pydantic ---
class EventResult(BaseModel):
    title: str
    date: str
    location: Optional[str] = "N/A"
    distance: float

class QueryRequest(BaseModel):
    query: str = Field(..., description="Requête de recherche (ex: 'concert à La Vapeur').")
    k: int = Field(3, ge=1, le=10, description="Nombre de résultats à retourner.")

# --- Endpoints ---
@app.post("/search", response_model=dict)
def search_events(request: QueryRequest):
    """Recherche des événements similaires à la requête."""
    try:
        # Vectoriser la requête
        query_embedding = model.encode(
            [request.query],
            normalize_embeddings=True,
            batch_size=1
        ).astype('float32')

        # Recherche FAISS
        D, I = index.search(query_embedding, k=request.k)

        # Formater les résultats
        results = []
        for dist, idx in zip(D[0], I[0]):
            event = metadata.iloc[idx]
            results.append({
                "title": event["title"],
                "date": str(event["date_begin"]),
                "location": event.get("location_name", "N/A"),
                "distance": float(dist)
            })

        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        log.error(f"❌ Erreur lors de la recherche : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Pour le développement ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)