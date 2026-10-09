import logging
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_PATH = Path("data/vector_store")
METADATA_PATH = Path("data/vector_store/metadata.parquet")
MODEL_NAME = "BAAI/bge-m3"  # Modèle multilingue pour les embeddings

# Chargement des ressources au démarrage
index = None
metadata_df = None

def load_resources():
    """Charge l'index FAISS et les métadonnées au démarrage de l'API."""
    global index, metadata_df

    try:
        # Chargement des embeddings (modèle multilingue)
        embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # Chargement de l'index FAISS
        index = FAISS.load_local(
            folder_path=str(INDEX_PATH),
            embeddings=embeddings,
            index_name="faiss_index",
            allow_dangerous_deserialization=True  # Nécessaire pour charger un index local
        )

        # Chargement des métadonnées
        metadata_df = pd.read_parquet(METADATA_PATH)

        logger.info("✅ Ressources chargées avec succès.")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des ressources : {e}")
        raise

# Initialisation de l'API
app = FastAPI(title="RAG API pour Open Agenda")
load_resources()

# Modèle de requête
class QueryRequest(BaseModel):
    query: str
    k: int = 3  # Nombre de résultats par défaut

@app.get("/health")
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "OK", "index_size": len(metadata_df) if metadata_df is not None else 0}

@app.post("/search")
def search_events(request: QueryRequest):
    """
    Recherche des événements similaires à la requête en utilisant FAISS.
    Retourne les k résultats les plus pertinents avec leurs métadonnées.
    """
    if index is None:
        raise HTTPException(status_code=500, detail="Index FAISS non chargé.")

    try:
        # Recherche dans l'index FAISS
        docs_and_scores = index.similarity_search_with_score(
            request.query,
            k=request.k
        )

        # Extraction des métadonnées pour chaque résultat
        results = []
        for doc, score in docs_and_scores:
            # Récupération des métadonnées du chunk (via son ID)
            chunk_id = doc.metadata.get("chunk_id", "")
            chunk_data = metadata_df[metadata_df["chunk_id"] == chunk_id].iloc[0].to_dict()

            # Ajout du score de similarité
            chunk_data["score"] = float(score)
            results.append(chunk_data)

        return {"results": results}
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche : {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)