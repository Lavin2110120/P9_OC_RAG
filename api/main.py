from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from scripts.mistral_qa import generate_response
import logging
import sys
from dotenv import load_dotenv  
import os

# Chargez les variables d'environnement depuis .env
load_dotenv()  # <-- Charge le fichier .env

# Vérifiez que la clé API est bien chargée
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("La variable MISTRAL_API_KEY est manquante dans .env ou n'est pas définie.")

# Initialisez le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="Chatbot RAG pour Événements Culturels",
    description="API pour interagir avec un chatbot RAG basé sur FAISS et Mistral.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Modèle de requête ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Requête utilisateur pour le chatbot.", example="Quels sont les concerts à Dijon en octobre ?")
    k: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de résultats à récupérer depuis FAISS (entre 1 et 10).",
        example=3,
    )

# --- Endpoint principal pour le chat ---
@app.post(
    "/chat",
    response_model=Dict[str, Any],
    responses={
        200: {"description": "Réponse générée avec succès."},
        400: {"description": "Requête invalide."},
        500: {"description": "Erreur interne du serveur."},
    },
    summary="Interroger le chatbot RAG",
    description="Génère une réponse augmentée à partir des événements indexés dans FAISS.",
)
async def chat(request: QueryRequest):
    """
    Endpoint pour interagir avec le chatbot RAG.

    Args:
        request (QueryRequest): Requête contenant la question utilisateur et le nombre de résultats souhaités.

    Returns:
        Dict[str, Any]: Réponse générée par le chatbot.
    """
    try:
        logger.info(f"Requête reçue : {request.query} (k={request.k})")

        # Validation supplémentaire : Vérifier que la requête n'est pas vide
        if not request.query.strip():
            logger.warning("Requête vide reçue.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="La requête ne peut pas être vide.",
            )

        # Génération de la réponse
        response = generate_response(request.query)
        logger.info(f"Réponse générée : {response}")

        return {"response": response, "k": request.k}

    except HTTPException as he:
        logger.error(f"Erreur HTTP : {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la génération de la réponse : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne : {str(e)}",
        )

# --- Endpoint de santé ---
@app.get(
    "/health",
    response_model=Dict[str, str],
    summary="Vérifier l'état de l'API",
    description="Endpoint pour vérifier que l'API est opérationnelle.",
)
async def health_check():
    """Vérifie que l'API est en bon état de fonctionnement."""
    logger.info("Requête de santé reçue.")
    return {"status": "OK", "message": "L'API est opérationnelle."}

# --- Endpoint pour tester la connexion à Mistral ---
@app.get(
    "/test-mistral",
    summary="Tester la connexion à Mistral",
    description="Vérifie que la connexion à l'API Mistral fonctionne.",
)
async def test_mistral():
    """Teste la connexion à l'API Mistral."""
    try:
        # Appel minimal pour vérifier la connexion
        test_response = generate_response("Test de connexion.")
        logger.info(f"Connexion à Mistral validée. Réponse : {test_response[:50]}...")
        return {"status": "OK", "message": "Connexion à Mistral fonctionnelle."}
    except Exception as e:
        logger.error(f"Échec de la connexion à Mistral : {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Échec de la connexion à Mistral : {str(e)}",
        )

# --- Point d'entrée pour exécuter l'API ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,
    )