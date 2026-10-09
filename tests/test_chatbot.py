from fastapi.testclient import TestClient
from api.main import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_chat_endpoint():
    """Teste que l'endpoint /chat fonctionne."""
    response = client.post("/chat", json={"query": "Quels sont les concerts à Dijon en octobre ?"})
    assert response.status_code == 200, f"❌ Erreur : {response.json()}"
    assert "response" in response.json(), "❌ Réponse manquante dans la réponse JSON."
    logger.info(f"✅ Test de l'endpoint /chat : OK. Réponse : {response.json()['response']}")

def test_health_endpoint():
    """Teste que l'endpoint /health fonctionne."""
    response = client.get("/health")
    assert response.status_code == 200, "❌ L'endpoint /health a échoué."
    assert response.json() == {"status": "OK"}, "❌ Réponse inattendue."
    logger.info("✅ Test de l'endpoint /health : OK.")

if __name__ == "__main__":
    test_health_endpoint()
    test_chat_endpoint()