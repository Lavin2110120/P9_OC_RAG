# tests/api_test.py
import requests
import pytest
from fastapi.testclient import TestClient
from api.main import app

# --- Configuration ---
client = TestClient(app)
API_BASE_URL = "http://localhost:8000"

# --- Tests ---
def test_ask_endpoint():
    """Teste l'endpoint /ask avec une requête valide."""
    response = client.post(
        "/ask",
        json={"query": "Quels sont les concerts à Dijon ?", "k": 3},
    )
    assert response.status_code == 200
    assert "response" in response.json()
    assert isinstance(response.json()["response"], str)

def test_ask_empty_query():
    """Teste l'endpoint /ask avec une requête vide."""
    response = client.post(
        "/ask",
        json={"query": "", "k": 3},
    )
    assert response.status_code == 400
    assert "La requête ne peut pas être vide" in response.json()["detail"]

def test_rebuild_endpoint():
    """Teste l'endpoint /rebuild."""
    response = client.post("/rebuild")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

def test_health_endpoint():
    """Teste l'endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

# --- Test avec requests (pour une API en cours d'exécution) ---
@pytest.mark.skip(reason="Nécessite que l'API soit en cours d'exécution.")
def test_api_running():
    """Teste que l'API est accessible via requests."""
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200