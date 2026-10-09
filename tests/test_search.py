from fastapi.testclient import TestClient
from api.main import app
import pytest

client = TestClient(app)

def test_search_success():
    """Teste une recherche réussie."""
    response = client.post("/search", json={"query": "concert", "k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert "title" in data["results"][0]

def test_search_invalid_k():
    """Teste une valeur de k invalide."""
    response = client.post("/search", json={"query": "test", "k": 0})
    assert response.status_code == 422  # Validation error

def test_search_empty_query():
    """Teste une requête vide."""
    response = client.post("/search", json={"query": "", "k": 1})
    assert response.status_code == 200  # Devrait retourner des résultats (même si peu pertinents)