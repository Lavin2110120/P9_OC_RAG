import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Teste que l'endpoint de statut répond correctement."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert "message" in data

def test_ask_endpoint_valid():
    """Teste une requête valide vers l'endpoint /ask."""
    response = client.post("/ask", json={"query": "Quels sont les concerts à La Vapeur ?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

def test_ask_endpoint_empty():
    """Teste le rejet d'une requête vide."""
    response = client.post("/ask", json={"query": "   "})
    assert response.status_code == 400
    assert "La question ne peut pas être vide" in response.json()["detail"]

def test_rebuild_endpoint():
    """Teste le déclenchement de la reconstruction de l'index."""
    response = client.post("/rebuild")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

if __name__ == "__main__":
    print("Exécution des tests de l'API...")
    test_health_endpoint()
    print("✅ /health validé")
    test_ask_endpoint_empty()
    print("✅ Validation requête vide OK")
    test_rebuild_endpoint()
    print("✅ /rebuild validé")
    test_ask_endpoint_valid()
    print("✅ /ask validé avec succès")