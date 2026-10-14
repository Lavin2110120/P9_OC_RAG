import pytest
from fastapi.testclient import TestClient
from api.main import app
import os
import runpy
import pytest
from unittest.mock import patch, MagicMock
import importlib
import scripts.mistral_qa

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


def test_mistral_qa_missing_key(monkeypatch):
    # Supprimer la variable d'environnement
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    # Empêcher load_dotenv de remettre la clé (si .env existe)
    monkeypatch.setattr("scripts.mistral_qa.load_dotenv", MagicMock())
    # Forcer os.getenv à retourner None
    with patch("os.getenv", return_value=None):
        with pytest.raises(ValueError, match="MISTRAL_API_KEY est requis"):
            importlib.reload(scripts.mistral_qa)


@patch("scripts.mistral_qa.generate_response")
def test_mistral_qa_main_block(mock_generate_response, capsys):
    mock_generate_response.return_value = "Réponse mockée"
    from scripts.mistral_qa import main
    main()
    captured = capsys.readouterr()
    assert "Réponse mockée" in captured.out
    mock_generate_response.assert_called_once()