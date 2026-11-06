"""Tests HTTP de api.main, sans exécuter le RAG ni le pipeline réel."""
import importlib
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def api_client():
    """Client avec génération et pipeline externe mockés."""
    import api.main as module_api

    module_api = importlib.reload(module_api)
    generator = Mock(return_value="Réponse mockée")
    with (
        patch.object(module_api, "generate_response", generator),
        patch.object(module_api, "fetch_events_main"),
        patch.object(module_api, "clean_events_main"),
        patch.object(module_api, "vectorize_and_index"),
    ):
        with TestClient(module_api.app, raise_server_exceptions=False) as client:
            yield module_api, client, generator


def test_health_endpoint(api_client):
    _, client, _ = api_client
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OK"
    assert data["message"] == "API opérationnelle"
    assert "last_rebuild" in data
    assert "next_scheduled_rebuild" in data


def test_ask_appelle_le_generate_response_importe_par_main(api_client):
    _, client, generator = api_client
    response = client.post("/ask", json={"query": "Concerts à Dijon ?"})

    assert response.status_code == 200
    assert response.json() == {"response": "Réponse mockée"}
    generator.assert_called_once_with("Concerts à Dijon ?")


def test_ask_refuse_une_requete_comportant_seulement_des_espaces(api_client):
    _, client, generator = api_client
    response = client.post("/ask", json={"query": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "La requête est vide."
    generator.assert_not_called()


def test_ask_erreur_interne_retourne_500(api_client):
    module_api, client, generator = api_client
    generator.side_effect = RuntimeError("LLM down")

    response = client.post("/ask", json={"query": "test"})

    assert response.status_code == 500
    assert response.json()["detail"] == "LLM down"
    generator.assert_called_once_with("test")
