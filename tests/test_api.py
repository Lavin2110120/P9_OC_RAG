"""Tests de l'API FastAPI (mocks pour éviter appels réseau / Mistral)."""
import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Force-import so mock.patch dotted path can resolve.
import api.main  # noqa: F401

# Permet l'import `api.main` depuis ce fichier.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def api_client():
    """Importe api.main avec generate_response mocké."""
    with patch("api.main.generate_response", return_value="Réponse mockée"):
        if "api.main" in sys.modules:
            importlib.reload(sys.modules["api.main"])
        else:
            import api.main  # noqa: F401
        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)


def test_health_endpoint(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "OK"
    assert "message" in data


def test_ask_endpoint_valid(api_client):
    r = api_client.post("/ask", json={"query": "Quels concerts à Dijon ?"})
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0


def test_ask_endpoint_empty(api_client):
    r = api_client.post("/ask", json={"query": "   "})
    assert r.status_code == 400


def test_rebuild_endpoint(api_client):
    r = api_client.post("/rebuild")
    assert r.status_code == 200
    assert r.json()["status"] in ("success", "already_running")
