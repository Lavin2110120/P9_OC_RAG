"""Tests /rebuild/now, /rebuild/status, persistance last_rebuild et erreurs /ask."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.main  # noqa: F401
from tests.test_api import _seuil_env  # réutilise tes utilitaires si tu veux


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient
    import api.main as module_api
    importlib = pytest.importorskip("importlib")
    from unittest.mock import patch as p

    importlib.reload(module_api)
    with p.object(module_api, "generate_response", return_value="Réponse mockée"), \
         p.object(module_api, "fetch_events_main"), \
         p.object(module_api, "clean_events_main"), \
         p.object(module_api, "vectorize_and_index"):
        with TestClient(module_api.app) as client:
            yield client


def test_rebuild_now_execute_le_pipeline_mocke(api_client):
    r = api_client.post("/rebuild/now")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "completed_at" in r.json()


def test_rebuild_status_endpoint(api_client):
    r = api_client.get("/rebuild/status")
    assert r.status_code == 200
    data = r.json()
    assert data["is_running"] is False
    assert data["scheduler_running"] is True


def test_ask_erreur_interne_leve_500(api_client):
    import api.main as module_api
    with patch.object(module_api, "generate_response",
                      side_effect=RuntimeError("LLM down")):
        r = api_client.post("/ask", json={"query": "test"})
    assert r.status_code == 500
    assert "LLM down" in r.json()["detail"]


def test_health_affiche_last_rebuild(api_client):
    r = api_client.get("/health")
    data = r.json()
    assert "last_rebuild" in data
    assert "next_scheduled_rebuild" in data


def test_save_et_load_last_rebuild_roundtrip(tmp_path):
    from datetime import datetime
    import api.main as module_api

    with patch.object(module_api, "LAST_REBUILD_FILE",
                      tmp_path / "last_rebuild.json"):
        module_api.last_rebuild = datetime(2026, 10, 30, 12, 0, 0)
        module_api.save_last_rebuild()
        module_api.last_rebuild = None
        charge = module_api.load_last_rebuild()
        assert charge == datetime(2026, 10, 30, 12, 0, 0)


def test_load_last_rebuild_fichier_inexistant_renvoie_none(tmp_path):
    import api.main as module_api
    with patch.object(module_api, "LAST_REBUILD_FILE",
                      tmp_path / "inexistant.json"):
        assert module_api.load_last_rebuild() is None


def test_load_last_rebuild_json_corrompu_renvoie_none(tmp_path):
    import api.main as module_api
    fichier = tmp_path / "corrupt.json"
    fichier.write_text("{pas du json", encoding="utf-8")
    with patch.object(module_api, "LAST_REBUILD_FILE", fichier):
        assert module_api.load_last_rebuild() is None