"""Tests des routes et fonctions de rebuild réellement définies dans api.main.

Les étapes de collecte/nettoyage/vectorisation sont neutralisées : aucun accès
réseau, indexation FAISS ou appel à un modèle n'est effectué par ces tests.
"""
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def rebuild_api(tmp_path):
    """Client API, pipeline mocké et fichier de persistance temporaire."""
    import api.main as module_api

    module_api = importlib.reload(module_api)
    persist_path = tmp_path / "last_rebuild.json"
    with (
        patch.object(module_api, "LAST_REBUILD_FILE", persist_path),
        patch.object(module_api, "fetch_events_main") as fetch_mock,
        patch.object(module_api, "clean_events_main") as clean_mock,
        patch.object(module_api, "vectorize_and_index") as vectorize_mock,
    ):
        with TestClient(module_api.app) as client:
            yield module_api, client, (fetch_mock, clean_mock, vectorize_mock)


def test_routes_de_rebuild_definies_dans_main(rebuild_api):
    module_api, _, _ = rebuild_api
    routes = {
        (method, route.path)
        for route in module_api.app.routes
        for method in (getattr(route, "methods", None) or set())
    }
    assert ("POST", "/rebuild") in routes
    assert ("POST", "/rebuild/now") in routes
    assert ("GET", "/rebuild/status") in routes


def test_rebuild_now_execute_les_etapes_et_persiste(rebuild_api):
    module_api, client, pipeline_mocks = rebuild_api

    response = client.post("/rebuild/now")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    completed_at = datetime.fromisoformat(result["completed_at"])
    assert module_api.last_rebuild == completed_at
    assert module_api.LAST_REBUILD_FILE.is_file()
    stored = json.loads(module_api.LAST_REBUILD_FILE.read_text(encoding="utf-8"))
    assert stored["last_rebuild"] == completed_at.isoformat()
    for pipeline_mock in pipeline_mocks:
        pipeline_mock.assert_called_once_with()


def test_rebuild_en_arriere_plan_execute_le_pipeline_mocke(rebuild_api):
    _, client, pipeline_mocks = rebuild_api

    response = client.post("/rebuild")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Pipeline lancé en arrière-plan",
    }
    # Starlette TestClient exécute la tâche d'arrière-plan avant de retourner.
    for pipeline_mock in pipeline_mocks:
        pipeline_mock.assert_called_once_with()


def test_status_indique_etat_de_verrou_et_scheduler(rebuild_api):
    module_api, client, _ = rebuild_api

    response = client.get("/rebuild/status")

    assert response.status_code == 200
    state = response.json()
    assert state["is_running"] is False
    assert state["scheduler_running"] is True
    assert state["last_rebuild"] is None
    assert state["next_scheduled_rebuild"] is not None
    assert module_api.scheduler.running is True


def test_erreur_pipeline_est_convertie_en_resultat_erreur(rebuild_api):
    _, client, pipeline_mocks = rebuild_api
    pipeline_mocks[0].side_effect = RuntimeError("fetch failed")

    response = client.post("/rebuild/now")

    assert response.status_code == 200
    assert response.json() == {"status": "error", "detail": "fetch failed"}
    pipeline_mocks[0].assert_called_once_with()
    pipeline_mocks[1].assert_not_called()
    pipeline_mocks[2].assert_not_called()


def test_persistance_last_rebuild_roundtrip(rebuild_api):
    module_api, _, _ = rebuild_api
    moment = datetime(2026, 10, 30, 12, 0, 0)
    previous = module_api.last_rebuild
    try:
        module_api.last_rebuild = moment
        module_api.save_last_rebuild()
        module_api.last_rebuild = None
        assert module_api.load_last_rebuild() == moment
    finally:
        module_api.last_rebuild = previous


def test_chargement_retourne_none_si_fichier_absent_ou_json_invalide(
    rebuild_api, tmp_path
):
    module_api, _, _ = rebuild_api
    persist_path = tmp_path / "absent.json"
    with patch.object(module_api, "LAST_REBUILD_FILE", persist_path):
        assert module_api.load_last_rebuild() is None
        persist_path.write_text("{json invalide", encoding="utf-8")
        assert module_api.load_last_rebuild() is None
