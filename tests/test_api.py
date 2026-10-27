"""Tests de l'API FastAPI (mocks pour éviter appels réseau / Mistral)."""
import importlib
import math
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pytest

# Permet l'import `api.main` depuis ce fichier.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force-import so mock.patch dotted path can resolve.
import api.main  # noqa: F401


def _seuil_env(nom: str, defaut: float) -> float:
    """Lit un seuil de latence positif, configurable sans modifier les tests."""
    try:
        seuil = float(os.getenv(nom, str(defaut)))
    except ValueError as exc:
        raise ValueError(f"{nom} doit être un nombre exprimé en secondes") from exc
    if not math.isfinite(seuil) or seuil <= 0:
        raise ValueError(f"{nom} doit être un nombre fini strictement positif")
    return seuil


@pytest.fixture
def api_client():
    """Client API avec génération et tâches de rebuild neutralisées."""
    from fastapi.testclient import TestClient
    import api.main as module_api

    # Recharger avant les patchs : autrement reload() remplacerait les mocks.
    importlib.reload(module_api)
    with patch.object(module_api, "generate_response", return_value="Réponse mockée"), \
         patch.object(module_api, "fetch_events_main"), \
         patch.object(module_api, "clean_events_main"), \
         patch.object(module_api, "vectorize_and_index"):
        with TestClient(module_api.app) as client:
            yield client


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


def test_api_mockee_latence_unitaire(api_client):
    """Mesure uniquement le coût HTTP local avec une réponse entièrement mockée.

    Ce test vérifie un budget d'overhead API, pas la latence du système RAG.
    """
    limite = _seuil_env("API_MOCK_LATENCY_MAX_SECONDS", 2.0)
    durees = []
    for _ in range(3):
        debut = perf_counter()
        reponse = api_client.post("/ask", json={"query": "Concerts à Dijon ?"})
        durees.append(perf_counter() - debut)
        assert reponse.status_code == 200
        assert reponse.json()["response"] == "Réponse mockée"

    mediane = statistics.median(durees)
    assert mediane < limite, (
        f"Latence médiane API mockée (3 requêtes): {mediane:.3f}s, "
        f"seuil {limite:.3f}s; maximum observé {max(durees):.3f}s "
        "(ce résultat ne mesure pas le RAG réel)"
    )


def test_benchmark_opt_in_latence_rag_reel():
    """Benchmark API/RAG réel, désactivé par défaut et susceptible d'appels payants.

    Exécution depuis la racine du projet :
    RUN_REAL_RAG_LATENCY=1 pytest -q tests/test_api.py::test_benchmark_opt_in_latence_rag_reel

    Il faut aussi un index/métadonnées FAISS, le modèle d'embedding disponible et
    MISTRAL_API_KEY. Le seuil de bout en bout se règle avec
    RAG_LATENCY_MAX_SECONDS (défaut : 60 secondes).
    """
    if os.getenv("RUN_REAL_RAG_LATENCY") != "1":
        pytest.skip("benchmark réel désactivé; définir RUN_REAL_RAG_LATENCY=1")

    from config import settings

    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY absent; aucun appel réel ne sera lancé")
    if not Path(settings.INDEX_FILE).is_file() or not Path(settings.METADATA_FILE).is_file():
        pytest.skip("index FAISS ou métadonnées Parquet absents")

    from fastapi.testclient import TestClient
    import api.main as module_api

    limite = _seuil_env("RAG_LATENCY_MAX_SECONDS", 60.0)
    with TestClient(module_api.app) as client:
        debut = perf_counter()
        reponse = client.post(
            "/ask", json={"query": "Quels concerts sont proposés à Dijon ?"}
        )
        duree = perf_counter() - debut

    assert reponse.status_code == 200, reponse.text
    assert isinstance(reponse.json().get("response"), str)
    assert reponse.json()["response"].strip()
    assert duree < limite, (
        f"Latence API/RAG réelle: {duree:.3f}s, seuil {limite:.3f}s"
    )
