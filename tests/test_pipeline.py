"""Tests du pipeline (FAISS, clean_events, settings).

Les tests de retrieval mockent le chargement d'index et les embeddings : ils
ne téléchargent pas le modèle et n'évaluent pas les performances du système réel.
"""
import math
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.documents import Document

# Force-import so mock.patch dotted paths resolve via getattr.
import api.faiss_retriever  # noqa: F401
import scripts.clean_events  # noqa: F401
import scripts.fetch_events  # noqa: F401


# ===================== tests config/settings =====================
def test_settings_defines_expected_paths():
    from config import settings
    for attr in ("RAW_DIR", "PROCESSED_DIR", "PROCESSED_FILE",
                 "RAW_FILE", "INDEX_FILE", "METADATA_FILE", "VECTOR_STORE_DIR"):
        assert hasattr(settings, attr), f"settings.{attr} manquant"


# ===================== tests clean_events =====================
def _sample_raw_event():
    return {
        "uid": "u1",
        "title": {"fr": "Concert rock"},
        "description": {"fr": "Super concert"},
        "longDescription": {"fr": "Détails du concert"},
        "keywords": {"fr": ["rock", "musique"]},
        "timings": [{"start": "2026-10-14T20:00:00", "end": "2026-10-14T23:00:00"}],
        "location": {
            "name": "La Vapeur",
            "city": "Dijon",
            "department": "Côte-d'Or",
            "postalCode": "21000",
            "address": "12 rue test",
            "latitude": 47.3,
            "longitude": 5.0,
        },
        "canonicalUrl": "https://example.com/e1",
        "_agenda_title": "Dijon Culture",
    }


def test_clean_events_end_to_end(tmp_path):
    """Crée un parquet brut puis vérifie l'aplatissement et le nettoyage."""
    from datetime import timedelta

    from config import settings
    from scripts.clean_events import clean_dataframe, flatten_event

    event = _sample_raw_event()
    start = settings.DATE_MIN_DT + timedelta(days=1)
    end = start + timedelta(hours=3)
    event["timings"] = [{"begin": start.isoformat(), "end": end.isoformat()}]

    raw = pd.DataFrame([event] * 3)
    raw_path = tmp_path / "events_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    # Charge et aplati l'événement brut en conservant begin/end dans la fenêtre.
    df = pd.DataFrame([flatten_event(event)])
    assert df.loc[0, "date_begin"] == start.isoformat()
    assert df.loc[0, "date_end"] == end.isoformat()

    df = clean_dataframe(df)

    assert "text" in df.columns, "Colonne 'text' manquante"
    assert len(df) >= 1
    assert all(df["title"].str.len() > 0)
    assert all(df["text"].str.contains("Titre"))


# ===================== tests faiss_retriever (mockés) =====================
MOCK_EMBEDDING_DIM = 384  # mock du modèle configuré dans api.faiss_retriever


@pytest.fixture
def mock_retriever_env():
    """Mocke précisément les fonctions consommées par FAISSRetriever.

    ``encode([query], convert_to_numpy=True)`` reçoit une seule requête et
    renvoie donc une matrice 2D d'une ligne; le code fourni ne valide pas la
    dimension de l'embedding contre l'index.
    """
    with patch("api.faiss_retriever.faiss.read_index") as mock_index, \
         patch("api.faiss_retriever.pd.read_parquet") as mock_meta, \
         patch("api.faiss_retriever.SentenceTransformer") as mock_st:
        # Index FAISS simulé
        idx_mock = MagicMock(ntotal=2)
        mock_index.return_value = idx_mock

        # Le code ne lit dans les métadonnées que ces champs.
        mock_meta.return_value = pd.DataFrame({
            "uid": ["a", "b"],
            "title": ["Concert A", "Concert B"],
            "text": ["doc A", "doc B"],
            "location_name": ["La Vapeur", "Le César"],
            "date_begin": ["2026-10-14", "2026-10-15"],
        })

        # La méthode encode reçoit une liste d'une requête et renvoie (1, d).
        st_instance = MagicMock()
        st_instance.encode.return_value = np.ones(
            (1, MOCK_EMBEDDING_DIM), dtype="float32"
        )
        mock_st.return_value = st_instance
        yield


def test_retriever_init_loads_index_and_metadata(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None
    r = FAISSRetriever()
    assert r.index.ntotal == 2
    assert len(r.metadata) == 2


def test_retrieve_returns_documents(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None
    r = FAISSRetriever()
    r.index.search.return_value = (
        np.array([[0.9, 0.8]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )
    docs = r.retrieve("concert", k=2)
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata["uid"] == "a"


def test_retrieve_k_zero_returns_empty(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None
    r = FAISSRetriever()
    assert r.retrieve("test", k=0) == []


def test_retrieve_handles_negative_index(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None
    r = FAISSRetriever()
    r.index.search.return_value = (
        np.array([[0.9, 0.8]], dtype="float32"),
        np.array([[-1, 0]], dtype="int64"),
    )
    docs = r.retrieve("test", k=2)
    assert len(docs) == 1


def test_retriever_init_rejects_index_metadata_count_mismatch(mock_retriever_env):
    """Le code fourni valide le compte de vecteurs, pas la dimension."""
    from api.faiss_retriever import FAISSRetriever

    FAISSRetriever._instance = None
    mock_idx = MagicMock(ntotal=3)
    with patch("api.faiss_retriever.faiss.read_index", return_value=mock_idx):
        with pytest.raises(ValueError, match="Incohérence : index=3 vecteurs"):
            FAISSRetriever()


def test_retrieve_latence_unitaire_avec_dependances_mockees(mock_retriever_env):
    """Mesure le retrieval avec index/modèle entièrement simulés.

    Ce test rapide ne mesure ni l'inférence BGE-small, ni FAISS réel, ni le RAG.
    """
    try:
        limite = float(os.getenv("PIPELINE_MOCK_RETRIEVE_MAX_SECONDS", "2.0"))
    except ValueError as exc:
        raise ValueError(
            "PIPELINE_MOCK_RETRIEVE_MAX_SECONDS doit être exprimé en secondes"
        ) from exc
    assert math.isfinite(limite) and limite > 0, (
        "PIPELINE_MOCK_RETRIEVE_MAX_SECONDS doit être fini et positif"
    )

    from api.faiss_retriever import FAISSRetriever

    FAISSRetriever._instance = None
    retriever = FAISSRetriever()
    retriever.index.search.return_value = (
        np.array([[0.9, 0.8]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )
    # Une première exécution hors chronométrage réduit l'effet du démarrage Python.
    assert len(retriever.retrieve("concert", k=2)) == 2
    durees = []
    for _ in range(10):
        debut = perf_counter()
        docs = retriever.retrieve("concert", k=2)
        durees.append(perf_counter() - debut)
        assert len(docs) == 2
    mediane = statistics.median(durees)

    assert mediane < limite, (
        f"Latence médiane de retrieval mocké (10 appels): {mediane:.3f}s, "
        f"seuil {limite:.3f}s; maximum observé {max(durees):.3f}s "
        "(ce résultat n'est pas une latence FAISS/BGE réelle)"
    )
