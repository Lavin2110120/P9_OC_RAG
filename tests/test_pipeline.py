"""Tests du pipeline.

Le modèle d'embedding et l'index FAISS sont mockés.
La dimension fictive utilisée dans les tests est 1024.
"""
import importlib
import sys
from pathlib import Path
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

from datetime import timedelta
from config import settings

start = settings.DATE_MIN_DT + timedelta(days=30)
end = start + timedelta(hours=3)

# ===================== tests config/settings =====================
def test_settings_defines_expected_paths():
    from config import settings
    for attr in ("RAW_DIR", "PROCESSED_DIR", "PROCESSED_FILE",
                 "RAW_FILE", "INDEX_FILE", "METADATA_FILE", "VECTOR_STORE_DIR"):
        assert hasattr(settings, attr), f"settings.{attr} manquant"


def test_retriever_defines_embedding_configuration():
    from api.faiss_retriever import MODEL_NAME, EMBEDDING_DEVICE

    assert MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert EMBEDDING_DEVICE == "cpu"


# ===================== tests clean_events =====================
def _sample_raw_event():
    return {
        "uid": "u1",
        "title": {"fr": "Concert rock"},
        "description": {"fr": "Super concert"},
        "longDescription": {"fr": "Détails du concert"},
        "keywords": {"fr": ["rock", "musique"]},
        "timings": [{"begin": start.isoformat(), "end": end.isoformat()}],
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


def test_clean_events_end_to_end():
    from scripts.clean_events import clean_dataframe, flatten_event

    raw_events = [_sample_raw_event() for _ in range(3)]
    df = pd.DataFrame([flatten_event(event) for event in raw_events])

    cleaned = clean_dataframe(df)

    # Les trois événements ont le même UID.
    assert len(cleaned) == 1
    assert cleaned.loc[0, "title"] == "Concert rock"
    assert "Titre : Concert rock" in cleaned.loc[0, "text"]
    assert pd.notna(cleaned.loc[0, "date_begin"])


# ===================== tests faiss_retriever (mockés) =====================
BGE_M3_DIM = 1024  # dimension de BAAI/bge-m3


@pytest.fixture
def mock_retriever_env():
    """Mock complet du chargement FAISS + modèle d'embedding BGE-m3.

    - ``index.d = 1024`` (dimension FAISS, cohérente avec BGE-m3)
    - ``model.get_sentence_embedding_dimension()`` → 1024
    - ``model.max_seq_length`` est assignable (settings)
    - ``model.encode(...)`` → vecteur normalisé 2D
    """
    with patch("api.faiss_retriever.faiss.read_index") as mock_index, \
         patch("api.faiss_retriever.pd.read_parquet") as mock_meta, \
         patch("api.faiss_retriever.SentenceTransformer") as mock_st:
        # Index FAISS simulé
        idx_mock = MagicMock(ntotal=2, d=BGE_M3_DIM)
        mock_index.return_value = idx_mock

        # Métadonnées (incluant les colonnes embedding_* pour la cohérence)
        mock_meta.return_value = pd.DataFrame({
            "uid": ["a", "b"],
            "title": ["Concert A", "Concert B"],
            "text": ["doc A", "doc B"],
            "location_name": ["La Vapeur", "Le César"],
            "date_begin": ["2026-10-14", "2026-10-15"],
            "embedding_model": ["BAAI/bge-m3", "BAAI/bge-m3"],
            "embedding_dim": [BGE_M3_DIM, BGE_M3_DIM],
        })

        # SentenceTransformer simulé (BGE-m3)
        st_instance = MagicMock()
        st_instance.get_sentence_embedding_dimension.return_value = BGE_M3_DIM
        st_instance.max_seq_length = 512
        st_instance.encode.return_value = np.full((1, BGE_M3_DIM),1 / np.sqrt(BGE_M3_DIM),dtype="float32",)
        mock_st.return_value = st_instance
        yield


def test_retriever_init_loads_index_and_metadata(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None
    r = FAISSRetriever()
    assert r.index.ntotal == 2
    assert r.index.d == BGE_M3_DIM
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


def test_retrieve_caps_k_to_index_size(mock_retriever_env):
    from api.faiss_retriever import FAISSRetriever

    FAISSRetriever._instance = None
    r = FAISSRetriever()

    r.index.search.return_value = (
        np.array([[0.9, 0.8]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )

    docs = r.retrieve("concert", k=100)

    assert len(docs) == 2
    assert r.index.search.call_args.args[1] == r.index.ntotal
