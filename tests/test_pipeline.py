"""Tests du pipeline (FAISS, clean_events, settings).

Aucun téléchargement de BAAI/bge-m3 : tous les modèles sont mockés.
La dimension mockée (1024) correspond à BGE-m3.
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


# ===================== tests config/settings =====================
def test_settings_defines_expected_paths():
    from config import settings
    for attr in ("RAW_DIR", "PROCESSED_DIR", "PROCESSED_FILE",
                 "RAW_FILE", "INDEX_FILE", "METADATA_FILE", "VECTOR_STORE_DIR"):
        assert hasattr(settings, attr), f"settings.{attr} manquant"


def test_settings_centralises_embedding_model():
    """Le nom du modèle et la dimension sont centralisés dans settings."""
    from config import settings
    assert hasattr(settings, "EMBEDDING_MODEL"), "EMBEDDING_MODEL manquant"
    assert hasattr(settings, "EMBEDDING_DIM"), "EMBEDDING_DIM manquant"
    assert hasattr(settings, "EMBEDDING_MAX_SEQ_LENGTH"), "EMBEDDING_MAX_SEQ_LENGTH manquant"
    assert hasattr(settings, "EMBEDDING_BATCH_SIZE"), "EMBEDDING_BATCH_SIZE manquant"
    assert hasattr(settings, "EMBEDDING_NORMALIZE"), "EMBEDDING_NORMALIZE manquant"
    # Le défaut doit être bge-m3.
    assert "bge-m3" in settings.EMBEDDING_MODEL
    # Le nom du modèle ne doit pas être celui d'une variante antérieure.
    assert settings.EMBEDDING_MODEL != "sentence-transformers/all-MiniLM-L6-v2"
    # Dimension par défaut = 1024 (info ; la dim réelle est déduite du modèle).
    assert settings.EMBEDDING_DIM == 1024
    # bge-m3 n'a pas besoin de préfixe de requête : normalisé activé.
    assert settings.EMBEDDING_NORMALIZE is True


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
    """Crée un parquet brut au format de fetch_events puis appelle clean_dataframe."""
    from scripts.clean_events import clean_dataframe

    raw = pd.DataFrame([_sample_raw_event()] * 3)
    raw_path = tmp_path / "events_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    # Charge et aplati comme fetch_events
    from scripts.fetch_events import flatten_event
    df = pd.DataFrame([flatten_event(e) for e in [_sample_raw_event()]])
    df = clean_dataframe(df)

    assert "text" in df.columns, "Colonne 'text' manquante"
    assert len(df) >= 1
    assert all(df["title"].str.len() > 0)
    assert all(df["text"].str.contains("Titre"))


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
        st_instance.encode.return_value = np.array([[0.1] * BGE_M3_DIM,
                                                    [0.2] * BGE_M3_DIM],
                                                   dtype="float32")
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


def test_retriever_init_rejects_dim_mismatch(mock_retriever_env):
    """Si l'index est dans une autre dimension que le modèle, on doit avoir une erreur claire."""
    from api.faiss_retriever import FAISSRetriever
    FAISSRetriever._instance = None

    # Casser la cohérence : index dim 384, modèle dim 1024
    mock_idx = MagicMock(ntotal=2, d=384)
    with patch("api.faiss_retriever.faiss.read_index", return_value=mock_idx):
        with pytest.raises(ValueError, match="Incohérence de dimension"):
            FAISSRetriever()
