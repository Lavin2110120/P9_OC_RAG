"""Tests des branches FAISSRetriever avec index, métadonnées et embeddings factices."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _construct(metadata, ntotal=1):
    import api.faiss_retriever as module
    index = MagicMock(ntotal=ntotal, d=3)
    with patch.object(module.faiss, "read_index", return_value=index), \
         patch.object(module.pd, "read_parquet", return_value=metadata):
        return module.FAISSRetriever()


def test_init_rejette_metadonnees_sans_colonne_text():
    with pytest.raises(ValueError, match="Colonne 'text' absente"):
        _construct(pd.DataFrame({"uid": ["x"]}))


def test_get_instance_cree_une_seule_instance():
    import api.faiss_retriever as module
    cls = module.FAISSRetriever
    cls._instance = None
    with patch.object(cls, "__init__", return_value=None) as init:
        first = cls.get_instance()
        second = cls.get_instance()
    assert first is second
    init.assert_called_once_with()
    cls._instance = None


def test_retrieve_reshape_embedding_1d_and_rejects_dimension():
    import api.faiss_retriever as module

    model = MagicMock()
    model.encode.return_value = np.array([0.1, 0.2], dtype="float64")

    with patch.object(module, "SentenceTransformer", return_value=model):
        retriever = _construct(pd.DataFrame({"text": ["doc"]}))

        with pytest.raises(
            ValueError,
            match=r"Dimension incompatible : requête=2, index=3",
        ):
            retriever.retrieve("test")

    model.encode.assert_called_once_with(
        ["test"],
        convert_to_numpy=True,
        normalize_embeddings=module.settings.EMBEDDING_NORMALIZE)

def test_retrieve_vide_si_index_vide():
    retriever = _construct(pd.DataFrame({"text": []}), ntotal=0)
    assert retriever.retrieve("sans index") == []
