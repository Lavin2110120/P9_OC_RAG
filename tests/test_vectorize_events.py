"""Tests de vectorize_events avec embeddings simulés."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.vectorize_events as ve


def test_load_data_lit_le_parquet(tmp_path):
    df_expected = pd.DataFrame({"text": ["a"]})
    file = tmp_path / "p.parquet"
    df_expected.to_parquet(file, index=False)
    pd.testing.assert_frame_equal(ve.load_data(file), df_expected)


def test_prepare_texts_ok():
    df = pd.DataFrame({"text": ["doc 1", "doc 2"]})
    assert ve.prepare_texts(df) == ["doc 1", "doc 2"]


def test_prepare_texts_colonne_absente_leve_valueerror():
    with pytest.raises(ValueError, match="colonne 'text' est absente"):
        ve.prepare_texts(pd.DataFrame({"autre": ["x"]}))


def test_prepare_texts_dataframe_vide_leve_valueerror():
    with pytest.raises(ValueError, match="aucun événement"):
        ve.prepare_texts(pd.DataFrame({"text": []}))


def test_prepare_texts_textes_vides_leve_valueerror():
    with pytest.raises(ValueError, match="texte vide"):
        ve.prepare_texts(pd.DataFrame({"text": ["ok", "   "]}))


def test_generate_embeddings_appelle_le_model_encode():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.ones((2, 4), dtype="float64")
    with patch("scripts.vectorize_events.SentenceTransformer",
               return_value=mock_model) as mock_st:
        out = ve.generate_embeddings(["a", "b"], "mon-modele")
    mock_st.assert_called_once_with("mon-modele")
    assert out.dtype == np.float32
    assert out.shape == (2, 4)


def test_build_faiss_index_dimensions():
    embeddings = np.ones((3, 8), dtype="float32")
    index = ve.build_faiss_index(embeddings)
    assert index.ntotal == 3
    assert index.d == 8


def test_save_index_et_save_metadata(tmp_path):
    embeddings = np.ones((2, 4), dtype="float32")
    index = ve.build_faiss_index(embeddings)
    idx_path = tmp_path / "index.bin"
    ve.save_index(index, idx_path)
    assert idx_path.exists()

    df = pd.DataFrame({"uid": ["a", "b"]})
    meta_path = tmp_path / "meta.parquet"
    ve.save_metadata(df, meta_path)
    assert meta_path.exists()


def test_main_incoherence_embeddings_metadonnees_leve_valueerror():
    df = pd.DataFrame({"text": ["a", "b"]})  # 2 lignes
    with patch("scripts.vectorize_events.load_data", return_value=df), \
         patch("scripts.vectorize_events.generate_embeddings",
               return_value=np.ones((3, 4), dtype="float32")):  # 3 embeddings ≠ 2
        with pytest.raises(ValueError, match="ne correspond pas"):
            ve.main()


def test_main_ok_ecrit_index_et_metadata(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"]})
    with patch("scripts.vectorize_events.load_data", return_value=df), \
         patch("scripts.vectorize_events.generate_embeddings",
               return_value=np.ones((2, 4), dtype="float32")), \
         patch.object(ve.settings, "VECTOR_STORE_DIR", tmp_path, create=True), \
         patch.object(ve.settings, "INDEX_FILE",
                      tmp_path / "index.bin", create=True), \
         patch.object(ve.settings, "METADATA_FILE",
                      tmp_path / "meta.parquet", create=True):
        ve.main()
    assert (tmp_path / "index.bin").exists()
    assert (tmp_path / "meta.parquet").exists()