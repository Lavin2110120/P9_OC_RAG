import sys
import tempfile
import faiss
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from scripts.faiss_retriever import FAISSRetriever

# ----------------------------------------------------------------------
# Fixtures et helpers
# ----------------------------------------------------------------------

@pytest.fixture
def mock_parquet_data():
    """Faux jeu de données pour tester la validation des colonnes."""
    return pd.DataFrame({
        "uid": ["123", "456"],
        "title": ["Concert Test", "Expo Test"],
        "description": ["Super concert", "Belle expo"],
        "location_name": ["La Vapeur", "Musée des Beaux-Arts"],
        "date_begin": ["2026-10-14", "2026-10-15"]
    })

@pytest.fixture
def real_index_fixture(tmp_path):
    """
    Crée un véritable index FAISS en mémoire et les métadonnées associées.
    Le répertoire temporaire `tmp_path` contiendra :
      - faiss_index.bin
      - metadata.parquet
    La fixture retourne (index_dir, meta_df) pour les tests.
    """
    # Dimensions correspondant à BAAI/bge-small-en-v1.5 (384)
    dim = 384
    # Créer 3 vecteurs aléatoires
    vecs = np.random.rand(3, dim).astype("float32")
    # Index FlatL2 pour tests simples
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)

    # Sauvegarde de l'index
    index_path = tmp_path / "faiss_index.bin"
    faiss.write_index(index, str(index_path))

    # Métadonnées associées
    meta_df = pd.DataFrame({
        "text": [
            "Concert de rock à La Vapeur",
            "Exposition d'art moderne à Dijon",
            "Théâtre classique à Beaune"
        ],
        "title": [
            "Rock en Force",
            "L'Art Contemporain",
            "Molière en Scène"
        ],
        "location_name": ["La Vapeur", "Dijon", "Beaune"],
        "date_begin": ["2026-10-20", "2026-11-01", "2026-12-05"]
    })
    meta_path = tmp_path / "metadata.parquet"
    meta_df.to_parquet(meta_path, index=False)

    return tmp_path, meta_df

# ----------------------------------------------------------------------
# Tests unitaires avec mocks
# ----------------------------------------------------------------------

@patch("pandas.read_parquet")
def test_processed_events_logic(mock_read_parquet, mock_parquet_data):
    """Vérifie la validation des colonnes sans le vrai fichier."""
    mock_read_parquet.return_value = mock_parquet_data
    df = pd.read_parquet("fake_path.parquet")
    assert not df.empty
    required = ["uid", "title", "description", "location_name", "date_begin"]
    for col in required:
        assert col in df.columns, f"Colonne '{col}' absente."

@patch("scripts.faiss_retriever.FAISSRetriever.retrieve")
def test_retriever_mock(mock_retrieve):
    """Vérifie que la méthode retrieve renvoie le bon format."""
    mock_retrieve.return_value = [
        Document(page_content="Concert Test à Dijon",
                 metadata={"title": "Concert", "location_name": "Dijon"})
    ]
    retriever = FAISSRetriever()
    results = retriever.retrieve("concert", k=1)

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].metadata["title"] == "Concert"
    mock_retrieve.assert_called_once_with("concert", k=1)

@patch("scripts.faiss_retriever.faiss.read_index")
@patch("scripts.faiss_retriever.pd.read_parquet")
def test_faiss_init_missing_text_column(mock_read_parquet, mock_read_index):
    """Doit lever une erreur si la colonne 'text' est absente."""
    mock_read_index.return_value = MagicMock(ntotal=2)
    mock_read_parquet.return_value = pd.DataFrame({"title": ["a", "b"]})
    with pytest.raises(ValueError, match="colonne 'text' est absente"):
        FAISSRetriever()

@patch("scripts.faiss_retriever.faiss.read_index")
@patch("scripts.faiss_retriever.pd.read_parquet")
def test_faiss_init_misaligned(mock_read_parquet, mock_read_index):
    """Doit lever une erreur si l'index et les métadonnées ne sont pas alignés."""
    mock_read_index.return_value = MagicMock(ntotal=3)
    mock_read_parquet.return_value = pd.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError, match="pas alignés"):
        FAISSRetriever()

@patch("scripts.faiss_retriever.faiss.read_index")
@patch("scripts.faiss_retriever.pd.read_parquet")
@patch("scripts.faiss_retriever.SentenceTransformer")
def test_retrieve_k_zero(mock_sentence, mock_read_parquet, mock_read_index):
    """k=0 doit retourner une liste vide."""
    mock_read_index.return_value = MagicMock(ntotal=5)
    mock_read_parquet.return_value = pd.DataFrame({"text": ["x"] * 5})
    mock_sentence.return_value.encode.return_value = np.array([0.1, 0.2])
    retriever = FAISSRetriever()
    assert retriever.retrieve("test", k=0) == []

@patch("scripts.faiss_retriever.faiss.read_index")
@patch("scripts.faiss_retriever.pd.read_parquet")
@patch("scripts.faiss_retriever.SentenceTransformer")
def test_retrieve_negative_index(mock_sentence, mock_read_parquet, mock_read_index):
    """Les indices négatifs doivent être ignorés."""
    mock_index = MagicMock(ntotal=2)
    mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[-1, 0]]))
    mock_read_index.return_value = mock_index
    mock_read_parquet.return_value = pd.DataFrame({
        "text": ["doc0", "doc1"],
        "title": ["t0", "t1"]
    })
    mock_sentence.return_value.encode.return_value = np.array([0.1, 0.2])
    retriever = FAISSRetriever()
    results = retriever.retrieve("test", k=2)
    # Seul l'index 0 est conservé
    assert len(results) == 1
    assert results[0].metadata["title"] == "t0"

# ----------------------------------------------------------------------
# Test unitaire avec un vrai index FAISS (et embeddings mockés)
# ----------------------------------------------------------------------

@patch("scripts.faiss_retriever.SentenceTransformer")
def test_retrieve_with_real_index(mock_sentence, real_index_fixture):
    """
    Utilise un véritable index FAISS (créé dans un dossier temporaire).
    Les embeddings sont mockés pour renvoyer des vecteurs prédéfinis.
    Ce test couvre l'intégralité du flux de retrieve().
    """
    tmp_path, meta_df = real_index_fixture

    # On force le chargement de l'index depuis le dossier temporaire
    with patch("scripts.faiss_retriever.INDEX_FILE",
               tmp_path / "faiss_index.bin"):
        with patch("scripts.faiss_retriever.METADATA_PATH",
                   tmp_path / "metadata.parquet"):
            # L'encodeur doit retourner un vecteur de dimension 384
            # pour correspondre à l'index.
            mock_model = mock_sentence.return_value
            mock_model.encode.return_value = np.random.rand(384).astype("float32")

            # Instanciation du retriever (le vrai index est chargé)
            retriever = FAISSRetriever()

            # Vérification de la bonne initialisation
            assert retriever.index.ntotal == 3
            assert len(retriever.metadata) == 3

            # Appel à retrieve
            results = retriever.retrieve("concert La Vapeur", k=2)

            # On attend 2 résultats, correspondant aux 2 plus proches vecteurs
            assert len(results) == 2
            for doc in results:
                assert isinstance(doc, Document)
                assert "title" in doc.metadata
                assert "location_name" in doc.metadata
                # Le texte doit faire partie du contenu original
                assert doc.page_content in meta_df["text"].values

            # Vérifier que encode a bien été appelé avec la requête
            mock_model.encode.assert_called_once_with("concert La Vapeur")

# ----------------------------------------------------------------------
# Tests d'intégration (utilisent les vrais fichiers, s'ils existent)
# ----------------------------------------------------------------------

# Chemins réels pour les tests d'intégration
DATA_PROCESSED_PATH = Path("data/processed/events.parquet")
INDEX_DIR = Path("data/vector_store")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.parquet"

@pytest.mark.skipif(
    not DATA_PROCESSED_PATH.exists(),
    reason="Fichier events.parquet introuvable (Integration)"
)
def test_processed_events_real():
    """Vérifie la présence et le format du vrai events.parquet."""
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    assert not df.empty, "Le fichier events.parquet est vide."

    required_cols = ["uid", "title", "description", "location_name", "date_begin"]
    for col in required_cols:
        assert col in df.columns, f"Colonne '{col}' absente de events.parquet."

@pytest.mark.skipif(
    not INDEX_FILE.exists() or not METADATA_PATH.exists(),
    reason="Fichiers FAISS introuvables (Integration)"
)
def test_faiss_files_real():
    """Vérifie que le vrai index et les métadonnées de l'index existent."""
    meta_df = pd.read_parquet(METADATA_PATH)
    assert not meta_df.empty, "Le fichier de métadonnées est vide."

@pytest.mark.skipif(
    not INDEX_FILE.exists(),
    reason="Index FAISS introuvable (Integration)"
)
def test_retriever_search_real():
    """Vérifie le fonctionnement du vrai Retriever FAISS."""
    retriever = FAISSRetriever()
    assert retriever.index.ntotal > 0, "L'index FAISS est vide."

    results = retriever.retrieve("concert La Vapeur", k=3)
    assert len(results) == 3, f"Attendu 3 résultats, obtenu {len(results)}."
    for r in results:
        assert isinstance(r, Document)
        assert "title" in r.metadata
        assert "location_name" in r.metadata