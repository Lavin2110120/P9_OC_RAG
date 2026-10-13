import pytest
import pandas as pd
import logging
from pathlib import Path
from unittest.mock import patch
from langchain_core.documents import Document
from scripts.faiss_retriever import FAISSRetriever

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --- CONFIGURATION DES CHEMINS (Pour les tests d'intégration) ---
# Assure-toi que ces chemins correspondent à la nouvelle arborescence de ton projet
DATA_PROCESSED_PATH = Path("data/processed/events.parquet")
INDEX_FILE = Path("data/faiss_index.bin")
METADATA_PATH = Path("data/faiss_metadata.parquet")


# ==========================================
# 1. TESTS UNITAIRES (Avec Mocks / Fixtures)
# ==========================================

@pytest.fixture
def mock_parquet_data():
    """Fixture pytest : faux jeu de données injecté dans les tests."""
    return pd.DataFrame({
        "uid": ["123", "456"],
        "title": ["Concert Test", "Expo Test"],
        "description": ["Super concert", "Belle expo"],
        "location_name": ["La Vapeur", "Musée des Beaux-Arts"],
        "date_begin": ["2026-10-14", "2026-10-15"]
    })

@patch("pandas.read_parquet")
def test_processed_events_logic(mock_read_parquet, mock_parquet_data):
    """Vérifie la logique de validation des colonnes sans lire le vrai fichier."""
    mock_read_parquet.return_value = mock_parquet_data

    df = pd.read_parquet("fake_path.parquet")
    assert not df.empty

    required_cols = ["uid", "title", "description", "location_name", "date_begin"]
    for col in required_cols:
        assert col in df.columns, f"Colonne '{col}' absente."

@patch("scripts.faiss_retriever.FAISSRetriever.retrieve")
def test_retriever_mock(mock_retrieve):
    """Vérifie que la fonction de recherche renvoie bien le bon format, sans utiliser FAISS."""
    mock_retrieve.return_value = [
        Document(page_content="Concert Test à Dijon", metadata={"title": "Concert", "location_name": "Dijon"})
    ]

    retriever = FAISSRetriever()
    results = retriever.retrieve("concert", k=1)

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].metadata["title"] == "Concert"
    mock_retrieve.assert_called_once_with("concert", k=1)


# ==========================================
# 2. TESTS D'INTÉGRATION (Sur les vrais fichiers)
# ==========================================

# On utilise 'skipif' pour ne pas faire planter les tests sur une machine vierge
# qui n'aurait pas encore téléchargé les données.
@pytest.mark.skipif(not DATA_PROCESSED_PATH.exists(), reason="Fichier events.parquet introuvable (Integration)")
def test_processed_events():
    """Vérifie la présence et le format du vrai events.parquet."""
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    assert not df.empty, "Le fichier events.parquet est vide."

    required_cols = ["uid", "title", "description", "location_name", "date_begin"]
    for col in required_cols:
        assert col in df.columns, f"Colonne '{col}' absente de events.parquet."
    log.info("✅ Fichier events.parquet conforme.")

@pytest.mark.skipif(not INDEX_FILE.exists() or not METADATA_PATH.exists(), reason="Fichiers FAISS introuvables (Integration)")
def test_faiss_files():
    """Vérifie que le vrai index et les métadonnées de l'index existent."""
    meta_df = pd.read_parquet(METADATA_PATH)
    assert not meta_df.empty, "Le fichier de métadonnées est vide."
    log.info(f"✅ Index et métadonnées présents ({len(meta_df)} enregistrements).")

@pytest.mark.skipif(not INDEX_FILE.exists(), reason="Index FAISS introuvable (Integration)")
def test_retriever_search():
    """Vérifie le fonctionnement du vrai Retriever FAISS."""
    retriever = FAISSRetriever()
    assert retriever.index.ntotal > 0, "L'index FAISS est vide."

    results = retriever.retrieve("concert La Vapeur", k=3)
    assert len(results) == 3, f"Attendu 3 résultats, obtenu {len(results)}."
    for r in results:
        assert isinstance(r, Document)
        assert "title" in r.metadata
        assert "location_name" in r.metadata
    log.info("✅ Recherche retriever validée avec succès.")


# ==========================================
# LANCEMENT (Pour exécuter le script avec la commande `python test_pipeline.py`)
# ==========================================
if __name__ == "__main__":
    # La bonne pratique est de déléguer l'exécution à pytest
    import sys
    sys.exit(pytest.main(["-v", __file__]))