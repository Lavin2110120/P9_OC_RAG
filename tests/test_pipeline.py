import pandas as pd
import numpy as np
import logging
import faiss
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des fichiers (adaptés à votre structure)
DATA_PROCESSED_PATH = Path("data/processed/events.parquet")
CHUNKS_PATH = Path("data/processed/chunks.parquet")
INDEX_DIR = Path("data/vector_store")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.parquet"

# Configuration des embeddings
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"

# --- Classe pour adapter FAISS ---
class LocalFAISSIndex:
    """Classe pour charger et interroger un index FAISS local avec des métadonnées."""

    def __init__(self, index_path: Path, metadata_df: pd.DataFrame, embeddings_model: SentenceTransformer):
        self.index = faiss.read_index(str(index_path))
        self.metadata_df = metadata_df
        self.embeddings_model = embeddings_model

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Effectue une recherche de similarité et retourne des documents avec métadonnées."""
        query_embedding = self.embeddings_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding])
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                metadata = self.metadata_df.iloc[idx].to_dict()
                text = metadata.pop("text", "")
                results.append(Document(page_content=text, metadata=metadata))
        return results

# --- Fonctions utilitaires ---
def load_embeddings() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

def load_index() -> LocalFAISSIndex:
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Fichier {INDEX_FILE} introuvable.")
    metadata_df = pd.read_parquet(METADATA_PATH)
    embeddings = load_embeddings()
    return LocalFAISSIndex(INDEX_FILE, metadata_df, embeddings)

def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Fichier {METADATA_PATH} introuvable.")
    return pd.read_parquet(METADATA_PATH)

# --- Tests des données ---
def test_data_processed():
    """Vérifie que le fichier `events.parquet` traité existe et est cohérent."""
    assert DATA_PROCESSED_PATH.exists(), f"❌ Fichier {DATA_PROCESSED_PATH} introuvable."
    df = pd.read_parquet(DATA_PROCESSED_PATH)
    assert not df.empty, "❌ Le fichier traité est vide."
    assert "text" in df.columns, "❌ Le champ 'text' est manquant dans les données traitées."
    assert "uid" in df.columns, "❌ Le champ 'uid' est manquant dans les données traitées."
    logger.info("✅ Test des données traitées : OK.")

def test_chunks():
    """Vérifie que les chunks sont correctement générés (adapté à vos données)."""
    assert CHUNKS_PATH.exists(), f"❌ Fichier {CHUNKS_PATH} introuvable."
    df = pd.read_parquet(CHUNKS_PATH)
    assert not df.empty, "❌ Aucun chunk généré."
    assert "text" in df.columns, "❌ Le champ 'text' est manquant dans les chunks."
    assert "uid" in df.columns, "❌ Le champ 'uid' est manquant dans les chunks."
    assert all(df["text"].str.len() > 0), "❌ Certains chunks sont vides."
    logger.info(f"✅ Test des chunks : OK ({len(df)} chunks valides).")

# --- Tests de l'index ---
def test_index_exists():
    """Vérifie que l'index FAISS et les métadonnées existent."""
    assert INDEX_DIR.exists(), f"❌ Dossier {INDEX_DIR} introuvable."
    assert INDEX_FILE.exists(), f"❌ Fichier {INDEX_FILE} introuvable."
    assert METADATA_PATH.exists(), f"❌ Fichier {METADATA_PATH} introuvable."
    logger.info("✅ Test de l'existence de l'index : OK.")

def test_index_size():
    """Vérifie que l'index a la bonne taille."""
    index = load_index()
    metadata_df = load_metadata()
    assert index.ntotal == len(metadata_df), \
        f"❌ Incohérence : {index.ntotal} entrées dans l'index vs {len(metadata_df)} métadonnées."
    logger.info(f"✅ Test de la taille de l'index : OK ({len(metadata_df)} entrées).")

def test_index_metadata():
    """Vérifie que les métadonnées de l'index sont complètes (adapté à vos champs)."""
    metadata_df = load_metadata()
    required_fields = ["uid", "title", "date_begin", "location_name"]  # Champs présents dans vos données
    for field in required_fields:
        assert field in metadata_df.columns, f"❌ Champ '{field}' manquant dans les métadonnées."
    logger.info("✅ Test des métadonnées de l'index : OK.")

# --- Tests de recherche ---
def test_search_basic():
    """Teste une recherche basique dans l'index (adapté à vos métadonnées)."""
    index = load_index()
    query = "concert à Dijon"
    results = index.similarity_search(query, k=3)

    assert len(results) == 3, f"❌ La recherche a retourné {len(results)} résultats au lieu de 3."
    for i, doc in enumerate(results, 1):
        assert isinstance(doc, Document), f"❌ Résultat {i} n'est pas un Document."
        assert "uid" in doc.metadata, f"❌ Métadonnées 'uid' manquantes pour le résultat {i}."
        assert "title" in doc.metadata, f"❌ Métadonnées 'title' manquantes pour le résultat {i}."
        logger.info(f"  {i}. {doc.metadata.get('title', 'Sans titre')} (uid: {doc.metadata.get('uid')})")

    logger.info("✅ Test de recherche basique : OK.")

def test_search_empty_query():
    """Teste une recherche avec une requête vide."""
    index = load_index()
    results = index.similarity_search("", k=3)
    assert len(results) == 3, "❌ La recherche avec une requête vide a échoué."
    logger.info("✅ Test de recherche avec requête vide : OK.")

def test_search_no_results():
    """Teste une recherche avec une requête aléatoire."""
    index = load_index()
    query = "xyzabc123"
    results = index.similarity_search(query, k=3)
    assert len(results) == 3, "❌ La recherche a retourné un nombre inattendu de résultats."
    logger.info("✅ Test de recherche sans résultats : OK.")

# --- Exécution des tests ---
def run_all_tests():
    logger.info("🔍 Début des tests du pipeline RAG...")

    logger.info("\n--- Tests des données ---")
    test_data_processed()
    test_chunks()

    logger.info("\n--- Tests de l'index ---")
    test_index_exists()
    test_index_size()
    test_index_metadata()

    logger.info("\n--- Tests de recherche ---")
    test_search_basic()
    test_search_empty_query()
    test_search_no_results()

    logger.info("\n✅ Tous les tests ont réussi !")

if __name__ == "__main__":
    run_all_tests()