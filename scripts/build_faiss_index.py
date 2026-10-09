import numpy as np
import pandas as pd
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des fichiers
CHUNKS_PATH = Path("data/processed/chunks.parquet")
INDEX_DIR = Path("data/vector_store")
METADATA_PATH = INDEX_DIR / "metadata.parquet"
INDEX_NAME = "faiss_index"

# Configuration des embeddings
MODEL_NAME = "BAAI/bge-m3"  # Modèle multilingue (meilleur pour le français)
EMBEDDING_DEVICE = "cpu"  # Utiliser "cuda" si GPU disponible
NORMALIZE_EMBEDDINGS = True  # Normaliser les vecteurs pour une meilleure similarité

def load_chunks() -> pd.DataFrame:
    """Charge les chunks depuis le fichier Parquet."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Fichier {CHUNKS_PATH} introuvable. Exécutez d'abord `chunks_events.py`.")

    df = pd.read_parquet(CHUNKS_PATH)

    # Vérification des données
    if df.empty:
        raise ValueError("Aucun chunk trouvé dans le fichier. Vérifiez `chunks_events.py`.")
    if "text" not in df.columns:
        raise ValueError("Le champ 'text' est manquant dans les chunks.")
    if "chunk_id" not in df.columns:
        raise ValueError("Le champ 'chunk_id' est manquant dans les chunks.")

    logger.info(f"✅ Chargés {len(df)} chunks depuis {CHUNKS_PATH}.")
    return df

def build_faiss_index(df: pd.DataFrame) -> tuple:
    """
    Construit un index FAISS à partir des chunks et sauvegarde :
    - L'index FAISS (`faiss_index` dans `INDEX_DIR`).
    - Les métadonnées associées (`metadata.parquet`).
    Retourne l'index et le DataFrame des métadonnées.
    """
    # Préparation des documents LangChain
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "chunk_id": row["chunk_id"],
            "uid": row["uid"],
            "title": row.get("title", ""),
            "date_begin": str(row.get("date_begin", "")),
            "date_end": str(row.get("date_end", "")),
            "location_name": row.get("location_name", ""),
            "city": row.get("city", ""),
            "address": row.get("address", ""),
            "categories": row.get("categories", []),
        }
        documents.append(Document(page_content=row["text"], metadata=metadata))

    # Initialisation des embeddings (modèle multilingue)
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS}
    )

    # Création de l'index FAISS
    index = FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    # Sauvegarde de l'index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.save_local(folder_path=str(INDEX_DIR), index_name=INDEX_NAME)
    logger.info(f"✅ Index FAISS sauvegardé dans {INDEX_DIR / INDEX_NAME}.")

    # Sauvegarde des métadonnées
    df.to_parquet(METADATA_PATH)
    logger.info(f"✅ Métadonnées sauvegardées dans {METADATA_PATH}.")

    return index, df

def verify_index(index, metadata_df: pd.DataFrame):
    """Vérifie l'intégrité de l'index FAISS."""
    # Vérification de la taille
    if len(index.docstore._dict) != len(metadata_df):
        raise ValueError(f"Incohérence : {len(index.docstore._dict)} entrées dans l'index vs {len(metadata_df)} métadonnées.")

    # Test de recherche
    test_query = "concert à Dijon"
    try:
        results = index.similarity_search(test_query, k=3)
        logger.info(f"✅ Recherche de test réussie pour '{test_query}' (top 3 résultats).")
        for i, doc in enumerate(results, 1):
            logger.info(f"  {i}. {doc.metadata.get('title', 'Sans titre')} (score: {doc.metadata.get('score', 'N/A')})")
    except Exception as e:
        logger.error(f"❌ Échec de la recherche de test : {e}")
        raise

def main():
    """Pipeline complet : chargement des chunks, construction de l'index, vérification."""
    try:
        # Chargement des chunks
        df_chunks = load_chunks()

        # Construction de l'index FAISS
        index, metadata_df = build_faiss_index(df_chunks)

        # Vérification de l'index
        verify_index(index, metadata_df)

        logger.info("✅ Pipeline terminé avec succès !")
    except Exception as e:
        logger.error(f"❌ Erreur fatale : {e}")
        raise

if __name__ == "__main__":
    main()