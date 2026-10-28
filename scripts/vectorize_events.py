"""Vectorise le parquet nettoyé et construit l'index FAISS + métadonnées.

Usage :
    python -m scripts.vectorize_events
"""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from config import settings

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_data(file_path: Path) -> pd.DataFrame:
    log.info("Lecture : %s", file_path)
    return pd.read_parquet(file_path)


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Réutilise le texte construit par clean_events.py."""
    if "text" not in df.columns:
        raise ValueError(
            "La colonne 'text' est absente du parquet nettoyé. "
            "Relancez : python -m scripts.clean_events"
        )

    texts = df["text"].fillna("").astype(str).tolist()

    if not texts:
        raise ValueError(
            "Le parquet nettoyé ne contient aucun événement."
        )

    empty_positions = [
        position
        for position, text in enumerate(texts)
        if not text.strip()
    ]
    if empty_positions:
        raise ValueError(
            f"{len(empty_positions)} événement(s) ont un texte vide. "
            "Vérifiez le nettoyage avant de créer l'index."
        )

    return texts


def generate_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    log.info("Chargement du modèle %s...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Génération des embeddings...")
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    log.info("Construction de l'index FAISS (dim=%d, n=%d)", dimension, len(embeddings))
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_index(index, index_path: Path) -> None:
    faiss.write_index(index, str(index_path))
    log.info("✅ Index : %s", index_path)


def save_metadata(df: pd.DataFrame, metadata_path: Path) -> None:
    df.to_parquet(metadata_path, index=False)
    log.info("✅ Metadata : %s", metadata_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_data(settings.PROCESSED_FILE)
    log.info("Documents : %d", len(df))

    texts = prepare_texts(df)

    embeddings = generate_embeddings(texts, EMBEDDING_MODEL)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(df):
        raise ValueError(
            "Le nombre d'embeddings ne correspond pas "
            "au nombre de lignes de métadonnées."
        )
    index = build_faiss_index(embeddings)

    settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    save_index(index, settings.INDEX_FILE)
    save_metadata(df, settings.METADATA_FILE)

    test_query = "concert La Vapeur octobre 2026"
    log.info("Test de recherche : %r", test_query)


if __name__ == "__main__":
    main()
