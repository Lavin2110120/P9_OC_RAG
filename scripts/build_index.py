"""Vectorisation des événements et construction de l'index FAISS.

Usage :
    python scripts/build_index.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INDEX_DIR = settings.ROOT_DIR / "data" / "faiss_index"
CHUNK_SIZE = 1000      # caractères
CHUNK_OVERLAP = 100
BATCH_SIZE = 50        # documents par appel d'embedding (limite les erreurs de débit)


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Transforme chaque événement en Document avec ses métadonnées, puis découpe."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for row in df.itertuples():
        meta = {
            "uid": row.uid,
            "title": row.title,
            "city": row.city,
            "date_begin": str(row.date_begin),
            "date_end": str(row.date_end),
            "url": row.url,
        }
        for chunk in splitter.split_text(row.text):
            docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def main() -> None:
    if not settings.MISTRAL_API_KEY:
        sys.exit("❌ MISTRAL_API_KEY manquante dans .env")
    if not settings.PROCESSED_FILE.exists():
        sys.exit("❌ Données absentes : lance clean_events.py d'abord")

    df = pd.read_parquet(settings.PROCESSED_FILE)
    docs = build_documents(df)
    log.info("%d événements → %d chunks", len(df), len(docs))

    embeddings = MistralAIEmbeddings(
        model="mistral-embed", mistral_api_key=settings.MISTRAL_API_KEY
    )

    # Construction par lots pour limiter les erreurs de débit de l'API
    store = FAISS.from_documents(docs[:BATCH_SIZE], embeddings)
    for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
        store.add_documents(docs[i : i + BATCH_SIZE])
        log.info("  %d / %d chunks vectorisés", min(i + BATCH_SIZE, len(docs)), len(docs))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    log.info("✅ Index FAISS sauvegardé dans %s (%d vecteurs)", INDEX_DIR, store.index.ntotal)


if __name__ == "__main__":
    main()