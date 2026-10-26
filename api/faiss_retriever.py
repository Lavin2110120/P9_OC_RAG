"""Retriever FAISS singleton basé sur un index + métadonnées Parquet."""
import logging
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"


class FAISSRetriever:
    """Singleton chargeant l'index FAISS et les métadonnées au démarrage."""

    _instance: Optional["FAISSRetriever"] = None

    def __init__(self) -> None:
        self.index = faiss.read_index(str(settings.INDEX_FILE))

        metadata_path = settings.METADATA_FILE
        self.metadata_df = pd.read_parquet(metadata_path)

        if "text" not in self.metadata_df.columns:
            raise ValueError(
                f"Colonne 'text' absente de {metadata_path}. "
                "Régénérez via `python -m scripts.vectorize_events`."
            )

        if self.index.ntotal != len(self.metadata_df):
            raise ValueError(
                f"Incohérence : index={self.index.ntotal} vecteurs, "
                f"métadonnées={len(self.metadata_df)} lignes."
            )

        if "uid" in self.metadata_df.columns:
            self.metadata_df["uid"] = self.metadata_df["uid"].astype(str)

        logger.info(
            "FAISSRetriever prêt : %d vecteurs, %d lignes de métadonnées.",
            self.index.ntotal, len(self.metadata_df),
        )

    @property
    def metadata(self) -> pd.DataFrame:
        return self.metadata_df

    @classmethod
    def get_instance(cls) -> "FAISSRetriever":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        if k <= 0 or self.index.ntotal == 0:
            return []

        model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)
        query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k_effective = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_effective)

        documents: List[Document] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            row = self.metadata_df.iloc[idx]
            text = row.get("text", "") or ""
            doc = Document(
                page_content=text,
                metadata={
                    "uid": row.get("uid", ""),
                    "title": row.get("title", ""),
                    "location_name": row.get("location_name", ""),
                    "date_begin": str(row.get("date_begin", "")),
                    "distance": float(dist),
                },
            )
            documents.append(doc)

        return documents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Requête à tester")
    parser.add_argument("-k", type=int, default=3, help="Nombre de résultats")
    args = parser.parse_args()

    retriever = FAISSRetriever()
    results = retriever.retrieve(args.query, k=args.k)
    if not results:
        print("Aucun résultat.")
    else:
        for i, doc in enumerate(results, 1):
            m = doc.metadata
            print(f"{i}. {m.get('title', '?')} (uid={m.get('uid', '?')}, "
                  f"date={m.get('date_begin', '?')})")
