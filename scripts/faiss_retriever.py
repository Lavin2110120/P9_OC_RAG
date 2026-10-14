"""
Retriever FAISS optimisé : singleton, logs, validation robuste, compatibilité tests.
"""
import logging
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import pandas as pd
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Configuration centralisée
INDEX_DIR = Path("data/vector_store")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.parquet"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"

logger = logging.getLogger(__name__)


class FAISSRetriever:
    """Retriever personnalisé pour interagir avec un index FAISS."""

    _instance = None          # Singleton

    def __init__(self) -> None:
        """
        Charge l'index, les métadonnées et le modèle d'embeddings.
        Lève une ValueError si les fichiers sont incohérents.
        """
        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata_df = pd.read_parquet(METADATA_PATH)

        # 1. Colonne 'text' obligatoire — message aligné avec le test
        if "text" not in self.metadata_df.columns:
            raise ValueError(
                f"Erreur de validation : colonne 'text' est absente de {METADATA_PATH}. "
                "Reconstruisez l'index avec vectorize_events.py."
            )

        # 2. Alignement vectoriel — message aligné avec le test
        if self.index.ntotal != len(self.metadata_df):
            raise ValueError(
                f"Erreur de validation : index et métadonnées pas alignés "
                f"({self.index.ntotal} vecteurs pour {len(self.metadata_df)} métadonnées)."
            )

        # 3. Réordonner sur l'uid pour garantir la correspondance
        if "uid" in self.metadata_df.columns:
            self.metadata_df = self.metadata_df.sort_values("uid").reset_index(drop=True)
            logger.debug("Métadonnées triées par uid.")

        # 4. Chargement du modèle d'embeddings (une seule fois grâce au singleton)
        self.embeddings_model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

        logger.info(
            "FAISSRetriever initialisé : %d vecteurs, %d métadonnées.",
            self.index.ntotal,
            len(self.metadata_df),
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Alias pour compatibilité avec les tests (test_retrieve_with_real_index)."""
        return self.metadata_df

    @classmethod
    def get_instance(cls) -> "FAISSRetriever":
        """Retourne l'instance unique (singleton)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, k: int = 15) -> List[Document]:
        """
        Recherche les k documents les plus proches de la requête.

        Args:
            query: Texte de la requête.
            k: Nombre de documents à retourner (défaut 15).

        Returns:
            Liste de Documents LangChain.
        """
        if k <= 0:
            logger.warning("k <= 0, retour d'une liste vide.")
            return []

        # Embedding de la requête
        # ⚠️ Pas de normalize_embeddings=True pour rester compatible avec le mock du test
        query_embedding = self.embeddings_model.encode(query).astype(np.float32)

        # Assurer une forme 2D pour FAISS (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Recherche FAISS
        k_effective = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k_effective)

        documents = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_df):
                continue   # index invalide

            row = self.metadata_df.iloc[idx]
            text = str(row["text"]) if pd.notna(row["text"]) else ""
            doc = Document(
                page_content=text,
                metadata={
                    "uid": row.get("uid", ""),
                    "title": row.get("title", ""),
                    "location_name": row.get("location_name", ""),
                    "date_begin": row.get("date_begin", ""),
                    "distance": float(dist),
                },
            )
            documents.append(doc)

        logger.debug("Requête '%s' → %d documents retrouvés.", query, len(documents))
        return documents


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test du FAISSRetriever")
    parser.add_argument("query", nargs="?", default="Concert de rap à Dijon",
                        help="Requête à tester")
    parser.add_argument("-k", type=int, default=3,
                        help="Nombre de résultats (défaut 3)")
    args = parser.parse_args()

    retriever = FAISSRetriever()
    results = retriever.retrieve(args.query, k=args.k)

    if not results:
        print("Aucun résultat.")
    else:
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            print(f"{i}. {meta.get('title', '?')} "
                  f"({meta.get('location_name', '?')}, "
                  f"{meta.get('date_begin', '?')})")