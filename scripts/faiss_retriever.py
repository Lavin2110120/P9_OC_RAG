import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Configuration
INDEX_DIR = Path("data/vector_store")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.parquet"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"

class FAISSRetriever:
    """Retriever personnalisé pour interagir avec votre index FAISS."""

    def __init__(self):
        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata_df = pd.read_parquet(METADATA_PATH)
        self.embeddings_model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Récupère les `k` événements les plus pertinents pour une requête."""
        query_embedding = self.embeddings_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding])

        # Recherche dans FAISS
        distances, indices = self.index.search(query_embedding, k)

        # Construction des résultats
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                metadata = self.metadata_df.iloc[idx].to_dict()
                text = metadata.pop("text", "")
                results.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
        return results

    def get_retriever(self):
        """Retourne un retriever compatible avec LangChain."""
        return self.retrieve