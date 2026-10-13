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
        if "text" not in self.metadata_df.columns:
            raise ValueError(
                f"La colonne 'text' est absente de {METADATA_PATH}. "
                "Reconstruisez l'index avec vectorize_events.py."
            )
        if self.index.ntotal != len(self.metadata_df):
            raise ValueError(
                "L'index FAISS et les métadonnées ne sont pas alignés : "
                f"{self.index.ntotal} vecteurs pour {len(self.metadata_df)} lignes."
            )
        self.embeddings_model = SentenceTransformer(MODEL_NAME, device=EMBEDDING_DEVICE)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Récupère les `k` événements les plus pertinents pour une requête."""
        query_embedding = self.embeddings_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding])

        # Recherche dans FAISS (ne pas demander plus de voisins que l'index)
        if k <= 0:
            return []
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)

        # Construction des résultats
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:
                metadata = self.metadata_df.iloc[int(idx)].to_dict()
                text = metadata.pop("text")
                if pd.isna(text):
                    text = ""
                else:
                    text = str(text)
                results.append(Document(
                    page_content=text,
                    metadata=metadata
                ))
        return results

    def get_retriever(self):
        """Retourne un retriever compatible avec LangChain."""
        return self.retrieve

if __name__ == "__main__":
    print("Initialisation du Retriever...")
    retriever = FAISSRetriever()
    
    query = "Concert de rap à Dijon"
    print(f"\nRecherche pour : '{query}'")
    
    results = retriever.retrieve(query, k=2)
    for i, doc in enumerate(results):
        print(f"\n--- Résultat {i+1} ---")
        print(f"Titre : {doc.metadata.get('title')}")
        print(f"Lieu : {doc.metadata.get('location_name')}")
        print(f"Date : {doc.metadata.get('date_begin')}")