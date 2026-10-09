# api/rag_system.py
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_mistralai import MistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
INDEX_DIR = Path("data/vector_store")
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_PATH = INDEX_DIR / "metadata.parquet"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MISTRAL_API_KEY = "votre_clé_api_mistral"  # À remplacer ou charger depuis une variable d'environnement
MISTRAL_MODEL = "mistral-small"

# --- Chargement des ressources ---
class RAGSystem:
    def __init__(self):
        self.embeddings_model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.mistral_llm = MistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
        self._load_faiss_index()

    def _load_faiss_index(self):
        """Charge l'index FAISS et les métadonnées."""
        if not INDEX_FILE.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("Index FAISS ou métadonnées introuvables. Veuillez reconstruire l'index.")

        self.index = faiss.read_index(str(INDEX_FILE))
        self.metadata_df = pd.read_parquet(METADATA_PATH)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Récupère les `k` documents les plus pertinents pour une requête."""
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

    def generate_response(self, query: str, k: int = 3) -> str:
        """Génère une réponse augmentée à partir de la requête."""
        # Récupération des documents pertinents
        docs = self.retrieve(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt pour Mistral
        prompt = ChatPromptTemplate.from_template(
            """
            Tu es un assistant spécialisé dans les événements culturels.
            Réponds à la question suivante en utilisant UNIQUEMENT le contexte fourni.
            Si tu ne connais pas la réponse, dis-le clairement.

            **Contexte :**
            {context}

            **Question :** {query}

            **Réponse :**
            """
        )

        # Chaîne de traitement
        chain = (
            {"context": RunnablePassthrough(), "query": RunnablePassthrough()}
            | prompt
            | self.mistral_llm
        )

        return chain.invoke({"context": context, "query": query})

    def rebuild_index(self):
        """Reconstruit l'index FAISS à partir des données brutes."""
        from scripts.build_faiss_index import build_index
        import pandas as pd

        # Charger les données brutes (à adapter selon votre structure)
        df = pd.read_parquet("data/processed/events.parquet")
        index, metadata = build_index(df)

        # Sauvegarder l'index et les métadonnées
        faiss.write_index(index, str(INDEX_FILE))
        metadata.to_parquet(METADATA_PATH)

        # Recharger l'index
        self._load_faiss_index()
        return "Index reconstruit avec succès."