"""Vectorisation des événements avec sentence-transformers et FAISS."""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURATION ---
# Chemin vers votre fichier Parquet (à adapter si nécessaire)
PARQUET_FILE = Path("C:/Users/15GIRAV/Moi/Boulot/OpenClassrooms/P9_OC_RAG/data/processed/events.parquet")
# Dossier de sortie pour l'index vectoriel
VECTOR_STORE_DIR = Path("data/vector_store")
# Modèle d'embedding (léger et performant)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Créer le dossier de sortie si inexistant
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers de sortie
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.parquet"

# Configurer le logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --- FONCTIONS ---
def load_data(file_path: Path) -> pd.DataFrame:
    """Charge les données depuis le fichier Parquet."""
    log.info(f"Chargement des données depuis {file_path}...")
    return pd.read_parquet(file_path)

def prepare_texts(df: pd.DataFrame) -> list[str]:
    """Prépare les textes à vectoriser en combinant les champs pertinents."""
    texts = []
    for _, row in df.iterrows():
        # Combinaison des champs pour un contexte riche
        text = f"""
        Titre: {row['title']}
        Description: {row['description']}
        Détails: {row['long_description']}
        Mots-clés: {row['keywords']}
        Lieu: {row['location_name']}
        Date: {row['date_begin']}
        """
        texts.append(text.strip())
    return texts

def generate_embeddings(texts: list[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Génère les embeddings pour une liste de textes."""
    log.info(f"Chargement du modèle {model_name}...")
    model = SentenceTransformer(model_name)

    log.info("Génération des embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,          # Taille des lots pour optimiser la mémoire
        show_progress_bar=True, # Barre de progression
        convert_to_numpy=True,  # Retourne un tableau NumPy
        normalize_embeddings=True  # Normalisation pour FAISS
    )
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Construit un index FAISS pour la recherche vectorielle."""
    dimension = embeddings.shape[1]
    log.info(f"Construction de l'index FAISS (dimension: {dimension})...")

    # IndexFlatL2 : recherche exacte avec distance L2 (parfait pour commencer)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index_and_metadata(index: faiss.Index, df: pd.DataFrame, index_path: Path, metadata_path: Path):
    """Sauvegarde l'index FAISS et les métadonnées."""
    log.info(f"Sauvegarde de l'index FAISS dans {index_path}...")
    faiss.write_index(index, str(index_path))

    # Sauvegarder uniquement les métadonnées utiles pour le RAG
    metadata = df[["uid", "title", "date_begin", "location_name", "url", "description"]].copy()
    log.info(f"Sauvegarde des métadonnées dans {metadata_path}...")
    metadata.to_parquet(metadata_path)

# --- EXÉCUTION ---
def main():
    try:
        # 1. Charger les données
        df = load_data(PARQUET_FILE)
        log.info(f"✅ {len(df)} événements chargés.")

        # 2. Préparer les textes
        texts = prepare_texts(df)
        log.info(f"✅ {len(texts)} textes préparés pour la vectorisation.")

        # 3. Générer les embeddings
        embeddings = generate_embeddings(texts)
        log.info(f"✅ Embeddings générés (shape: {embeddings.shape}).")

        # 4. Construire l'index FAISS
        index = build_faiss_index(embeddings)

        # 5. Sauvegarder
        save_index_and_metadata(index, df, INDEX_FILE, METADATA_FILE)
        log.info(f"✅ Index et métadonnées sauvegardés dans {VECTOR_STORE_DIR}.")

        # 6. Test rapide : rechercher un événement similaire
        test_query = "concert La Vapeur octobre 2026"
        test_embedding = generate_embeddings([test_query], EMBEDDING_MODEL)
        D, I = index.search(test_embedding, k=3)  # Top 3 résultats
        log.info("\n🔍 Test de recherche avec la requête : '%s'", test_query)
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            log.info(f"  {i+1}. {df.iloc[idx]['title']} (distance: {dist:.4f})")

    except Exception as e:
        log.error(f"❌ Erreur: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()