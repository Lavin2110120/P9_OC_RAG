# scripts/build_faiss_index.py
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# --- CONFIGURATION ---
INPUT_FILE = Path("data/processed/chunks.parquet")
VECTOR_STORE_DIR = Path("data/vector_store")
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # Doit correspondre à la dimension du modèle
BATCH_SIZE = 32


INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Charge les chunks ou les événements."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    if "text" not in df.columns:
        if "description" not in df.columns:
            raise ValueError("Ni 'text' ni 'description' trouvés dans les données.")
        df["text"] = df["description"]
    return df

def build_index(df: pd.DataFrame) -> Tuple[faiss.Index, pd.DataFrame]:
    """Génère les embeddings et construit un index FAISS adapté à la taille des données."""
    # Charger le modèle
    log.info(f"Chargement du modèle {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Générer les embeddings par lots
    embeddings = []
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]["text"].tolist()
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings).astype('float32')
    log.info(f"✅ {len(embeddings)} embeddings générés (dimension: {embeddings.shape[1]}).")

    # Vérifier la dimension
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Dimension des embeddings ({embeddings.shape[1]}) != EMBEDDING_DIM ({EMBEDDING_DIM}). "
            f"Mettez à jour EMBEDDING_DIM ou changez de modèle."
        )
    
    if len(embeddings) < 1000:  # Seuil pour basculer sur IndexFlatL2
        log.info("🔹 Utilisation de IndexFlatL2 (dataset trop petit pour IVF).")
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
    else:
        # Pour les grands datasets, utilisez IVF avec nlist adapté
        nlist = min(100, len(embeddings) // 10)  # nlist = 10% de la taille (ex: 100 pour 1000 vecteurs)
        log.info(f"🔹 Entraînement IVF avec nlist={nlist}.")
        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_L2)
        index.train(embeddings)  # Entraînement avec TOUS les vecteurs

    # Ajout des vecteurs (commun aux deux cas)
    index.add(embeddings)
    log.info(f"✅ {index.ntotal} vecteurs ajoutés à l'index.")

    # Sauvegarde
    faiss.write_index(index, str(INDEX_FILE))
    df.to_parquet(METADATA_FILE)
    return index, df

def verify_index(index: faiss.Index, df: pd.DataFrame) -> None:
    """Vérifie l'intégrité de l'index FAISS."""
    # 1. Vérifier la taille
    if index.ntotal != len(df):
        raise ValueError(
            f"Incohérence : {index.ntotal} vecteurs dans FAISS vs {len(df)} dans les métadonnées."
        )
    log.info(f"✅ Taille cohérente : {index.ntotal} entrées.")

    # 2. Vérifier que les vecteurs ne sont pas nuls
    sample_indices = np.random.choice(index.ntotal, min(10, index.ntotal), replace=False).astype(np.int64).tolist()
    sample_vectors = np.array([index.reconstruct(i) for i in sample_indices])
    if np.any(np.isnan(sample_vectors)):
        raise ValueError("❌ Certains vecteurs contiennent des NaN.")
    log.info("✅ Aucun vecteur NaN détecté.")

    # 3. Tester une recherche basique
    test_query = "test"
    test_embedding = np.random.rand(1, EMBEDDING_DIM).astype('float32')
    faiss.normalize_L2(test_embedding)
    D, I = index.search(test_embedding, k=1)
    if D[0][0] < 0:
        raise ValueError("❌ Distance négative détectée (problème de normalisation).")
    log.info("✅ Recherche de test réussie.")

def test_index(index: faiss.Index, metadata_df: pd.DataFrame, query: str = "concert à La Vapeur") -> None:
    """Teste une requête sur l'index."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query], normalize_embeddings=True)

    D, I = index.search(query_embedding, k=3)
    print(f"\n🔍 Requête: '{query}'\n")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        event = metadata_df.iloc[idx]
        date_str = event["date_begin"].strftime('%Y-%m-%d') if pd.notna(event["date_begin"]) else "N/A"
        print(f"{i+1}. {event['title']} (le {date_str}) - Distance: {dist:.4f}")

if __name__ == "__main__":
    try:
        df = load_data()
        index, metadata = build_index(df)
        verify_index(index, metadata)  # ✅ Appel de la vérification
        test_index(index, metadata)
    except Exception as e:
        log.error(f"❌ Erreur fatale : {e}")
        raise