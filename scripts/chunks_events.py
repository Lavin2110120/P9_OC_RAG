# scripts/chunk_events.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict

# --- CONFIGURATION ---
INPUT_FILE = Path("data/processed/events.parquet")
OUTPUT_FILE = Path("data/processed/chunks.parquet")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_and_validate_data() -> pd.DataFrame:
    """Charge les données et vérifie leur intégrité."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    # Vérifier les colonnes obligatoires
    required_columns = ["uid", "description", "title", "date_begin", "location_name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes : {missing_columns}")

    # Supprimer les lignes sans description
    df = df.dropna(subset=["description"])
    log.info(f"✅ {len(df)} événements chargés (après suppression des NaN).")
    return df

def split_into_chunks(df: pd.DataFrame) -> List[Dict]:
    """Découpe les descriptions en chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )

    chunks = []
    for _, row in df.iterrows():
        try:
            for i, chunk in enumerate(text_splitter.split_text(row["description"])):
                chunks.append({
                    "uid": f"{row['uid']}_{i}",
                    "event_uid": row["uid"],
                    "text": chunk,
                    "title": row["title"],
                    "date_begin": row["date_begin"],
                    "location_name": row["location_name"],
                })
        except Exception as e:
            log.warning(f"⚠️ Erreur lors du découpage de l'événement {row['uid']}: {e}")

    log.info(f"✅ {len(chunks)} chunks générés.")
    return chunks

if __name__ == "__main__":
    try:
        df = load_and_validate_data()
        chunks = split_into_chunks(df)
        chunks_df = pd.DataFrame(chunks)
        chunks_df.to_parquet(OUTPUT_FILE)
        log.info(f"✅ Chunks sauvegardés dans {OUTPUT_FILE}.")
    except Exception as e:
        log.error(f"❌ Erreur fatale : {e}")
        raise