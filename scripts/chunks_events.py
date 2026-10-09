import pandas as pd
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
DATA_PROCESSED_PATH = Path("data/processed/events.parquet")
CHUNKS_OUTPUT_PATH = Path("data/processed/chunks.parquet")

# Configuration du découpage en chunks
CHUNK_SIZE = 500      # Taille maximale d'un chunk (en caractères)
CHUNK_OVERLAP = 50    # Chevauchement entre les chunks (en caractères)

def split_text_into_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Découpe le champ 'text' de chaque événement en chunks, tout en conservant les métadonnées.
    Retourne un DataFrame où chaque ligne est un chunk avec ses métadonnées associées.
    """
    # Initialisation du text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]  # Séparateurs pour découper intelligemment
    )

    chunks_data = []

    for _, row in df.iterrows():
        # Découpage du texte en chunks
        chunks = text_splitter.create_documents([row["text"]])

        for i, chunk in enumerate(chunks):
            chunk_data = {
                "uid": row["uid"],  # Identifiant unique de l'événement
                "chunk_id": f"{row['uid']}_{i}",  # Identifiant unique du chunk
                "text": chunk.page_content,  # Contenu du chunk
                "title": row.get("title", ""),
                "date_begin": row.get("date_begin", ""),
                "date_end": row.get("date_end", ""),
                "location_name": row.get("location_name", ""),
                "city": row.get("city", ""),
                "address": row.get("address", ""),
                "categories": row.get("categories", []),
                "original_index": row.name,  # Index de l'événement original (pour traçabilité)
            }
            chunks_data.append(chunk_data)

    return pd.DataFrame(chunks_data)

def main():
    """Charge les événements nettoyés, découpe en chunks et sauvegarde."""
    # Chargement des données nettoyées
    df_clean = pd.read_parquet(DATA_PROCESSED_PATH)

    # Découpage en chunks
    df_chunks = split_text_into_chunks(df_clean)

    # Sauvegarde
    df_chunks.to_parquet(CHUNKS_OUTPUT_PATH)
    print(f"✅ Chunks générés et sauvegardés dans {CHUNKS_OUTPUT_PATH} ({len(df_chunks)} chunks).")

if __name__ == "__main__":
    main()