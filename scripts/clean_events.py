import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- Configuration ---
DATA_RAW_PATH = Path("data/raw/events.parquet")
DATA_PROCESSED_PATH = Path("data/processed/events.parquet")
DATE_MIN = "2025-10-01"  # Date minimale pour filtrer les événements
DATE_MAX = "2027-12-31"  # Date maximale pour filtrer les événements

def clean_html(text: Optional[str]) -> str:
    """
    Nettoie un texte HTML en :
    1. Supprimant les balises HTML (y compris <script>, <style>).
    2. Décodant les entités HTML (&, &eacute;, etc.).
    3. Normalisant les espaces (1 espace entre les mots, pas d'espaces en début/fin).
    4. Remplaçant les sauts de ligne multiples par un seul espace.
    """
    if pd.isna(text):
        return ""

    # Suppression des balises HTML avec BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(" ", strip=False)

    # Décodage des entités HTML (ex: & -> &)
    cleaned_text = cleaned_text.replace("&", "&")

    # Normalisation des espaces : remplacer les séquences d'espaces/tabulations/sauts de ligne par un seul espace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def build_text(row: pd.Series) -> str:
    """
    Construit un champ 'text' pour chaque événement en combinant :
    - Titre
    - Description (nettoyée)
    - Long description (nettoyée)
    - Lieu
    - Dates (début et fin)
    - Catégories
    """
    parts = []

    # Titre
    if pd.notna(row["title"]):
        parts.append(f"Titre: {row['title']}")

    # Description
    if pd.notna(row["description"]):
        parts.append(f"Description: {clean_html(row['description'])}")

    # Long description
    if pd.notna(row["long_description"]):
        parts.append(f"Détails: {clean_html(row['long_description'])}")

    # Lieu
    if pd.notna(row["location_name"]):
        parts.append(f"Lieu: {row['location_name']}")
    if pd.notna(row["city"]):
        parts.append(f"Ville: {row['city']}")
    if pd.notna(row["address"]):
        parts.append(f"Adresse: {row['address']}")

    # Dates
    if pd.notna(row["date_begin"]):
        parts.append(f"Début: {row['date_begin']}")
    if pd.notna(row["date_end"]):
        parts.append(f"Fin: {row['date_end']}")

    # Catégories
    if pd.notna(row["categories"]):
        categories = eval(row["categories"]) if isinstance(row["categories"], str) else row["categories"]
        if isinstance(categories, list) and categories:
            parts.append(f"Catégories: {', '.join(categories)}")

    return " | ".join(parts)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame des événements :
    1. Supprime les doublons (sur 'uid').
    2. Filtre les événements hors période [DATE_MIN, DATE_MAX].
    3. Nettoie les champs texte (HTML, espaces, etc.).
    4. Construit le champ 'text' pour le RAG.
    5. Conserve toutes les métadonnées utiles.
    """
    # Suppression des doublons
    df = df.drop_duplicates(subset=["uid"])

    # Filtrage temporel : garder les événements qui commence avant DATE_MAX ET finissent après DATE_MIN
    df = df.copy()  # Éviter SettingWithCopyWarning
    df["date_begin"] = pd.to_datetime(df["date_begin"], errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")

    # Remplacer date_end par date_begin si vide
    df["date_end"] = df["date_end"].fillna(df["date_begin"])

    # Filtrer les événements dans la période
    mask = (
        (df["date_begin"] <= DATE_MAX) &
        (df["date_end"] >= DATE_MIN)
    )
    df = df[mask]

    # Nettoyage des champs texte
    text_columns = ["title", "description", "long_description", "location_name", "city", "address"]
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_html)

    # Construction du champ 'text' pour le RAG
    df["text"] = df.apply(build_text, axis=1)

    # Suppression des lignes sans texte (pas de titre, description, ou long_description)
    df = df[df["text"].str.len() > 0]

    return df

def main():
    """Charge, nettoie et sauvegarde les événements."""
    # Chargement des données brutes
    df_raw = pd.read_parquet(DATA_RAW_PATH)

    # Nettoyage
    df_clean = clean_dataframe(df_raw)

    # Sauvegarde
    df_clean.to_parquet(DATA_PROCESSED_PATH)
    print(f"✅ Données nettoyées sauvegardées dans {DATA_PROCESSED_PATH} ({len(df_clean)} événements).")

if __name__ == "__main__":
    main()