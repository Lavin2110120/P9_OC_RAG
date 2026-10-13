"""Nettoyage des événements bruts et construction du champ 'text' pour le RAG."""
import logging
import warnings

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from config import settings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TEXT_COLUMNS = ["title", "description", "long_description", "location_name", "city", "address"]


def clean_html(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def build_text(row: pd.Series) -> str:
    """Texte enrichi (titre, dates, lieu, description) pour l'indexation."""
    parts = []
    if row["title"]:
        parts.append(row["title"])
    if pd.notna(row["date_begin"]):
        dates = f"Du {row['date_begin']:%d/%m/%Y}"
        if pd.notna(row["date_end"]):
            dates += f" au {row['date_end']:%d/%m/%Y}"
        parts.append(dates + ".")
    lieu = ", ".join(p for p in [row["location_name"], row["address"], row["city"]] if p)
    if lieu:
        parts.append(f"Lieu : {lieu}.")
    for col in ("description", "long_description"):
        if row[col]:
            parts.append(row[col])
    return " ".join(parts).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["uid"]).copy()

    # Dates : l'API renvoie des ISO 8601 avec fuseau → on normalise en UTC
    for col in ("date_begin", "date_end"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    date_min = pd.Timestamp(settings.DATE_MIN).tz_localize("UTC")
    date_max = pd.Timestamp(settings.DATE_MAX).tz_localize("UTC") + pd.Timedelta(days=1)
    end_or_begin = df["date_end"].fillna(df["date_begin"])
    mask = df["date_begin"].isna() | ((df["date_begin"] <= date_max) & (end_or_begin >= date_min))
    df = df[mask].copy()

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_html)

    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def main():
    log.info(f"Chargement de {settings.RAW_FILE}")
    df_raw = pd.read_parquet(settings.RAW_FILE)
    log.info(f"✅ {len(df_raw)} événements chargés")

    df_clean = clean_dataframe(df_raw)
    log.info(f"✅ {len(df_clean)} événements nettoyés")

    settings.PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(settings.PROCESSED_FILE, index=False)
    log.info(f"✅ Données sauvegardées dans {settings.PROCESSED_FILE}")


if __name__ == "__main__":
    main()