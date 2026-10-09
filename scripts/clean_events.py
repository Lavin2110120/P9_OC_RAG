"""Nettoyage et structuration des événements bruts (Côte-d'Or)."""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import re
import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from config.geo import extract_location, is_in_target_zone, normalize  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# --- Catégorisation améliorée ---
CATEGORY_KEYWORDS = {
    "musique": ["concert", "musique", "jazz", "classique", "pop", "rap", "rock", "electro", "chanson", "opéra", "festival de musique", "la vapeur"],
    "theatre": ["théâtre", "théâtral", "spectacle", "comédie", "tragédie", "one man show", "pièce"],
    "festival": ["festival", "foire", "salons", "rencontres"],
    "sante_bien_etre": ["santé", "bien-être", "prévention", "retraite", "addictions", "méditation", "yoga"],
    "sport": ["sport", "vélo", "mai à vélo", "course", "marathon", "football", "rugby", "tennis", "asept fc"],
    "biodiversite": ["nature", "botanique", "flore", "faune", "écologie", "environnement", "jardin"],
    "vivre_ensemble": ["jeune public", "seniors", "dijon", "famille", "intergénérationnel", "associatif"],
    "exposition": ["exposition", "vernissage", "galerie", "musée", "art"],
    "cinema": ["cinéma", "film", "projection", "court-métrage"],
    "danse": ["danse", "ballet", "hip-hop", "contemporain"],
    "patrimoine": ["patrimoine", "visite guidée", "monument", "château", "église"],
}
EXCLUDED_CATEGORIES: list[str] = []  # Liste des catégories à exclure (ex: ["sport"])
CATEGORY_PRIORITY = ["musique", "theatre", "festival", "exposition", "cinema", "danse", "patrimoine", "sante_bien_etre", "sport", "biodiversite", "vivre_ensemble"]

# Pré-calcul des mots-clés normalisés pour éviter de recalculer à chaque appel
_NORMALIZED = {cat: {normalize(k) for k in kws} for cat, kws in CATEGORY_KEYWORDS.items()}

# --- Fonctions utilitaires ---
def get_fr(value: Any) -> Optional[str]:
    """Extrait la valeur en français d'un champ potentiellement multilingue."""
    if isinstance(value, dict):
        return value.get("fr") or next(iter(value.values()), None)
    return value

def clean_html(text: str) -> str:
    """Nettoie le HTML et normalise les espaces."""
    if text is None:
        return ""

    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', '', text)

    # Normaliser TOUS les espaces (y compris \n, \t, etc.) en un seul espace
    text = re.sub(r'\s+', ' ', text)

    # Supprimer les espaces en début/fin
    return text.strip()

def clean_list(items: Optional[list]) -> list[str]:
    """Nettoie une liste (supprime les valeurs None/vides et convertit en str)."""
    if not items:
        return []
    return [str(item).strip() for item in items if item and str(item).strip()]

def categorize_all(title: str, keywords: str) -> list[str]:
    """Attribue une ou plusieurs catégories à un événement en fonction de son titre et mots-clés.
    Args:
        title: Titre de l'événement.
        keywords: Mots-clés associés (séparés par des virgules).
    Returns:
        Liste des catégories (triées par priorité).
    """
    if not title and not keywords:
        return ["autre"]

    # Normalisation du titre et des mots-clés
    title_norm = normalize(title)
    keywords_norm = normalize(keywords)

    # Extraction des tokens depuis les mots-clés (séparés par des virgules)
    tokens = {normalize(v.strip()) for v in keywords.split(",") if v.strip()}

    # Recherche des catégories correspondantes
    cats = []
    for cat in CATEGORY_PRIORITY:
        # Vérifie si un token correspond à un mot-clé de la catégorie
        if tokens & _NORMALIZED[cat]:
            cats.append(cat)
        # Ou si le titre/mots-clés contiennent un mot-clé de la catégorie
        elif any(k in title_norm or k in keywords_norm for k in _NORMALIZED[cat]):
            cats.append(cat)

    return cats if cats else ["autre"]

def flatten_event(event: dict) -> dict:
    """Aplatit un événement brut OpenAgenda en un dictionnaire structuré.
    Args:
        event: Événement brut au format OpenAgenda.
    Returns:
        Dictionnaire aplati avec des champs standardisés.
    """
    loc = extract_location(event)
    timings = event.get("timings") or []

    # Récupération des champs avec gestion des valeurs manquantes
    title = clean_html(get_fr(event.get("title")) or "")
    description = clean_html(get_fr(event.get("description")) or "")
    long_description = clean_html(get_fr(event.get("longDescription")) or "")
    keywords = clean_list(get_fr(event.get("keywords")) or [])
    conditions = clean_html(get_fr(event.get("conditions")) or "")
    date_range = clean_html(get_fr(event.get("dateRange")) or "")

    # Gestion des dates (begin/end)
    date_begin = timings[0].get("begin") if timings else None
    date_end = timings[-1].get("end") if timings else None

    # Gestion des agendas (évite les doublons)
    agendas = list(set(str(a) for a in event.get("_agendas", []) if a))
    agenda_title = event.get("_agenda_title") or ""

    return {
        "uid": event.get("uid"),
        "title": title,
        "description": description,
        "long_description": long_description,
        "keywords": ", ".join(keywords) if keywords else "",
        "conditions": conditions,
        "date_range": date_range,
        "date_begin": date_begin,
        "date_end": date_end,
        "location_name": loc.get("name") or "",
        "address": loc.get("address") or "",
        "city": loc.get("city") or "",
        "postal_code": str(loc.get("postal_code") or "").strip(),
        "department": loc.get("department") or "",
        "region": loc.get("region") or "",
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
        "url": event.get("canonicalUrl") or "",
        "agendas": ", ".join(agendas) if agendas else "",
        "agenda": agenda_title,
    }

def build_text(row: pd.Series) -> str:
    """Construit un texte complet pour un événement (utilisé pour le RAG).
    Args:
        row: Ligne d'un DataFrame Pandas.
    Returns:
        Texte structuré avec toutes les informations pertinentes.
    """
    # Construction du lieu
    lieu_parts = [str(p) for p in [row["location_name"], row["address"], row["city"]] if p]
    lieu = ", ".join(lieu_parts) if lieu_parts else ""

    # Liste des parties à inclure (seulement si non vides)
    parts = []
    if row["title"]:
        parts.append(f"Titre : {row['title']}")
    if row["agendas"]:
        parts.append(f"Agendas : {row['agendas']}")
    if row["categories_txt"]:
        parts.append(f"Catégories : {row['categories_txt']}")
    if row["description"]:
        parts.append(f"Description : {row['description']}")
    if row["long_description"]:
        parts.append(f"Détails : {row['long_description']}")
    if row["date_range"]:
        parts.append(f"Dates : {row['date_range']}")
    if lieu:
        parts.append(f"Lieu : {lieu}")
    if row["keywords"]:
        parts.append(f"Mots-clés : {row['keywords']}")
    if row["conditions"]:
        parts.append(f"Conditions : {row['conditions']}")
    if row["url"]:
        parts.append(f"URL : {row['url']}")

    return "\n".join(parts)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et filtre le DataFrame des événements.
    Args:
        df: DataFrame brut avec les événements aplatis.
    Returns:
        DataFrame nettoyé et filtré.
    """
    n0 = len(df)
    log.info("Début du nettoyage avec %d événements bruts.", n0)

    # 1. Suppression des doublons (sur UID)
    df = df.drop_duplicates(subset="uid").copy()
    n1 = len(df)
    log.info("→ %d événements après suppression des doublons.", n1)

    # 2. Filtrage des événements sans titre ou description
    df = df[df["title"].str.len() > 0].copy()
    df = df[(df["description"].str.len() > 0) | (df["long_description"].str.len() > 0)].copy()
    n2 = len(df)
    log.info("→ %d événements après filtrage des champs vides.", n2)

    # 3. Conversion des dates et filtrage par période
    df["date_begin"] = pd.to_datetime(df["date_begin"], utc=True, errors="coerce")
    df["date_end"] = pd.to_datetime(df["date_end"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_begin"]).copy()  # On garde seulement ceux avec une date de début

    # Filtrage par date (date_end >= DATE_MIN)
    date_min = pd.Timestamp(settings.DATE_MIN, tz="UTC")
    df = df[df["date_end"].fillna(df["date_begin"]) >= date_min].copy()
    n3 = len(df)
    log.info("→ %d événements après filtrage par période.", n3)

    # 4. Filtrage géographique (si activé)
    if settings.ENABLE_GEO_FILTER:
        df = df[
            df.apply(
                lambda r: is_in_target_zone(
                    r["city"], r["department"], r["postal_code"], r["address"], r["latitude"], r["longitude"]
                ),
                axis=1,
            )
        ].copy()
        n4 = len(df)
        log.info("→ %d événements après filtrage géographique.", n4)
    else:
        n4 = n3

    # 5. Remplissage des valeurs manquantes pour les champs texte
    text_cols = ["location_name", "address", "city", "postal_code", "department", "region", "keywords", "conditions", "date_range", "url", "agendas"]
    df[text_cols] = df[text_cols].fillna("")

    # 6. Catégorisation
    df["categories"] = df.apply(lambda r: categorize_all(r["title"], r["keywords"]), axis=1)
    df["category"] = df["categories"].str[0]  # Catégorie principale
    df["categories_txt"] = df["categories"].str.join(", ")

    # 7. Exclusion des catégories non souhaitées
    if EXCLUDED_CATEGORIES:
        df = df[~df["category"].isin(EXCLUDED_CATEGORIES)].copy()
        n5 = len(df)
        log.info("→ %d événements après exclusion des catégories indésirables.", n5)
    else:
        n5 = n4

    # 8. Construction du texte pour le RAG
    df["text"] = df.apply(build_text, axis=1)

    # 9. Suppression des événements avec un texte vide (très rare)
    df = df[df["text"].str.len() > 0].copy()
    n_final = len(df)

    log.info(
        "Nettoyage terminé : %d bruts → %d dédoublonnés → %d champs valides → %d période → %d zone → %d catégories → %d final",
        n0, n1, n2, n3, n4, n5, n_final,
    )
    return df.reset_index(drop=True)

def main() -> None:
    """Point d'entrée du script."""
    try:
        # Vérification de l'existence du fichier brut
        if not settings.RAW_FILE.exists():
            raise FileNotFoundError(f"Fichier brut introuvable : {settings.RAW_FILE}")

        # Chargement des données
        with open(settings.RAW_FILE, "r", encoding="utf-8") as f:
            raw_events = json.load(f)

        if not raw_events:
            log.warning("Aucun événement brut trouvé dans %s.", settings.RAW_FILE)
            return

        # Transformation en DataFrame
        df = pd.DataFrame([flatten_event(e) for e in raw_events])

        # Nettoyage et sauvegarde
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        clean_df = clean_dataframe(df)
        clean_df.to_parquet(settings.PROCESSED_FILE, index=False)

        log.info("✅ %d événements nettoyés sauvegardés dans %s", len(clean_df), settings.PROCESSED_FILE)

    except FileNotFoundError as e:
        log.error("❌ %s", e)
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.error("❌ Fichier JSON corrompu : %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("❌ Erreur inattendue : %s", e, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()