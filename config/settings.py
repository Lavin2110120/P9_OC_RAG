"""Paramètres centralisés du projet."""
import os
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILE = RAW_DIR / "events.parquet"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "events.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- API
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
OPENAGENDA_BASE_URL = os.getenv("OPENAGENDA_BASE_URL", "https://api.openagenda.com/v2")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
USER_AGENT = "puls-events-poc/1.0"
LANG = "fr"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "100"))  # max 300 côté API
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2"))

# --- Périmètre géographique : Côte-d'Or (21), Dijon inclus
ENABLE_GEO_FILTER = True
TARGET_CITY = "Dijon"
TARGET_CITIES = {
    "Dijon", "Beaune", "Auxonne", "Montbard", "Semur-en-Auxois",
    "Châtillon-sur-Seine", "Nuits-Saint-Georges", "Chenôve", "Talant",
    "Quetigny", "Longvic", "Fontaine-lès-Dijon", "Saint-Apollinaire",
}
TARGET_DEPARTMENT = "Côte-d'Or"
TARGET_DEPT_CODE = "21"

# Bounding box approximative du département (lat/lng)
GEO_BBOX = {"north": 48.05, "south": 46.90, "east": 5.55, "west": 4.05}

# Découverte d'agendas (fallback si l'endpoint global /events est indisponible)
AGENDA_QUERIES = ["Dijon", "Côte-d'Or", "Beaune", "Auxonne", "Montbard",
                  "Semur-en-Auxois", "Châtillon-sur-Seine", "Nuits-Saint-Georges"]
AGENDA_KEYWORDS = ["dijon", "cote d'or", "cote-d'or", "bourgogne", "beaune"]
AGENDA_EXCLUDE_KEYWORDS = ["test", "archives", "[archives]"]
# --- Fenêtre temporelle : aujourd'hui → +12 mois
TODAY = date.today()
DATE_MIN = TODAY
DATE_MAX = TODAY + timedelta(days=365)

# --- Catégorisation (utilisée par d'autres scripts)
CATEGORY_KEYWORDS = {
    "musique": ["la vapeur", "concert", "pop", "rap", "rock", "electro", "chanson"],
    "sante_bien_etre": ["prevention sante", "retraite", "bien etre", "bien vieillir", "addictions"],
    "sport": ["asept fc b", "mai a velo"],
    "biodiversite": ["botanique", "flore"],
    "theatre": ["theatre"],
    "festival": ["festival"],
    "vivre_ensemble": ["jeune public", "seniors", "dijon"],
}
EXCLUDED_CATEGORIES = ["sante_bien_etre", "vivre_ensemble"]
CATEGORY_PRIORITY = ["musique", "theatre", "festival", "sante_bien_etre", "sport", "biodiversite", "vivre_ensemble"]


def require_openagenda_key() -> str:
    """Retourne la clé OpenAgenda ou lève une erreur explicite."""
    if not OPENAGENDA_API_KEY:
        raise RuntimeError("OPENAGENDA_API_KEY manquante (.env ou variable d'environnement)")
    return OPENAGENDA_API_KEY