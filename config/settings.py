"""Paramètres centralisés du projet."""
import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILE = RAW_DIR / "events_raw.json"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "events.parquet"

OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
OPENAGENDA_BASE_URL = os.getenv("OPENAGENDA_BASE_URL", "https://api.openagenda.com/v2")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
USER_AGENT = "puls-events-poc/1.0"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "100"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2"))

# --- Périmètre géographique : Côte-d'Or, Dijon inclus
TARGET_CITY = "Dijon"  # compatibilité avec les scripts existants
# --- Liste des villes autorisées en Côte-d'Or (21) ---
TARGET_CITIES = {
    "Dijon", "Beaune", "Auxonne", "Montbard", "Semur-en-Auxois",
    "Châtillon-sur-Seine", "Nuits-Saint-Georges"
}
TARGET_DEPARTMENT = "Côte-d'Or"
TARGET_DEPT_CODE = "21"
ONLY_DIJON = False
ENABLE_GEO_FILTER = True
TARGET_ZONE = {
    "department": "21",  # Côte-d'Or
    "city": "dijon",
    "keywords": ["dijon", "cote d or", "21", "bourgogne"],
}
AGENDA_SEARCH = "Côte-d'Or"

# --- Requêtes larges : elles servent à découvrir les agendas candidats
AGENDA_QUERIES = ["Dijon Côte-d'Or", "Beaune 21", "Auxonne Bourgogne","Montbard", "Semur-en-Auxois", "Châtillon-sur-Seine", "Nuits-Saint-Georges"]
# Vide = tous les agendas découverts ; sinon correspondance partielle sur le titre.
AGENDA_WHITELIST = []

# --- Fenêtre configurable : aujourd'hui jusqu'à environ 12 mois
TODAY = date.today()
DATE_MIN = date.fromisoformat(os.getenv("DATE_MIN", TODAY.isoformat()))
DATE_MAX = date.fromisoformat(os.getenv("DATE_MAX", (TODAY + timedelta(days=365)).isoformat()))

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def require_openagenda_key() -> str:
    """Retourne la clé OpenAgenda ou lève une erreur explicite."""
    if not OPENAGENDA_API_KEY:
        raise RuntimeError("OPENAGENDA_API_KEY manquante (.env ou variable d'environnement)")
    return OPENAGENDA_API_KEY
