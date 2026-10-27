"""Paramètres centralisés du projet."""
import os
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

# --- Chemins (indépendants du dossier courant)
ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = ROOT
load_dotenv(ROOT / ".env")

RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "events_raw.json"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "events.parquet"
VECTOR_STORE_DIR = ROOT / "data" / "vector_store"
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.parquet"

# --- Clés API
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAGENDA_BASE_URL = "https://api.openagenda.com/v2"

# --- Clé API pour l'agenda Open Agenda (facultatif — fetch_events en a aussi une)
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY", "")

# --- Périmètre géographique
TARGET_CITY = "Dijon"
TARGET_CITIES = [TARGET_CITY]
TARGET_DEPARTMENT = "Côte-d'Or"
TARGET_DEPT_CODE = "21"
AGENDA_SEARCH = "Dijon"

# --- Rayon (km) autour du centre de Dijon pour le filtre géospatial.
# Cahier des charges : "zone = Dijon + 20 km".
RADIUS_KM = float(os.getenv("RADIUS_KM", "20"))

# --- Période : 1 an d'historique + 1 an à venir.
# IMPORTANT : on fige la date de référence pour rendre le filtre reproductible
# (sinon DATE_MIN/DATE_MAX bougent à chaque exécution). Override possible via
# la variable d'env REFERENCE_DATE au format ISO (YYYY-MM-DD).
_ref_env = os.getenv("REFERENCE_DATE")
if _ref_env:
    try:
        TODAY = date.fromisoformat(_ref_env)
    except ValueError:
        TODAY = date.today()
else:
    TODAY = date.today()

# On borne la période à la journée courante en UTC pour comparer avec
# pd.to_datetime(..., utc=True).
DATE_MIN = datetime(TODAY.year - 1, TODAY.month, TODAY.day, tzinfo=timezone.utc).date()
DATE_MAX = datetime(TODAY.year + 1, TODAY.month, TODAY.day, tzinfo=timezone.utc).date()

# Bornes "datetime UTC" prêtes à l'emploi (utilisables directement avec pandas).
DATE_MIN_DT = datetime(TODAY.year - 1, TODAY.month, TODAY.day, tzinfo=timezone.utc)
DATE_MAX_DT = datetime(TODAY.year + 1, TODAY.month, TODAY.day, tzinfo=timezone.utc)

# --- Pagination / retries API
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "100"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))

# --- Liste blanche d'agendas Open Agenda (mots-clés normalisés)
AGENDA_WHITELIST = [
    "dijon",
    "cote d or",
    "cote-d or",
    "bourgogne",
    "la vapeur",
    "atheneum",
    "ceci",
    "cdc",
    "mjc",
]
