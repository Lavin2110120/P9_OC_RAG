"""Configuration centralisée du projet (zone Dijon, rayon de 20 km).

Les secrets sont lus uniquement depuis les variables d'environnement du
processus ; ce module ne charge ni ne lit de fichier .env.
"""
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)

# --- Chemins (conservés depuis la configuration du projet)
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
RAW_FILE = RAW_DIR / "events_raw.parquet"
PROCESSED_FILE = PROCESSED_DIR / "events.parquet"
INDEX_FILE = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_STORE_DIR / "metadata.parquet"

# --- API
OPENAGENDA_API_KEY = os.environ.get("OPENAGENDA_API_KEY")
OPENAGENDA_BASE_URL = "https://api.openagenda.com/v2"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "100"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "4"))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", "0.2"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "30"))
RETRY_BACKOFF_BASE = float(os.environ.get("RETRY_BACKOFF_BASE", "2"))
USER_AGENT = os.environ.get("USER_AGENT", "puls-events-poc/1.0")
LANG = os.environ.get("LANG", "fr")

# Paramètres des versions historiques de la collecte.
AGENDA_SEARCH = "Dijon"
AGENDA_QUERIES = ["Dijon", "Côte-d'Or"]
AGENDA_KEYWORDS = ["Dijon", "Côte-d'Or", "Bourgogne"]
AGENDA_EXCLUDE_KEYWORDS = []
AGENDA_WHITELIST = [
    "dijon", "cote d or", "cote-d or", "bourgogne", "la vapeur",
    "atheneum", "ceci", "cdc", "mjc",
]

# --- Modèle d'embedding. La dimension est déduite du modèle / index : aucune
# constante EMBEDDING_DIM n'est déclarée ici.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_SEQ_LENGTH = 512
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = True
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".cache" / "huggingface"))

# --- Géographie : centre approximatif de Dijon et rayon source de vérité
TARGET_CITY = "Dijon"
TARGET_CITIES = [TARGET_CITY]
TARGET_DEPARTMENT = "Côte-d'Or"
TARGET_DEPT_CODE = "21"
ONLY_DIJON = False
ENABLE_GEO_FILTER = True
AGENDA_LATITUDE = 47.3220
AGENDA_LONGITUDE = 5.0415
AGENDA_RADIUS_KM = 20.0
RADIUS_KM = AGENDA_RADIUS_KM  # alias de compatibilité pour geo.py
# Liste de repli commune/code postal issue de la variante Dijon + 20 km.
TARGET_COMMUNES = [
    "Dijon", "Chenôve", "Talant", "Fontaine-lès-Dijon", "Longvic",
    "Saint-Apollinaire", "Quetigny", "Chevigny-Saint-Sauveur",
    "Marsannay-la-Côte", "Perrigny-lès-Dijon", "Ahuy",
    "Plombières-lès-Dijon", "Daix", "Hauteville-lès-Dijon",
    "Neuilly-lès-Dijon", "Sennecey-lès-Dijon", "Ruffey-lès-Echirey",
    "Varois-et-Chaignot", "Couternon", "Arc-sur-Tille", "Genlis",
    "Gevrey-Chambertin",
]
TARGET_POSTAL_PREFIXES = [
    "21000", "21100", "21110", "21121", "21160", "21240", "21300",
    "21370", "21560", "21600", "21800", "21850",
]
# Bbox rectangulaire englobant le cercle de 20 km (latitude/longitude degrés).
_lat_delta = AGENDA_RADIUS_KM / 111.32
_lon_delta = AGENDA_RADIUS_KM / (111.32 * __import__("math").cos(__import__("math").radians(AGENDA_LATITUDE)))
GEO_BBOX = {
    "south": AGENDA_LATITUDE - _lat_delta,
    "north": AGENDA_LATITUDE + _lat_delta,
    "west": AGENDA_LONGITUDE - _lon_delta,
    "east": AGENDA_LONGITUDE + _lon_delta,
}

# --- Fenêtre temporelle. DATE_MAX conservé pour les consommateurs historiques.
TODAY = date.today()
DATE_MIN = TODAY - timedelta(days=365)
DATE_MAX = TODAY + timedelta(days=365)
DATE_MIN_DT = datetime.combine(DATE_MIN, time.min, tzinfo=timezone.utc)
DATE_MAX_DT = datetime.combine(DATE_MAX, time.max, tzinfo=timezone.utc)


def require_openagenda_key() -> str:
    """Retourne la clé OpenAgenda ou signale son absence clairement."""
    if not OPENAGENDA_API_KEY:
        raise RuntimeError("OPENAGENDA_API_KEY manquante (variable d'environnement)")
    return OPENAGENDA_API_KEY
