"""Filtre géographique : Dijon / Côte-d'Or + rayon 20 km autour de Dijon."""
import math
import re
import unicodedata

from config import settings  # adapte l'import à ton arborescence


# Centre approximatif de Dijon (Place de la Libération) — utilisé pour le
# filtre géographique en rayon (20 km par défaut autour du centre).
DIJON_LATITUDE = 47.3220
DIJON_LONGITUDE = 5.0415


def normalize(text) -> str:
    """minuscules, sans accents, apostrophes/tirets -> espace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[’'`\-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


_CITY = normalize(settings.TARGET_CITY)
_DEPT_NAMES = {normalize(settings.TARGET_DEPARTMENT), "cote d or"}
_DEPT_CODE = settings.TARGET_DEPT_CODE

# Rayon (km) autour de Dijon : configurable via settings.RADIUS_KM,
# 20 km par défaut (cf. cahier des charges "zone = Dijon + 20 km").
_RADIUS_KM = float(getattr(settings, "RADIUS_KM", 20.0))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance grand-cercle en kilomètres entre deux points (lat, lon en degrés)."""
    try:
        lat1f = float(lat1)
        lon1f = float(lon1)
        lat2f = float(lat2)
        lon2f = float(lon2)
    except (TypeError, ValueError):
        return float("inf")
    r = 6371.0
    p1, p2 = math.radians(lat1f), math.radians(lat2f)
    dp = math.radians(lat2f - lat1f)
    dl = math.radians(lon2f - lon1f)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def is_dijon(city) -> bool:
    return normalize(city) == _CITY


def _is_within_radius(latitude, longitude) -> bool:
    """Vrai si le point (lat, lon) est à moins de _RADIUS_KM du centre de Dijon."""
    if latitude in (None, "", 0) and longitude in (None, "", 0):
        return False
    d = haversine_km(latitude, longitude, DIJON_LATITUDE, DIJON_LONGITUDE)
    return d <= _RADIUS_KM


def is_in_target_zone(city=None, department=None, postal_code=None,
                      latitude=None, longitude=None) -> bool:
    """Vrai si l'événement est dans la zone cible.

    Ordre de priorité :
      1. Lat/lon à moins de RADIUS_KM du centre de Dijon (si disponibles).
      2. Ville = Dijon (match exact).
      3. Département = "Côte-d'Or" (avec repli sur "cote d or").
      4. Code postal commençant par 21.
    """
    # 1. Filtre géospatial (rayon) — utilise lat/lon si valides
    if _is_within_radius(latitude, longitude):
        return True

    # 2. Filtre ville stricte
    if is_dijon(city):
        return True

    # 3. Mode "Dijon uniquement" → on s'arrête là
    if getattr(settings, "ONLY_DIJON", False):
        return False

    # 4. Filtre département
    if department is not None and normalize(department) in _DEPT_NAMES:
        return True

    # 5. Secours : code postal 21xxx
    return str(postal_code or "").strip().startswith(_DEPT_CODE)
