"""Filtre géographique : Dijon / Côte-d'Or + rayon configurable autour de Dijon."""
import math
import re
import unicodedata

try:  # import comme config.geo
    from . import settings
except ImportError:  # compatibilité si geo.py est importé comme module isolé
    import settings


# Centre approximatif de Dijon (Place de la Libération).
DIJON_LATITUDE = 47.3220
DIJON_LONGITUDE = 5.0415


def normalize(text) -> str:
    """Minuscules, sans accents, apostrophes/tirets transformés en espaces."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[’'`\-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


_CITY = normalize(settings.TARGET_CITY)
_DEPT_NAMES = {normalize(settings.TARGET_DEPARTMENT), "cote d or"}
_DEPT_CODE = settings.TARGET_DEPT_CODE
_RADIUS_KM = float(getattr(settings, "RADIUS_KM", 20.0))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distance grand-cercle en km; renvoie l'infini pour des coordonnées invalides."""
    try:
        lat1f, lon1f, lat2f, lon2f = map(float, (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError, OverflowError):
        return float("inf")
    if not all(math.isfinite(v) for v in (lat1f, lon1f, lat2f, lon2f)):
        return float("inf")
    if not (-90 <= lat1f <= 90 and -90 <= lat2f <= 90
            and -180 <= lon1f <= 180 and -180 <= lon2f <= 180):
        return float("inf")

    r = 6371.0
    p1, p2 = math.radians(lat1f), math.radians(lat2f)
    dp = math.radians(lat2f - lat1f)
    dl = math.radians(lon2f - lon1f)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    # Protection contre les écarts d'arrondi en bordure de l'intervalle [0, 1].
    a = min(1.0, max(0.0, a))
    return 2 * r * math.asin(math.sqrt(a))


def is_dijon(city) -> bool:
    return normalize(city) == _CITY


def _within_radius(latitude, longitude) -> bool:
    """Vrai si une paire complète de coordonnées valides est dans le rayon."""
    if latitude in (None, "") or longitude in (None, ""):
        return False
    distance = haversine_km(latitude, longitude, DIJON_LATITUDE, DIJON_LONGITUDE)
    return distance <= _RADIUS_KM


# Garder le nom privé déjà présent, au cas où un appelant l'utilise.
_is_within_radius = _within_radius


def is_in_target_zone(city=None, department=None, postal_code=None,
                      latitude=None, longitude=None) -> bool:
    """Vrai si l'événement est dans la zone géographique cible.

    Priorité: rayon GPS, ville, département (sauf mode ONLY_DIJON), code postal.
    """
    if _within_radius(latitude, longitude):
        return True
    if is_dijon(city):
        return True
    if getattr(settings, "ONLY_DIJON", False):
        return False
    if department is not None and normalize(department) in _DEPT_NAMES:
        return True
    return str(postal_code or "").strip().startswith(_DEPT_CODE)
