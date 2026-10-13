"""Filtrage géographique : Côte-d'Or / Dijon."""
import unicodedata
from config import settings


def _norm(value) -> str:
    """Minuscule, sans accents, sans espaces superflus."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in text if not unicodedata.combining(c)).lower().strip()


_TARGET_CITIES = {_norm(c) for c in settings.TARGET_CITIES}
_TARGET_DEPT = _norm(settings.TARGET_DEPARTMENT)


def in_bbox(lat, lng) -> bool:
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return False
    b = settings.GEO_BBOX
    return b["south"] <= lat <= b["north"] and b["west"] <= lng <= b["east"]


def location_in_zone(location: dict | None) -> bool:
    if not settings.ENABLE_GEO_FILTER:
        return True
    if not location:
        return False

    postal = str(location.get("postalCode") or "").strip()
    if postal:
        return postal.startswith(settings.TARGET_DEPT_CODE)  # décision ferme

    if _norm(location.get("city")) in _TARGET_CITIES:
        return True
    dept = _norm(location.get("department"))
    if dept and dept in (_TARGET_DEPT, settings.TARGET_DEPT_CODE):
        return True
    return in_bbox(location.get("latitude"), location.get("longitude"))


def event_in_zone(event: dict) -> bool:
    return location_in_zone(event.get("location"))


def agenda_in_zone(agenda: dict) -> bool:
    """Filtre large sur le titre/description (uniquement pour la découverte d'agendas)."""
    if not settings.ENABLE_GEO_FILTER:
        return True
    text = _norm(agenda.get("title")) + " " + _norm(agenda.get("description"))
    return any(_norm(kw) in text for kw in settings.AGENDA_KEYWORDS) or \
        any(city in text for city in _TARGET_CITIES)