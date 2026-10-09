"""Fonctions communes de normalisation et de filtrage géographique."""
import re
import unicodedata

from config import settings

DEPT_LAT = (46.9, 48.05)
DEPT_LON = (4.05, 5.55)
_POSTAL_RE = re.compile(r"\b(\d{5})\b")


def normalize(value) -> str:
    """Minuscules, sans accents, tirets/apostrophes et ponctuation normalisés."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _location_dict(loc: dict) -> dict:
    loc = loc if isinstance(loc, dict) else {}
    return {
        "name": loc.get("name"),
        "city": loc.get("city"),
        "department": loc.get("department") or loc.get("adminLevel2") or loc.get("adminLevel1"),
        "admin_level1": loc.get("adminLevel1"),
        "admin_level2": loc.get("adminLevel2"),
        "admin_level3": loc.get("adminLevel3"),
        "region": loc.get("region"),
        "postal_code": loc.get("postalCode"),
        "address": loc.get("address"),
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
    }


def extract_location(ev: dict) -> dict:
    """Extrait une localisation canonique depuis ``location`` ou ``locations``.

    Pour une liste, la première localisation est renvoyée ; ``event_in_zone``
    teste toutes les localisations afin de ne pas perdre un lieu secondaire.
    """
    loc = ev.get("location")
    if isinstance(loc, dict) and loc:
        return _location_dict(loc)
    locations = ev.get("locations") or []
    return _location_dict(locations[0]) if isinstance(locations, list) and locations else _location_dict({})


def _all_locations(ev: dict) -> list[dict]:
    locations = []
    if isinstance(ev.get("location"), dict):
        locations.append(_location_dict(ev["location"]))
    for loc in ev.get("locations") or []:
        if isinstance(loc, dict):
            locations.append(_location_dict(loc))
    return locations or [_location_dict({})]


def _postal_codes(postal_code, address) -> list[str]:
    raw = str(postal_code or "").strip()
    digits = re.sub(r"\D", "", raw.split(".")[0])
    codes = [digits] if len(digits) == 5 else []
    codes.extend(_POSTAL_RE.findall(str(address or "")))
    return codes


def _in_bbox(latitude, longitude) -> bool:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return False
    return DEPT_LAT[0] <= lat <= DEPT_LAT[1] and DEPT_LON[0] <= lon <= DEPT_LON[1]


def is_in_target_zone(city=None, department=None, postal_code=None, address=None,
                       latitude=None, longitude=None, **_ignored) -> bool:
    """Indique si une localisation est en Côte-d'Or, avec preuves graduées."""
    if not getattr(settings, "ENABLE_GEO_FILTER", True):
        return True
    city_n = normalize(city)
    dept_n = normalize(department)
    target_cities = {normalize(c) for c in getattr(settings, "TARGET_CITIES", [])}
    target_cities.add(normalize(getattr(settings, "TARGET_CITY", "Dijon")))
    city_ok = any(city_n == c or city_n.startswith(c + " ") for c in target_cities if c)
    postal_ok = any(code.startswith("21") for code in _postal_codes(postal_code, address))
    dept_names = {normalize(getattr(settings, "TARGET_DEPARTMENT", "Côte-d'Or")), "cote d or", "cote dor"}
    dept_ok = dept_n in dept_names

    if getattr(settings, "ONLY_DIJON", False):
        return city_ok
    if city_ok or postal_ok or dept_ok:
        return True
    # Un code postal ou un département explicitement différent est une preuve négative.
    if _postal_codes(postal_code, address) or dept_n:
        return False
    # GPS uniquement en dernier recours, notamment pour Beaune sans ville/code postal.
    return _in_bbox(latitude, longitude)


def event_in_zone(ev: dict) -> bool:
    """Teste toutes les localisations d'un événement brut OpenAgenda."""
    return any(is_in_target_zone(**loc) for loc in _all_locations(ev))

def agenda_in_zone(agenda: dict) -> bool:
    """Vérifie si un agenda est dans la zone cible (Côte-d'Or) via location ou fallback."""
    location = agenda.get("location") or {}
    if location:
        return is_in_target_zone(
            city=location.get("city"),
            department=location.get("department"),
            postal_code=location.get("postalCode"),
            address=location.get("address"),
            latitude=location.get("latitude"),
            longitude=location.get("longitude")
        )
    # Fallback : vérification dans le titre ou la description
    title = normalize(agenda.get("title") or "")
    description = normalize(agenda.get("description") or "")
    return any(
        kw in title or kw in description
        for kw in {"dijon", "cote d or", "21", "bourgogne"}
    )