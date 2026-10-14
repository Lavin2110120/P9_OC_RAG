"""Récupération des événements OpenAgenda autour de Dijon / Côte-d'Or."""
import logging
import time
from datetime import datetime, time as dtime

import pandas as pd
import requests

from config import settings, geo

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

COLUMNS = [
    "uid", "slug", "title", "description", "long_description", "keywords",
    "date_begin", "date_end", "location_name", "address", "city", "postal_code",
    "latitude", "longitude", "agenda_uid", "agenda_title", "online_link",
]


# ---------------------------------------------------------------- HTTP helpers
def setup_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": settings.USER_AGENT})
    return session


def get_json(session: requests.Session, url: str, params: dict) -> dict | None:
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=settings.REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.exceptions.RetryError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            # Ne jamais propager l'URL : elle contient la clé API
            raise requests.exceptions.HTTPError(
                f"HTTP {exc.response.status_code} sur {url}", response=exc.response
            ) from None
        except requests.exceptions.RequestException as exc:
            wait = settings.RETRY_BACKOFF_BASE ** attempt
            log.warning(f"Tentative {attempt}/{settings.MAX_RETRIES} échouée "
                        f"({type(exc).__name__}), retry dans {wait:.0f}s")
            time.sleep(wait)
    return None


def paginate(session, url, params, key):
    params = dict(params)
    items, total = [], None
    while True:
        data = get_json(session, url, params)
        if not data:
            break
        batch = data.get(key, [])
        items.extend(batch)
        total = data.get("total", total)
        after = data.get("after")
        if not batch or not after:
            break
        params["after[]"] = after
        time.sleep(settings.REQUEST_DELAY)
    if total is not None and len(items) < total:
        log.warning(f"Pagination incomplète : {len(items)}/{total}")
    return items


# ---------------------------------------------------------------- Fetch
def event_params() -> dict:
    bbox = settings.GEO_BBOX
    return {
        "key": settings.OPENAGENDA_API_KEY,
        "size": settings.PAGE_SIZE,
        "detailed": 1,
        "timings[gte]": datetime.combine(settings.DATE_MIN, dtime.min).isoformat(),
        "timings[lte]": datetime.combine(settings.DATE_MAX, dtime.max).isoformat(),
        "geo[northEast][lat]": bbox["north"],
        "geo[northEast][lng]": bbox["east"],
        "geo[southWest][lat]": bbox["south"],
        "geo[southWest][lng]": bbox["west"],
    }


def fetch_global_events(session: requests.Session) -> list[dict]:
    """Endpoint global : tous les événements de la plateforme dans la bounding box."""
    url = f"{settings.OPENAGENDA_BASE_URL}/events"
    return paginate(session, url, event_params(), key="events")


def fetch_agenda_list(session: requests.Session) -> list[dict]:
    """Fallback : découverte des agendas par mots-clés."""
    url = f"{settings.OPENAGENDA_BASE_URL}/agendas"
    agendas: dict = {}
    for query in settings.AGENDA_QUERIES:
        params = {"key": settings.OPENAGENDA_API_KEY, "size": settings.PAGE_SIZE, "search": query}
        for agenda in paginate(session, url, params, key="agendas"):
            agendas[agenda["uid"]] = agenda
    return list(agendas.values())


def fetch_agenda_events(session: requests.Session, agenda_uid) -> list[dict]:
    url = f"{settings.OPENAGENDA_BASE_URL}/agendas/{agenda_uid}/events"
    return paginate(session, url, event_params(), key="events")


def fetch_all_events(session: requests.Session) -> list[dict]:
    try:
        events = fetch_global_events(session)
        log.info(f"Endpoint global : {len(events)} événements dans la bounding box")
        return events
    except requests.exceptions.HTTPError as exc:
        log.warning(f"Endpoint global indisponible ({exc}), bascule sur la découverte d'agendas")

    def is_excluded_agenda(agenda: dict) -> bool:
        title = geo._norm(agenda.get("title"))
        return any(kw in title for kw in settings.AGENDA_EXCLUDE_KEYWORDS)

    agendas = [a for a in fetch_agenda_list(session) if geo.agenda_in_zone(a) and not is_excluded_agenda(a)]
    log.info(f"{len(agendas)} agendas retenus")
    events = []
    for agenda in agendas:
        try:
            found = fetch_agenda_events(session, agenda["uid"])
        except requests.exceptions.HTTPError as exc:
            log.warning(f"  • {agenda.get('title')} : erreur {exc}")
            continue
        log.info(f"  • {agenda.get('title')} : {len(found)} récupérés")
        events.extend(found)
    return events


def _lang(value, lang: str = settings.LANG):
    """Les champs texte OpenAgenda sont multilingues : {'fr': '...', 'en': '...'}."""
    if isinstance(value, dict):
        return value.get(lang) or next((v for v in value.values() if v), "")
    return value or ""


def extract_event_data(event: dict) -> dict:
    first = event.get("firstTiming") or (event.get("timings") or [{}])[0] or {}
    last = event.get("lastTiming") or first
    location = event.get("location") or {}
    agenda = event.get("originAgenda") or {}
    keywords = _lang(event.get("keywords"))
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    return {
        "uid": event.get("uid"),
        "slug": event.get("slug"),
        "title": _lang(event.get("title")),
        "description": _lang(event.get("description")),
        "long_description": _lang(event.get("longDescription")),
        "keywords": list(keywords or []),
        "date_begin": first.get("begin"),
        "date_end": last.get("end"),
        "location_name": location.get("name"),
        "address": location.get("address"),
        "city": location.get("city"),
        "postal_code": location.get("postalCode"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "agenda_uid": agenda.get("uid"),
        "agenda_title": agenda.get("title"),
        "online_link": event.get("onlineAccessLink"),
    }


def main():
    settings.require_openagenda_key()
    session = setup_session()

    events = fetch_all_events(session)
    records = [extract_event_data(e) for e in events if geo.event_in_zone(e)]
    log.info(f"Filtre géographique : {len(records)} événements retenus sur {len(events)}")

    df = pd.DataFrame(records, columns=COLUMNS).drop_duplicates(subset=["uid"])
    settings.RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.RAW_FILE, index=False)
    log.info(f"TOTAL : {len(df)} événements uniques sauvegardés dans {settings.RAW_FILE}")


if __name__ == "__main__":
    main()