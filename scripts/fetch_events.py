"""Récupération des événements Open Agenda (Dijon / Côte-d'Or + 20 km)."""
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from config import settings
from config.geo import is_in_target_zone, normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://api.openagenda.com/v2"
RAW_PATH = Path("data/raw/events_raw.json")

_session = requests.Session()
_session.headers.update({
    "User-Agent": "puls-events-poc/1.0",
    "key": settings.OPENAGENDA_API_KEY or ""})


def _to_api_param(d) -> str:
    """Convertit une date en ISO 8601 avec timezone UTC (attendu par l'API).

    L'API Open Agenda veut des chaînes ISO 8601 ; sans offset, la
    comparaison serveur peut être ambiguë. On envoie donc du UTC explicite.
    """
    if hasattr(d, "tzinfo") and d.tzinfo is not None:
        return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)


def _get(url: str, params: dict | None = None) -> dict:
    """GET avec retry, backoff exponentiel et gestion du 429."""
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            r = _session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("429 reçu, pause %ss", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(settings.REQUEST_DELAY)
            return r.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            log.warning("Tentative %d/%d échouée (%s), pause %ss",
                        attempt, settings.MAX_RETRIES, type(e).__name__, wait)
            time.sleep(wait)
    raise RuntimeError(f"Échec de la requête : {url}")


def search_agendas(query: str) -> list[dict]:
    agendas, after = [], None
    while True:
        params = {"search": query, "size": 50}
        if after:
            params["after"] = after
        data = _get(f"{BASE_URL}/agendas", params)
        agendas.extend(data.get("agendas", []))
        after = data.get("after")
        if not after or not data.get("agendas"):
            break
    return agendas


def select_agendas(agendas: list[dict]) -> list[dict]:
    """Garde uniquement la liste blanche, dédoublonnée par uid."""
    wl = [normalize(w) for w in settings.AGENDA_WHITELIST]
    selected = {}
    for a in agendas:
        title = normalize(a.get("title"))
        if any(w in title for w in wl):
            selected[a["uid"]] = a
    return list(selected.values())


def fetch_agenda_events(agenda_uid: int) -> list[dict]:
    """Événements d'un agenda, filtrés par période côté serveur.

    On envoie DATE_MIN / DATE_MAX en UTC ISO pour lever toute ambiguïté
    côté serveur (« timezone-naive comparisons »).
    """
    events, after = [], None
    while True:
        params = {
            "size": settings.PAGE_SIZE,
            "timings[gte]": _to_api_param(settings.DATE_MIN_DT),
            "timings[lte]": _to_api_param(settings.DATE_MAX_DT),
            "detailed": 1,
        }
        if after:
            params["after"] = after
        data = _get(f"{BASE_URL}/agendas/{agenda_uid}/events", params)
        batch = data.get("events", [])
        events.extend(batch)
        after = data.get("after")
        if not batch or not after:
            break
    return events


def event_in_zone(ev: dict) -> bool:
    loc = ev.get("location") or {}
    return is_in_target_zone(
        city=loc.get("city"),
        department=loc.get("department"),
        postal_code=loc.get("postalCode"),
        latitude=loc.get("latitude"),
        longitude=loc.get("longitude"),
    )


def main() -> None:
    if not settings.OPENAGENDA_API_KEY:
        raise SystemExit("OPENAGENDA_API_KEY manquante (variable d'environnement / .env)")

    log.info(
        "Fenêtre API : %s → %s (référence TODAY=%s)",
        settings.DATE_MIN_DT.isoformat(),
        settings.DATE_MAX_DT.isoformat(),
        settings.TODAY,
    )

    candidates = {}
    for q in ["Dijon", "Côte-d'Or", "Bourgogne"]:
        for a in search_agendas(q):
            candidates[a["uid"]] = a
    log.info("%d agendas candidats", len(candidates))

    agendas = select_agendas(list(candidates.values()))
    log.info("%d agendas retenus (liste blanche)", len(agendas))

    all_events, seen = [], set()
    for a in agendas:
        try:
            events = fetch_agenda_events(a["uid"])
        except RuntimeError as e:
            log.error("Agenda %s ignoré : %s", a["uid"], e)
            continue
        kept = 0
        for ev in events:
            uid = ev.get("uid")
            if uid in seen or not event_in_zone(ev):
                continue
            seen.add(uid)
            ev["_agenda_title"] = a.get("title")
            all_events.append(ev)
            kept += 1
        log.info("  • %-45s %4d récupérés -> %4d dans la zone",
                 (a.get("title") or "")[:45], len(events), kept)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(all_events, ensure_ascii=False), encoding="utf-8")
    log.info("TOTAL : %d événements uniques sauvegardés", len(all_events))


if __name__ == "__main__":
    main()
