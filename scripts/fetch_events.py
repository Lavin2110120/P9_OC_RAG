"""Collecte des événements OpenAgenda (Côte-d'Or) depuis l'API v2."""
import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings
from config.geo import event_in_zone, agenda_in_zone

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.USER_AGENT, "key": settings.require_openagenda_key()})
    return session

def _get(session: requests.Session, url: str, params=None) -> dict:
    """GET avec retry réseau/429/5xx, jamais sur les autres 4xx."""
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=settings.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt == settings.MAX_RETRIES:
                break
            log.warning("Tentative %d/%d : erreur réseau (%s)", attempt, settings.MAX_RETRIES, type(exc).__name__)
            time.sleep(settings.RETRY_BACKOFF_BASE ** attempt)
            continue
        if response.status_code in (429,) or response.status_code >= 500:
            if attempt == settings.MAX_RETRIES:
                break
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else settings.RETRY_BACKOFF_BASE ** attempt
            log.warning("Tentative %d/%d : HTTP %d, pause %.1fs", attempt, settings.MAX_RETRIES, response.status_code, wait)
            time.sleep(wait)
            continue
        if response.status_code in (401, 403):
            raise PermissionError(f"Accès refusé (HTTP {response.status_code}) : vérifie OPENAGENDA_API_KEY")
        response.raise_for_status()
        time.sleep(settings.REQUEST_DELAY)
        return response.json()
    raise RuntimeError(f"Échec après {settings.MAX_RETRIES} tentatives : {url}")

def _with_cursor(params: dict, after):
    if after:
        params["after[]" if isinstance(after, list) else "after"] = after
    return params

def search_agendas(session: requests.Session, query: str) -> list[dict]:
    agendas, after = [], None
    while True:
        data = _get(session, f"{settings.OPENAGENDA_BASE_URL}/agendas",
                    _with_cursor({"search": query, "size": 50}, after))
        batch = data.get("agendas", [])
        agendas.extend(batch)
        after = data.get("after")
        if not batch or not after:
            return agendas

def select_agendas(agendas: list[dict]) -> list[dict]:
    """Dédoublonne, applique la liste blanche et filtre géographiquement."""
    whitelist = [str(v).casefold() for v in getattr(settings, "AGENDA_WHITELIST", []) if v]
    selected = {}
    for agenda in agendas:
        title = str(agenda.get("title") or "")
        if not whitelist or any(item in title.casefold() for item in whitelist):
            if agenda.get("uid") is not None:
                # 👇 FILTRE GÉOGRAPHIQUE STRICT : on vérifie la ville ET le département
                location = agenda.get("location", {})
                city = location.get("city", "").casefold()
                department = location.get("department", "").casefold()

                # Villes cibles en Côte-d'Or (21)
                target_cities = {"dijon", "beaune", "auxonne", "montbard", "semur-en-auxois",
                                "châtillon-sur-seine", "nuits-saint-georges"}
                target_dept = "21"

                # Si la ville est dans la liste OU le département est 21
                if (city in target_cities) or (department == target_dept):
                    selected[agenda["uid"]] = agenda
                else:
                    log.debug(f"Agenda {title} ignoré : ville={city}, département={department}")
    return list(selected.values())

def fetch_agenda_events(session: requests.Session, agenda_uid: int) -> list[dict]:
    events, after = [], None
    while True:
        params = _with_cursor({"size": settings.PAGE_SIZE, "detailed": 1,
                               "monolingual": "fr", "timings[gte]": settings.DATE_MIN.isoformat(),
                               "timings[lte]": settings.DATE_MAX.isoformat()}, after)
        data = _get(session, f"{settings.OPENAGENDA_BASE_URL}/agendas/{agenda_uid}/events", params)
        batch = data.get("events", [])
        events.extend(batch)
        after = data.get("after")
        if not batch or not after:
            return events

def main() -> None:
    try:
        session = _build_session()
        candidates = {}
        for query in settings.AGENDA_QUERIES:
            for agenda in search_agendas(session, query):
                if agenda.get("uid") is not None:
                    candidates[agenda["uid"]] = agenda

        # 👇 FILTRE GÉOGRAPHIQUE AVANT RÉCUPÉRATION DES ÉVÉNEMENTS
        agendas = select_agendas(list(candidates.values()))
        log.info("Filtre géographique appliqué : %d agendas retenus sur %d candidats",
                len(agendas), len(candidates))

    except PermissionError as exc:
        raise SystemExit(f"❌ {exc}")
    except (RuntimeError, requests.RequestException) as exc:
        raise SystemExit(f"❌ Découverte des agendas impossible : {type(exc).__name__}")

    # --- Récupération des événements (seulement pour les agendas filtrés) ---
    by_uid = {}
    total_new = total_duplicates = total_outside = 0

    for agenda in agendas:
        try:
            events = fetch_agenda_events(session, agenda["uid"])
        except (RuntimeError, PermissionError, requests.RequestException) as exc:
            log.error("Agenda %s ignoré : %s", agenda.get("title"), type(exc).__name__)
            continue

        new = duplicates = outside = 0
        for event in events:
            uid = event.get("uid")
            if uid in by_uid:
                by_uid[uid].setdefault("_agendas", []).append(agenda.get("title"))
                duplicates += 1
                continue

            # 👇 FILTRE GÉOGRAPHIQUE STRICT SUR LES ÉVÉNEMENTS
            if settings.ENABLE_GEO_FILTER and not event_in_zone(event):
                outside += 1
                log.debug("Événement %s rejeté : hors zone", uid)
                continue

            event["_agenda_title"] = agenda.get("title")
            event["_agendas"] = [agenda.get("title")]
            by_uid[uid] = event
            new += 1

        total_new += new
        total_duplicates += duplicates
        total_outside += outside

        if events:
            log.info("  • %-45s %4d récupérés | %4d nouveaux | %4d doublons | %4d hors zone",
                    (agenda.get("title") or "")[:45], len(events), new, duplicates, outside)
        else:
            log.debug("  • %-45s agenda vide (filtre géographique ?)", (agenda.get("title") or "")[:45])

    # Sauvegarde finale
    all_events = list(by_uid.values())
    settings.RAW_FILE.write_text(json.dumps(all_events, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("TOTAL : %d événements uniques sauvegardés dans %s", len(all_events), settings.RAW_FILE)
    log.info("Statistiques : %d nouveaux | %d doublons | %d hors zone ignorés",
             total_new, total_duplicates, total_outside)

if __name__ == "__main__":
    main()