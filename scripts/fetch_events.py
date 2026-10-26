"""Récupération des événements Open Agenda (Dijon / Côte-d'Or + 20 km)."""

import json
import logging
import sys
import time
from datetime import timezone
from pathlib import Path

import requests

# Racine du projet ajoutée au sys.path pour résoudre l'import « config »
# quel que soit le dossier courant d'exécution.
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings
from config.geo import is_in_target_zone, normalize

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BASE_URL = settings.OPENAGENDA_BASE_URL
RAW_PATH = settings.RAW_FILE

_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": "puls-events-poc/1.0",
        "key": settings.OPENAGENDA_API_KEY or "",
    }
)


def clean_unicode(value):
    """Remplace les caractères Unicode isolés sans altérer les caractères valides."""
    if isinstance(value, str):
        # L'encodage avec backslashreplace préserve les caractères Unicode valides
        # et échappe les surrogates isolés, que json.dumps pourra sérialiser.
        return value.encode("utf-8", errors="backslashreplace").decode("utf-8")
    if isinstance(value, list):
        return [clean_unicode(item) for item in value]
    if isinstance(value, dict):
        return {
            clean_unicode(key): clean_unicode(item)
            for key, item in value.items()
        }
    return value


def _to_api_param(value) -> str:
    """Convertit une date en ISO 8601, avec timezone UTC si elle est définie."""
    if getattr(value, "tzinfo", None) is not None:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _get(url: str, params: dict | None = None) -> dict:
    """Effectue une requête GET avec retries et gestion du statut 429."""
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            response = _session.get(
                url,
                params=params,
                timeout=30,
                verify=False,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = int(retry_after) if retry_after else 2**attempt
                except ValueError:
                    wait = 2**attempt

                log.warning("429 reçu, pause %ss", wait)
                time.sleep(wait)
                continue

            response.raise_for_status()
            time.sleep(settings.REQUEST_DELAY)
            return response.json()

        except requests.RequestException as exc:
            wait = 2**attempt
            log.warning(
                "Tentative %d/%d échouée (%r), pause %ss",
                attempt,
                settings.MAX_RETRIES,
                exc,
                wait,
            )
            if attempt < settings.MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"Échec de la requête : {url}")


def search_agendas(query: str) -> list[dict]:
    """Recherche tous les agendas correspondant à une requête."""
    agendas = []
    after = None

    while True:
        params = {"search": query, "size": 50}
        if after:
            params["after"] = after

        data = _get(f"{BASE_URL}/agendas", params)
        batch = data.get("agendas", [])
        agendas.extend(batch)

        after = data.get("after")
        if not after or not batch:
            break

    return agendas


def select_agendas(agendas: list[dict]) -> list[dict]:
    """Garde les agendas correspondant à la liste blanche, sans doublons."""
    whitelist = [normalize(item) for item in settings.AGENDA_WHITELIST]
    selected = {}

    for agenda in agendas:
        title = normalize(agenda.get("title"))
        uid = agenda.get("uid")

        if uid is not None and any(word in title for word in whitelist):
            selected[uid] = agenda

    return list(selected.values())


def fetch_agenda_events(agenda_uid: int) -> list[dict]:
    """Récupère les événements d'un agenda dans la période configurée."""
    events = []
    after = None

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


def event_in_zone(event: dict) -> bool:
    """Indique si un événement est dans la zone géographique cible."""
    location = event.get("location") or {}
    return is_in_target_zone(
        city=location.get("city"),
        department=location.get("department"),
        postal_code=location.get("postalCode"),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
    )


def main() -> None:
    if not settings.OPENAGENDA_API_KEY:
        raise SystemExit(
            "OPENAGENDA_API_KEY manquante (variable d'environnement / .env)"
        )

    log.info(
        "Fenêtre API : %s → %s (référence TODAY=%s)",
        settings.DATE_MIN_DT.isoformat(),
        settings.DATE_MAX_DT.isoformat(),
        settings.TODAY,
    )

    candidates = {}
    for query in ["Dijon", "Côte-d'Or", "Bourgogne"]:
        for agenda in search_agendas(query):
            uid = agenda.get("uid")
            if uid is not None:
                candidates[uid] = agenda

    log.info("%d agendas candidats", len(candidates))

    agendas = select_agendas(list(candidates.values()))
    log.info("%d agendas retenus (liste blanche)", len(agendas))

    all_events = []
    seen = set()

    for agenda in agendas:
        try:
            events = fetch_agenda_events(agenda["uid"])
        except RuntimeError as exc:
            log.error("Agenda %s ignoré : %s", agenda.get("uid"), exc)
            continue

        kept = 0
        for event in events:
            uid = event.get("uid")
            if uid is None or uid in seen or not event_in_zone(event):
                continue

            seen.add(uid)
            event["_agenda_title"] = agenda.get("title")
            all_events.append(event)
            kept += 1

        log.info(
            "  • %-45s %4d récupérés -> %4d dans la zone",
            (agenda.get("title") or "")[:45],
            len(events),
            kept,
        )

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_events = clean_unicode(all_events)
    RAW_PATH.write_text(
        json.dumps(safe_events, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info("TOTAL : %d événements uniques sauvegardés", len(all_events))


if __name__ == "__main__":
    main()