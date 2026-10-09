"""Diagnostic : à quoi ressemble `location` dans l'agenda La Vapeur ?"""
import json
import requests
from config import settings

# 1. Retrouver l'UID de l'agenda La Vapeur
r = requests.get(
    f"{settings.OPENAGENDA_BASE_URL}/agendas",
    params={"key": settings.OPENAGENDA_API_KEY, "search": "La Vapeur", "size": 20},
    timeout=30,
)
r.raise_for_status()
agendas = [a for a in r.json()["agendas"] if "vapeur" in a["title"].lower()]
for a in agendas:
    print("AGENDA :", a["uid"], a["title"])

uid = agendas[0]["uid"]

# 2. Regarder le champ location des 3 premiers événements
r = requests.get(
    f"{settings.OPENAGENDA_BASE_URL}/agendas/{uid}/events",
    params={"key": settings.OPENAGENDA_API_KEY, "size": 3},
    timeout=30,
)
r.raise_for_status()
for ev in r.json()["events"]:
    print("---")
    print("titre   :", ev.get("title"))
    print("location:", json.dumps(ev.get("location"), ensure_ascii=False, indent=2))