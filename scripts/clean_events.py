"""Nettoyage et structuration des événements bruts (Dijon / Côte-d'Or + 20 km).

Pipeline RAG événements :
    1. fetch_events.py  →  data/raw/events_raw.json  (puis parquet)
    2. clean_events.py  →  data/processed/events.parquet  (ce script)
    3. vectorize_events.py → data/vector_store/{faiss_index.bin, metadata.parquet}

Notes sur les dates :
    Open Agenda renvoie des dates ISO avec un offset horaire (souvent +01:00
    en hiver et +02:00 en été). `pd.to_datetime(..., utc=True, errors="coerce")`
    résout l'erreur « Mixed timezones detected » en ramenant toutes les
    valeurs en UTC. Les bornes DATE_MIN / DATE_MAX sont comparées en UTC.

Colonnes de date explorées (dans l'ordre de priorité) :
    - ``timings[].begin`` / ``timings[].end`` (le plus fiable — Open Agenda v2)
    - ``firstdate`` / ``lastdate`` (repli pour agendas sans ``timings``)
    - ``date_start`` / ``date_end`` (repli supplémentaire)

Usage :
    python -m scripts.clean_events
"""
import json
import logging
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings
from config.geo import is_in_target_zone  # signature (city, dept, postal, lat, lon)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Mapping issu du classement manuel des mots-clés Open Agenda (valeurs normalisées :
# minuscules, sans accents, sans tirets/apostrophes → cf. normalize()).
CATEGORY_KEYWORDS = {
    "musique": ["la vapeur", "concert", "pop", "rap", "rock", "electro", "chanson"],
    "sante_bien_etre": ["prevention sante", "retraite", "bien etre", "bien vieillir", "addictions"],
    "sport": ["asept fc b", "mai a velo"],
    "biodiversite": ["botanique", "flore"],
    "theatre": ["theatre"],
    "festival": ["festival"],
    "vivre_ensemble": ["jeune public", "seniors", "dijon"],
}
# Filtre éditorial/culturel existant, conservé : ce n'est pas une borne de dates.
EXCLUDED_CATEGORIES: list[str] = ["sante_bien_etre", "vivre_ensemble"]

# Ordre de priorité : la 1re catégorie qui matche l'emporte.
# "vivre_ensemble" est volontairement en DERNIER car "dijon" est présent partout.
CATEGORY_PRIORITY = ["musique", "theatre", "festival", "sante_bien_etre", "sport", "biodiversite", "vivre_ensemble"]


# ---------- Fonctions utilitaires (testables unitairement) ----------

def normalize(value) -> str:
    """Minuscules, sans accents, sans tirets/apostrophes : pour comparer des noms."""
    if not value or not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(c for c in text if not unicodedata.combining(c))
    for ch in "-'’":
        text = text.replace(ch, " ")
    return " ".join(text.lower().split())


def get_fr(value):
    """Extrait la version française d'un champ potentiellement multilingue."""
    if isinstance(value, dict):
        return value.get("fr") or next(iter(value.values()), None)
    return value


def clean_html(text) -> str:
    """Supprime les balises HTML / Markdown simples et normalise les espaces."""
    if not text or not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    return " ".join(text.replace("**", "").split())


# ---------- Extraction tolérante des colonnes de date ----------

# Colonnes alternatives explorées si ``timings`` est absent ou vide.
_BEGIN_CANDIDATES = ("begin", "start", "from", "dateStart", "date_start")
_END_CANDIDATES = ("end", "finish", "to", "dateEnd", "date_end")


def _extract_from_timings(timings, date_min=None, future_after=None):
    """Renvoie les bornes utiles d'un événement à occurrences multiples.

    Si les premiers créneaux sont échus mais que l'événement possède encore
    des créneaux dans la période collectée, le début aplati devient le premier
    créneau non échu. La forme du dictionnaire retourné par ``flatten_event``
    reste inchangée.
    """
    valid = [slot for slot in (timings or []) if isinstance(slot, dict)]
    if not valid:
        return None, None

    def stamp(value):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return parsed if not pd.isna(parsed) else None

    lower_bound = stamp(date_min) if date_min is not None else None
    future_cutoff = stamp(future_after) if future_after is not None else None
    ordered = sorted(
        enumerate(valid),
        key=lambda pair: stamp(pair[1].get("begin")) or pd.Timestamp.max.tz_localize("UTC"),
    )
    future_slots, in_history_window = [], []
    for _, slot in ordered:
        begin, end = stamp(slot.get("begin")), stamp(slot.get("end"))
        if future_cutoff is not None and end is not None and end >= future_cutoff:
            future_slots.append(slot)
        if lower_bound is None or (end is not None and end >= lower_bound):
            in_history_window.append(slot)
    # Priorité à la prochaine occurrence active/future; s'il n'y en a pas,
    # garder la première occurrence de la fenêtre historique conservée.
    first = (future_slots or in_history_window or [ordered[0][1]])[0]
    ends = [slot.get("end") for slot in valid if slot.get("end") not in (None, "")]
    if ends:
        parsed_ends = [(stamp(value), value) for value in ends]
        valid_ends = [(when, value) for when, value in parsed_ends if when is not None]
        last_end = max(valid_ends, key=lambda item: item[0])[1] if valid_ends else ends[-1]
    else:
        last_end = None
    return first.get("begin"), last_end


def _extract_from_dict_keys(obj, candidates):
    """Renvoie la première valeur non-vide parmi les clés candidates."""
    if not isinstance(obj, dict):
        return None
    for k in candidates:
        v = obj.get(k)
        if v not in (None, "", []):
            return v
    return None


def _extract_dates(e: dict):
    """Tente plusieurs stratégies pour récupérer (date_begin, date_end) d'un event.

    Ordre :
      1. ``timings[].begin`` / ``timings[].end``
      2. ``firstdate`` / ``lastdate``
      3. ``date_start`` / ``date_end``
    """
    timings = e.get("timings") or []
    reference_day = getattr(settings, "TODAY", None)
    future_after = (
        datetime.combine(reference_day, datetime.min.time(), tzinfo=timezone.utc)
        if reference_day is not None else None
    )
    b, en = _extract_from_timings(
        timings, getattr(settings, "DATE_MIN_DT", None), future_after
    )
    if b is not None or en is not None:
        return b, en

    b = _extract_from_dict_keys(e, ("firstdate", "firstDate", "date_start", "dateStart"))
    en = _extract_from_dict_keys(e, ("lastdate", "lastDate", "date_end", "dateEnd"))
    if b is not None or en is not None:
        return b, en

    # Dernier repli : certaines réponses exposent un objet ``dates`` unique.
    dates = e.get("dates")
    if isinstance(dates, dict):
        return dates.get("begin"), dates.get("end")

    return None, None


def flatten_event(e: dict) -> dict:
    """Aplatit un événement brut Open Agenda en dictionnaire plat.

    Gère 3 formats de date (timings, firstdate/lastdate, date_start/date_end).
    """
    loc = e.get("location") or {}
    keywords = get_fr(e.get("keywords")) or []

    date_begin, date_end = _extract_dates(e)
    timings = e.get("timings") or []

    return {
        "uid": e.get("uid"),
        "title": clean_html(get_fr(e.get("title"))),
        "description": clean_html(get_fr(e.get("description"))),
        "long_description": clean_html(get_fr(e.get("longDescription"))),
        "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords),
        "conditions": clean_html(get_fr(e.get("conditions"))),
        "date_range": clean_html(get_fr(e.get("dateRange"))),
        "date_begin": date_begin,
        "date_end": date_end,
        "location_name": loc.get("name"),
        "address": loc.get("address"),
        "city": loc.get("city"),
        "postal_code": loc.get("postalCode"),
        "department": loc.get("department"),
        "region": loc.get("region"),
        "latitude": loc.get("latitude"),
        "longitude": loc.get("longitude"),
        "url": e.get("canonicalUrl"),
        "agenda": e.get("_agenda_title"),
        "_had_timings": bool(timings),
    }


# Pré-calcul : on normalise les mots-clés du mapping une seule fois
_NORMALIZED = {cat: {normalize(k) for k in kws} for cat, kws in CATEGORY_KEYWORDS.items()}


def categorize_all(title, keywords) -> list[str]:
    kw_set = {normalize(k) for k in str(keywords or "").split(",") if k.strip()}
    cats = [c for c in CATEGORY_PRIORITY if kw_set & _NORMALIZED[c]]
    return cats or ["autre"]


def build_text(row: pd.Series) -> str:
    """Construit le texte qui sera vectorisé à l'étape 3."""
    lieu = ", ".join(p for p in [row["location_name"], row["address"], row["city"]] if p)
    parts = [
        f"Titre : {row['title']}",
        f"Catégories : {row['categories_txt']}" if row["categories_txt"] else "",
        f"Description : {row['description']}" if row["description"] else "",
        f"Détails : {row['long_description']}" if row["long_description"] else "",
        f"Dates : {row['date_range']}" if row["date_range"] else "",
        f"Lieu : {lieu}" if lieu else "",
        f"Mots-clés : {row['keywords']}" if row["keywords"] else "",
        f"Conditions : {row['conditions']}" if row["conditions"] else "",
    ]
    return "\n".join(p for p in parts if p)


def _to_utc(series: pd.Series) -> pd.Series:
    """Conversion datetime tolérante en UTC.

    - ``errors="coerce"`` : valeurs invalides → NaT (jamais d'exception).
    - ``utc=True`` : harmonise les fuseaux en UTC (résout « Mixed timezones »).
    """
    return pd.to_datetime(series, utc=True, errors="coerce")


def _log_date_diagnostics(df: pd.DataFrame, label: str) -> None:
    """Affiche un diagnostic clair sur la colonne de dates."""
    if "date_begin" not in df.columns:
        log.warning("[%s] Pas de colonne date_begin", label)
        return
    n_total = len(df)
    n_nat = int(df["date_begin"].isna().sum())
    if n_nat:
        log.warning("[%s] %d/%d dates_begin NaT après parsing UTC", label, n_nat, n_total)
    else:
        log.info("[%s] 0/%d dates_begin NaT après parsing UTC", label, n_total)
    valid = df["date_begin"].dropna()
    if not valid.empty:
        log.info(
            "[%s] date_begin min=%s max=%s (tz=%s)",
            label, valid.min(), valid.max(),
            getattr(valid.iloc[0], "tzinfo", None),
        )
    if n_total:
        sample = df[["uid", "date_begin", "date_end"]].head(3).to_string(index=False)
        log.info("[%s] 3 exemples (avant filtre):\n%s", label, sample)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique toutes les règles de qualité, de catégorisation et de filtrage."""
    n0 = len(df)
    log.info("Pipeline clean_events — %d événements bruts", n0)

    # 1. Doublons (un même événement peut figurer dans plusieurs agendas)
    df = df.drop_duplicates(subset="uid").copy()
    n1 = len(df)
    log.info("[1/8] Dédoublonnage : %d → %d", n0, n1)

    # 2. Qualité minimale : un titre suffit; descriptions peuvent manquer
    # sur les fiches OpenAgenda, et ne doivent pas supprimer un événement valide.
    for col in ["title", "description", "long_description"]:
        df[col] = df[col].fillna("")
    df = df[df["title"].str.len() > 0].copy()
    n2 = len(df)
    log.info("[2/8] Champs obligatoires : %d → %d", n1, n2)

    # ---- DIAGNOSTIC période (AVANT le filtre) ----
    log.info(
        "[3/8] Fenêtre utilisée : DATE_MIN=%s → sans borne supérieure "
        "(DATE_MAX configurée=%s, référence TODAY=%s)",
        getattr(settings, "DATE_MIN_DT", settings.DATE_MIN),
        getattr(settings, "DATE_MAX_DT", settings.DATE_MAX),
        settings.TODAY,
    )

    # 3. Dates : conversion robuste en UTC puis filtrage de la période
    df["date_begin"] = _to_utc(df["date_begin"])
    df["date_end"] = _to_utc(df["date_end"])
    _log_date_diagnostics(df, "3-avant")

    n_before_dropna = len(df)
    df = df.dropna(subset=["date_begin"])
    log.info("[3/8] Après dropna(date_begin) : %d → %d", n_before_dropna, len(df))

    # Borne basse UTC uniquement. DATE_MAX reste exposée dans settings pour
    # compatibilité, mais ne doit plus exclure les événements futurs lointains.
    if hasattr(settings, "DATE_MIN_DT"):
        date_min = pd.Timestamp(settings.DATE_MIN_DT)
    else:
        date_min = pd.Timestamp(settings.DATE_MIN, tz="UTC")

    # Garde tout événement encore en cours à DATE_MIN ou commençant ensuite.
    end_or_begin = df["date_end"].fillna(df["date_begin"])
    mask_period = end_or_begin >= date_min
    n_dropped_period = int((~mask_period).sum())
    if n_dropped_period:
        log.warning(
            "[3/8] Filtre période élimine %d événements "
            "(date_end < %s)",
            n_dropped_period, date_min,
        )
    df = df[mask_period].copy()
    n3 = len(df)
    log.info("[3/8] Après filtre période : %d → %d", len(df) + n_dropped_period, n3)
    _log_date_diagnostics(df, "3-après")

    # 4. Filtre géographique — utilise TOUS les critères (ville, dept, postal, lat/lon).
    #    On évite l'écueil du shadowing en important directement depuis config.geo.
    mask = df.apply(
        lambda r: is_in_target_zone(
            city=r.get("city"),
            department=r.get("department"),
            postal_code=r.get("postal_code"),
            latitude=r.get("latitude"),
            longitude=r.get("longitude"),
        ),
        axis=1,
    )
    n_dropped_zone = int((~mask).sum())
    if n_dropped_zone:
        log.warning("[4/8] Filtre zone élimine %d événements", n_dropped_zone)
    df = df[mask].copy()
    n4 = len(df)
    log.info("[4/8] Après filtre zone : %d → %d", len(df) + n_dropped_zone, n4)

    # 5. Valeurs manquantes non bloquantes → chaînes vides
    text_cols = ["location_name", "address", "city", "postal_code", "department",
                 "region", "keywords", "conditions", "date_range", "url"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # 6. Catégorisation (type d'événement)
    df["categories"] = df.apply(lambda r: categorize_all(r["title"], r["keywords"]), axis=1)
    df["category"] = df["categories"].str[0]                 # catégorie principale
    df["categories_txt"] = df["categories"].str.join(", ")   # pour le texte et les métadonnées

    # 7. Exclusion des catégories hors périmètre culturel (optionnel)
    if EXCLUDED_CATEGORIES:
        df = df[~df["category"].isin(EXCLUDED_CATEGORIES)].copy()
    n5 = len(df)
    log.info("[7/8] Après exclusion catégories : %d → %d", n4, n5)

    # 8. Texte final pour l'embedding
    df["text"] = df.apply(build_text, axis=1)

    log.info(
        "Nettoyage : %d bruts → %d après dédoublonnage → %d après champs obligatoires "
        "→ %d après période → %d après zone → %d après exclusion catégories",
        n0, n1, n2, n3, n4, n5,
    )
    if not df.empty:
        log.info("Répartition par catégorie :\n%s", df["category"].value_counts().to_string())
    return df.reset_index(drop=True)


def main() -> None:
    raw_path = settings.RAW_FILE
    if not raw_path.exists():
        raise SystemExit(f"Fichier brut introuvable : {raw_path}")
    log.info("Chargement de %s", raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    log.info("✅ %d événements chargés depuis %s", len(raw), raw_path)

    df = pd.DataFrame([flatten_event(e) for e in raw])
    df = clean_dataframe(df)

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.PROCESSED_FILE, index=False)

    log.info("✅ %d événements sauvegardés dans %s", len(df), settings.PROCESSED_FILE)
    if not df.empty:
        log.info("Période : %s → %s", df["date_begin"].min(), df["date_begin"].max())
        log.info("Villes :\n%s", df["city"].value_counts().head(10).to_string())
        log.info("Longueur moyenne du texte : %.0f caractères", df["text"].str.len().mean())


if __name__ == "__main__":
    main()

