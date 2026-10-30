"""Tests unitaires des helpers de clean_events (lignes 95-136, 272-281...)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clean_events import (
    build_text, categorize_all, clean_html,
    _extract_dates, _extract_from_dict_keys, _extract_from_timings,
    flatten_event, get_fr, normalize,
)


# ---------- normalize ----------
def test_normalize_minuscules_sans_accents_ni_tirets():
    assert normalize("Théâtre-Café d'Été") == "theatre cafe d ete"


def test_normalize_vide_ou_non_string():
    assert normalize(None) == ""
    assert normalize(123) == ""
    assert normalize("   ") == ""


# ---------- get_fr ----------
def test_get_fr_dictionnaire_avec_fr():
    assert get_fr({"fr": "Bonjour", "en": "Hello"}) == "Bonjour"


def test_get_fr_dictionnaire_sans_fr_prend_premiere_valeur():
    assert get_fr({"en": "Hello"}) == "Hello"


def test_get_fr_non_dict_passthrough():
    assert get_fr("simple") == "simple"


# ---------- clean_html ----------
def test_clean_html_supprime_balises_et_normalise():
    assert clean_html("<p>Bonjour **le** monde</p>") == "Bonjour le monde"


def test_clean_html_vide_ou_non_string():
    assert clean_html(None) == ""
    assert clean_html(42) == ""


# ---------- extraction de dates ----------
def test_extract_from_timings_vide_ou_invalide():
    assert _extract_from_timings([]) == (None, None)
    assert _extract_from_timings(None) == (None, None)
    assert _extract_from_timings(["pas un dict"]) == (None, None)


def test_extract_from_timings_multiple_prend_premier_et_dernier():
    timings = [{"begin": "2026-01-01", "end": "2026-01-02"},
               {"begin": "2026-02-01", "end": "2026-02-03"}]
    assert _extract_from_timings(timings) == ("2026-01-01", "2026-02-03")


def test_extract_from_dict_keys_premiere_cle_non_vide():
    obj = {"firstdate": "", "date_start": "2026-01-01"}
    assert _extract_from_dict_keys(obj, ("firstdate", "date_start")) == "2026-01-01"


def test_extract_from_dict_keys_aucune_cle():
    assert _extract_from_dict_keys({"autre": 1}, ("firstdate",)) is None
    assert _extract_from_dict_keys("pas un dict", ("firstdate",)) is None


def test_extract_dates_repli_dates_object():
    e = {"dates": {"begin": "2026-03-01", "end": "2026-03-02"}}
    assert _extract_dates(e) == ("2026-03-01", "2026-03-02")


def test_extract_dates_rien_trouve():
    assert _extract_dates({}) == (None, None)


# ---------- flatten_event : valeurs manquantes ----------
def test_flatten_event_champs_absents_ne_plante_pas():
    out = flatten_event({})
    assert out["title"] == ""
    assert out["uid"] is None
    assert out["_had_timings"] is False


# ---------- categorize_all ----------
def test_categorize_all_match_musique():
    assert "musique" in categorize_all("Concert", "rock, chanson")


def test_categorize_all_aucun_match_renvoie_autre():
    assert categorize_all("Titre", "inconnu") == ["autre"]


def test_categorize_all_avec_accent_et_casse():
    assert "theatre" in categorize_all("Théâtre", "Théâtre")


# ---------- build_text ----------
def test_build_text_contient_les_sections_renseignees():
    row = pd.Series({
        "title": "Concert", "categories_txt": "musique",
        "description": "Desc", "long_description": "Long",
        "date_range": "14 oct", "location_name": "La Vapeur",
        "address": "12 rue", "city": "Dijon",
        "keywords": "rock", "conditions": "Entrée libre",
    })
    text = build_text(row)
    for fragment in ("Titre : Concert", "Catégories : musique",
                     "Lieu : La Vapeur, 12 rue, Dijon", "Conditions :"):
        assert fragment in text


def test_build_text_ignore_les_sections_vides():
    row = pd.Series({
        "title": "T", "categories_txt": "", "description": "",
        "long_description": "", "date_range": "", "location_name": "",
        "address": "", "city": "", "keywords": "", "conditions": "",
    })
    assert build_text(row) == "Titre : T"