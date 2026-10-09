"""Tests de la collecte et de la qualité des données."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.clean_events import clean_html, flatten_event, get_fr, is_in_target_zone

@pytest.mark.parametrize("city,dept,attendu", [
    ("Dijon", "Côte-d'Or", True),
    ("DIJON", None, True),
    ("Dijon Cedex", None, True),
    ("Beaune", "Cote d'Or", True),     # variante sans accent
    ("Lyon", "Rhône", False),
    (None, None, False),
])
def test_is_in_target_zone(city, dept, attendu):
    assert is_in_target_zone(city, dept) is attendu

# ---------- Tests des fonctions (sans réseau) ----------

def test_get_fr_multilingue():
    assert get_fr({"fr": "Concert", "en": "Gig"}) == "Concert"
    assert get_fr({"en": "Gig"}) == "Gig"
    assert get_fr("texte") == "texte"
    assert get_fr(None) is None


def test_clean_html():
    assert clean_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert clean_html(None) == ""
    assert clean_html("  trop   d'espaces ") == "trop d'espaces"


def test_flatten_event_minimal():
    raw = {
        "uid": 1,
        "title": {"fr": "Expo"},
        "location": {"city": "Lyon"},
        "timings": [{"begin": "2026-05-01T10:00:00+02:00", "end": "2026-05-01T18:00:00+02:00"}],
    }
    flat = flatten_event(raw)
    assert flat["title"] == "Expo"
    assert flat["city"] == "Lyon"
    assert flat["date_begin"].startswith("2026-05-01")


def test_flatten_event_champs_manquants():
    """Un événement incomplet ne doit pas faire planter le script."""
    flat = flatten_event({"uid": 2})
    assert flat["title"] == ""
    assert flat["date_begin"] is None


# ---------- Tests sur le jeu de données produit ----------

@pytest.fixture(scope="module")
def df():
    if not settings.PROCESSED_FILE.exists():
        pytest.skip("Données non générées : lancer fetch_events.py puis clean_events.py")
    return pd.read_parquet(settings.PROCESSED_FILE)


def test_dataset_non_vide(df):
    assert len(df) > 0


def test_colonnes_attendues(df):
    expected = {"uid", "title", "text", "city", "date_begin", "date_end", "url",
                "category", "categories_txt"}
    assert expected.issubset(df.columns)


def test_uid_uniques(df):
    assert df["uid"].is_unique


def test_textes_non_vides(df):
    assert (df["title"].str.len() > 0).all()
    assert (df["text"].str.len() > 20).all()


def test_periode_respectee(df):
    date_min = pd.Timestamp(settings.DATE_MIN, tz="UTC")
    fin = df["date_end"].fillna(df["date_begin"])
    assert (fin >= date_min).all(), "Des événements de plus d'un an sont présents"


def test_zone_geographique(df):
    ok = df.apply(lambda r: is_in_target_zone(r["city"], r["department"]), axis=1)
    assert ok.all(), "Des événements hors zone sont présents"


def test_pas_de_html_residuel(df):
    assert not df["text"].str.contains(r"<[a-z]+[^>]*>", regex=True).any()