"""Tests de fetch_events entièrement mockés (aucun appel réseau)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.fetch_events as fe


# ---------- clean_unicode ----------
def test_clean_unicode_string_valide():
    assert fe.clean_unicode("Concert rock") == "Concert rock"


def test_clean_unicode_surrogate_isole():
    assert "\\ud83d" in fe.clean_unicode("bad\ud83dvalue")


def test_clean_unicode_structures_imbriquees():
    data = {"a": ["x\ud800", 1], "b": {"c": "ok"}}
    out = fe.clean_unicode(data)
    assert out["b"]["c"] == "ok"
    assert out["a"][1] == 1


def test_clean_unicode_non_string_passthrough():
    assert fe.clean_unicode(42) == 42


# ---------- _to_api_param ----------
def test_to_api_param_datetime_aware_converti_en_utc():
    from datetime import datetime, timedelta, timezone
    dt = datetime(2026, 10, 14, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
    assert fe._to_api_param(dt) == "2026-10-14T10:00:00Z"


def test_to_api_param_datetime_naif():
    from datetime import datetime
    assert fe._to_api_param(datetime(2026, 10, 14)) == "2026-10-14T00:00:00"


def test_to_api_param_non_datetime():
    assert fe._to_api_param("2026-10-14") == "2026-10-14"

# ---------- event_in_zone ----------
def test_event_in_zone_delegue_a_is_in_target_zone():
    event = {"location": {"city": "Dijon"}}
    with patch("scripts.fetch_events.is_in_target_zone", return_value=True) as mock:
        assert fe.event_in_zone(event) is True
    mock.assert_called_once_with(
        city="Dijon", department=None,
        postal_code=None, latitude=None, longitude=None,
    )


def test_event_in_zone_sans_location():
    with patch("scripts.fetch_events.is_in_target_zone", return_value=False):
        assert fe.event_in_zone({}) is False


# ---------- _get : retries et 429 ----------
def _mock_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_data or {}
    if status_code < 400:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = fe.requests.HTTPError("err")
    return resp


def test_get_success_premiere_tentative():
    with patch.object(fe, "_session") as session, \
         patch.object(fe.settings, "REQUEST_DELAY", 0, create=True):
        session.get.return_value = _mock_response(200, {"ok": True})
        assert fe._get("http://x") == {"ok": True}


def test_get_429_puis_succes_avec_retry_after():
    with patch.object(fe, "_session") as session, \
         patch.object(fe.settings, "REQUEST_DELAY", 0, create=True), \
         patch("scripts.fetch_events.time.sleep") as mock_sleep:
        session.get.side_effect = [
            _mock_response(429, headers={"Retry-After": "3"}),
            _mock_response(200, {"ok": True}),
        ]
        assert fe._get("http://x") == {"ok": True}
        mock_sleep.assert_called_with(3)


def test_get_429_puis_succes_avec_retry_after():
    with patch.object(fe, "_session") as session, \
         patch.object(fe.settings, "REQUEST_DELAY", 0, create=True), \
         patch("scripts.fetch_events.time.sleep") as mock_sleep:
        session.get.side_effect = [
            _mock_response(429, headers={"Retry-After": "3"}),
            _mock_response(200, {"ok": True}),
        ]
        assert fe._get("http://x") == {"ok": True}
        mock_sleep.assert_any_call(3)


def test_get_echec_definitif_leve_runtimeerror():
    with patch.object(fe, "_session") as session, \
         patch.object(fe.settings, "MAX_RETRIES", 2, create=True), \
         patch("scripts.fetch_events.time.sleep"):
        session.get.side_effect = fe.requests.ConnectionError("boom")
        with pytest.raises(RuntimeError, match="Échec de la requête"):
            fe._get("http://x")


# ---------- pagination ----------
def test_search_agendas_pagine_jusqu_a_epuisement():
    pages = [
        {"agendas": [{"uid": 1, "title": "a"}], "after": "tok"},
        {"agendas": [{"uid": 2, "title": "b"}], "after": None},
    ]
    with patch("scripts.fetch_events._get", side_effect=pages):
        result = fe.search_agendas("Dijon")
    assert [a["uid"] for a in result] == [1, 2]


def test_fetch_agenda_events_pagine():
    pages = [
        {"events": [{"uid": "e1"}], "after": "tok"},
        {"events": [{"uid": "e2"}], "after": None},
    ]
    with patch("scripts.fetch_events._get", side_effect=pages):
        result = fe.fetch_agenda_events(42)
    assert [e["uid"] for e in result] == ["e1", "e2"]


# ---------- main : orchestration et écriture du JSON ----------
def test_main_sans_api_key_leve_systemexit(tmp_path):
    with patch.object(fe.settings, "OPENAGENDA_API_KEY", "", create=True):
        with pytest.raises(SystemExit, match="OPENAGENDA_API_KEY"):
            fe.main()


def test_main_ecrit_events_dans_raw_file(tmp_path):
    raw_file = tmp_path / "events_raw.json"
    agenda = {"uid": 1, "title": "Dijon Culture"}
    event = {"uid": "e1", "location": {"city": "Dijon"}}

    # On patche fe.RAW_PATH (capturé au niveau module, pas settings.RAW_FILE).
    with patch.object(fe.settings, "OPENAGENDA_API_KEY", "key", create=True), \
         patch.object(fe, "RAW_PATH", raw_file), \
         patch("scripts.fetch_events.search_agendas", return_value=[agenda]), \
         patch("scripts.fetch_events.fetch_agenda_events", return_value=[event]), \
         patch("scripts.fetch_events.is_in_target_zone", return_value=True):
        fe.main()

    data = json.loads(raw_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["uid"] == "e1"
    assert data[0]["_agenda_title"] == "Dijon Culture"

def test_main_dedoublonne_les_events(tmp_path):
    raw_file = tmp_path / "events_raw.json"
    agenda = {"uid": 1, "title": "Dijon Culture"}
    with patch.object(fe.settings, "OPENAGENDA_API_KEY", "key", create=True), \
         patch.object(fe, "RAW_PATH", raw_file), \
         patch("scripts.fetch_events.search_agendas", return_value=[agenda]), \
         patch("scripts.fetch_events.fetch_agenda_events",
               return_value=[{"uid": "e1", "location": {}},
                             {"uid": "e1", "location": {}}]), \
         patch("scripts.fetch_events.is_in_target_zone", return_value=True):
        fe.main()
    assert len(json.loads(raw_file.read_text(encoding="utf-8"))) == 1

# ---------- select_agendas ----------
def test_select_agendas_dedoublonne_par_uid_et_exclut_les_uids_nuls():
    agendas = [
        {"uid": 1, "title": "Dijon Culture"},
        {"uid": 1, "title": "Dijon Culture"},  # doublon
        {"uid": 2, "title": "Agenda sans mot-clé géo"},  # gardé : pas de filtre par titre
        {"uid": None, "title": "Dijon Culture"},  # sans uid
    ]
    result = fe.select_agendas(agendas)
    assert sorted(a["uid"] for a in result) == [1, 2]


def test_select_agendas_liste_vide():
    assert fe.select_agendas([]) == []


# ---------- main : échec d'agenda ----------
def test_main_echec_agenda_leve_runtimeerror_collecte_incomplete(tmp_path):
    raw_file = tmp_path / "events_raw.json"
    agenda = {"uid": 1, "title": "Dijon Culture"}
    with patch.object(fe.settings, "OPENAGENDA_API_KEY", "key", create=True), \
         patch.object(fe, "RAW_PATH", raw_file), \
         patch("scripts.fetch_events.search_agendas", return_value=[agenda]), \
         patch("scripts.fetch_events.fetch_agenda_events",
               side_effect=RuntimeError("timeout")), \
         patch("scripts.fetch_events.is_in_target_zone", return_value=True):
        with pytest.raises(RuntimeError, match="Collecte incomplète : échec de l'agenda 1"):
            fe.main()
    # Aucun fichier partiel ne doit avoir été écrit
    assert not raw_file.exists()