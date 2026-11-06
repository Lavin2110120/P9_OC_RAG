"""Tests complémentaires des branches de nettoyage et d'orchestration."""
import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest


def test_timings_sans_fin_historique_et_fin_invalide_renvoie_none():
    from scripts.clean_events import _extract_from_timings
    assert _extract_from_timings([{"begin": "2020-01-01", "end": None}],
                                 date_min="2025-01-01", future_after="2026-01-01") == ("2020-01-01", None)
    assert _extract_from_timings([{"begin": "x", "end": "invalide"}]) == ("x", "invalide")


def test_extract_dates_repli_firstdate_et_date_start():
    from scripts.clean_events import _extract_dates
    assert _extract_dates({"firstdate": "a", "lastdate": "b"}) == ("a", "b")
    assert _extract_dates({"date_start": "c", "date_end": "d"}) == ("c", "d")


def test_log_date_diagnostics_absence_nat_et_dataframe_vide(caplog):
    from scripts.clean_events import _log_date_diagnostics
    _log_date_diagnostics(pd.DataFrame({"x": []}), "sans-date")
    assert "Pas de colonne date_begin" in caplog.text
    with_date = pd.DataFrame({"uid": ["x"], "date_begin": [pd.NaT], "date_end": [pd.NaT]})
    _log_date_diagnostics(with_date, "nat")
    _log_date_diagnostics(with_date.iloc[:0], "vide")
    assert "1/1 dates_begin NaT" in caplog.text


def test_clean_dataframe_fallback_date_min_and_elimine_periode_et_zone(monkeypatch):
    import scripts.clean_events as clean
    from config import settings
    monkeypatch.delattr(settings, "DATE_MIN_DT", raising=False)
    monkeypatch.setattr(settings, "DATE_MIN", date(2025, 1, 1), raising=False)
    monkeypatch.setattr(settings, "DATE_MAX", date(2030, 1, 1), raising=False)
    monkeypatch.setattr(settings, "TODAY", date(2026, 1, 1), raising=False)
    monkeypatch.setattr(clean, "is_in_target_zone", lambda **kwargs: kwargs["city"] == "Dijon")
    frame = pd.DataFrame([
        {"uid":"old", "title":"Old", "description":"", "long_description":"", "date_begin":"2024-01-01", "date_end":"2024-01-02", "city":"Dijon", "keywords":"", "location_name":"", "address":"", "postal_code":"", "department":"", "region":"", "conditions":"", "date_range":"", "url":"", "long_description":"", "description":""},
        {"uid":"far", "title":"Far", "description":"", "long_description":"", "date_begin":"2027-01-01", "date_end":"2027-01-02", "city":"Dijon", "keywords":"", "location_name":"", "address":"", "postal_code":"", "department":"", "region":"", "conditions":"", "date_range":"", "url":"", "long_description":"", "description":""},
        {"uid":"outside", "title":"Outside", "description":"", "long_description":"", "date_begin":"2027-01-01", "date_end":"2027-01-02", "city":"ailleurs", "keywords":"", "location_name":"", "address":"", "postal_code":"", "department":"", "region":"", "conditions":"", "date_range":"", "url":"", "long_description":"", "description":""},
    ])
    result = clean.clean_dataframe(frame)
    assert result["uid"].tolist() == ["far"]


def test_clean_dataframe_diagnostic_dates_manquantes():
    from scripts.clean_events import clean_dataframe
    row = {"uid":"x", "title":"Sans date", "description":"", "long_description":"", "date_begin":None, "date_end":None, "city":"Dijon"}
    with pytest.raises(ValueError, match="Cannot set a DataFrame without columns"):
        clean_dataframe(pd.DataFrame([row]))


def test_main_fichier_absent_leve_systemexit(monkeypatch, tmp_path):
    from config import settings
    from scripts import clean_events
    monkeypatch.setattr(settings, "RAW_FILE", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="Fichier brut introuvable"):
        clean_events.main()


def test_main_lit_nettoie_et_ecrit_parquet(monkeypatch, tmp_path):
    from config import settings
    from scripts import clean_events
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps([{
        "uid":"ok", "title":{"fr":"Concert rock"}, "keywords":{"fr":["rock"]},
        "timings":[{"begin":"2027-03-01T20:00:00+01:00", "end":"2027-03-01T22:00:00+01:00"}],
        "location":{"city":"Dijon", "name":"Salle", "address":"Rue", "postalCode":"21000"}
    }]), encoding="utf-8")
    out_dir = tmp_path / "processed"
    out_path = out_dir / "events.parquet"
    monkeypatch.setattr(settings, "RAW_FILE", raw_path)
    monkeypatch.setattr(settings, "PROCESSED_DIR", out_dir)
    monkeypatch.setattr(settings, "PROCESSED_FILE", out_path)
    monkeypatch.setattr(clean_events, "is_in_target_zone", lambda **kwargs: True)
    clean_events.main()
    saved = pd.read_parquet(out_path)
    assert saved["uid"].tolist() == ["ok"]
    assert "Titre : Concert rock" in saved.loc[0, "text"]
