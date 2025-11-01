from pathlib import Path

import pytest

from ufc_predictor import scraping


def test_resolve_output_csv_defaults_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(scraping, "DATA_DIR", tmp_path)
    resolved = scraping._resolve_output_csv(None)
    expected = tmp_path / "fight_stats_raw.csv"
    assert resolved == expected


@pytest.mark.parametrize("relative", ["fight_stats_raw.csv", Path("exports/raw.csv")])
def test_resolve_output_csv_respects_relative_paths(monkeypatch, tmp_path, relative):
    monkeypatch.setattr(scraping, "DATA_DIR", tmp_path)
    resolved = scraping._resolve_output_csv(relative)
    assert resolved == tmp_path / Path(relative)


def test_resolve_output_csv_keeps_absolute_paths(tmp_path):
    absolute = tmp_path / "custom/raw.csv"
    resolved = scraping._resolve_output_csv(absolute)
    assert resolved == absolute
