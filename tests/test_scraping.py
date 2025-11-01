from pathlib import Path

import pytest

from ufc_predictor import scraping


def test_resolve_output_csv_defaults_to_data_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = scraping._resolve_output_csv(None)
    expected = scraping.DATA_DIR / "fight_stats_raw.csv"
    assert resolved == expected


@pytest.mark.parametrize("relative", ["fight_stats_raw.csv", Path("exports/raw.csv")])
def test_resolve_output_csv_respects_relative_paths(monkeypatch, tmp_path, relative):
    monkeypatch.chdir(tmp_path)
    resolved = scraping._resolve_output_csv(relative)
    assert resolved == tmp_path / Path(relative)
