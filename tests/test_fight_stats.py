import pandas as pd
import runpy
import pytest

import ufc_predictor.fight_stats as fight_stats_module
from ufc_predictor.fight_stats import compute_last_five_stats, _resolve_export_dir


def test_compute_last_five_stats_missing_csv(tmp_path, capsys):
    missing_file = tmp_path / "missing.csv"
    df = compute_last_five_stats(missing_file)
    captured = capsys.readouterr()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "not found" in captured.out.lower()


def test_main_exits_when_no_data(monkeypatch, capsys):
    def fake_read_csv(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ufc_predictor.fight_stats", run_name="__main__")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "No stats generated; skipping export." in captured.out


def test_filters_and_normalizes(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "Fighter": ["F1", "F2", "F3", "F4"],
            "Winner": ["W", "NC", "W", "D"],
            "Method": [
                "Decision - Unanimous",
                "TKO - Doctor's Stoppage",
                "TKO - Doctor's Stoppage",
                "DQ",
            ],
            "Weight Class": [
                "Bantamweight Bout",
                "Featherweight Bout",
                "Lightweight Bout",
                "Heavyweight Bout",
            ],
            "KD": [0, 0, 0, 0],
            "Sig. Str.": ["0 of 0"] * 4,
            "Total Str.": ["0 of 0"] * 4,
            "TD": ["0 of 0"] * 4,
            "Sub. Att": [0, 0, 0, 0],
            "Reversal": [0, 0, 0, 0],
            "Control Time": ["0:00"] * 4,
            "Head": ["0 of 0"] * 4,
            "Body": ["0 of 0"] * 4,
            "Leg": ["0 of 0"] * 4,
            "Distance": ["0 of 0"] * 4,
            "Clinch": ["0 of 0"] * 4,
            "Ground": ["0 of 0"] * 4,
            "Fight_lenght": ["0:01"] * 4,
            "Format": ["3 Rnd"] * 4,
        }
    )
    csv = tmp_path / "raw.csv"
    data.to_csv(csv, index=False)

    monkeypatch.setattr(fight_stats_module, "DATA_DIR", tmp_path)
    df = compute_last_five_stats(csv)

    assert len(df) == 2
    assert set(df["Weight Class"]) == {"Bantamweight", "Lightweight"}
    assert set(df["Method"]) == {"Decision", "KO/TKO"}
    assert not set(df["Winner"]) & {"NC", "D"}


def test_uses_custom_output_directory(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "Fighter": ["F1"],
            "Winner": ["W"],
            "Method": ["Decision - Unanimous"],
            "Weight Class": ["Bantamweight"],
            "KD": [0],
            "Sig. Str.": ["1 of 1"],
            "Total Str.": ["1 of 1"],
            "TD": ["0 of 0"],
            "Sub. Att": [0],
            "Reversal": [0],
            "Control Time": ["0:10"],
            "Head": ["1 of 1"],
            "Body": ["0 of 0"],
            "Leg": ["0 of 0"],
            "Distance": ["1 of 1"],
            "Clinch": ["0 of 0"],
            "Ground": ["0 of 0"],
            "Fight_lenght": ["0:10"],
            "Format": ["3 Rnd"],
        }
    )
    csv = tmp_path / "raw.csv"
    data.to_csv(csv, index=False)

    output_dir = tmp_path / "exports"
    monkeypatch.setattr(fight_stats_module, "DATA_DIR", tmp_path / "unused")

    df = compute_last_five_stats(csv, output_dir=output_dir)

    assert not df.empty
    assert (output_dir / "fight_stats.csv").is_file()
    assert (output_dir / "df_estadisticas_ultimos_5.csv").is_file()


def test_resolve_export_dir_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(fight_stats_module, "DATA_DIR", tmp_path)
    resolved = _resolve_export_dir("exports")
    assert resolved == tmp_path / "exports"


def test_weight_class_regex_mapping(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "Fighter": ["F1", "F2", "F3", "F4"],
            "Winner": ["W"] * 4,
            "Method": ["Decision - Majority"] * 4,
            "Weight Class": [
                "Road To UFC 1 Bantamweight Tournament Title Bout",
                "UFC 10 Tournament Title Bout",
                "UFC Women's Featherweight Title Bout",
                "Catch Weight Bout",
            ],
            "KD": [0, 0, 0, 0],
            "Sig. Str.": ["0 of 0"] * 4,
            "Total Str.": ["0 of 0"] * 4,
            "TD": ["0 of 0"] * 4,
            "Sub. Att": [0, 0, 0, 0],
            "Reversal": [0, 0, 0, 0],
            "Control Time": ["0:00"] * 4,
            "Head": ["0 of 0"] * 4,
            "Body": ["0 of 0"] * 4,
            "Leg": ["0 of 0"] * 4,
            "Distance": ["0 of 0"] * 4,
            "Clinch": ["0 of 0"] * 4,
            "Ground": ["0 of 0"] * 4,
            "Fight_lenght": ["0:01"] * 4,
            "Format": ["3 Rnd"] * 4,
        }
    )
    csv = tmp_path / "raw.csv"
    data.to_csv(csv, index=False)

    monkeypatch.setattr(fight_stats_module, "DATA_DIR", tmp_path)
    df = compute_last_five_stats(csv)

    assert len(df) == 4
    assert set(df["Weight Class"]) == {
        "Bantamweight",
        "Openweight",
        "Women's Featherweight",
        "Catchweight",
    }
