import pandas as pd
import runpy
import pytest

from ufc_predictor.fight_stats import compute_last_five_stats


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
