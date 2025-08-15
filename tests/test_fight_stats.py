import pandas as pd

from ufc_predictor.fight_stats import compute_last_five_stats


def test_compute_last_five_stats_missing_csv(tmp_path, capsys):
    missing_file = tmp_path / "missing.csv"
    df = compute_last_five_stats(missing_file)
    captured = capsys.readouterr()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "not found" in captured.out.lower()
