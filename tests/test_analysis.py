import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis import aggregate_last_five_stats


def test_aggregate_last_five_stats_missing_columns():
    df = pd.DataFrame({"fighter": ["A"]})
    with pytest.raises(ValueError):
        aggregate_last_five_stats(df)


def test_aggregate_last_five_stats_tail_and_date_conversion():
    df = pd.DataFrame(
        {
            "fighter": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6, freq="D").astype(str),
            "KD": range(6),
        }
    )

    result = aggregate_last_five_stats(df)

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result.loc[result["fighter"] == "A", "KD"].iloc[0] == pytest.approx(3.0)
    assert result.loc[result["fighter"] == "A", "date"].iloc[0] == pd.Timestamp("2020-01-06")
