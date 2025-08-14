import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import (
    convert_height_to_cm,
    convert_weight_to_kg,
    convert_reach_to_cm,
    add_rolling_features,
)
import pandas as pd


def test_convert_height_to_cm_malformed():
    assert convert_height_to_cm("--") is None
    assert convert_height_to_cm("") is None
    assert convert_height_to_cm("bad format") is None


def test_convert_weight_to_kg_malformed():
    assert convert_weight_to_kg("--") is None
    assert convert_weight_to_kg("") is None
    assert convert_weight_to_kg("bad format") is None


def test_convert_reach_to_cm_malformed():
    assert convert_reach_to_cm("--") is None
    assert convert_reach_to_cm("") is None
    assert convert_reach_to_cm("bad format") is None


def test_add_rolling_features():
    df = pd.DataFrame(
        {
            "fighter_name": ["A", "A", "A", "B", "A", "B"],
            "KD": [0, 1, 2, 0, 3, 1],
            "Sig Str": [10, 20, 30, 5, 40, 15],
            "TD": [0, 1, 0, 0, 2, 0],
            "Control": [0, 50, 100, 0, 150, 0],
            "result": ["W", "L", "W", "W", "L", "L"],
        }
    )
    res = add_rolling_features(df)
    assert res.loc[0, "KD"] == 0
    assert res.loc[1, "KD"] == pytest.approx(0.5)
    assert res.loc[2, "KD"] == pytest.approx(1.0)
    assert res.loc[4, "KD"] == pytest.approx(1.5)
    assert res.loc[3, "form_last_5"] == 1.0
    assert res.loc[5, "form_last_5"] == pytest.approx(0.5)
