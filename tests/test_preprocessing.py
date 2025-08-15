import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import (
    convert_height_to_cm,
    convert_weight_to_kg,
    convert_reach_to_cm,
    preprocess_fighters,
)

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


def _make_df(birthdate):
    return pd.DataFrame(
        {
            "full_name": ["Test Fighter"],
            "height": [None],
            "weight": [None],
            "reach": [None],
            "stance": ["Orthodox"],
            "wins": [0],
            "losses": [0],
            "draws": [0],
            "birthdate": [birthdate],
        }
    )


def test_preprocess_fighters_birthdate_nan():
    df = _make_df(float("nan"))
    result = preprocess_fighters(df)
    assert result.loc[0, "birthdate"] == "N/A"


def test_preprocess_fighters_birthdate_none():
    df = _make_df(None)
    result = preprocess_fighters(df)
    assert result.loc[0, "birthdate"] == "N/A"

