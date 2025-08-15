import pandas as pd
import pytest

from ufc_predictor.preprocessing import (
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


def test_preprocess_fighters_birthdate_null_values():
    df = pd.DataFrame(
        {
            "full_name": ["F1", "F2"],
            "height": ["6' 0\"", "5' 8\""],
            "weight": ["170 lbs.", "145 lbs."],
            "reach": ["70\"", "68\""],
            "stance": ["Orthodox", "Southpaw"],
            "wins": [10, 5],
            "losses": [2, 3],
            "draws": [0, 1],
            "birthdate": [None, float("nan")],
        }
    )

    processed = preprocess_fighters(df)

    assert (processed.loc[processed["full_name"] == "F1", "birthdate"].item()) == "N/A"
    assert (processed.loc[processed["full_name"] == "F2", "birthdate"].item()) == "N/A"

