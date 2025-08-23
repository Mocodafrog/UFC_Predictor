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


def test_preprocess_fighters_imputes_missing_and_flags():
    df = pd.DataFrame(
        {
            "full_name": ["F1", "F2", "F3"],
            "height": ["6' 0\"", None, "5' 8\""],
            "weight": ["170 lbs.", "145 lbs.", None],
            "reach": ["70\"", "68\"", None],
            "stance": ["Orthodox", "Southpaw", "Orthodox"],
            "wins": [10, 5, 7],
            "losses": [2, 3, 4],
            "draws": [0, 1, 0],
            "birthdate": ["Jan 01, 1990", "Feb 02, 1992", None],
        }
    )

    processed = preprocess_fighters(df)

    assert processed.loc[processed["full_name"] == "F2", "height_missing"].item()
    assert not processed.loc[processed["full_name"] == "F1", "height_missing"].item()
    assert processed.loc[processed["full_name"] == "F3", "weight_missing"].item()
    assert processed.loc[processed["full_name"] == "F3", "reach_missing"].item()

    expected_height = (convert_height_to_cm("6' 0\"") + convert_height_to_cm("5' 8\"")) / 2
    expected_weight = (convert_weight_to_kg("170 lbs.") + convert_weight_to_kg("145 lbs.")) / 2
    expected_reach = (convert_reach_to_cm("70\"") + convert_reach_to_cm("68\"")) / 2

    assert processed.loc[processed["full_name"] == "F2", "height_cm"].item() == pytest.approx(
        expected_height
    )
    assert processed.loc[processed["full_name"] == "F3", "weight_kg"].item() == pytest.approx(
        expected_weight
    )
    assert processed.loc[processed["full_name"] == "F3", "reach_cm"].item() == pytest.approx(
        expected_reach
    )

