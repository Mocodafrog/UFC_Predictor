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


def test_convert_height_to_cm_variants():
    assert convert_height_to_cm("6'2\"") == pytest.approx(187.96, abs=0.01)
    assert convert_height_to_cm("6ft2in") == pytest.approx(187.96, abs=0.01)
    assert convert_height_to_cm("6ft") == pytest.approx(182.88, abs=0.01)
    assert convert_height_to_cm("180cm") is None


def test_convert_weight_to_kg_malformed():
    assert convert_weight_to_kg("--") is None
    assert convert_weight_to_kg("") is None
    assert convert_weight_to_kg("bad format") is None


def test_convert_weight_to_kg_variants():
    expected = 170 * 0.453592
    assert convert_weight_to_kg("170 lb") == pytest.approx(expected, abs=0.01)
    assert convert_weight_to_kg("170lbs") == pytest.approx(expected, abs=0.01)
    assert convert_weight_to_kg("170lb") == pytest.approx(expected, abs=0.01)
    assert convert_weight_to_kg("170 lbs.") == pytest.approx(expected, abs=0.01)
    assert convert_weight_to_kg("80kg") is None


def test_convert_reach_to_cm_malformed():
    assert convert_reach_to_cm("--") is None
    assert convert_reach_to_cm("") is None
    assert convert_reach_to_cm("bad format") is None


def test_convert_reach_to_cm_variants():
    expected = 70 * 2.54
    assert convert_reach_to_cm('70"') == pytest.approx(expected, abs=0.01)
    assert convert_reach_to_cm('70in') == pytest.approx(expected, abs=0.01)
    assert convert_reach_to_cm('70 in.') == pytest.approx(expected, abs=0.01)
    assert convert_reach_to_cm('180cm') is None


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

