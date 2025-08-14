import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import (
    convert_height_to_cm,
    convert_weight_to_kg,
    convert_reach_to_cm,
    add_fight_number,
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


def test_add_fight_number():
    df = pd.DataFrame(
        {
            "fighter_name": ["A", "A", "B", "A"],
            "date": pd.to_datetime(
                ["2021-01-01", "2021-06-01", "2020-01-01", "2020-01-01"]
            ),
        }
    )

    result = add_fight_number(df)
    expected = pd.DataFrame(
        {
            "fighter_name": ["A", "A", "A", "B"],
            "date": pd.to_datetime(
                ["2020-01-01", "2021-01-01", "2021-06-01", "2020-01-01"]
            ),
            "fight_number": [1, 2, 3, 1],
        }
    )

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
