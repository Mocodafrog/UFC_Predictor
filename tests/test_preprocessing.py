import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import (
    convert_height_to_cm,
    convert_weight_to_kg,
    convert_reach_to_cm,
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

