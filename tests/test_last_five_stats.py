import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing import compute_last_five_stats


def test_compute_last_five_stats_form_and_means():
    data = {
        "Fighter": ["Test Fighter"] * 6,
        "Result": ["W", "L", "W", "L", "W", "L"],
        "KD": [1, 2, 3, 4, 5, 6],
        "TD": [6, 5, 4, 3, 2, 1],
    }
    df = pd.DataFrame(data)
    stats = compute_last_five_stats(df)

    sixth_fight = stats.iloc[5]
    assert sixth_fight["form_last_5"] == pytest.approx(0.6)
    assert sixth_fight["KD"] == pytest.approx(3.0)
    assert sixth_fight["TD"] == pytest.approx(4.0)
