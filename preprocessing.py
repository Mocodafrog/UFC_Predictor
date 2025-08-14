"""Data cleaning and preprocessing utilities for UFC fighters."""
from __future__ import annotations
import re
import pandas as pd


def convert_height_to_cm(height: str) -> float | None:
    """Convert a height value like ``6' 2\"`` to centimeters."""
    if height and height != "--":
        try:
            feet, inches = height.split("' ")
            inches = inches.replace('"', '')
            return int(feet) * 30.48 + int(inches) * 2.54
        except ValueError:
            return None
    return None


def convert_weight_to_kg(weight: str) -> float | None:
    """Convert a weight value like ``170 lbs."`` to kilograms."""
    if weight and weight != "--":
        try:
            return float(weight.replace(' lbs.', '')) * 0.453592
        except ValueError:
            return None
    return None


def convert_reach_to_cm(reach: str) -> float | None:
    """Convert a reach measurement in inches to centimeters."""
    if reach and reach != "--":
        try:
            return float(reach.replace('"', '')) * 2.54
        except ValueError:
            return None
    return None


def preprocess_fighters(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the fighters DataFrame returned by :func:`scrape_fighters`.

    The function removes duplicated names, assigns a numeric ``fighter_id`` and
    converts physical metrics to metric units.

    Parameters
    ----------
    df:
        Raw DataFrame obtained from ``scrape_fighters``.

    Returns
    -------
    pandas.DataFrame
        Cleaned and processed DataFrame ready for storage or further analysis.
    """
    duplicated = df[df.duplicated(subset=["full_name"], keep=False)]
    df = df[~df["full_name"].isin(duplicated["full_name"])]

    df["fighter_id"] = pd.factorize(df["full_name"])[0] + 1
    df["height_cm"] = df["height"].apply(convert_height_to_cm)
    df["weight_kg"] = df["weight"].apply(convert_weight_to_kg)
    df["reach_cm"] = df["reach"].apply(convert_reach_to_cm)

    df["height_cm"].fillna(0, inplace=True)
    df["weight_kg"].fillna(0, inplace=True)
    df["reach_cm"].fillna(0, inplace=True)

    df["birthdate"] = df["birthdate"].apply(
        lambda x: re.search(r"\w{3} \d{2}, \d{4}", x).group(0)
        if re.search(r"\w{3} \d{2}, \d{4}", x)
        else "N/A"
    )

    return df[
        [
            "fighter_id",
            "full_name",
            "height_cm",
            "weight_kg",
            "reach_cm",
            "stance",
            "wins",
            "losses",
            "draws",
            "birthdate",
        ]
    ]

def compute_last_five_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages for the previous five fights of each fighter.

    Parameters
    ----------
    df:
        DataFrame containing at least ``Fighter``, ``Result`` and numeric
        statistics (e.g. ``KD``, ``TD``).

    Returns
    -------
    pandas.DataFrame
        DataFrame where numeric columns represent the average of the previous
        five fights and includes a ``form_last_5`` column with the win ratio in
        the same window. The original ``Result`` column is dropped in the
        returned DataFrame.
    """
    df = df.copy()
    if "Fighter" not in df.columns or "Result" not in df.columns:
        raise ValueError("DataFrame must contain 'Fighter' and 'Result' columns")

    # Map results to numeric wins (1) and losses (0)
    df["_win"] = df["Result"].map({"W": 1, "L": 0}).fillna(df["Result"])

    numeric_cols = df.select_dtypes(include="number").columns.difference(["_win"])
    grouped = df.groupby("Fighter")

    # Proportion of wins in the previous five fights
    df["form_last_5"] = grouped["_win"].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())

    for col in numeric_cols:
        df[col] = grouped[col].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())

    return df.drop(columns=["Result", "_win"])
