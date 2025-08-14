"""Data cleaning and preprocessing utilities for UFC data."""
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


def add_rolling_features(df_fights: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling five-fight averages and form for each fighter.

    Parameters
    ----------
    df_fights:
        DataFrame containing per-fight statistics. It must include the
        columns ``fighter_name`` and ``result`` in addition to numeric
        columns such as ``KD``, ``Sig Str``, ``TD`` and ``Control``.

    Returns
    -------
    pandas.DataFrame
        ``df_fights`` enriched with the rolling means for the selected
        numeric columns and a ``form_last_5`` column indicating the
        average win ratio over the last five fights for each fighter.
    """

    cols = [c for c in ["KD", "Sig Str", "TD", "Control"] if c in df_fights.columns]
    rolling = (
        df_fights.groupby("fighter_name")[cols]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_fights[cols] = rolling

    wins = (df_fights["result"] == "W").astype(int)
    df_fights["form_last_5"] = (
        wins.groupby(df_fights["fighter_name"])  # type: ignore[arg-type]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df_fights
