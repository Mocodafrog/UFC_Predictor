
from __future__ import annotations
import re
import pandas as pd
from sklearn.impute import SimpleImputer


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
    """Clean and enrich the fighters DataFrame returned by :func:`ufc_predictor.scraping.scrape_fighters`.

    The function removes duplicated names, assigns a numeric ``fighter_id`` and
    converts physical metrics to metric units.

    Parameters
    ----------
    df:
        Raw DataFrame obtained from ``ufc_predictor.scraping.scrape_fighters``.

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

    df["height_missing"] = df["height_cm"].isna()
    df["weight_missing"] = df["weight_kg"].isna()
    df["reach_missing"] = df["reach_cm"].isna()

    imputer = SimpleImputer(strategy="median")
    df[["height_cm", "weight_kg", "reach_cm"]] = imputer.fit_transform(
        df[["height_cm", "weight_kg", "reach_cm"]]
    )

    df["birthdate"] = df["birthdate"].apply(
        lambda x: (
            re.search(r"\w{3} \d{2}, \d{4}", str(x)).group(0)
            if pd.notnull(x)
            and re.search(r"\w{3} \d{2}, \d{4}", str(x))
            else "N/A"
        )
    )

    return df[
        [
            "fighter_id",
            "full_name",
            "height_cm",
            "height_missing",
            "weight_kg",
            "weight_missing",
            "reach_cm",
            "reach_missing",
            "stance",
            "wins",
            "losses",
            "draws",
            "birthdate",
        ]
    ]
