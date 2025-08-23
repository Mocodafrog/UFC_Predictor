
from __future__ import annotations

import re
import pandas as pd
from sklearn.impute import SimpleImputer


def convert_height_to_cm(height: str) -> float | None:
    """Convert height strings like ``6'2"`` or ``6ft2in`` to centimeters."""

    if not height or height == "--":
        return None

    match = re.match(
        r"^\s*(?P<feet>\d+)\s*(?:ft|')\s*(?P<inches>\d*)\s*(?:in|\"|inches)?\s*$",
        height,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    feet = int(match.group("feet"))
    inches = int(match.group("inches")) if match.group("inches") else 0
    return feet * 30.48 + inches * 2.54


def convert_weight_to_kg(weight: str) -> float | None:
    """Convert weight strings like ``170lb`` or ``170 lbs.`` to kilograms."""

    if not weight or weight == "--":
        return None

    match = re.match(
        r"^\s*(?P<weight>\d+(?:\.\d+)?)\s*(?:(?:lbs?|pounds?)\.?)?\s*$",
        weight,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group("weight")) * 0.453592


def convert_reach_to_cm(reach: str) -> float | None:
    """Convert reach measurements like ``70in`` or ``70\"`` to centimeters."""

    if not reach or reach == "--":
        return None

    match = re.match(
        r"^\s*(?P<reach>\d+(?:\.\d+)?)\s*(?:in|\"|inches)?\.?\s*$",
        reach,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group("reach")) * 2.54


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
