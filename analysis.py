"""Utility functions for fight statistics analysis."""
from __future__ import annotations

import pandas as pd


MIN_COLUMNS = {"fighter", "date"}


def compute_last_five_stats(df_fights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics for each fighter's last five fights.

    Parameters
    ----------
    df_fights:
        DataFrame containing fight level statistics. It must include at
        least the columns in :data:`MIN_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per fighter containing the mean of the
        numeric statistics for the fighter's last five bouts. The ``date``
        column corresponds to the most recent fight considered.

    Raises
    ------
    ValueError
        If any of the required columns are missing from ``df_fights``.
    """

    missing = MIN_COLUMNS - set(df_fights.columns)
    if missing:
        raise ValueError(f"df_fights is missing required columns: {missing}")

    df = df_fights.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    last_five = df.groupby("fighter").tail(5)

    def _aggregate(group: pd.DataFrame) -> pd.Series:
        numeric = group.select_dtypes(include="number").mean()
        numeric["fighter"] = group["fighter"].iloc[0]
        numeric["date"] = group["date"].max()
        return numeric

    result = last_five.groupby("fighter", as_index=False).apply(_aggregate)
    result.reset_index(drop=True, inplace=True)
    return result
