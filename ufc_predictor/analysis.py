"""Utility functions for fight statistics analysis."""
from __future__ import annotations

import pandas as pd


MIN_COLUMNS = {"fighter", "date"}


def aggregate_last_five_stats(df_fights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics for each fighter's last five fights.

    Unlike :func:`ufc_predictor.fight_stats.compute_last_five_stats`, which returns a row per
    bout with rolling averages, this utility collapses the last five bouts of
    each fighter into a single aggregated record.

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


def get_next_5_results(
    df_fights: pd.DataFrame, fighter_col: str = "fighter", result_col: str = "Winner"
) -> pd.Series:
    """Return the outcomes of the next five fights for each row.

    Parameters
    ----------
    df_fights:
        DataFrame containing at least ``fighter_col`` and ``result_col``. The
        DataFrame is assumed to be ordered chronologically for each fighter.
    fighter_col:
        Name of the column identifying the fighter.
    result_col:
        Name of the column with the bout result (e.g. ``"Winner"``).

    Returns
    -------
    pandas.Series
        Series aligned with ``df_fights`` where each entry is a string with the
        results of the fighter's next five bouts. Results are encoded using
        ``"W"`` for win, ``"L"`` for loss, ``"D"`` for draw and ``"N"`` for any
        unexpected value or missing fight.
    """

    required = {fighter_col, result_col}
    missing = required - set(df_fights.columns)
    if missing:
        raise ValueError(f"df_fights is missing required columns: {missing}")

    mapping = {"W": "W", "L": "L", "D": "D"}
    mapped = df_fights[result_col].map(lambda x: mapping.get(x, "N"))

    next_results = pd.Series(index=df_fights.index, dtype=object)
    temp = df_fights.assign(_mapped=mapped)
    for _, group in temp.groupby(fighter_col):
        results = group["_mapped"].tolist()
        indices = group.index.tolist()
        for i, idx in enumerate(indices):
            future = results[i + 1 : i + 6]
            padded = future + ["N"] * (5 - len(future))
            next_results.loc[idx] = "".join(padded)

    return next_results
