from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# Regex-based mapping to collapse a variety of historical and tournament
# weight-class labels into a canonical set. Patterns are evaluated in order so
# that specific classes (e.g. ``Women's Flyweight``) take precedence over more
# generic ones like ``Flyweight``.
WEIGHT_CLASS_MAP = {
    r".*Women's Bantamweight.*": "Women's Bantamweight",
    r".*Women's Featherweight.*": "Women's Featherweight",
    r".*Women's Flyweight.*": "Women's Flyweight",
    r".*Women's Strawweight.*": "Women's Strawweight",
    r".*(?<!Women's )Bantamweight.*": "Bantamweight",
    r".*(?<!Women's )Featherweight.*": "Featherweight",
    r".*(?<!Women's )Flyweight.*": "Flyweight",
    r".*Light Heavyweight.*": "Light Heavyweight",
    r".*Lightweight.*": "Lightweight",
    r".*Middleweight.*": "Middleweight",
    r".*Welterweight.*": "Welterweight",
    r".*Super Heavyweight.*": "Super Heavyweight",
    r".*Heavyweight.*": "Heavyweight",
    r".*Catch Weight.*": "Catchweight",
    r".*Open Weight.*": "Openweight",
    r".*Tournament.*": "Openweight",
    r".*Superfight.*": "Openweight",
    r".*Grand Prix.*": "Openweight",
}


def _extract_numeric_pair(text: str) -> tuple[int, int]:
    """Return the landed and attempted values from strings like ``"20 of 53"``.

    If the value cannot be parsed, ``(0, 0)`` is returned.
    """
    match = re.match(r"\s*(\d+)\s*of\s*(\d+)", str(text))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def _time_to_seconds(text: str) -> int:
    """Convert a ``MM:SS`` time string into total seconds."""
    match = re.search(r"(\d+):(\d+)", str(text))
    if match:
        minutes, seconds = match.groups()
        return int(minutes) * 60 + int(seconds)
    return 0


def compute_last_five_stats(
    csv_path: str | Path = DATA_DIR / "fight_stats_raw.csv",
    *,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a rolling dataset for the last five fights of each fighter.

    This routine keeps one row per bout and appends rolling averages for
    numerous statistics as well as a ``form_last_5`` column with the number of
    recent wins. After the metrics are created the resulting DataFrame is
    written to ``fight_stats.csv`` inside ``output_dir`` (or ``data/`` by
    default) and a reduced dataset containing the rolling means is exported to
    ``df_estadisticas_ultimos_5.csv`` in the same directory.
    For a per-fighter aggregated summary, see
    :func:`ufc_predictor.analysis.aggregate_last_five_stats`.

    If ``csv_path`` does not exist, an empty :class:`~pandas.DataFrame` is
    returned and a message is printed.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}")
        return pd.DataFrame()

    df = df[~df["Winner"].isin(["NC", "D"]) & (df["Method"] != "DQ")]
    if "Format" in df.columns:
        format_numeric = pd.to_numeric(
            df["Format"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        df = df[format_numeric.isin([3, 5])].copy()
    df["Weight Class"] = df["Weight Class"].replace(WEIGHT_CLASS_MAP, regex=True)
    df["Method"] = df["Method"].replace(
        {
            "Decision - Majority": "Decision",
            "Decision - Split": "Decision",
            "Decision - Unanimous": "Decision",
            "TKO - Doctor's Stoppage": "KO/TKO",
        }
    )

    # Sequential date placeholder to preserve chronological order per fighter
    df["date"] = df.groupby("Fighter").cumcount()

    # Compute rolling win form
    df["win"] = (df["Winner"] == "W").astype(int)
    df["form_last_5"] = (
        df.groupby("Fighter")["win"]
        .rolling(window=5, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # Parse time columns to seconds
    df["Control Time Sec"] = df["Control Time"].apply(_time_to_seconds)
    df["Total_fight_length_sec"] = df["Fight_lenght"].apply(_time_to_seconds)

    # Parse "landed of attempted" statistics
    pair_cols = [
        "Sig. Str.",
        "Total Str.",
        "TD",
        "Head",
        "Body",
        "Leg",
        "Distance",
        "Clinch",
        "Ground",
    ]
    for col in pair_cols:
        landed, atmp = zip(*df[col].map(_extract_numeric_pair))
        base = col.lower().replace(" ", "_").replace(".", "")
        df[f"landed_{base}"] = landed
        df[f"atmp_{base}"] = atmp

    # Compute landed strikes per minute for selected categories
    fight_length_min = (
        df["Total_fight_length_sec"].replace(0, pd.NA).div(60).fillna(1)
    )
    # The division above assumes a minimum fight duration of one minute.
    # Using ``pd.NA`` first prevents division by zero while ``fillna(1)``
    # provides a sensible default for extremely short or missing fights.
    lpm_bases = [
        "sig_str",
        "total_str",
        "head",
        "body",
        "leg",
        "distance",
        "clinch",
        "ground",
    ]
    lpm_cols: list[str] = []
    for base in lpm_bases:
        col_name = f"{base}_lpm"
        df[col_name] = df[f"landed_{base}"] / fight_length_min
        lpm_cols.append(col_name)

    # Compute accuracy for landed versus attempted metrics
    for base in [
        "sig_str",
        "total_str",
        "td",
        "head",
        "body",
        "leg",
        "distance",
        "clinch",
        "ground",
    ]:
        attempts = df[f"atmp_{base}"].replace(0, np.nan)
        df[f"{base}_acc"] = (df[f"landed_{base}"] / attempts).fillna(0)

    # Columns for which we will compute rolling means
    rolling_cols = [
        "KD",
        "Sub. Att",
        "Reversal",
        "Control Time Sec",
        "Total_fight_length_sec",
    ]
    for col in pair_cols:
        base = col.lower().replace(" ", "_").replace(".", "")
        rolling_cols.extend([f"landed_{base}", f"atmp_{base}"])
    rolling_cols.extend(lpm_cols)


    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[f"{col}_rolling_mean"] = (
            df.groupby("Fighter")[col]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Persist the full dataset and a reduced version with only rolling means
    export_dir = Path(output_dir) if output_dir is not None else DATA_DIR

    df.to_csv(export_dir / "fight_stats.csv", index=False)

    reduced_cols = [
        "Fighter",
        "Format",
        "Weight Class",
        "date",
        "form_last_5",
    ] + [c for c in df.columns if c.endswith("_rolling_mean")]
    reduced = df[reduced_cols]
    reduced.to_csv(export_dir / "df_estadisticas_ultimos_5.csv", index=False)

    return df


if __name__ == "__main__":
    fight_stats = compute_last_five_stats()
    if fight_stats.empty:
        print("No stats generated; skipping export.")
        raise SystemExit(1)
