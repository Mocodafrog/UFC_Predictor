from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")


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


def compute_last_five_stats(csv_path: str | Path = DATA_DIR / "fight_stats_raw.csv") -> pd.DataFrame:
    """Compute rolling statistics for the last five fights of each fighter.

    The function expects the raw fight statistics CSV produced by the scraping
    utilities.  It parses numeric values, computes per-fighter rolling means for
    several statistics and a ``form_last_5`` column indicating the number of
    wins in the last five fights.
    """
    df = pd.read_csv(csv_path)

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

    for col in rolling_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[f"{col}_rolling_mean"] = (
            df.groupby("Fighter")[col]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Rename fighter column for clarity
    df.rename(columns={"Fighter": "fighter_name"}, inplace=True)

    return df


if __name__ == "__main__":
    fight_stats = compute_last_five_stats()
    DATA_DIR.mkdir(exist_ok=True)
    fight_stats.to_csv(DATA_DIR / "fight_stats.csv", index=False)

    reduced_cols = [
        "fighter_name",
        "date",
        "form_last_5",
    ] + [c for c in fight_stats.columns if c.endswith("_rolling_mean")]
    reduced = fight_stats[reduced_cols]
    reduced.to_csv(DATA_DIR / "df_estadisticas_ultimos_5.csv", index=False)
