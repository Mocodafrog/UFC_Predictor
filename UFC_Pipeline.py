"""Command-line entry point for the UFC data pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from ufc_predictor.fight_stats import compute_last_five_stats
from ufc_predictor.preprocessing import preprocess_fighters
from ufc_predictor.scraping import scrape_fight_stats, scrape_fighters


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    """Run the scraping and preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="UFC data scraping pipeline")
    parser.add_argument(
        "--timeout", type=int, default=10, help="HTTP request timeout in seconds"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests in seconds"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "data"),
        help=(
            "Directory where pipeline artifacts will be written. The files "
            "fight_stats_raw.csv, fight_stats.csv, df_estadisticas_ultimos_5.csv, "
            "fighters_raw.csv and fighters.csv will be saved in this directory. "
            "If a columnas_X.csv file already exists, it should also reside here."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fighters_raw = scrape_fighters(timeout=args.timeout, delay=args.delay)

    fighters_raw_output_path = output_dir / "fighters_raw.csv"
    fighters_raw.to_csv(fighters_raw_output_path, index=False)
    print(f"Saved {len(fighters_raw)} raw fighters to {fighters_raw_output_path}")

    fighters_clean = preprocess_fighters(fighters_raw)
    fighters_output_path = output_dir / "fighters.csv"
    fighters_clean.to_csv(fighters_output_path, index=False)
    print(f"Saved {len(fighters_clean)} fighters to {fighters_output_path}")

    fight_stats_raw_path = output_dir / "fight_stats_raw.csv"
    fight_stats = scrape_fight_stats(
        timeout=args.timeout,
        delay=args.delay,
        output_csv=str(fight_stats_raw_path),
    )
    if fight_stats.empty:
        print("Fight stats scraping did not return any data.")
        return

    print(
        f"Saved {len(fight_stats)} fight statistic rows to {fight_stats_raw_path}"
    )

    processed_stats = compute_last_five_stats(
        csv_path=fight_stats_raw_path,
        output_dir=output_dir,
    )
    if processed_stats.empty:
        print("Processed fight stats dataset is empty after filtering; skipping export.")
    else:
        print(
            "Exported processed fight stats to "
            f"{output_dir / 'fight_stats.csv'} and {output_dir / 'df_estadisticas_ultimos_5.csv'}"
        )


if __name__ == "__main__":
    main()
