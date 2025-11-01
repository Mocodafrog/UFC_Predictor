"""Command-line entry point for the UFC data pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from ufc_predictor.fight_stats import compute_last_five_stats
from ufc_predictor.preprocessing import preprocess_fighters
from ufc_predictor.scraping import scrape_fight_stats, scrape_fighters


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
        "--fighters-output",
        type=str,
        default="fighters.csv",
        help="Path where the processed fighters CSV will be stored",
    )
    parser.add_argument(
        "--fight-stats-output",
        type=str,
        default="data/fight_stats_raw.csv",
        help="Path where the raw fight stats CSV will be stored",
    )
    args = parser.parse_args()

    fighters_raw = scrape_fighters(timeout=args.timeout, delay=args.delay)
    fighters_clean = preprocess_fighters(fighters_raw)
    fighters_output_path = Path(args.fighters_output)
    fighters_output_path.parent.mkdir(parents=True, exist_ok=True)
    fighters_clean.to_csv(fighters_output_path, index=False)
    print(f"Saved {len(fighters_clean)} fighters to {fighters_output_path}")

    fight_stats_output_path = Path(args.fight_stats_output)
    fight_stats = scrape_fight_stats(
        timeout=args.timeout,
        delay=args.delay,
        output_csv=str(fight_stats_output_path),
    )
    if fight_stats.empty:
        print("Fight stats scraping did not return any data.")
        return

    print(
        f"Saved {len(fight_stats)} fight statistic rows to {fight_stats_output_path}"
    )

    processed_stats = compute_last_five_stats(
        csv_path=fight_stats_output_path,
        output_dir=fight_stats_output_path.parent,
    )
    if processed_stats.empty:
        print("Processed fight stats dataset is empty after filtering; skipping export.")
    else:
        export_dir = fight_stats_output_path.parent
        print(
            "Exported processed fight stats to "
            f"{export_dir / 'fight_stats.csv'} and {export_dir / 'df_estadisticas_ultimos_5.csv'}"
        )


if __name__ == "__main__":
    main()
