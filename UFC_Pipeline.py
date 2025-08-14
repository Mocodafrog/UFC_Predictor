"""Command-line entry point for the UFC data pipeline."""
import argparse
import os

from scraping import scrape_fighters, scrape_fight_stats
from preprocessing import preprocess_fighters, compute_last_five_stats

def main() -> None:
    """Run the scraping and preprocessing pipeline.

    This function parses command-line arguments, scrapes the fighter data and
    stores the cleaned dataset on disk.
    """
    parser = argparse.ArgumentParser(description="UFC data scraping pipeline")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument(
        "--fighters-output",
        type=str,
        default=os.path.join("data", "fighters.csv"),
        help="Path where the processed fighters CSV will be stored",
    )
    parser.add_argument(
        "--fight-stats-output",
        type=str,
        default=os.path.join("data", "fight_stats.csv"),
        help="Path where the processed fight stats CSV will be stored",
    )
    parser.add_argument(
        "--last-five-output",
        type=str,
        default=os.path.join("data", "df_estadisticas_ultimos_5.csv"),
        help="Path where the last-five stats CSV will be stored",
    )
    parser.add_argument("--start-event", type=int, default=None, help="First event to scrape")
    parser.add_argument("--end-event", type=int, default=None, help="Last event to scrape")
    parser.add_argument(
        "--max-fights",
        type=int,
        default=None,
        help="Maximum number of fights to process",
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    fighters_raw = scrape_fighters(timeout=args.timeout, delay=args.delay)
    fighters_clean = preprocess_fighters(fighters_raw)
    fighters_clean.to_csv(args.fighters_output, index=False)

    fight_stats_raw = scrape_fight_stats(
        start_event=args.start_event,
        end_event=args.end_event,
        max_fights=args.max_fights,
        timeout=args.timeout,
        delay=args.delay,
    )
    fight_stats, last_five = compute_last_five_stats(fight_stats_raw)
    fight_stats.to_csv(args.fight_stats_output, index=False)
    last_five.to_csv(args.last_five_output, index=False)

    print(f"Saved {len(fighters_clean)} fighters to {args.fighters_output}")
    print(f"Saved {len(fight_stats)} fight stats to {args.fight_stats_output}")
    print(
        "Saved last-five stats for "
        f"{last_five['Fighter'].nunique() if not last_five.empty else 0} fighters "
        f"to {args.last_five_output}"
    )

if __name__ == "__main__":
    main()
