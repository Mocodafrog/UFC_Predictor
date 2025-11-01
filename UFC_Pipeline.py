"""Command-line entry point for the UFC data pipeline."""
import argparse
from ufc_predictor.scraping import scrape_fight_stats, scrape_fighters
from ufc_predictor.preprocessing import preprocess_fighters

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
    fighters_clean.to_csv(args.fighters_output, index=False)
    print(f"Saved {len(fighters_clean)} fighters to {args.fighters_output}")

    fight_stats = scrape_fight_stats(
        timeout=args.timeout,
        delay=args.delay,
        output_csv=args.fight_stats_output,
    )
    if fight_stats.empty:
        print("Fight stats scraping did not return any data.")
    else:
        print(
            f"Saved {len(fight_stats)} fight statistic rows to {args.fight_stats_output}"
        )

if __name__ == "__main__":
    main()
