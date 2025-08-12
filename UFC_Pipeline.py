"""Command-line entry point for the UFC data pipeline."""
import argparse
from scraping import scrape_fighters
from preprocessing import preprocess_fighters

def main() -> None:
    """Run the scraping and preprocessing pipeline.

    This function parses command-line arguments, scrapes the fighter data and
    stores the cleaned dataset on disk.
    """
    parser = argparse.ArgumentParser(description="UFC data scraping pipeline")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP request timeout in seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument(
        "--output",
        type=str,
        default="fighters.csv",
        help="Path where the processed CSV will be stored",
    )
    args = parser.parse_args()

    fighters_raw = scrape_fighters(timeout=args.timeout, delay=args.delay)
    fighters_clean = preprocess_fighters(fighters_raw)
    fighters_clean.to_csv(args.output, index=False)
    print(f"Saved {len(fighters_clean)} fighters to {args.output}")

if __name__ == "__main__":
    main()
