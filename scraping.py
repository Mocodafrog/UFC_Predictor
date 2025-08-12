"""Web scraping utilities for UFC statistics."""
import time
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup


def fetch(
    session: requests.Session,
    url: str,
    retries: int = 3,
    backoff: int = 1,
    timeout: int = 10,
) -> Optional[requests.Response]:
    """Perform an HTTP GET request with retries.

    Args:
        session: Active :class:`requests.Session` used for the request.
        url: Target URL.
        retries: Number of retry attempts for failed requests.
        backoff: Base number of seconds to wait between retries.
        timeout: Timeout for each request in seconds.

    Returns:
        The :class:`requests.Response` object if successful, otherwise ``None``.
    """
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            # Retry on common transient errors
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"Request failed for {url} with status {response.status_code}")
                return None
        except requests.RequestException as exc:  # pragma: no cover - network side effects
            if attempt == retries - 1:
                print(f"Error fetching {url}: {exc}")
            else:
                time.sleep(backoff * (attempt + 1))
    return None


def fetch_soup(
    session: requests.Session,
    url: str,
    retries: int = 3,
    backoff: int = 1,
    timeout: int = 10,
) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup parsed document."""
    response = fetch(session, url, retries=retries, backoff=backoff, timeout=timeout)
    return BeautifulSoup(response.content, "html.parser") if response else None


def scrape_fighters(timeout: int = 10, delay: float = 1.0) -> pd.DataFrame:
    """Scrape fighter information from UFCStats.

    Parameters
    ----------
    timeout:
        Request timeout in seconds for each HTTP call.
    delay:
        Seconds to wait between requests to avoid hitting the server too hard.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing fighter statistics such as height, weight and record.
    """
    full_names, heights, weights, reaches = [], [], [], []
    stances, wins, losses, draws, birthdates = [], [], [], [], []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/119.0 Safari/537.36"
                )
            }
        )

        for char in "abcdefghijklmnopqrstuvwxyz":
            url = f"http://ufcstats.com/statistics/fighters?char={char}&page=all"
            soup = fetch_soup(session, url, timeout=timeout)
            if not soup:
                continue
            table = soup.select_one("table")
            if not table:
                continue
            for row in table.select("tr:nth-of-type(n+2)"):
                cols = row.find_all("td")
                link = cols[1].find("a")
                if not link:
                    continue
                profile_url = link["href"]
                profile_soup = fetch_soup(session, profile_url, timeout=timeout)
                if not profile_soup:
                    continue
                full_name_tag = profile_soup.select_one("span.b-content__title-highlight")
                birthdate_tag = profile_soup.select_one(
                    ".b-list__info-box_style_small-width li:nth-of-type(5)"
                )
                full_names.append(full_name_tag.text.strip() if full_name_tag else "N/A")
                birthdates.append(birthdate_tag.text.strip() if birthdate_tag else "N/A")
                heights.append(cols[3].text.strip())
                weights.append(cols[4].text.strip())
                reaches.append(cols[5].text.strip())
                stances.append(cols[6].text.strip())
                wins.append(cols[7].text.strip())
                losses.append(cols[8].text.strip())
                draws.append(cols[9].text.strip())
                time.sleep(delay)  # be polite with the server

    return pd.DataFrame(
        {
            "full_name": full_names,
            "birthdate": birthdates,
            "height": heights,
            "weight": weights,
            "reach": reaches,
            "stance": stances,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }
    )
