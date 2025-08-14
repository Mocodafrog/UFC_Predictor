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


def scrape_fight_stats(
    timeout: int = 10,
    delay: float = 1.0,
    max_events: Optional[int] = None,
    output_csv: str = "data/fight_stats_raw.csv",
) -> pd.DataFrame:
    """Scrape individual fight statistics from completed UFC events.

    Parameters
    ----------
    timeout:
        Timeout in seconds for each HTTP request.
    delay:
        Seconds to wait between requests to avoid server overloading.
    max_events:
        Optional maximum number of events to scrape. Useful for testing.
    output_csv:
        Path where the resulting CSV will be written.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing per-fighter statistics for each fight.
    """

    results: list[dict[str, str]] = []

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

        events_url = "http://ufcstats.com/statistics/events/completed"
        events_soup = fetch_soup(session, events_url, timeout=timeout)
        if not events_soup:
            return pd.DataFrame()

        event_links = [
            a["href"]
            for a in events_soup.select("tr.b-statistics__table-row a")
            if a.get("href")
        ]

        for idx, event_url in enumerate(event_links):
            if max_events is not None and idx >= max_events:
                break

            time.sleep(delay)
            event_soup = fetch_soup(session, event_url, timeout=timeout)
            if not event_soup:
                continue

            event_name_tag = event_soup.select_one("h2.b-content__title")
            event_name = event_name_tag.get_text(strip=True) if event_name_tag else ""

            fight_rows = event_soup.select("tr.b-fight-details__table-row")
            for row in fight_rows:
                fight_link_tag = row.find(
                    "a", href=lambda x: x and "fight-details" in x
                )
                if not fight_link_tag:
                    continue

                fight_url = fight_link_tag["href"]
                fight_title = fight_link_tag.get_text(strip=True)
                weight_class_tag = row.select_one(
                    "td.b-fight-details__table-col:nth-of-type(1)"
                )
                weight_class = (
                    weight_class_tag.get_text(strip=True) if weight_class_tag else ""
                )

                time.sleep(delay)
                fight_soup = fetch_soup(session, fight_url, timeout=timeout)
                if not fight_soup:
                    continue

                info: dict[str, str] = {}
                for li in fight_soup.select("ul.b-fight-details__list li"):
                    key_tag = li.select_one("i")
                    if not key_tag:
                        continue
                    key = key_tag.get_text(strip=True).rstrip(":")
                    value = li.get_text(strip=True).replace(key_tag.get_text(strip=True), "").strip()
                    info[key] = value

                stats_table = fight_soup.select_one("table.b-fight-details__table")
                if not stats_table:
                    continue

                headers = [
                    th.get_text(strip=True) for th in stats_table.select("thead th")
                ]

                for tr in stats_table.select("tbody tr"):
                    fighter_tag = tr.select_one(
                        "td.b-fight-details__table-col:first-child a"
                    )
                    if not fighter_tag:
                        continue
                    fighter_name = fighter_tag.get_text(strip=True)

                    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(cells) != len(headers):
                        continue
                    data = dict(zip(headers, cells))

                    results.append(
                        {
                            "Event": event_name,
                            "Fight": fight_title,
                            "Fighter": fighter_name,
                            "Weight Class": weight_class,
                            "Winner": "W"
                            if "b-fight-details__table-row__withdrawn" not in tr.get("class", [])
                            and "b-fight-details__table-row--loser" not in tr.get("class", [])
                            and "b-fight-details__table-row--draw" not in tr.get("class", [])
                            and "b-fight-details__table-row--no-contest" not in tr.get("class", [])
                            and "b-fight-details__table-row--ko" not in tr.get("class", [])
                            and "b-fight-details__table-row--winner" in tr.get("class", [])
                            else "L",
                            "KD": data.get("KD", ""),
                            "Sig. Str.": data.get("Sig. str.", ""),
                            "Total Str.": data.get("Total str.", ""),
                            "TD": data.get("TD", ""),
                            "Sub. Att": data.get("Sub. att", ""),
                            "Reversal": data.get("Rev.", ""),
                            "Control Time": data.get("Ctrl", ""),
                            "Head": data.get("Head", ""),
                            "Body": data.get("Body", ""),
                            "Leg": data.get("Leg", ""),
                            "Distance": data.get("Distance", ""),
                            "Clinch": data.get("Clinch", ""),
                            "Ground": data.get("Ground", ""),
                            "Method": info.get("Method", ""),
                            "Fight_lenght": info.get("Time", ""),
                            "Rounds": info.get("Round", ""),
                            "Format": info.get("Time format", ""),
                            "Referee": info.get("Referee", ""),
                        }
                    )

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
    return df
