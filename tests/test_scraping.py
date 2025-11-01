from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

import ufc_predictor.scraping as scraping


def test_scrape_fight_stats_raises_when_events_missing(monkeypatch):
    monkeypatch.setattr(scraping, "fetch_soup", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError) as exc:
        scraping.scrape_fight_stats(max_events=1, delay=0)

    assert "events listing" in str(exc.value)


def test_scrape_fight_stats_parses_unique_event_links(tmp_path: Path, monkeypatch):
    events_html = """
    <table>
      <tr class="b-statistics__table-row">
        <td><a href="http://example.com/event-details/test">Event</a></td>
        <td><a href="http://example.com/event-details/test">Results</a></td>
      </tr>
    </table>
    """

    event_detail_html = """
    <html>
      <h2 class="b-content__title">Test Event</h2>
      <table>
        <tr class="b-fight-details__table-row">
          <td class="b-fight-details__table-col">Bantamweight Bout</td>
          <td class="b-fight-details__table-col">
            <a href="http://example.com/fight-details/test">Fighter One vs Fighter Two</a>
          </td>
        </tr>
      </table>
    </html>
    """

    fight_detail_html = """
    <html>
      <ul class="b-fight-details__list">
        <li><i>Method:</i>Decision - Unanimous</li>
        <li><i>Time:</i>5:00</li>
        <li><i>Time format:</i>3 Rnd</li>
        <li><i>Round:</i>3</li>
        <li><i>Referee:</i>Ref</li>
      </ul>
      <table class="b-fight-details__table">
        <thead>
          <tr>
            <th>Fighter</th>
            <th>KD</th>
            <th>Sig. str.</th>
            <th>Total str.</th>
            <th>TD</th>
            <th>Sub. att</th>
            <th>Rev.</th>
            <th>Ctrl</th>
            <th>Head</th>
            <th>Body</th>
            <th>Leg</th>
            <th>Distance</th>
            <th>Clinch</th>
            <th>Ground</th>
          </tr>
        </thead>
        <tbody>
          <tr class="b-fight-details__table-row b-fight-details__table-row--winner">
            <td class="b-fight-details__table-col"><a>Fighter One</a></td>
            <td>0</td>
            <td>10 of 20</td>
            <td>20 of 40</td>
            <td>1 of 2</td>
            <td>0</td>
            <td>0</td>
            <td>1:00</td>
            <td>5 of 10</td>
            <td>3 of 5</td>
            <td>2 of 5</td>
            <td>7 of 15</td>
            <td>2 of 3</td>
            <td>1 of 2</td>
          </tr>
          <tr class="b-fight-details__table-row b-fight-details__table-row--loser">
            <td class="b-fight-details__table-col"><a>Fighter Two</a></td>
            <td>0</td>
            <td>5 of 15</td>
            <td>10 of 30</td>
            <td>0 of 1</td>
            <td>0</td>
            <td>0</td>
            <td>0:30</td>
            <td>2 of 6</td>
            <td>2 of 4</td>
            <td>1 of 3</td>
            <td>4 of 12</td>
            <td>1 of 2</td>
            <td>0 of 1</td>
          </tr>
        </tbody>
      </table>
    </html>
    """

    calls: list[str] = []

    def fake_fetch_soup(session, url, **kwargs):
        calls.append(url)
        if "events/completed" in url:
            return BeautifulSoup(events_html, "html.parser")
        if "event-details" in url:
            return BeautifulSoup(event_detail_html, "html.parser")
        if "fight-details" in url:
            return BeautifulSoup(fight_detail_html, "html.parser")
        return None

    monkeypatch.setattr(scraping, "fetch_soup", fake_fetch_soup)

    output_path = tmp_path / "fight_stats_raw.csv"
    df = scraping.scrape_fight_stats(max_events=1, delay=0, output_csv=str(output_path))

    assert len(df) == 2
    assert output_path.is_file()

    event_detail_calls = [url for url in calls if "event-details" in url]
    assert event_detail_calls == ["http://example.com/event-details/test"]
