"""
Player stats scraper for sports-reference.com / ESPN.

Usage:
    python scraper/stats_scraper.py [--sport basketball|football] [--output PATH]

NOTE: Stub implementation. Actual scraping must respect robots.txt and ToS.
"""

import argparse
import csv
import time

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
DEFAULT_OUTPUT = "data/raw/player_stats.csv"


def scrape_basketball_stats(url: str) -> list[dict]:
    """Scrape basketball player stats."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players = []
    table = soup.select_one("#per_game_stats") or soup.select_one("table")
    if not table:
        return players

    for row in table.select("tbody tr:not(.thead)"):
        cols = row.select("td")
        if len(cols) < 10:
            continue
        try:
            players.append({
                "player_name": row.select_one("td[data-stat='player']").get_text(strip=True),
                "sport": "basketball",
                "games_played": int(cols[2].get_text(strip=True) or 0),
                "ppg": float(cols[-1].get_text(strip=True) or 0),
                "apg": float(cols[-3].get_text(strip=True) or 0),
                "rpg": float(cols[-4].get_text(strip=True) or 0),
            })
        except (ValueError, AttributeError):
            continue

    return players


def scrape_football_stats(url: str) -> list[dict]:
    """Scrape football player stats."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players = []
    table = soup.select_one("#passing") or soup.select_one("table")
    if not table:
        return players

    for row in table.select("tbody tr:not(.thead)"):
        cols = row.select("td")
        if len(cols) < 5:
            continue
        try:
            players.append({
                "player_name": row.select_one("td[data-stat='player']").get_text(strip=True),
                "sport": "football",
                "games_played": int(cols[1].get_text(strip=True) or 0),
                "yards": float(cols[3].get_text(strip=True).replace(",", "") or 0),
                "tds": int(cols[4].get_text(strip=True) or 0),
            })
        except (ValueError, AttributeError):
            continue

    return players


def scrape(sport: str = "basketball", output: str = DEFAULT_OUTPUT):
    """Main scraping entrypoint. Stub: prints instructions."""
    print(f"Stats scraper for {sport}")
    print("NOTE: This is a stub. To use with real data:")
    print("  1. Provide URLs for the target stats pages")
    print("  2. Update CSS selectors to match current page markup")
    print("  3. Respect rate limits and robots.txt")
    print(f"  Output would be saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape player stats")
    parser.add_argument("--sport", choices=["basketball", "football"], default="basketball")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    scrape(sport=args.sport, output=args.output)
