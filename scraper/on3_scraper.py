"""
On3 Sports NIL valuation scraper for college basketball players.

Scrapes NIL valuations from On3's NIL rankings page. The school is matched
later by player name in the dataset builder.

Usage:
    python scraper/on3_scraper.py [--pages 5] [--output data/raw/on3_valuations.csv]
"""

import argparse
import csv
import os
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.on3.com/nil/rankings/player/college/basketball/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_OUTPUT = "data/raw/on3_valuations.csv"


def _parse_valuation(text: str) -> int | None:
    """Parse NIL valuation string like '$1.2M' or '$450K' to integer dollars."""
    if not text:
        return None
    text = text.strip().replace("$", "").replace(",", "").upper()
    try:
        if "M" in text:
            return int(float(text.replace("M", "")) * 1_000_000)
        elif "K" in text:
            return int(float(text.replace("K", "")) * 1_000)
        else:
            return int(float(text))
    except (ValueError, TypeError):
        return None


def scrape_page(url: str) -> list[dict]:
    """Scrape a single On3 NIL rankings page using requests + BeautifulSoup.

    On3 serves the data server-side rendered, so JavaScript is not required.
    """
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players = []

    # Each player is wrapped in a div with class containing "NilPlayerRankingItem_itemContainer"
    items = soup.select('[class*="NilPlayerRankingItem_itemContainer"]')

    for item in items:
        try:
            # Rank
            rank_el = item.select_one('[class*="NilPlayerRankingItem_playerRank"]')
            ranking = None
            if rank_el:
                rank_text = rank_el.get_text(strip=True)
                try:
                    ranking = int(re.sub(r"[^\d]", "", rank_text))
                except ValueError:
                    pass

            # Position
            pos_el = item.select_one('[class*="NilPlayerRankingItem_position"]')
            position = pos_el.get_text(strip=True) if pos_el else None

            # Name (in an anchor tag)
            name_el = item.select_one('[class*="NilPlayerRankingItem_name"] a')
            if not name_el:
                name_el = item.select_one('[class*="NilPlayerRankingItem_name"]')
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            if not name:
                continue

            # Player profile link (useful for fetching school later if needed)
            profile_link = None
            if name_el.name == "a":
                profile_link = name_el.get("href")

            # Class year, height, weight (combined as e.g. "JR/6-7/205")
            details_el = item.select_one('[class*="NilPlayerRankingItem_details"]')
            class_year, height, weight = None, None, None
            if details_el:
                detail_text = details_el.get_text(strip=True)
                parts = [p.strip() for p in detail_text.split("/")]
                if len(parts) >= 1:
                    class_year = parts[0]
                if len(parts) >= 2:
                    height = parts[1]
                if len(parts) >= 3:
                    weight = parts[2]

            # On3 rating (talent score)
            rating_el = item.select_one('[class*="StarRating_overallRating"]')
            on3_rating = None
            if rating_el:
                try:
                    on3_rating = float(rating_el.get_text(strip=True))
                except ValueError:
                    pass

            # NIL valuation
            val_el = item.select_one('[class*="NilPlayerRankingItem_valuationCurrency"]')
            if not val_el:
                val_el = item.select_one('[class*="valuationContainer"]')
            valuation = _parse_valuation(val_el.get_text(strip=True)) if val_el else None

            if valuation is None:
                continue

            players.append({
                "player_name": name,
                "position": position,
                "class_year": class_year,
                "height": height,
                "weight": weight,
                "on3_rating": on3_rating,
                "nil_valuation": valuation,
                "nil_ranking": ranking,
                "profile_url": profile_link,
                "sport": "basketball",
                "date_scraped": date.today().isoformat(),
            })
        except Exception:
            continue

    return players


def scrape(
    pages: int = 5,
    output: str = DEFAULT_OUTPUT,
) -> list[dict]:
    """Scrape On3 NIL rankings for basketball players.

    Args:
        pages: Number of pages to scrape (each page has ~100 players).
        output: Path to save the CSV.

    Returns:
        List of all scraped player dicts.
    """
    all_players = []
    seen_names = set()  # Dedupe across pages

    for page in range(1, pages + 1):
        url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
        print(f"Scraping page {page}: {url}")

        try:
            players = scrape_page(url)
            new_count = 0
            for p in players:
                key = p["player_name"].lower()
                if key not in seen_names:
                    seen_names.add(key)
                    all_players.append(p)
                    new_count += 1
            print(f"  Found {len(players)} players ({new_count} new)")

            if len(players) == 0:
                print("  No players found, stopping pagination")
                break
        except Exception as e:
            print(f"  Error on page {page}: {e}")

        time.sleep(3)

    if all_players:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fieldnames = [
            "player_name", "position", "class_year", "height", "weight",
            "on3_rating", "nil_valuation", "nil_ranking", "profile_url",
            "sport", "date_scraped",
        ]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_players)
        print(f"\nSaved {len(all_players)} players to {output}")
    else:
        print("\nNo players scraped.")

    return all_players


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape On3 NIL valuations for basketball")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    scrape(pages=args.pages, output=args.output)
