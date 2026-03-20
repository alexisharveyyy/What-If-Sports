"""
On3 Sports NIL valuation scraper.

Usage:
    python scraper/on3_scraper.py [--headless] [--pages N] [--output PATH]

NOTE: This is a stub for educational purposes. Actual scraping should
respect robots.txt and the site's terms of service.
"""

import argparse
import csv
import time
from datetime import date

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.on3.com/nil/rankings/player/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
DEFAULT_OUTPUT = "data/raw/on3_valuations.csv"


def scrape_page_bs4(url: str) -> list[dict]:
    """Scrape a single page using requests + BeautifulSoup."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    players = []
    # NOTE: Actual CSS selectors depend on On3's current markup.
    # These are illustrative and would need updating.
    for row in soup.select(".rankingsPage_listItem"):
        try:
            name = row.select_one(".rankingsPage_playerName").get_text(strip=True)
            school = row.select_one(".rankingsPage_school").get_text(strip=True)
            sport = row.select_one(".rankingsPage_sport").get_text(strip=True)
            valuation_text = row.select_one(".rankingsPage_nil").get_text(strip=True)
            valuation = int(valuation_text.replace("$", "").replace(",", ""))
            ranking_text = row.select_one(".rankingsPage_rank").get_text(strip=True)
            ranking = int(ranking_text)

            players.append({
                "player_name": name,
                "sport": sport,
                "school": school,
                "nil_valuation": valuation,
                "nil_ranking": ranking,
                "date_scraped": date.today().isoformat(),
            })
        except (AttributeError, ValueError):
            continue

    return players


def scrape_page_selenium(url: str) -> list[dict]:
    """Scrape a JS-rendered page using Selenium."""
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".rankingsPage_listItem"))
        )
        soup = BeautifulSoup(driver.page_source, "html.parser")
    finally:
        driver.quit()

    # Reuse the same parsing logic
    return scrape_page_bs4.__wrapped__(soup) if hasattr(scrape_page_bs4, "__wrapped__") else []


def scrape(pages: int = 1, headless: bool = False, output: str = DEFAULT_OUTPUT):
    """Main scraping entrypoint."""
    all_players = []
    scrape_fn = scrape_page_selenium if headless else scrape_page_bs4

    for page in range(1, pages + 1):
        url = f"{BASE_URL}?page={page}"
        print(f"Scraping page {page}: {url}")

        try:
            players = scrape_fn(url)
            all_players.extend(players)
            print(f"  Found {len(players)} players")
        except Exception as e:
            print(f"  Error on page {page}: {e}")

        # Rate limiting
        time.sleep(1.5)

    if all_players:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_players[0].keys())
            writer.writeheader()
            writer.writerows(all_players)
        print(f"Saved {len(all_players)} players to {output}")
    else:
        print("No players scraped.")

    return all_players


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape On3 NIL valuations")
    parser.add_argument("--headless", action="store_true", help="Use Selenium for JS-rendered pages")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to scrape")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    scrape(pages=args.pages, headless=args.headless, output=args.output)
