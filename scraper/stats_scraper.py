"""
College basketball stats scraper using Sports Reference (sports-reference.com/cbb/).

Scrapes per-game stats for every player on every team in the target conferences:
ACC, SEC, Big Ten, Big 12, Pac-12, Independent.

Usage:
    python scraper/stats_scraper.py [--season 2025] [--output data/raw/player_stats.csv]

NOTE: Respects robots.txt with rate limiting between requests.
"""

import argparse
import csv
import os
import re
import time
from datetime import date

import requests
from bs4 import BeautifulSoup, Comment


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
DEFAULT_OUTPUT = "data/raw/player_stats.csv"
BASE_URL = "https://www.sports-reference.com/cbb"

# Maps conference display names to Sports Reference conference slugs
CONFERENCE_SLUGS = {
    "ACC": "acc",
    "SEC": "sec",
    "Big Ten": "big-ten",
    "Big 12": "big-12",
    "Pac-12": "pac-12",
}

# Independent schools tracked separately (no major D1 basketball independents currently)
# Note: Notre Dame is in the ACC for basketball, not Independent.
INDEPENDENT_SCHOOLS = []

# Maps Sports Reference data-stat names → our column names
STAT_FIELDS = {
    "games": "games_played",
    "games_started": "games_started",
    "mp_per_g": "mpg",
    "fg_per_g": "fg_per_game",
    "fga_per_g": "fga_per_game",
    "fg_pct": "fg_pct",
    "fg3_per_g": "three_pt_per_game",
    "fg3a_per_g": "three_pt_att_per_game",
    "fg3_pct": "three_pt_pct",
    "ft_per_g": "ft_per_game",
    "fta_per_g": "fta_per_game",
    "ft_pct": "ft_pct",
    "orb_per_g": "orb",
    "drb_per_g": "drb",
    "trb_per_g": "rpg",
    "ast_per_g": "apg",
    "stl_per_g": "spg",
    "blk_per_g": "bpg",
    "tov_per_g": "tov",
    "pf_per_g": "fouls",
    "pts_per_g": "ppg",
}

# Sports Reference's player-name data-stat changed to "name_display" in newer pages.
PLAYER_NAME_STATS = ["name_display", "player"]


def _safe_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _safe_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _get_with_retry(url: str, retries: int = 3, backoff: float = 5.0) -> requests.Response | None:
    """GET a URL with retry on connection failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"    Connection failed after {retries} attempts: {e}")
                return None
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                return None
            print(f"    HTTP error: {e}")
            return None
    return None


def get_conference_teams(conference_slug: str, season: int) -> list[dict]:
    """Get all teams in a conference for a given season.

    Uses the conference standings page at /cbb/conferences/{slug}/men/{year}.html
    """
    url = f"{BASE_URL}/conferences/{conference_slug}/men/{season}.html"
    resp = _get_with_retry(url)
    if resp is None:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.select_one("#standings")

    if not table:
        # Check inside HTML comments
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.select_one("#standings")
            if table:
                break

    if not table:
        return []

    teams = []
    seen = set()
    for row in table.select("tbody tr"):
        link = row.select_one("th[data-stat='school_name'] a") or row.select_one(
            "td[data-stat='school_name'] a"
        )
        if not link:
            continue

        href = link.get("href", "")
        # Extract slug from href like /cbb/schools/duke/men/2025.html
        match = re.search(r"/schools/([^/]+)/", href)
        if not match:
            continue

        slug = match.group(1)
        if slug in seen:
            continue
        seen.add(slug)

        teams.append({
            "name": link.get_text(strip=True),
            "slug": slug,
        })

    return teams


def scrape_team_roster_stats(school_slug: str, season: int) -> list[dict]:
    """Scrape per-game stats for all players on a team.

    Uses the players_per_game table from the team's season page.
    """
    # Sports Reference uses /cbb/schools/{slug}/men/{season}.html for current pages
    urls_to_try = [
        f"{BASE_URL}/schools/{school_slug}/men/{season}.html",
        f"{BASE_URL}/schools/{school_slug}/{season}.html",
    ]

    soup = None
    for url in urls_to_try:
        resp = _get_with_retry(url, retries=2)
        if resp is not None:
            soup = BeautifulSoup(resp.text, "html.parser")
            break

    if soup is None:
        return []

    # Search for the players_per_game table
    table = soup.select_one("#players_per_game")

    # Sports Reference often hides tables inside HTML comments
    if not table:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment_soup = BeautifulSoup(comment, "html.parser")
            table = comment_soup.select_one("#players_per_game")
            if table:
                break

    if not table:
        return []

    players = []
    for row in table.select("tbody tr"):
        # Skip header/separator rows
        row_classes = row.get("class") or []
        if "thead" in row_classes or "over_header" in row_classes:
            continue

        # Find the player name — try both old and new field names
        player_cell = None
        for stat_name in PLAYER_NAME_STATS:
            player_cell = row.select_one(
                f"th[data-stat='{stat_name}'] a"
            ) or row.select_one(
                f"td[data-stat='{stat_name}'] a"
            ) or row.select_one(
                f"th[data-stat='{stat_name}']"
            ) or row.select_one(
                f"td[data-stat='{stat_name}']"
            )
            if player_cell:
                break

        if not player_cell:
            continue

        player_name = player_cell.get_text(strip=True)
        if not player_name or player_name.lower() in ("team", "opponent", "school", ""):
            continue

        # Get position if available
        pos_cell = row.select_one("td[data-stat='pos']")
        position = pos_cell.get_text(strip=True) if pos_cell else None

        player_data = {
            "player_name": player_name,
            "position": position,
            "sport": "basketball",
        }

        # Extract each stat field using data-stat attributes
        for sr_stat, our_name in STAT_FIELDS.items():
            cell = row.select_one(f"td[data-stat='{sr_stat}']")
            if cell:
                text = cell.get_text(strip=True)
                if our_name in ("games_played", "games_started"):
                    player_data[our_name] = _safe_int(text)
                else:
                    player_data[our_name] = _safe_float(text)

        # Only include players with at least 1 game played
        if player_data.get("games_played") and player_data["games_played"] > 0:
            players.append(player_data)

    return players


def scrape_all_conferences(
    season: int = 2025,
    output: str = DEFAULT_OUTPUT,
    conferences: dict | None = None,
    independent_schools: list | None = None,
) -> list[dict]:
    """Scrape stats for all players in target conferences."""
    if conferences is None:
        conferences = CONFERENCE_SLUGS
    if independent_schools is None:
        independent_schools = INDEPENDENT_SCHOOLS

    all_players = []
    date_scraped = date.today().isoformat()

    for conf_name, conf_slug in conferences.items():
        print(f"\n{'='*60}")
        print(f"Scraping conference: {conf_name} ({conf_slug})")
        print(f"{'='*60}")

        teams = get_conference_teams(conf_slug, season)
        print(f"Found {len(teams)} teams in {conf_name}")

        if not teams:
            print(f"  (Conference may not exist for season {season})")
            continue

        time.sleep(2)

        for team in teams:
            print(f"  Scraping {team['name']} ({team['slug']})...")
            try:
                players = scrape_team_roster_stats(team["slug"], season)
                for p in players:
                    p["school"] = team["name"]
                    p["conference"] = conf_name
                    p["date_scraped"] = date_scraped
                all_players.extend(players)
                print(f"    Found {len(players)} players")
            except Exception as e:
                print(f"    Error: {e}")

            time.sleep(3)  # Rate limiting

    # Handle independent schools
    if independent_schools:
        print(f"\n{'='*60}")
        print("Scraping Independent schools")
        print(f"{'='*60}")

        for school_slug in independent_schools:
            print(f"  Scraping {school_slug}...")
            try:
                players = scrape_team_roster_stats(school_slug, season)
                school_name = school_slug.replace("-", " ").title()
                for p in players:
                    p["school"] = school_name
                    p["conference"] = "Independent"
                    p["date_scraped"] = date_scraped
                all_players.extend(players)
                print(f"    Found {len(players)} players")
            except Exception as e:
                print(f"    Error: {e}")

            time.sleep(3)

    # Save to CSV
    if all_players:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        fieldnames = [
            "player_name", "position", "sport", "school", "conference",
            "games_played", "games_started", "mpg",
            "ppg", "apg", "rpg", "spg", "bpg",
            "fg_pct", "three_pt_pct", "ft_pct",
            "fg_per_game", "fga_per_game",
            "three_pt_per_game", "three_pt_att_per_game",
            "ft_per_game", "fta_per_game",
            "orb", "drb", "tov", "fouls",
            "date_scraped",
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
    parser = argparse.ArgumentParser(description="Scrape college basketball player stats")
    parser.add_argument(
        "--season", type=int, default=2025,
        help="Season year (e.g. 2025 for 2024-25, 2026 for 2025-26)",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    scrape_all_conferences(season=args.season, output=args.output)
