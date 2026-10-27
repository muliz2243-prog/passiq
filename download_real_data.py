# ============================================================
# PassIQ — download_real_data.py
# Downloads one real match events file from StatsBomb Open Data
# (https://github.com/statsbomb/open-data) using only the
# Python standard library — no extra dependencies required.
#
# Style: LOTA-F — explicit, single-direction data flow.
# ============================================================

# 1. Imports
import json
import os
import sys
import urllib.request
from dataclasses import dataclass


# 2. Constants
BASE_URL:       str = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL: str = f"{BASE_URL}/competitions.json"
OUTPUT_DIR:     str = "data/real"

# Target: La Liga — StatsBomb has free open data for multiple seasons.
# competition_id=11 is La Liga; season_id=90 is 2020/21.
# Change these to explore other competitions.
TARGET_COMPETITION_ID: int = 11
TARGET_SEASON_ID:      int = 90


# 3. Data structures
@dataclass
class MatchInfo:
    match_id:          int
    competition_name:  str
    season_name:       str
    home_team:         str
    away_team:         str
    match_date:        str
    home_score:        int
    away_score:        int


# 4. Pure functions, in the order the driver calls them
# ----------------------------------------------------------

def download_json(url: str) -> list | dict:
    """
    Downloads and parses a JSON resource from the given URL.
    Terminates the process with a descriptive message on network or parse errors.
    """
    print(f"  Downloading: {url}")
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            raw_bytes: bytes = response.read()
    except Exception as err:
        print(f"ERROR: could not download '{url}'. Detail: {err}")
        sys.exit(1)

    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as err:
        print(f"ERROR: invalid JSON from '{url}'. Detail: {err}")
        sys.exit(1)


def find_season(
    competitions: list[dict],
    competition_id: int,
    season_id: int,
) -> dict:
    """
    Returns the competition entry matching the given IDs.
    Terminates if not found — the IDs are constants set at the top of this file.
    """
    for entry in competitions:
        if entry.get("competition_id") == competition_id and \
           entry.get("season_id") == season_id:
            return entry

    print(
        f"ERROR: competition_id={competition_id}, season_id={season_id} "
        f"not found in StatsBomb open data."
    )
    print("Available options with free data:")
    for entry in competitions:
        print(
            f"  competition_id={entry['competition_id']:>4}  "
            f"season_id={entry['season_id']:>4}  "
            f"{entry['competition_name']} — {entry['season_name']}"
        )
    sys.exit(1)


def build_matches_url(competition_id: int, season_id: int) -> str:
    """Leaf: constructs the URL for the matches index of one season."""
    return f"{BASE_URL}/matches/{competition_id}/{season_id}.json"


def parse_match_info(raw: dict, competition_name: str, season_name: str) -> MatchInfo:
    """
    Extracts the fields we care about from a raw match dict.
    Defaults to empty strings / zeros for absent optional fields.
    """
    return MatchInfo(
        match_id         = int(raw["match_id"]),
        competition_name = competition_name,
        season_name      = season_name,
        home_team        = raw.get("home_team", {}).get("home_team_name", "Unknown"),
        away_team        = raw.get("away_team", {}).get("away_team_name", "Unknown"),
        match_date       = raw.get("match_date", ""),
        home_score       = int(raw.get("home_score", 0)),
        away_score       = int(raw.get("away_score", 0)),
    )


def select_match(matches_raw: list[dict], competition_name: str, season_name: str) -> MatchInfo:
    """
    Parses all matches, prints the index, and returns the first one.
    For the hackathon prototype we always pick match index 0.
    Change the index here to target a specific match.
    """
    match_index: int = 0

    all_matches: list[MatchInfo] = [
        parse_match_info(raw, competition_name, season_name)
        for raw in matches_raw
    ]

    print(f"\nFound {len(all_matches)} matches. Selecting index {match_index}:\n")
    for i, m in enumerate(all_matches[:10]):   # show first 10
        marker: str = ">>>" if i == match_index else "   "
        print(
            f"  {marker} [{i:>2}] {m.match_date}  "
            f"{m.home_team} {m.home_score}–{m.away_score} {m.away_team}  "
            f"(id={m.match_id})"
        )
    if len(all_matches) > 10:
        print(f"  ... and {len(all_matches) - 10} more.")

    return all_matches[match_index]


def build_events_url(match_id: int) -> str:
    """Leaf: constructs the URL for the events file of one match."""
    return f"{BASE_URL}/events/{match_id}.json"


def build_output_path(match_info: MatchInfo, output_dir: str) -> str:
    """Leaf: constructs the local file path for the downloaded events."""
    safe_home: str = match_info.home_team.replace(" ", "_").replace("/", "-")
    safe_away: str = match_info.away_team.replace(" ", "_").replace("/", "-")
    filename:  str = f"{match_info.match_id}_{safe_home}_vs_{safe_away}.json"
    return os.path.join(output_dir, filename)


def save_json(data: list | dict, output_path: str) -> None:
    """
    Writes the parsed JSON back to disk.
    Does NOT re-download if the file already exists — safe to re-run.
    """
    if os.path.exists(output_path):
        print(f"File already exists, skipping download: '{output_path}'")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        print(f"ERROR: could not write '{output_path}'. Detail: {err}")
        sys.exit(1)

    print(f"Saved: '{output_path}'")


def print_events_summary(output_path: str) -> None:
    """
    Reads the saved file and prints a quick summary of event types.
    Helps verify the file is valid StatsBomb format.
    """
    with open(output_path, encoding="utf-8") as f:
        events: list[dict] = json.load(f)

    counts: dict[str, int] = {}
    for event in events:
        name: str = event.get("type", {}).get("name", "Unknown")
        counts[name] = counts.get(name, 0) + 1

    print(f"\nEvent type breakdown ({len(events)} total events):")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>5}  {name}")


# 5. Driver — orchestrates stages in order, contains no business logic
# --------------------------------------------------------------------

def main() -> None:
    print("=== PassIQ — download_real_data.py ===\n")

    # Stage 1: fetch competitions index
    competitions: list[dict] = download_json(COMPETITIONS_URL)

    # Stage 2: find the target competition/season entry
    season_entry: dict = find_season(competitions, TARGET_COMPETITION_ID, TARGET_SEASON_ID)
    competition_name: str = season_entry["competition_name"]
    season_name:      str = season_entry["season_name"]
    print(f"Competition: {competition_name} — {season_name}")

    # Stage 3: fetch matches index for the selected season
    matches_url:    str       = build_matches_url(TARGET_COMPETITION_ID, TARGET_SEASON_ID)
    matches_raw:    list[dict] = download_json(matches_url)

    # Stage 4: select the target match
    match_info: MatchInfo = select_match(matches_raw, competition_name, season_name)
    print(
        f"\nSelected: {match_info.home_team} {match_info.home_score}–"
        f"{match_info.away_score} {match_info.away_team}  "
        f"({match_info.match_date}, id={match_info.match_id})"
    )

    # Stage 5: download events file
    output_path: str       = build_output_path(match_info, OUTPUT_DIR)
    events_url:  str       = build_events_url(match_info.match_id)
    events_data: list[dict] = download_json(events_url)

    # Stage 6: save to disk
    save_json(events_data, output_path)

    # Stage 7: verify — print event type summary
    print_events_summary(output_path)

    print(f"\n=== Done. To analyze in PassIQ, enter this path in the app: ===")
    print(f"  {output_path}")
    print(f"  (from Rust: ../{output_path})")


if __name__ == "__main__":
    main()
