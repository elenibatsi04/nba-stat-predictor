import argparse
import time
import pandas as pd
from curl_cffi import requests as cffi_requests
from .config import RAW_DIR, SEASONS

URL = "https://stats.nba.com/stats/leaguegamelog"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def fetch_season(season: str, retries: int = 4, pause: float = 4.0) -> pd.DataFrame:
    params = {
        "Counter": "1000",
        "DateFrom": "",
        "DateTo": "",
        "Direction": "DESC",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": "Regular Season",
        "Sorter": "DATE",
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = cffi_requests.get(
                URL,
                params=params,
                headers=HEADERS,
                impersonate="chrome124",
                timeout=30,
            )
            resp.raise_for_status()
            payload = resp.json()["resultSets"][0]
            return pd.DataFrame(payload["rowSet"], columns=payload["headers"])
        except Exception as e:
            last_err = e
            print(f"  [retry {attempt + 1}/{retries}] {season}: {type(e).__name__}")
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {season}: {last_err}") from last_err


def download(seasons=SEASONS, force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = RAW_DIR / f"logs_{season}.csv"
        is_current = season == seasons[-1]
        if path.exists() and not force and not is_current:
            print(f"[skip] {season} (cached)")
            continue
        print(f"[fetch] {season} ...")
        fetch_season(season).to_csv(path, index=False)
        time.sleep(1.5)


def load_logs() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("logs_*.csv"))
    if not files:
        raise FileNotFoundError("No raw data found. Run `python -m src.data` first.")
    frames = [pd.read_csv(f, dtype={"GAME_ID": str, "SEASON_ID": str}) for f in files]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    download(force=ap.parse_args().force)
