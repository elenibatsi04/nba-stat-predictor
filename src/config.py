from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEASONS = [f"{y}-{str(y + 1)[2:]}" for y in range(2017, 2027)]
TARGETS = ["PTS", "REB", "AST"]
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
COVERAGE = 0.8
MIN_ROLL_MIN = 12.0


def season_to_id(season: str) -> str:
    return "2" + season[:4]
