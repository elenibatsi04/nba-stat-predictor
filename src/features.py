import numpy as np
import pandas as pd
from .config import MIN_ROLL_MIN, PROC_DIR, TARGETS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["OPP_ABBR"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]{3})")

    df["DAYS_REST"] = (
        df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=7)
    )
    df["IS_B2B"] = (df["DAYS_REST"] == 1).astype(int)

    stat_cols = ["MIN", "PTS", "REB", "AST", "FGA", "FG3A", "FTA", "TOV", "PLUS_MINUS"]
    grouped = df.groupby("PLAYER_ID", group_keys=False)

    for w in (3, 10, 25):
        rolled = (
            grouped[stat_cols]
            .apply(lambda g: g.shift(1).rolling(w, min_periods=1).mean())
            .add_suffix(f"_ROLL_{w}")
        )
        df = pd.concat([df, rolled], axis=1)

    for stat in TARGETS:
        df[f"{stat}_PER_MIN_10"] = df[f"{stat}_ROLL_10"] / df["MIN_ROLL_10"].replace(0, np.nan)

    team_allowed = (
        df.groupby(["SEASON_ID", "OPP_ABBR", "GAME_ID", "GAME_DATE"], as_index=False)[TARGETS]
        .sum()
        .sort_values(["OPP_ABBR", "GAME_DATE"])
    )
    for stat in TARGETS:
        team_allowed[f"OPP_{stat}_ALLOWED_15"] = (
            team_allowed.groupby("OPP_ABBR")[stat]
            .transform(lambda s: s.shift(1).rolling(15, min_periods=3).mean())
        )

    df = df.merge(
        team_allowed[["GAME_ID", "OPP_ABBR"] + [f"OPP_{s}_ALLOWED_15" for s in TARGETS]],
        on=["GAME_ID", "OPP_ABBR"],
        how="left",
    )

    df = df[df["MIN_ROLL_10"] >= MIN_ROLL_MIN].dropna(subset=["PTS_ROLL_10"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from .data import load_logs

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    feats = build_features(load_logs())
    out_path = PROC_DIR / "features.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"Saved {len(feats):,} rotation player-games to {out_path}")
