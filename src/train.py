import argparse
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .config import MODEL_DIR, PROC_DIR, QUANTILES, TARGETS, season_to_id

NON_FEATURE_COLS = {
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "OPP_ABBR",
    "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
    "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS", "FANTASY_PTS", "VIDEO_AVAILABLE",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def make_estimator(backend: str, quantile: float):
    if backend == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            subsample_for_bin=50000,
            random_state=42,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
    )


def train(val_season: str = "2025-26", backend: str = "sklearn") -> None:
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    val_id = season_to_id(val_season)

    train_df = df[df["SEASON_ID"] < val_id]
    feature_cols = get_feature_cols(df)

    out_dir = MODEL_DIR / backend
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {"feature_cols": feature_cols, "val_season": val_season, "models": {}}

    for target in TARGETS:
        bundle["models"][target] = {}
        for q in QUANTILES:
            model = make_estimator(backend, q)
            model.fit(train_df[feature_cols], train_df[target])
            bundle["models"][target][q] = model
        print(f"[trained] {target} across quantiles {QUANTILES}")

    joblib.dump(bundle, out_dir / "quantile_bundle.joblib")
    print(f"Saved model bundle to {out_dir / 'quantile_bundle.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", default="2025-26")
    ap.add_argument("--backend", choices=["sklearn", "lightgbm"], default="sklearn")
    args = ap.parse_args()
    train(val_season=args.val_season, backend=args.backend)
