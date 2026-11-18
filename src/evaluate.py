import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from .config import COVERAGE, MODEL_DIR, PROC_DIR, QUANTILES, REPORT_DIR, TARGETS, season_to_id


def evaluate(backend: str = "sklearn") -> None:
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    bundle = joblib.load(MODEL_DIR / backend / "quantile_bundle.joblib")

    val_df = df[df["SEASON_ID"] == season_to_id(bundle["val_season"])].copy()
    X_val = val_df[bundle["feature_cols"]]

    rows = []
    for target in TARGETS:
        y_true = val_df[target].values
        preds = {q: bundle["models"][target][q].predict(X_val) for q in QUANTILES}

        q_matrix = np.sort(np.column_stack([preds[q] for q in QUANTILES]), axis=1)
        q_map = dict(zip(QUANTILES, q_matrix.T))

        mae = mean_absolute_error(y_true, q_map[0.5])
        in_interval = (y_true >= q_map[0.1]) & (y_true <= q_map[0.9])
        empirical_cov = float(np.mean(in_interval))
        avg_width = float(np.mean(q_map[0.9] - q_map[0.1]))
        pinball_50 = mean_pinball_loss(y_true, q_map[0.5], alpha=0.5)

        rows.append({
            "Target": target,
            "Median_MAE": round(mae, 3),
            "Pinball_50": round(pinball_50, 3),
            f"Target_Cov ({COVERAGE:.0%})": COVERAGE,
            "Empirical_Cov": round(empirical_cov, 3),
            "Mean_80_Width": round(avg_width, 2),
        })

    report = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_DIR / f"eval_{backend}.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sklearn", "lightgbm"], default="sklearn")
    evaluate(backend=ap.parse_args().backend)
