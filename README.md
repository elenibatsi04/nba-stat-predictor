# NBA Player Stat Quantile Predictor

An end-to-end machine learning pipeline that forecasts NBA player **Points (`PTS`)**, **Rebounds (`REB`)**, and **Assists (`AST`)** with 80% prediction intervals (`0.1` to `0.9` quantiles) and an interactive Streamlit explorer.

## Pipeline Architecture
1. **`src/data.py`** — Downloads and caches historical NBA regular-season player game logs (`2017-18` through `2026-27`).
2. **`src/features.py`** — Computes leak-free (`shift(1)`) rolling player averages (`3`, `10`, `25` games), per-minute efficiency rates, rest/back-to-back indicators, and opponent defensive baselines.
3. **`src/train.py`** — Trains quantile regression models (`HistGradientBoostingRegressor` or `LightGBM`) using strict temporal train/validation splits.
4. **`src/evaluate.py`** — Evaluates median MAE, pinball loss, and empirical 80% interval calibration.
5. **`app/streamlit_app.py`** — Interactive web UI to inspect player projections and recent game logs.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.data
python -m src.features
python -m src.train --backend sklearn --val-season 2025-26
python -m src.evaluate --backend sklearn
streamlit run app/streamlit_app.py
