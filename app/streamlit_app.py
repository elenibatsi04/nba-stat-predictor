import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import MODEL_DIR, PROC_DIR, QUANTILES, TARGETS

st.set_page_config(page_title="NBA Stat Predictor", layout="wide")
st.title("NBA Player Stat Quantile Predictor")


@st.cache_resource
def load_assets(backend: str = "sklearn"):
    df = pd.read_parquet(PROC_DIR / "features.parquet")
    bundle = joblib.load(MODEL_DIR / backend / "quantile_bundle.joblib")
    return df, bundle


df, bundle = load_assets()
players = sorted(df["PLAYER_NAME"].unique())
selected_player = st.sidebar.selectbox("Select Player", players)

player_df = df[df["PLAYER_NAME"] == selected_player].sort_values("GAME_DATE")
latest_row = player_df.iloc[[-1]]
X_latest = latest_row[bundle["feature_cols"]]

st.subheader(f"{selected_player} — Next Game Projection (Based on Latest Form)")
cols = st.columns(len(TARGETS))

for col, target in zip(cols, TARGETS):
    raw_q = [bundle["models"][target][q].predict(X_latest)[0] for q in QUANTILES]
    sorted_q = np.sort(raw_q)
    q_dict = dict(zip(QUANTILES, sorted_q))

    with col:
        st.metric(
            label=f"Projected {target} (Median)",
            value=f"{q_dict[0.5]:.1f}",
            delta=f"80% CI: [{q_dict[0.1]:.1f} – {q_dict[0.9]:.1f}]",
        )
        st.caption(f"IQR (25th–75th): {q_dict[0.25]:.1f} – {q_dict[0.75]:.1f}")

st.subheader("Recent 15 Games")
st.dataframe(
    player_df[["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]].tail(15),
    use_container_width=True,
    hide_index=True,
)
