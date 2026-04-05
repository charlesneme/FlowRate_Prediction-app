# app.py — Volve Field Multi-Phase Flow Rate Predictor
# Gradient Boosting model trained on Volve production data (2008-2016)
# Predicts: BORE_OIL_VOL, BORE_GAS_VOL, BORE_WAT_VOL (Sm³/day)
#
# Run locally:  streamlit run app.py
# Deploy:       streamlit cloud / huggingface spaces
#
# Required files in same directory:
#   gradient_boosting_model.pkl
#   minmax_scaler.pkl
#   features_list.pkl
#   well_columns.pkl

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Phase Flow Rate Predictor for Volve Field",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model artefacts ───────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model    = joblib.load("gradient_boosting_model.pkl")
    scaler   = joblib.load("minmax_scaler.pkl")
    features = joblib.load("features_list.pkl")
    wells    = joblib.load("well_columns.pkl")
    return model, scaler, features, wells

try:
    model, scaler, FEATURES_ALL, WELL_COLS = load_artefacts()
    artefacts_loaded = True
except FileNotFoundError as e:
    st.error(
        f"❌ Model file not found: {e}\n\n"
        "Place `gradient_boosting_model.pkl`, `minmax_scaler.pkl`, "
        "`features_list.pkl`, and `well_columns.pkl` in the same "
        "directory as `app.py`, then restart."
    )
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────
TARGETS    = ["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]
TARGET_LABELS = ["Oil Volume (Sm³/day)", "Gas Volume (Sm³/day)", "Water Volume (Sm³/day)"]
TARGET_UNITS  = ["Sm³/day", "Sm³/day", "Sm³/day"]
TARGET_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
TARGET_ICONS  = ["🛢️", "💨", "💧"]

WELL_NAMES = [
    "NO 15/9-F-1 C",
    "NO 15/9-F-11 H",
    "NO 15/9-F-12 H",
    "NO 15/9-F-14 H",
    "NO 15/9-F-15 D",
    "NO 15/9-F-5 AH",
]

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🛢️ Multi-Phase Flow Rate Predictor for Volve Field Production")
st.markdown(
    """
    **Model Used:** Gradient Boosting Regressor &nbsp;|&nbsp;
    **Dataset:** Equinor Volve Open Data &nbsp;|&nbsp;
    **Designed by:** Charles James &nbsp;|&nbsp;
   
    """
)
st.divider()

# ── Sidebar: sensor inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Well & Sensor Inputs")
    st.caption("Adjust the values below and click **Predict** to generate flow rate estimates.")

    st.subheader("Well Identity")
    selected_well = st.selectbox(
        "Well bore",
        options=WELL_NAMES,
        index=2,
        help="Select the active well. One-hot encoded before prediction.",
    )

    st.subheader("Flow Conditions")
    on_stream_hrs = st.slider(
        "On-stream hours (hrs)",
        min_value=0.25, max_value=25.0, value=24.0, step=0.25,
        help="Hours the well actively flowed during this period.",
    )
    avg_choke_size_p = st.slider(
        "Average choke size (%)",
        min_value=0.0, max_value=100.0, value=59.0, step=0.5,
        help="Choke opening percentage — primary flow control mechanism.",
    )
    dp_choke_size = st.number_input(
        "Drill-pipe choke size (mm)",
        min_value=0.0, max_value=110.0, value=8.8, step=0.1,
        help="Physical choke diameter in millimetres.",
    )

    st.subheader("Pressure Parameters")
    avg_whp_p = st.number_input(
        "Avg wellhead pressure (bar)",
        min_value=0.0, max_value=125.0, value=38.0, step=0.5,
        help="Backpressure on the reservoir.",
    )
    avg_downhole_pressure = st.number_input(
        "Avg downhole pressure (bar)",
        min_value=0.0, max_value=310.0, value=230.0, step=1.0,
        help="Reservoir drive pressure at the bottom of the well.",
    )
    avg_dp_tubing = st.number_input(
        "Avg DP tubing (bar)",
        min_value=0.0, max_value=260.0, value=176.0, step=1.0,
        help="Pressure drop across the production tubing.",
    )
    avg_annulus_press = st.number_input(
        "Avg annulus pressure (bar)",
        min_value=0.0, max_value=31.0, value=18.0, step=0.1,
        help="Annulus pressure between tubing and outer casing. "
             "Rising values indicate gas migration.",
    )

    st.subheader("Temperature Parameters")
    avg_wht_p = st.number_input(
        "Avg wellhead temperature (°C)",
        min_value=0.0, max_value=95.0, value=82.0, step=0.5,
        help="Wellhead temperature — correlates with flow rate.",
    )
    avg_downhole_temperature = st.number_input(
        "Avg downhole temperature (°C)",
        min_value=0.0, max_value=110.0, value=105.0, step=0.5,
        help="Downhole temperature — fluid viscosity proxy.",
    )

    st.subheader("Temporal Context (Recent History)")
    st.caption(
        "These rolling averages and cumulative values encode the well's "
        "recent production trajectory — critical for accurate prediction "
        "during field decline."
    )

    col_r14, col_r90 = st.columns(2)
    with col_r14:
        oil_roll14 = st.number_input("Oil 14-day avg (Sm³)", 0.0, 6000.0, 700.0, 10.0)
        gas_roll14 = st.number_input("Gas 14-day avg (Sm³)", 0.0, 860000.0, 110000.0, 1000.0)
        wat_roll14 = st.number_input("Water 14-day avg (Sm³)", 0.0, 8100.0, 1400.0, 10.0)
    with col_r90:
        oil_roll90 = st.number_input("Oil 90-day avg (Sm³)", 0.0, 6000.0, 800.0, 10.0)
        gas_roll90 = st.number_input("Gas 90-day avg (Sm³)", 0.0, 860000.0, 120000.0, 1000.0)
        wat_roll90 = st.number_input("Water 90-day avg (Sm³)", 0.0, 8100.0, 1500.0, 10.0)

    col_lag1, col_lag7 = st.columns(2)
    with col_lag1:
        oil_lag1 = st.number_input("Oil yesterday (Sm³)", 0.0, 6000.0, 690.0, 10.0)
        gas_lag1 = st.number_input("Gas yesterday (Sm³)", 0.0, 860000.0, 108000.0, 1000.0)
        wat_lag1 = st.number_input("Water yesterday (Sm³)", 0.0, 8100.0, 1380.0, 10.0)
    with col_lag7:
        oil_lag7 = st.number_input("Oil 7 days ago (Sm³)", 0.0, 6000.0, 710.0, 10.0)
        gas_lag7 = st.number_input("Gas 7 days ago (Sm³)", 0.0, 860000.0, 112000.0, 1000.0)
        wat_lag7 = st.number_input("Water 7 days ago (Sm³)", 0.0, 8100.0, 1410.0, 10.0)

    # Intermediate windows (3, 7, 30, 60) — auto-interpolated from 14 and 90
    oil_roll3  = oil_roll14 * 1.01
    oil_roll7  = oil_roll14 * 1.005
    oil_roll30 = (oil_roll14 * 0.9 + oil_roll90 * 0.1)
    oil_roll60 = (oil_roll14 * 0.6 + oil_roll90 * 0.4)
    gas_roll3  = gas_roll14 * 1.01
    gas_roll7  = gas_roll14 * 1.005
    gas_roll30 = (gas_roll14 * 0.9 + gas_roll90 * 0.1)
    gas_roll60 = (gas_roll14 * 0.6 + gas_roll90 * 0.4)
    wat_roll3  = wat_roll14 * 1.01
    wat_roll7  = wat_roll14 * 1.005
    wat_roll30 = (wat_roll14 * 0.9 + wat_roll90 * 0.1)
    wat_roll60 = (wat_roll14 * 0.6 + wat_roll90 * 0.4)

    st.subheader("Depletion Context")
    cum_days  = st.number_input("Cumulative days on stream", 0, 3300, 800, 10)
    cum_oil   = st.number_input("Cumulative oil produced (Sm³)", 0.0, 3e6, 500000.0, 1000.0)
    cum_gas   = st.number_input("Cumulative gas produced (Sm³)", 0.0, 5e8, 80000000.0, 100000.0)
    cum_water = st.number_input("Cumulative water produced (Sm³)", 0.0, 5e6, 1000000.0, 10000.0)

    predict_btn = st.button("🔮  Predict Flow Rates", type="primary", use_container_width=True)

# ── Build feature vector ───────────────────────────────────────────────────
def build_feature_vector():
    # Decline ratios
    oil_decline = (oil_roll14 + 1) / (oil_roll90 + 1)
    gas_decline = (gas_roll14 + 1) / (gas_roll90 + 1)
    wat_decline = (wat_roll14 + 1) / (wat_roll90 + 1)

    row = {
        # Base sensor features
        "ON_STREAM_HRS":            on_stream_hrs,
        "AVG_DOWNHOLE_PRESSURE":    avg_downhole_pressure,
        "AVG_DOWNHOLE_TEMPERATURE": avg_downhole_temperature,
        "AVG_DP_TUBING":            avg_dp_tubing,
        "AVG_ANNULUS_PRESS":        avg_annulus_press,
        "AVG_CHOKE_SIZE_P":         avg_choke_size_p,
        "AVG_WHP_P":                avg_whp_p,
        "AVG_WHT_P":                avg_wht_p,
        "DP_CHOKE_SIZE":            dp_choke_size,
        # CUM_DAYS
        "CUM_DAYS":                 cum_days,
        # Rolling means — oil
        "BORE_OIL_VOL_roll3":       oil_roll3,
        "BORE_OIL_VOL_roll7":       oil_roll7,
        "BORE_OIL_VOL_roll14":      oil_roll14,
        "BORE_OIL_VOL_roll30":      oil_roll30,
        "BORE_OIL_VOL_roll60":      oil_roll60,
        "BORE_OIL_VOL_roll90":      oil_roll90,
        # Rolling means — gas
        "BORE_GAS_VOL_roll3":       gas_roll3,
        "BORE_GAS_VOL_roll7":       gas_roll7,
        "BORE_GAS_VOL_roll14":      gas_roll14,
        "BORE_GAS_VOL_roll30":      gas_roll30,
        "BORE_GAS_VOL_roll60":      gas_roll60,
        "BORE_GAS_VOL_roll90":      gas_roll90,
        # Rolling means — water
        "BORE_WAT_VOL_roll3":       wat_roll3,
        "BORE_WAT_VOL_roll7":       wat_roll7,
        "BORE_WAT_VOL_roll14":      wat_roll14,
        "BORE_WAT_VOL_roll30":      wat_roll30,
        "BORE_WAT_VOL_roll60":      wat_roll60,
        "BORE_WAT_VOL_roll90":      wat_roll90,
        # Lag features — oil
        "BORE_OIL_VOL_lag1":        oil_lag1,
        "BORE_OIL_VOL_lag7":        oil_lag7,
        # Lag features — gas
        "BORE_GAS_VOL_lag1":        gas_lag1,
        "BORE_GAS_VOL_lag7":        gas_lag7,
        # Lag features — water
        "BORE_WAT_VOL_lag1":        wat_lag1,
        "BORE_WAT_VOL_lag7":        wat_lag7,
        # Decline ratios
        "BORE_OIL_VOL_decline":     np.clip(oil_decline, 0.01, 10.0),
        "BORE_GAS_VOL_decline":     np.clip(gas_decline, 0.01, 10.0),
        "BORE_WAT_VOL_decline":     np.clip(wat_decline, 0.01, 10.0),
        # Cumulative production
        "CUM_BORE_OIL_VOL":         cum_oil,
        "CUM_BORE_GAS_VOL":         cum_gas,
        "CUM_BORE_WAT_VOL":         cum_water,
    }

    # One-hot well encoding — all zeros then set selected well to 1
    for wc in WELL_COLS:
        row[wc] = 0.0
    well_col = f"WELL_{selected_well}"
    if well_col in WELL_COLS:
        row[well_col] = 1.0

    # Build DataFrame in exact feature order
    df_input = pd.DataFrame([row])[FEATURES_ALL]
    return df_input


# ── Gauge chart ────────────────────────────────────────────────────────────
def gauge_chart(value, max_val, label, unit, color):
    fig, ax = plt.subplots(figsize=(3.2, 2.4), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    theta_start = np.pi
    theta_end   = 0.0
    theta_val   = theta_start - (value / max_val) * np.pi

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), lw=14, color="#2a2a2a",
            solid_capstyle="round")
    # Value arc
    if value > 0:
        theta_v = np.linspace(np.pi, theta_val, 200)
        ax.plot(np.cos(theta_v), np.sin(theta_v), lw=14, color=color,
                solid_capstyle="round")
    # Value text
    ax.text(0, 0.15, f"{value:,.1f}", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.text(0, -0.22, unit, ha="center", va="center",
            fontsize=8.5, color="#aaaaaa")
    ax.text(0, -0.55, label, ha="center", va="center",
            fontsize=8, color="#cccccc")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.15)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


# ── Bar comparison chart ───────────────────────────────────────────────────
def bar_chart(predictions):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ref_values = [715.9, 111238.0, 1476.4]  # field medians
    short_labs = ["Oil", "Gas", "Water"]
    for ax, pred, ref, c, lab in zip(axes, predictions, ref_values,
                                      TARGET_COLORS, short_labs):
        ax.set_facecolor("#0e1117")
        categories = ["Predicted", "Field Median"]
        values     = [pred, ref]
        bars = ax.bar(categories, values, color=[c, "#444444"],
                      edgecolor="none", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:,.0f}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")
        ax.set_title(lab, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("Sm³/day", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_facecolor("#0e1117")
        ax.yaxis.label.set_color("#aaaaaa")
    fig.suptitle("Predicted vs Field Median Daily Rate",
                 color="white", fontsize=10, y=1.01)
    plt.tight_layout()
    return fig


# ── Decline context chart ──────────────────────────────────────────────────
def decline_chart():
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    windows = [3, 7, 14, 30, 60, 90]
    oil_avgs = [oil_roll3, oil_roll7, oil_roll14, oil_roll30, oil_roll60, oil_roll90]
    gas_avgs = [v / 1000 for v in
                [gas_roll3, gas_roll7, gas_roll14, gas_roll30, gas_roll60, gas_roll90]]
    wat_avgs = [wat_roll3, wat_roll7, wat_roll14, wat_roll30, wat_roll60, wat_roll90]

    ax.plot(windows, oil_avgs, "o-", color="#1f77b4", lw=2,
            markersize=6, label="Oil (Sm³)")
    ax2 = ax.twinx()
    ax2.set_facecolor("#0e1117")
    ax2.plot(windows, gas_avgs, "s--", color="#ff7f0e", lw=2,
             markersize=6, label="Gas (×10³ Sm³)")
    ax.plot(windows, wat_avgs, "^:", color="#2ca02c", lw=2,
            markersize=6, label="Water (Sm³)")

    ax.set_xlabel("Rolling window (days)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Oil / Water (Sm³)", color="#aaaaaa", fontsize=9)
    ax2.set_ylabel("Gas (×10³ Sm³)", color="#ff7f0e", fontsize=9)
    ax.set_title("Production History Context — Rolling Averages by Window",
                 color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax2.tick_params(colors="#ff7f0e", labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#333333")
    for spine in ax2.spines.values(): spine.set_edgecolor("#333333")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
              fontsize=8, facecolor="#1a1a1a", edgecolor="#444",
              labelcolor="white")
    plt.tight_layout()
    return fig


# ── Main panel ─────────────────────────────────────────────────────────────
tab_predict, tab_context, tab_info = st.tabs(
    ["🔮 Prediction", "📈 Production Context", "ℹ️ Model Info"]
)

with tab_predict:
    if predict_btn:
        with st.spinner("Running Gradient Boosting prediction..."):
            df_input = build_feature_vector()
            X_scaled = scaler.transform(df_input.values)
            log_pred  = model.predict(X_scaled)
            preds     = np.expm1(log_pred[0])           # back-transform log → Sm³
            oil_pred, gas_pred, wat_pred = preds[0], preds[1], preds[2]

        st.success(f"✅ Prediction complete — Well: **{selected_well}**")
        st.divider()

        # ── Metric cards ──────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("🛢️ Oil Rate", f"{oil_pred:,.1f} Sm³/day",
                      delta=f"{oil_pred - oil_roll14:+,.1f} vs 14-day avg")
        with c2:
            st.metric("💨 Gas Rate", f"{gas_pred:,.1f} Sm³/day",
                      delta=f"{gas_pred - gas_roll14:+,.1f} vs 14-day avg")
        with c3:
            st.metric("💧 Water Rate", f"{wat_pred:,.1f} Sm³/day",
                      delta=f"{wat_pred - wat_roll14:+,.1f} vs 14-day avg")

        st.divider()

        # ── Gauge charts ──────────────────────────────────────────────────
        st.subheader("Flow Rate Gauges")
        g1, g2, g3 = st.columns(3)
        with g1:
            st.pyplot(gauge_chart(oil_pred, 6000, "Oil Production",
                                  "Sm³/day", "#1f77b4"), use_container_width=True)
        with g2:
            st.pyplot(gauge_chart(gas_pred, 860000, "Gas Production",
                                  "Sm³/day", "#ff7f0e"), use_container_width=True)
        with g3:
            st.pyplot(gauge_chart(wat_pred, 8100, "Water Production",
                                  "Sm³/day", "#2ca02c"), use_container_width=True)

        st.divider()

        # ── Bar comparison ────────────────────────────────────────────────
        st.subheader("Predicted vs Field Median")
        st.pyplot(bar_chart([oil_pred, gas_pred, wat_pred]),
                  use_container_width=True)

        st.divider()

        # ── Decline ratios ────────────────────────────────────────────────
        st.subheader("Decline Signal")
        dr1, dr2, dr3 = st.columns(3)
        oil_dr = np.clip((oil_roll14 + 1) / (oil_roll90 + 1), 0.01, 10)
        gas_dr = np.clip((gas_roll14 + 1) / (gas_roll90 + 1), 0.01, 10)
        wat_dr = np.clip((wat_roll14 + 1) / (wat_roll90 + 1), 0.01, 10)

        def decline_label(v):
            if v < 0.85:   return "🔴 Declining fast"
            elif v < 0.97: return "🟡 Declining"
            elif v < 1.03: return "🟢 Stable"
            else:          return "🔵 Recovering"

        with dr1:
            st.metric("Oil decline ratio (roll14/roll90)", f"{oil_dr:.3f}",
                      delta=decline_label(oil_dr), delta_color="off")
        with dr2:
            st.metric("Gas decline ratio", f"{gas_dr:.3f}",
                      delta=decline_label(gas_dr), delta_color="off")
        with dr3:
            st.metric("Water decline ratio", f"{wat_dr:.3f}",
                      delta=decline_label(wat_dr), delta_color="off")

        st.divider()

        # ── Summary table ─────────────────────────────────────────────────
        st.subheader("Prediction Summary")
        summary = pd.DataFrame({
            "Phase":           ["Oil", "Gas", "Water"],
            "Predicted (Sm³/day)": [f"{oil_pred:,.1f}",
                                    f"{gas_pred:,.1f}",
                                    f"{wat_pred:,.1f}"],
            "14-day Avg (Sm³/day)": [f"{oil_roll14:,.1f}",
                                     f"{gas_roll14:,.1f}",
                                     f"{wat_roll14:,.1f}"],
            "Change vs 14d avg":   [f"{oil_pred-oil_roll14:+,.1f}",
                                    f"{gas_pred-gas_roll14:+,.1f}",
                                    f"{wat_pred-wat_roll14:+,.1f}"],
            "Decline Ratio":       [f"{oil_dr:.3f}",
                                    f"{gas_dr:.3f}",
                                    f"{wat_dr:.3f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    else:
        st.info(
            "👈 Set your well and sensor inputs in the sidebar, "
            "then click **Predict Flow Rates** to generate simultaneous "
            "oil, gas, and water production estimates."
        )
        st.markdown("""
        ### How to use this app
        1. **Select well** — choose the active wellbore
        2. **Set sensor readings** — enter current wellhead and downhole measurements
        3. **Set recent history** — enter rolling averages and lag values from production records
        4. **Set depletion context** — cumulative production and days on stream
        5. **Click Predict** — the Gradient Boosting model returns all three flow rates instantly

        ### Model performance (confirmed on held-out test set)
        | Target | R² | MAE (Sm³/day) | RMSE (Sm³/day) |
        |---|---|---|---|
        | Oil | 0.9427 | ~905 | ~1,880 |
        | Gas | 0.9582 | ~905 | ~1,880 |
        | Water | 0.9509 | ~905 | ~1,880 |
        | **Average** | **0.9506** | **2,714** | **5,642** |
        """)

with tab_context:
    st.subheader("Production History Context")
    st.caption(
        "Rolling average inputs visualised across all six time windows. "
        "Slopes indicate the recent production trend being fed to the model."
    )
    st.pyplot(decline_chart(), use_container_width=True)

    st.divider()
    st.subheader("Input Feature Summary")
    input_summary = pd.DataFrame({
        "Parameter": [
            "Well", "On-stream hrs", "Choke size (%)", "Choke diameter (mm)",
            "Wellhead pressure (bar)", "Downhole pressure (bar)",
            "DP tubing (bar)", "Annulus pressure (bar)",
            "Wellhead temp (°C)", "Downhole temp (°C)",
            "Cumulative days", "Cum. oil (Sm³)", "Cum. gas (Sm³)", "Cum. water (Sm³)",
        ],
        "Value": [
            selected_well, on_stream_hrs, avg_choke_size_p, dp_choke_size,
            avg_whp_p, avg_downhole_pressure,
            avg_dp_tubing, avg_annulus_press,
            avg_wht_p, avg_downhole_temperature,
            cum_days, f"{cum_oil:,.0f}", f"{cum_gas:,.0f}", f"{cum_water:,.0f}",
        ],
    })
    st.dataframe(input_summary, use_container_width=True, hide_index=True)

with tab_info:
    st.subheader("Model Architecture")
    st.markdown("""
    **Algorithm:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)
    wrapped in `MultiOutputRegressor` to predict oil, gas, and water simultaneously.

    **Key hyperparameters:**
    | Parameter | Value | Rationale |
    |---|---|---|
    | `n_estimators` | 300 | 300 sequential trees; small steps accumulate accurate corrections |
    | `max_depth` | 4 | Limits each tree to 16 leaves — prevents memorisation of training patterns |
    | `learning_rate` | 0.02 | Conservative step size; 300 small corrections outperform fewer large ones |
    | `subsample` | 0.8 | Stochastic row sampling per tree — implicit regularisation |
    | `min_samples_leaf` | 5 | Eliminates noise-driven micro-splits from single observations |

    **Feature engineering (46 total features):**
    - 9 raw sensor features (pressure, temperature, choke settings)
    - 18 rolling means (6 windows × 3 targets, lag-safe with shift(1))
    - 6 lag features (yesterday + 7 days ago × 3 targets)
    - 3 decline ratios (roll14 / roll90 per target)
    - 3 cumulative production columns
    - 6 one-hot well identity columns
    - 1 CUM_DAYS

    **Training data:** Volve field, February 2008 – February 2015 (4,804 records)
    
    **Test period:** November 2015 – September 2016 (1,030 records) — late-field 
    depletion phase with 76% lower mean oil rate than training, representing the 
    hardest possible generalisation challenge.

    **Prediction pipeline:**
    1. Raw inputs assembled into 46-feature vector
    2. MinMaxScaler applied (fitted on training data only)
    3. Gradient Boosting predicts log-space targets
    4. `np.expm1()` back-transforms predictions to original Sm³ space
    """)

    st.divider()
    st.subheader("Dataset")
    st.markdown("""
    - **Source:** Equinor Volve Open Dataset (Norwegian Petroleum Directorate)
    - **Period:** September 2007 – December 2016
    - **Wells:** 5 active production wells (NO 15/9-F-1C, F-11H, F-12H, F-14H, F-15D, F-5AH)
    - **Records used:** 6,863 (after filtering injection, zero-oil, and zero-hours rows)
    
    """)
