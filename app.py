import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.data.loader import DataLoader
from src.data.external_loader import load_squad_value, load_manager_change_flags
from src.evaluation.backtester import WalkForwardBacktester
from src.features.feature_builder import FeatureBuilder
from src.features.form_features import FormFeatureBuilder
from src.forecasting.forecaster import SeasonForecaster
from src.forecasting.simulator import SeasonSimulator

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    layout="wide",
    page_title="EPL Forecast Dashboard",
    page_icon="⚽",
)

st.title("⚽ EPL Forecast Dashboard")
st.caption("Predicted table · Monte Carlo simulation · Walk-forward backtest · Historical trends")

# ======================
# SIDEBAR CONTROLS
# ======================
with st.sidebar:
    st.header("⚙️ Settings")

    model_type = st.selectbox(
        "Model",
        options=["hgbt", "rf", "lgbm", "xgb"],
        index=0,
        help="hgbt = HistGradientBoosting · rf = Random Forest · lgbm = LightGBM · xgb = XGBoost",
    )

    n_sims = st.slider("Simulations", min_value=500, max_value=10000, value=5000, step=500)

    use_poisson = st.toggle("Poisson goal model", value=True,
                            help="ON = Poisson goal sampling with EPL tiebreaking. OFF = legacy logistic.")

    st.divider()
    st.caption("Run backtest on the Backtest tab — it may take 30–60 s.")


# ======================
# DATA LOADING (cached)
# ======================
@st.cache_data(show_spinner="Loading data and building features…")
def load_dashboard_data(model_type: str, n_sims: int, use_poisson: bool):
    config = load_config("configs/default.yaml")
    config.model_type = model_type
    config.n_sims = n_sims
    config.use_poisson = use_poisson

    loader = DataLoader()
    builder = FeatureBuilder(rolling_windows=tuple(config.rolling_windows))
    form_builder = FormFeatureBuilder()

    season_df = loader.load_season_table(config.season_table)
    match_df = loader.load_match_table(config.match_table)

    match_features = builder.build_match_derived_features(match_df)
    form_df = form_builder.build(match_df)

    h2h_df = None
    if config.use_h2h_features:
        h2h_df = builder.build_h2h_features(match_df, season_df)

    feature_df = builder.build_team_season_features(
        season_df,
        match_features=match_features if config.use_match_features else None,
        form_features=form_df,
        h2h_features=h2h_df,
    )

    supervised_df = builder.make_supervised_frame(feature_df)

    squad_df = load_squad_value("data/external/team_squad_value_2024.csv")
    manager_df = load_manager_change_flags("data/external/manager_change_flags_2024.csv")
    supervised_df = supervised_df.merge(squad_df, on="team", how="left")
    supervised_df = supervised_df.merge(manager_df, on="team", how="left")
    supervised_df["manager_change_flag"] = (
        pd.to_numeric(supervised_df["manager_change_flag"], errors="coerce").fillna(0).astype(int)
    )
    promoted_avg_sv = supervised_df.loc[
        supervised_df["promoted_team_flag"] == 1, "squad_value_million"
    ].mean()
    supervised_df["squad_value_million"] = supervised_df["squad_value_million"].fillna(promoted_avg_sv)

    forbidden = {
        "team", "notes", "points", "position", "gf", "ga", "gd",
        "played", "won", "drawn", "lost",
        "target_points", "target_rank", "form_points", "form_gd", "h2h_ppg",
    }
    feature_columns = [
        c for c in supervised_df.columns
        if c not in forbidden
        and pd.api.types.is_numeric_dtype(supervised_df[c])
        and not c.startswith("match_")
    ]

    forecaster = SeasonForecaster(model_type=model_type, random_state=config.random_state)
    forecast = forecaster.forecast(
        supervised_df, feature_columns,
        predict_season=config.predict_season,
        promoted_teams=config.promoted_teams,
    )

    simulator = SeasonSimulator(
        random_state=config.random_state, use_poisson=use_poisson
    )
    sim_summary = simulator.simulate_many(forecast, n_sims=n_sims)

    fi = forecaster.model_.feature_importances()

    return config, supervised_df, season_df, forecast, sim_summary, feature_columns, fi


config, supervised_df, season_df, forecast, sim_summary, feature_columns, feat_imp = (
    load_dashboard_data(model_type, n_sims, use_poisson)
)


# ======================
# DISPLAY HELPERS
# ======================
def fmt_prob(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    return out


def band(rank: int) -> str:
    if rank <= 4:
        return "🟦 CL"
    if rank <= 6:
        return "🟩 Europa"
    if rank >= 18:
        return "🔴 Relegation"
    return "⬜ Mid-table"


# ======================
# TABS
# ======================
tab_forecast, tab_team, tab_backtest, tab_history, tab_features = st.tabs(
    ["📊 Forecast", "🔍 Team Deep Dive", "📈 Backtest", "🕰️ Historical", "🔧 Features"]
)


# ---- TAB 1: FORECAST ------------------------------------------------
with tab_forecast:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Season", int(config.predict_season))
    c2.metric("Teams", len(forecast))
    c3.metric("Features", len(feature_columns))
    c4.metric("Simulations", f"{n_sims:,}")

    st.divider()

    fc = forecast.copy()
    fc["Band"] = fc["predicted_rank"].apply(band)
    fc["predicted_points"] = fc["predicted_points"].round(1)

    sim_disp = sim_summary.copy()
    for col in ["title_prob", "top4_prob", "top6_prob", "relegation_prob"]:
        sim_disp[col] = (sim_disp[col] * 100).round(1)
    sim_disp["avg_points"] = sim_disp["avg_points"].round(1)
    sim_disp["avg_rank"] = sim_disp["avg_rank"].round(2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Predicted Table")
        st.dataframe(
            fc[["predicted_rank", "team", "predicted_points", "Band"]].rename(
                columns={"predicted_rank": "#", "team": "Team",
                         "predicted_points": "Pts (pred)", "Band": "Zone"}
            ),
            use_container_width=True, hide_index=True,
        )

    with col2:
        st.subheader("Simulation Probabilities")
        prob_cols = ["team", "avg_points", "avg_rank",
                     "title_prob", "top4_prob", "top6_prob", "relegation_prob"]
        st.dataframe(
            sim_disp[prob_cols].rename(columns={
                "team": "Team", "avg_points": "Avg Pts", "avg_rank": "Avg Rank",
                "title_prob": "Title %", "top4_prob": "Top 4 %",
                "top6_prob": "Top 6 %", "relegation_prob": "Rel %",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()

    # Plotly bar charts
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        fig = px.bar(
            sim_summary.sort_values("title_prob", ascending=False),
            x="team", y=sim_summary.sort_values("title_prob", ascending=False)["title_prob"] * 100,
            title="Title Probability (%)", labels={"y": "%", "team": ""},
            color_discrete_sequence=["#1f77b4"],
        )
        fig.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.bar(
            sim_summary.sort_values("top4_prob", ascending=False),
            x="team", y=sim_summary.sort_values("top4_prob", ascending=False)["top4_prob"] * 100,
            title="Top 4 Probability (%)", labels={"y": "%", "team": ""},
            color_discrete_sequence=["#2ca02c"],
        )
        fig.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        fig = px.bar(
            sim_summary.sort_values("relegation_prob", ascending=False),
            x="team", y=sim_summary.sort_values("relegation_prob", ascending=False)["relegation_prob"] * 100,
            title="Relegation Risk (%)", labels={"y": "%", "team": ""},
            color_discrete_sequence=["#d62728"],
        )
        fig.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig, use_container_width=True)


# ---- TAB 2: TEAM DEEP DIVE ------------------------------------------
with tab_team:
    team = st.selectbox("Select Team", forecast["team"].sort_values().tolist(), key="team_dd")

    tf = forecast[forecast["team"] == team].reset_index(drop=True)
    ts = sim_summary[sim_summary["team"] == team].reset_index(drop=True)
    team_hist = supervised_df[supervised_df["team"] == team].sort_values("season_end_year")
    feat_row = team_hist.tail(1).reset_index(drop=True)

    if tf.empty or ts.empty:
        st.error("No data for selected team.")
    else:
        pred_rank = int(tf.loc[0, "predicted_rank"])
        pred_pts = float(tf.loc[0, "predicted_points"])
        title_p = float(ts.loc[0, "title_prob"]) * 100
        top4_p = float(ts.loc[0, "top4_prob"]) * 100
        rel_p = float(ts.loc[0, "relegation_prob"]) * 100
        avg_rank = float(ts.loc[0, "avg_rank"])
        avg_pts = float(ts.loc[0, "avg_points"])

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Predicted Rank", pred_rank)
        m2.metric("Predicted Points", f"{pred_pts:.1f}")
        m3.metric("Title %", f"{title_p:.1f}%")
        m4.metric("Top 4 %", f"{top4_p:.1f}%")
        m5.metric("Relegation %", f"{rel_p:.1f}%")

        # Probability gauge chart
        fig = go.Figure(go.Bar(
            x=["Title", "Top 4", "Top 6", "Relegation"],
            y=[title_p, top4_p, float(ts.loc[0, "top6_prob"]) * 100, rel_p],
            marker_color=["gold", "#2ca02c", "#98df8a", "#d62728"],
            text=[f"{v:.1f}%" for v in [title_p, top4_p, float(ts.loc[0, "top6_prob"]) * 100, rel_p]],
            textposition="outside",
        ))
        fig.update_layout(title=f"{team} — Probability Breakdown", yaxis_title="%", height=320)
        st.plotly_chart(fig, use_container_width=True)

        # Feature snapshot
        st.subheader("Latest Feature Snapshot")
        snapshot_cols = [c for c in [
            "team", "season_end_year", "prev_points", "prev_position",
            "roll2_points_mean", "roll3_points_mean",
            "prev_form_points", "prev_form_gd",
            "prev_home_ppg", "prev_away_ppg",
            "prev_h2h_ppg",
            "squad_value_million", "manager_change_flag", "promoted_team_flag", "shock_flag",
        ] if c in feat_row.columns]
        st.dataframe(feat_row[snapshot_cols], use_container_width=True, hide_index=True)


# ---- TAB 3: BACKTEST ------------------------------------------------
with tab_backtest:
    st.info(
        "Walk-forward backtest trains on all data up to season N−1, "
        "predicts season N, and evaluates. Runs all historical test seasons. "
        "Click the button below to run (may take ~30–60 s)."
    )

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running walk-forward backtest…"):
            backtester = WalkForwardBacktester(
                model_type=model_type,
                min_train_season=config.min_train_season,
                random_state=config.random_state,
            )
            bt_df = backtester.run(supervised_df, feature_columns, config.predict_season)
            st.session_state["backtest_df"] = bt_df

    if "backtest_df" in st.session_state:
        bt_df = st.session_state["backtest_df"]
        adv = bt_df[bt_df["model_name"].str.startswith("advanced")].copy()
        base = bt_df[bt_df["model_name"].str.startswith("baseline")].copy()

        # Summary metrics
        cols = st.columns(5)
        cols[0].metric("Avg RMSE", f"{adv['rmse'].mean():.2f}", f"{adv['rmse'].mean() - base['rmse'].mean():.2f} vs baseline")
        cols[1].metric("Avg MAE", f"{adv['mae'].mean():.2f}")
        cols[2].metric("Avg Spearman ρ", f"{adv['spearman_rank_corr'].mean():.3f}")
        cols[3].metric("Top 4 Accuracy", f"{adv['top4_accuracy'].mean():.1%}")
        cols[4].metric("Relegation Accuracy", f"{adv['relegation_accuracy'].mean():.1%}")

        st.divider()

        # RMSE over time
        merged_bt = adv.merge(
            base[["test_season", "rmse", "mae"]].rename(columns={"rmse": "baseline_rmse", "mae": "baseline_mae"}),
            on="test_season", how="left"
        )
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Scatter(x=merged_bt["test_season"], y=merged_bt["rmse"],
                                       mode="lines+markers", name="Advanced"))
        fig_rmse.add_trace(go.Scatter(x=merged_bt["test_season"], y=merged_bt["baseline_rmse"],
                                       mode="lines+markers", name="Baseline", line=dict(dash="dash")))
        fig_rmse.update_layout(title="RMSE by Season", xaxis_title="Season", yaxis_title="RMSE", height=350)
        st.plotly_chart(fig_rmse, use_container_width=True)

        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            fig_top4 = px.bar(adv, x="test_season", y="top4_accuracy",
                              title="Top 4 Accuracy by Season", labels={"top4_accuracy": "Accuracy", "test_season": "Season"})
            fig_top4.update_layout(height=320)
            st.plotly_chart(fig_top4, use_container_width=True)

        with col_bt2:
            fig_brier = go.Figure()
            fig_brier.add_trace(go.Scatter(x=adv["test_season"], y=adv["brier_top4"],
                                           mode="lines+markers", name="Brier Top 4"))
            fig_brier.add_trace(go.Scatter(x=adv["test_season"], y=adv["brier_relegation"],
                                           mode="lines+markers", name="Brier Relegation"))
            fig_brier.update_layout(title="Brier Score by Season (lower = better)",
                                    xaxis_title="Season", height=320)
            st.plotly_chart(fig_brier, use_container_width=True)

        st.subheader("Full Backtest Table")
        st.dataframe(
            adv[["test_season", "rmse", "mae", "spearman_rank_corr",
                 "top4_accuracy", "relegation_accuracy", "champion_hit",
                 "brier_top4", "brier_relegation"]].round(3),
            use_container_width=True, hide_index=True,
        )


# ---- TAB 4: HISTORICAL TRENDS ---------------------------------------
with tab_history:
    all_teams = sorted(season_df["team"].unique())
    selected = st.multiselect(
        "Select teams", all_teams,
        default=["Manchester City", "Liverpool", "Arsenal", "Chelsea", "Manchester United"][:5],
    )

    if selected:
        hist = season_df[season_df["team"].isin(selected)].copy()
        hist = hist.sort_values(["team", "season_end_year"])

        col_h1, col_h2 = st.columns(2)

        with col_h1:
            fig_pts = px.line(
                hist, x="season_end_year", y="points", color="team",
                title="Points per Season", markers=True,
                labels={"season_end_year": "Season", "points": "Points"},
            )
            fig_pts.update_layout(height=380)
            st.plotly_chart(fig_pts, use_container_width=True)

        with col_h2:
            fig_pos = px.line(
                hist, x="season_end_year", y="position", color="team",
                title="Final Position per Season", markers=True,
                labels={"season_end_year": "Season", "position": "Position"},
            )
            fig_pos.update_yaxes(autorange="reversed")
            fig_pos.update_layout(height=380)
            st.plotly_chart(fig_pos, use_container_width=True)

        col_h3, col_h4 = st.columns(2)
        with col_h3:
            fig_gf = px.line(
                hist, x="season_end_year", y="gf", color="team",
                title="Goals Scored per Season", markers=True,
                labels={"season_end_year": "Season", "gf": "Goals For"},
            )
            fig_gf.update_layout(height=350)
            st.plotly_chart(fig_gf, use_container_width=True)

        with col_h4:
            fig_gd = px.line(
                hist, x="season_end_year", y="gd", color="team",
                title="Goal Difference per Season", markers=True,
                labels={"season_end_year": "Season", "gd": "GD"},
            )
            fig_gd.update_layout(height=350)
            st.plotly_chart(fig_gd, use_container_width=True)
    else:
        st.info("Select at least one team above.")


# ---- TAB 5: FEATURES ------------------------------------------------
with tab_features:
    st.subheader(f"Feature Importances ({model_type.upper()})")
    if feat_imp is not None and not feat_imp.empty:
        top_n = st.slider("Show top N features", 10, min(50, len(feat_imp)), 20)
        top = feat_imp.head(top_n).reset_index()
        top.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            top.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title=f"Top {top_n} Feature Importances",
            color="Importance", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=max(400, top_n * 22), showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances not available for this model type.")

    st.divider()
    st.subheader("All Features Used")
    st.write(f"**{len(feature_columns)} features total**")
    st.dataframe(pd.DataFrame({"Feature": feature_columns}), use_container_width=True, hide_index=True)
