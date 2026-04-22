"""
pipeline.py — CLI entry point that runs the full pipeline including backtesting.
Usage:  python -m src.pipeline  (from epl-forecasting/)
        python -m src.pipeline --config configs/default.yaml --backtest
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

from src.config import load_config
from src.data.loader import DataLoader
from src.data.external_loader import load_squad_value, load_manager_change_flags
from src.data.validator import DataValidator
from src.evaluation.backtester import WalkForwardBacktester
from src.features.feature_builder import FeatureBuilder
from src.features.form_features import FormFeatureBuilder
from src.forecasting.forecaster import SeasonForecaster
from src.forecasting.simulator import SeasonSimulator


# ------------------------------------------------------------------
# Shared data-loading / feature-building logic
# ------------------------------------------------------------------

def build_supervised(config):
    loader = DataLoader()
    builder = FeatureBuilder(rolling_windows=tuple(config.rolling_windows))
    form_builder = FormFeatureBuilder()
    validator = DataValidator()

    season_df = loader.load_season_table(config.season_table)
    match_df = loader.load_match_table(config.match_table)

    v_season = validator.validate_season_table(season_df)
    v_match = validator.validate_match_table(match_df)
    if v_season["issues"]:
        print(f"[WARNING] Season table issues: {v_season['issues']}")
    if v_match["issues"]:
        print(f"[WARNING] Match table issues: {v_match['issues']}")

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

    # External features (squad value + manager changes)
    squad_df = load_squad_value("data/external/team_squad_value_2024.csv")
    manager_df = load_manager_change_flags("data/external/manager_change_flags_2024.csv")

    supervised_df = supervised_df.merge(squad_df, on="team", how="left")
    supervised_df = supervised_df.merge(manager_df, on="team", how="left")
    supervised_df["manager_change_flag"] = (
        pd.to_numeric(supervised_df["manager_change_flag"], errors="coerce")
        .fillna(0).astype(int)
    )

    # Fill missing squad values with promoted-team average for backtesting
    promoted_avg_sv = supervised_df.loc[
        supervised_df["promoted_team_flag"] == 1, "squad_value_million"
    ].mean()
    supervised_df["squad_value_million"] = supervised_df["squad_value_million"].fillna(promoted_avg_sv)

    return supervised_df, season_df


def select_features(supervised_df: pd.DataFrame) -> list:
    forbidden = {
        "team", "notes", "points", "position", "gf", "ga", "gd",
        "played", "won", "drawn", "lost",
        "target_points", "target_rank", "form_points", "form_gd",
        "h2h_ppg",  # lag version (prev_h2h_ppg) is the safe feature
    }
    feature_columns = [
        c for c in supervised_df.columns
        if c not in forbidden and pd.api.types.is_numeric_dtype(supervised_df[c])
        and not c.startswith("match_")   # prevent current-season leakage
    ]
    return feature_columns


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EPL Forecast Pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--no-sim", action="store_true", help="Skip Monte Carlo simulation")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Pipeline] Building features…")
    supervised_df, season_df = build_supervised(config)
    feature_columns = select_features(supervised_df)
    print(f"[Pipeline] {len(feature_columns)} features selected.")

    # ----- Backtest -----
    if args.backtest:
        print(f"[Pipeline] Running walk-forward backtest ({config.model_type})…")
        backtester = WalkForwardBacktester(
            model_type=config.model_type,
            min_train_season=config.min_train_season,
            random_state=config.random_state,
        )
        backtest_df = backtester.run(supervised_df, feature_columns, config.predict_season)

        out_path = os.path.join(config.output_dir, "backtest_results.csv")
        backtest_df.to_csv(out_path, index=False)
        print(f"[Pipeline] Backtest saved → {out_path}")

        adv = backtest_df[backtest_df["model_name"].str.startswith("advanced")]
        print("\n=== Advanced Model Backtest Summary ===")
        print(adv[["test_season", "rmse", "mae", "spearman_rank_corr",
                    "top4_accuracy", "relegation_accuracy",
                    "brier_top4", "brier_relegation"]].to_string(index=False))

    # ----- Forecast -----
    print(f"\n[Pipeline] Forecasting {config.predict_season}…")
    forecaster = SeasonForecaster(
        model_type=config.model_type,
        random_state=config.random_state,
        tune_hyperparams=config.tune_hyperparams,
    )
    forecast = forecaster.forecast(
        supervised_df,
        feature_columns,
        predict_season=config.predict_season,
        promoted_teams=config.promoted_teams,
    )
    print(forecast.to_string(index=False))

    forecast_path = os.path.join(config.output_dir, "forecast.csv")
    forecast.to_csv(forecast_path, index=False)

    # ----- Simulation -----
    if not args.no_sim:
        print(f"\n[Pipeline] Running {config.n_sims} simulations…")
        simulator = SeasonSimulator(
            random_state=config.random_state,
            use_poisson=config.use_poisson,
        )
        sim_summary = simulator.simulate_many(forecast, n_sims=config.n_sims)
        sim_path = os.path.join(config.output_dir, "simulation_summary.csv")
        sim_summary.to_csv(sim_path, index=False)
        print(f"[Pipeline] Simulation saved → {sim_path}")
        print(sim_summary[["team", "avg_points", "avg_rank",
                            "title_prob", "top4_prob", "relegation_prob"]].to_string(index=False))

    # ----- Feature importances -----
    if hasattr(forecaster, "model_"):
        fi = forecaster.model_.feature_importances()
        if fi is not None:
            fi_path = os.path.join(config.output_dir, "feature_importances.csv")
            fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(fi_path, index=False)
            print(f"\n[Pipeline] Top 10 features:\n{fi.head(10).to_string()}")


if __name__ == "__main__":
    main()
