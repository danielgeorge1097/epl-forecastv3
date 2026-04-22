from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.baseline import BaselineModel
from src.models.tree_model import TreePointsModel


class WalkForwardBacktester:
    def __init__(
        self,
        model_type: str = "hgbt",
        min_train_season: int = 2001,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.min_train_season = min_train_season
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rank_predictions(points_pred: np.ndarray, teams: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"team": teams.values, "predicted_points": points_pred})
        df = df.sort_values(["predicted_points", "team"], ascending=[False, True]).reset_index(drop=True)
        df["predicted_rank"] = range(1, len(df) + 1)
        return df

    @staticmethod
    def overlap_ratio(a, b):
        return len(set(a) & set(b)) / len(set(a)) if a else np.nan

    @staticmethod
    def brier_score(actual_flags: np.ndarray, predicted_probs: np.ndarray) -> float:
        """
        Brier score for a binary outcome.
        Lower is better (0 = perfect, 1 = worst).
        Here predicted_probs is deterministic (0 or 1 from rank threshold),
        so this measures classification calibration.
        """
        return float(np.mean((predicted_probs - actual_flags) ** 2))

    def evaluate_one(
        self,
        test_df: pd.DataFrame,
        preds: np.ndarray,
        model_name: str,
        train_max_season: int,
    ) -> dict:
        ranked = self.rank_predictions(preds, test_df["team"])
        merged = test_df[["team", "target_points", "target_rank"]].merge(ranked, on="team", how="left")

        # Sort helpers
        by_actual = merged.sort_values("target_rank")
        by_pred = merged.sort_values("predicted_rank")

        actual_top4 = by_actual.head(4)["team"].tolist()
        pred_top4 = by_pred.head(4)["team"].tolist()
        actual_bottom3 = by_actual.tail(3)["team"].tolist()
        pred_bottom3 = by_pred.tail(3)["team"].tolist()
        actual_champion = by_actual.iloc[0]["team"]
        pred_champion = by_pred.iloc[0]["team"]

        # Brier scores (deterministic 0/1 predictions from rank)
        merged["actual_top4_flag"] = (merged["target_rank"] <= 4).astype(float)
        merged["pred_top4_flag"] = (merged["predicted_rank"] <= 4).astype(float)
        merged["actual_rel_flag"] = (merged["target_rank"] >= merged["target_rank"].max() - 2).astype(float)
        merged["pred_rel_flag"] = (merged["predicted_rank"] >= merged["predicted_rank"].max() - 2).astype(float)

        brier_top4 = self.brier_score(
            merged["actual_top4_flag"].values, merged["pred_top4_flag"].values
        )
        brier_relegation = self.brier_score(
            merged["actual_rel_flag"].values, merged["pred_rel_flag"].values
        )

        return {
            "train_max_season": int(train_max_season),
            "test_season": int(test_df["season_end_year"].iloc[0]),
            "model_name": model_name,
            "rmse": float(np.sqrt(mean_squared_error(merged["target_points"], merged["predicted_points"]))),
            "mae": float(mean_absolute_error(merged["target_points"], merged["predicted_points"])),
            "spearman_rank_corr": float(
                merged["target_rank"].corr(merged["predicted_rank"], method="spearman")
            ),
            "top4_accuracy": float(self.overlap_ratio(actual_top4, pred_top4)),
            "relegation_accuracy": float(self.overlap_ratio(actual_bottom3, pred_bottom3)),
            "champion_hit": float(actual_champion == pred_champion),
            "brier_top4": brier_top4,
            "brier_relegation": brier_relegation,
        }

    def run(
        self,
        supervised_df: pd.DataFrame,
        feature_columns: list,
        forecast_season: int,
    ) -> pd.DataFrame:
        rows = []
        seasons = sorted(supervised_df["season_end_year"].dropna().unique())
        test_seasons = [
            s for s in seasons
            if s < forecast_season and s >= self.min_train_season + 1
        ]

        for test_season in test_seasons:
            train_df = supervised_df[supervised_df["season_end_year"] < test_season].copy()
            test_df = supervised_df[supervised_df["season_end_year"] == test_season].copy()

            if train_df["season_end_year"].max() < self.min_train_season:
                continue

            # Baseline
            baseline = BaselineModel().fit(train_df)
            base_preds = baseline.predict(test_df)
            rows.append(
                self.evaluate_one(test_df, base_preds, "baseline_prev_points", train_df["season_end_year"].max())
            )

            # Advanced model
            advanced = TreePointsModel(model_type=self.model_type, random_state=self.random_state)
            advanced.fit(train_df, feature_columns)
            adv_preds = advanced.predict(test_df)
            rows.append(
                self.evaluate_one(test_df, adv_preds, f"advanced_{self.model_type}", train_df["season_end_year"].max())
            )

        return pd.DataFrame(rows)
