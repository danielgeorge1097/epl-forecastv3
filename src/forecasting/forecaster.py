from __future__ import annotations

from typing import List
import pandas as pd

from src.models.tree_model import TreePointsModel


class SeasonForecaster:
    """
    Trains on all historical seasons and produces a point forecast
    for the target season. Promoted teams are passed in from config
    rather than being hardcoded.
    """

    def __init__(
        self,
        model_type: str = "hgbt",
        random_state: int = 42,
        memory=None,
        tune_hyperparams: bool = False,
    ):
        self.model_type = model_type
        self.random_state = random_state
        self.memory = memory
        self.tune_hyperparams = tune_hyperparams

    def build_forecast_frame(
        self,
        supervised_df: pd.DataFrame,
        predict_season: int,
        promoted_teams: List[str],
    ) -> pd.DataFrame:
        """
        Build the prediction frame for `predict_season`.

        Teams = top-17 from last completed season + promoted_teams.
        Each team's feature row is their latest available history row,
        with season_end_year bumped to predict_season.
        """
        last_season = predict_season - 1
        last_completed = supervised_df[supervised_df["season_end_year"] == last_season].copy()

        if last_completed.empty:
            raise ValueError(
                f"No data found for season {last_season}. "
                "Cannot build forecast frame."
            )

        staying_up = last_completed[last_completed["target_rank"] <= 17]["team"].tolist()
        target_teams = list(dict.fromkeys(staying_up + promoted_teams))  # preserve order, dedupe

        history = supervised_df[supervised_df["team"].isin(target_teams)].copy()
        history = history.sort_values(["team", "season_end_year"])

        latest_rows = history.groupby("team", as_index=False).tail(1).copy()

        # Teams we still couldn't find any history for (brand-new promoted clubs)
        missing = set(target_teams) - set(latest_rows["team"])
        if missing:
            # Fill with league-average promoted-team stats
            promoted_avg = (
                supervised_df[supervised_df["promoted_team_flag"] == 1]
                .select_dtypes(include="number")
                .mean()
            )
            filler_rows = []
            for t in missing:
                row = promoted_avg.to_dict()
                row["team"] = t
                filler_rows.append(row)
            latest_rows = pd.concat(
                [latest_rows, pd.DataFrame(filler_rows)], ignore_index=True
            )

        latest_rows["season_end_year"] = predict_season
        latest_rows["target_points"] = pd.NA
        latest_rows["target_rank"] = pd.NA

        return latest_rows

    def forecast(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        predict_season: int,
        promoted_teams: List[str] = None,
    ) -> pd.DataFrame:
        train_df = df[df["season_end_year"] < predict_season].copy()
        pred_df = self.build_forecast_frame(
            df, predict_season, promoted_teams or []
        )

        model = TreePointsModel(
            model_type=self.model_type,
            random_state=self.random_state,
            memory=self.memory,
            tune_hyperparams=self.tune_hyperparams,
        )
        model.fit(train_df, feature_columns)
        preds = model.predict(pred_df)

        pred_df["predicted_points"] = preds
        pred_df = pred_df.sort_values(
            ["predicted_points", "team"], ascending=[False, True]
        ).reset_index(drop=True)
        pred_df["predicted_rank"] = range(1, len(pred_df) + 1)

        self.model_ = model  # store for feature importance access
        return pred_df[["team", "predicted_points", "predicted_rank"]]
