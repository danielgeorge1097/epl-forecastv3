from __future__ import annotations

import pandas as pd


class FormFeatureBuilder:
    def build(self, match_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        df = match_df.copy()

        home = pd.DataFrame({
            "team": df["home_team"],
            "date": df["date"],
            "season_end_year": df["season_end_year"],
            "gf": df["home_goals"],
            "ga": df["away_goals"],
        })

        away = pd.DataFrame({
            "team": df["away_team"],
            "date": df["date"],
            "season_end_year": df["season_end_year"],
            "gf": df["away_goals"],
            "ga": df["home_goals"],
        })

        long_df = pd.concat([home, away], ignore_index=True)
        long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

        long_df["points"] = (
            (long_df["gf"] > long_df["ga"]) * 3
            + (long_df["gf"] == long_df["ga"]) * 1
        )

        long_df["goal_diff"] = long_df["gf"] - long_df["ga"]

        long_df["form_points"] = (
            long_df.groupby("team")["points"]
            .rolling(window, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )

        long_df["form_gd"] = (
            long_df.groupby("team")["goal_diff"]
            .rolling(window, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )

        out = (
            long_df.sort_values("date")
            .groupby(["season_end_year", "team"], as_index=False)
            .tail(1)[["season_end_year", "team", "form_points", "form_gd"]]
            .reset_index(drop=True)
        )

        return out