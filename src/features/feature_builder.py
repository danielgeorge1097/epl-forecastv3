from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


class FeatureBuilder:
    def __init__(self, rolling_windows=(2, 3)):
        self.rolling_windows = rolling_windows

    # ------------------------------------------------------------------
    # Match-level aggregates (season totals + split by home/away)
    # ------------------------------------------------------------------

    def build_match_derived_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        def _make_long(df):
            home = pd.DataFrame({
                "season_end_year": df["season_end_year"],
                "team": df["home_team"],
                "venue": "home",
                "gf": df["home_goals"],
                "ga": df["away_goals"],
                "shots_for": df.get("home_shots"),
                "shots_against": df.get("away_shots"),
                "sot_for": df.get("home_shots_on_target"),
                "sot_against": df.get("away_shots_on_target"),
                "corners_for": df.get("home_corners"),
                "corners_against": df.get("away_corners"),
                "fouls_for": df.get("home_fouls"),
                "fouls_against": df.get("away_fouls"),
                "yellows": df.get("home_yellows"),
                "reds": df.get("home_reds"),
            })
            away = pd.DataFrame({
                "season_end_year": df["season_end_year"],
                "team": df["away_team"],
                "venue": "away",
                "gf": df["away_goals"],
                "ga": df["home_goals"],
                "shots_for": df.get("away_shots"),
                "shots_against": df.get("home_shots"),
                "sot_for": df.get("away_shots_on_target"),
                "sot_against": df.get("home_shots_on_target"),
                "corners_for": df.get("away_corners"),
                "corners_against": df.get("home_corners"),
                "fouls_for": df.get("away_fouls"),
                "fouls_against": df.get("home_fouls"),
                "yellows": df.get("away_yellows"),
                "reds": df.get("away_reds"),
            })
            return pd.concat([home, away], ignore_index=True)

        long_df = _make_long(match_df)
        long_df["points"] = np.select(
            [long_df["gf"] > long_df["ga"], long_df["gf"] == long_df["ga"]],
            [3, 1],
            default=0,
        )

        # Overall season aggregates
        agg = (
            long_df.groupby(["season_end_year", "team"])
            .agg(
                matches=("team", "size"),
                match_points=("points", "sum"),
                match_gf=("gf", "sum"),
                match_ga=("ga", "sum"),
                match_shots_for=("shots_for", "mean"),
                match_shots_against=("shots_against", "mean"),
                match_sot_for=("sot_for", "mean"),
                match_sot_against=("sot_against", "mean"),
                match_corners_for=("corners_for", "mean"),
                match_corners_against=("corners_against", "mean"),
                match_fouls_for=("fouls_for", "mean"),
                match_fouls_against=("fouls_against", "mean"),
                match_yellows=("yellows", "mean"),
                match_reds=("reds", "mean"),
            )
            .reset_index()
        )
        agg["match_gd"] = agg["match_gf"] - agg["match_ga"]
        agg["match_ppg"] = agg["match_points"] / agg["matches"]

        # Home / Away splits — much more predictive than combined
        home_df = long_df[long_df["venue"] == "home"]
        away_df = long_df[long_df["venue"] == "away"]

        def _split_agg(split, prefix):
            s = (
                split.groupby(["season_end_year", "team"])
                .agg(
                    pts=("points", "sum"),
                    gf=("gf", "sum"),
                    ga=("ga", "sum"),
                    n=("team", "size"),
                )
                .reset_index()
            )
            s[f"{prefix}_ppg"] = s["pts"] / s["n"]
            s[f"{prefix}_gf_pg"] = s["gf"] / s["n"]
            s[f"{prefix}_ga_pg"] = s["ga"] / s["n"]
            s[f"{prefix}_gd_pg"] = (s["gf"] - s["ga"]) / s["n"]
            s[f"{prefix}_win_rate"] = (
                split.groupby(["season_end_year", "team"])
                .apply(lambda g: (g["gf"] > g["ga"]).mean())
                .reset_index(drop=True)
            )
            return s[["season_end_year", "team",
                       f"{prefix}_ppg", f"{prefix}_gf_pg",
                       f"{prefix}_ga_pg", f"{prefix}_gd_pg",
                       f"{prefix}_win_rate"]]

        home_agg = _split_agg(home_df, "home")
        away_agg = _split_agg(away_df, "away")

        agg = agg.merge(home_agg, on=["season_end_year", "team"], how="left")
        agg = agg.merge(away_agg, on=["season_end_year", "team"], how="left")

        return agg

    # ------------------------------------------------------------------
    # H2H features — average points vs top-half teams
    # ------------------------------------------------------------------

    def build_h2h_features(self, match_df: pd.DataFrame, season_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each team-season, compute avg points earned against teams that
        finished in the top half of the league that season.
        This captures 'big game' performance.
        """
        # Get top-half teams per season
        median_pos = season_df.groupby("season_end_year")["position"].transform("median")
        top_half = season_df[season_df["position"] <= median_pos][["season_end_year", "team"]].copy()
        top_half_set = set(zip(top_half["season_end_year"], top_half["team"]))

        def _result_pts(gf, ga):
            if gf > ga:
                return 3
            if gf == ga:
                return 1
            return 0

        rows = []
        for _, row in match_df.iterrows():
            sy = row["season_end_year"]
            ht, at = row["home_team"], row["away_team"]
            hg, ag = row["home_goals"], row["away_goals"]

            # Home team vs top-half opponent
            if (sy, at) in top_half_set:
                rows.append({"season_end_year": sy, "team": ht,
                             "h2h_pts": _result_pts(hg, ag)})
            # Away team vs top-half opponent
            if (sy, ht) in top_half_set:
                rows.append({"season_end_year": sy, "team": at,
                             "h2h_pts": _result_pts(ag, hg)})

        if not rows:
            return pd.DataFrame(columns=["season_end_year", "team", "h2h_ppg"])

        h2h_df = pd.DataFrame(rows)
        h2h_agg = (
            h2h_df.groupby(["season_end_year", "team"])["h2h_pts"]
            .mean()
            .reset_index()
            .rename(columns={"h2h_pts": "h2h_ppg"})
        )
        return h2h_agg

    # ------------------------------------------------------------------
    # Season-level features with rolling lags
    # ------------------------------------------------------------------

    def build_team_season_features(
        self,
        season_df: pd.DataFrame,
        match_features: Optional[pd.DataFrame] = None,
        form_features: Optional[pd.DataFrame] = None,
        h2h_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        df = season_df.copy().sort_values(["team", "season_end_year"]).reset_index(drop=True)

        df["points_per_game"] = df["points"] / df["played"]
        df["gf_per_game"] = df["gf"] / df["played"]
        df["ga_per_game"] = df["ga"] / df["played"]
        df["gd_per_game"] = df["gd"] / df["played"]

        lag_cols = [
            "points", "position", "gd", "gf", "ga",
            "points_per_game", "gf_per_game", "ga_per_game", "gd_per_game",
        ]
        for col in lag_cols:
            df[f"prev_{col}"] = df.groupby("team")[col].shift(1)

        df["shock_flag"] = (
            (df["prev_points"].fillna(df["points"]) - df["points"]).abs() > 15
        ).astype(int)

        for window in self.rolling_windows:
            grouped = df.groupby("team")
            for col in ["points", "position", "gd", "points_per_game", "gf_per_game", "ga_per_game"]:
                df[f"roll{window}_{col}_mean"] = (
                    grouped[col].shift(1).rolling(window=window, min_periods=1).mean()
                    .reset_index(level=0, drop=True)
                )
                df[f"roll{window}_{col}_std"] = (
                    grouped[col].shift(1).rolling(window=window, min_periods=1).std()
                    .reset_index(level=0, drop=True)
                )

        df["prev_season_end_year"] = df.groupby("team")["season_end_year"].shift(1)
        df["was_in_prev_epl_season"] = (df["prev_season_end_year"] == df["season_end_year"] - 1).astype(int)
        df["promoted_team_flag"] = 1 - df["was_in_prev_epl_season"]

        df["career_avg_points_entering"] = (
            df.groupby("team")["points"].expanding().mean().shift(1)
            .reset_index(level=0, drop=True)
        )

        # Match-level aggregates + lagged home/away splits
        if match_features is not None:
            df = df.merge(match_features, on=["season_end_year", "team"], how="left")
            match_cols = [c for c in match_features.columns if c not in ["season_end_year", "team"]]
            df = df.sort_values(["team", "season_end_year"]).reset_index(drop=True)
            for col in match_cols:
                df[f"prev_{col}"] = df.groupby("team")[col].shift(1)

        # Form features
        if form_features is not None:
            df = df.merge(form_features, on=["season_end_year", "team"], how="left")
            df = df.sort_values(["team", "season_end_year"]).reset_index(drop=True)
            for col in ["form_points", "form_gd"]:
                df[f"prev_{col}"] = df.groupby("team")[col].shift(1)

        # H2H features
        if h2h_features is not None:
            df = df.merge(h2h_features, on=["season_end_year", "team"], how="left")
            df = df.sort_values(["team", "season_end_year"]).reset_index(drop=True)
            df["prev_h2h_ppg"] = df.groupby("team")["h2h_ppg"].shift(1)

        return df

    def make_supervised_frame(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        df = feature_df.copy()
        df["target_points"] = df["points"]
        df["target_rank"] = df["position"]
        return df
