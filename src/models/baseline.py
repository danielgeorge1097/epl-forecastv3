import pandas as pd


class BaselineModel:
    def fit(self, train_df: pd.DataFrame):
        league_avg = train_df.groupby("season_end_year")["target_points"].mean().mean()
        promoted_avg = train_df.loc[train_df["promoted_team_flag"] == 1, "target_points"].mean()
        self.default_points_ = float(promoted_avg if pd.notna(promoted_avg) else league_avg)
        return self

    def predict(self, x_df: pd.DataFrame):
        preds = x_df["prev_points"].copy()
        preds = preds.fillna(self.default_points_)
        return preds.to_numpy(dtype=float)