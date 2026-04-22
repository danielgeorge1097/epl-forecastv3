from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


class SeasonSimulator:
    """
    Monte Carlo season simulator with two modes:
      - Poisson goal model (use_poisson=True, default): samples home/away goals from
        Poisson distributions with attack/defense ratings derived from predicted points.
        Tiebreaking follows EPL rules: points → GD → GF → alphabetical.
      - Legacy logistic model (use_poisson=False): backward-compatible with old approach.
    """

    # EPL historical calibration constants
    HOME_ADVANTAGE = 1.30      # home team scores ~30% more goals on average
    MEAN_GOALS_PG = 1.45       # average goals per team per game (EPL historical)
    STRENGTH_SCALE = 0.022     # how aggressively pts diff maps to goal rate diff

    def __init__(
        self,
        random_state: int = 42,
        use_poisson: bool = True,
        strength_noise_std: float = 6.0,  # legacy param kept for backward compat
    ):
        self.rng = np.random.default_rng(random_state)
        self.use_poisson = use_poisson
        self.strength_noise_std = strength_noise_std

    # ------------------------------------------------------------------
    # Poisson model
    # ------------------------------------------------------------------

    def _compute_ratings(self, forecast_df: pd.DataFrame) -> dict:
        """
        Map predicted_points → (attack_rating, defense_rating) per team.
        Uses exponential scaling so a 10-pt gap yields ~20% goal rate diff.
        """
        pts = forecast_df["predicted_points"].values
        mean_pts = pts.mean()
        k = self.STRENGTH_SCALE

        attack = self.MEAN_GOALS_PG * np.exp(k * (pts - mean_pts))
        defense = self.MEAN_GOALS_PG * np.exp(-k * (pts - mean_pts))

        return {
            row["team"]: {"attack": attack[i], "defense": defense[i]}
            for i, (_, row) in enumerate(forecast_df.iterrows())
        }

    def _simulate_one_poisson(self, teams: list, ratings: dict) -> pd.DataFrame:
        """Simulate a full season with Poisson goal sampling."""
        table = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

        for home in teams:
            for away in teams:
                if home == away:
                    continue

                lam_h = ratings[home]["attack"] * ratings[away]["defense"] * self.HOME_ADVANTAGE
                lam_a = ratings[away]["attack"] * ratings[home]["defense"]

                hg = int(self.rng.poisson(lam_h))
                ag = int(self.rng.poisson(lam_a))

                table[home]["gf"] += hg
                table[away]["gf"] += ag
                table[home]["ga"] += ag   # home concedes away goals
                table[away]["ga"] += hg   # away concedes home goals

                if hg > ag:
                    table[home]["pts"] += 3
                elif hg == ag:
                    table[home]["pts"] += 1
                    table[away]["pts"] += 1
                else:
                    table[away]["pts"] += 3

        rows = [
            {
                "team": t,
                "sim_points": v["pts"],
                "sim_gf": v["gf"],
                "sim_gd": v["gf"] - v["ga"],
            }
            for t, v in table.items()
        ]
        df = pd.DataFrame(rows)
        # EPL tiebreaking: pts → GD → GF → alphabetical
        df = df.sort_values(
            ["sim_points", "sim_gd", "sim_gf", "team"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        df["sim_rank"] = np.arange(1, len(df) + 1)
        return df

    # ------------------------------------------------------------------
    # Legacy logistic model (kept for backward compat)
    # ------------------------------------------------------------------

    def sample_team_strength_map(self, forecast_df: pd.DataFrame) -> dict:
        sampled = {}
        for _, row in forecast_df.iterrows():
            noisy = row["predicted_points"] + self.rng.normal(0, self.strength_noise_std)
            sampled[row["team"]] = noisy
        return sampled

    def match_probabilities(self, home_strength: float, away_strength: float):
        gap = (home_strength + 2.5) - away_strength
        home_edge = 1 / (1 + np.exp(-gap / 12))
        away_edge = 1 - home_edge
        draw_p = max(0.16, 0.30 - abs(gap) * 0.0025)
        home_p = home_edge * (1 - draw_p)
        away_p = away_edge * (1 - draw_p)
        total = home_p + draw_p + away_p
        return home_p / total, draw_p / total, away_p / total

    def _simulate_one_legacy(self, teams: list, strength_map: dict) -> pd.DataFrame:
        table = {t: 0 for t in teams}
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                hp, dp, ap = self.match_probabilities(strength_map[home], strength_map[away])
                outcome = self.rng.choice(["H", "D", "A"], p=[hp, dp, ap])
                if outcome == "H":
                    table[home] += 3
                elif outcome == "D":
                    table[home] += 1
                    table[away] += 1
                else:
                    table[away] += 3

        df = pd.DataFrame({"team": list(table), "sim_points": list(table.values())})
        df = df.sort_values(["sim_points", "team"], ascending=[False, True]).reset_index(drop=True)
        df["sim_rank"] = np.arange(1, len(df) + 1)
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_many(self, forecast_df: pd.DataFrame, n_sims: int = 1000) -> pd.DataFrame:
        teams = forecast_df["team"].tolist()
        all_rows = []

        if self.use_poisson:
            ratings = self._compute_ratings(forecast_df)

        for sim_id in range(n_sims):
            if self.use_poisson:
                # Re-sample ratings with noise each run for uncertainty propagation
                noisy_pts = forecast_df.copy()
                noisy_pts["predicted_points"] = (
                    forecast_df["predicted_points"].values
                    + self.rng.normal(0, self.strength_noise_std, len(forecast_df))
                )
                ratings = self._compute_ratings(noisy_pts)
                sim_table = self._simulate_one_poisson(teams, ratings)
            else:
                strength_map = self.sample_team_strength_map(forecast_df)
                sim_table = self._simulate_one_legacy(teams, strength_map)

            sim_table["simulation"] = sim_id
            all_rows.append(sim_table)

        sims = pd.concat(all_rows, ignore_index=True)

        agg_cols = {"avg_points": ("sim_points", "mean"), "avg_rank": ("sim_rank", "mean")}
        if "sim_gd" in sims.columns:
            agg_cols["avg_gd"] = ("sim_gd", "mean")

        summary = (
            sims.groupby("team")
            .agg(
                avg_points=("sim_points", "mean"),
                avg_rank=("sim_rank", "mean"),
                title_prob=("sim_rank", lambda s: (s == 1).mean()),
                top4_prob=("sim_rank", lambda s: (s <= 4).mean()),
                top6_prob=("sim_rank", lambda s: (s <= 6).mean()),
                relegation_prob=("sim_rank", lambda s: (s >= 18).mean()),
                **({ "avg_gd": ("sim_gd", "mean")} if "sim_gd" in sims.columns else {}),
            )
            .reset_index()
            .sort_values(["avg_rank", "avg_points"], ascending=[True, False])
            .reset_index(drop=True)
        )
        return summary
