from __future__ import annotations

from typing import Dict, List
import pandas as pd


class DataValidator:
    def validate_season_table(self, season_df: pd.DataFrame) -> Dict[str, object]:
        issues: List[str] = []

        if season_df["season_end_year"].isna().any():
            issues.append("Some season_end_year values could not be parsed.")

        duplicate_count = season_df.duplicated(subset=["season_end_year", "team"]).sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} duplicate season/team rows.")

        null_team_count = (season_df["team"] == "UNKNOWN_TEAM").sum()
        if null_team_count > 0:
            issues.append(f"Found {null_team_count} rows with UNKNOWN_TEAM.")

        teams_per_season = season_df.groupby("season_end_year")["team"].nunique()
        unusual_team_counts = teams_per_season[(teams_per_season < 18) | (teams_per_season > 24)]
        if not unusual_team_counts.empty:
            issues.append(
                "Some seasons have suspicious team counts outside expected historical EPL range."
            )

        return {
            "row_count": int(len(season_df)),
            "season_min": int(season_df["season_end_year"].min()),
            "season_max": int(season_df["season_end_year"].max()),
            "teams_per_season": teams_per_season.to_dict(),
            "issues": issues,
        }

    def validate_match_table(self, match_df: pd.DataFrame) -> Dict[str, object]:
        issues: List[str] = []

        if match_df["season_end_year"].isna().any():
            issues.append("Some match seasons could not be parsed.")

        null_home = (match_df["home_team"] == "UNKNOWN_TEAM").sum()
        null_away = (match_df["away_team"] == "UNKNOWN_TEAM").sum()
        if null_home > 0 or null_away > 0:
            issues.append(
                f"UNKNOWN_TEAM values found in match table: home={null_home}, away={null_away}."
            )

        season_match_counts = match_df.groupby("season_end_year").size()

        suspicious_counts = season_match_counts[(season_match_counts < 300) | (season_match_counts > 420)]
        if not suspicious_counts.empty:
            issues.append(
                "Some seasons have suspicious match counts. Incomplete or non-EPL rows may exist."
            )

        return {
            "row_count": int(len(match_df)),
            "season_min": int(match_df["season_end_year"].min()),
            "season_max": int(match_df["season_end_year"].max()),
            "season_match_counts": season_match_counts.to_dict(),
            "issues": issues,
        }

    def validate_team_overlap(self, season_df: pd.DataFrame, match_df: pd.DataFrame) -> Dict[str, object]:
        season_teams = set(season_df["team"].unique())
        match_teams = set(match_df["home_team"].unique()).union(set(match_df["away_team"].unique()))

        only_in_season = sorted(season_teams - match_teams)
        only_in_match = sorted(match_teams - season_teams)

        return {
            "season_only_teams": only_in_season,
            "match_only_teams": only_in_match,
            "perfect_overlap": len(only_in_season) == 0 and len(only_in_match) == 0,
        }