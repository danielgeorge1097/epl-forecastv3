from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from src.data.normalizer import canonical_team_name, season_string_to_end_year, DEFAULT_TEAM_MAP


class DataLoader:
    REQUIRED_SEASON_COLUMNS = {
        "season_end_year",
        "team",
        "position",
        "played",
        "won",
        "drawn",
        "lost",
        "gf",
        "ga",
        "gd",
        "points",
    }

    MATCH_COLUMN_ALIASES = {
        "season": ["Season", "season"],
        "date": ["MatchDate", "Date", "date"],
        "home_team": ["HomeTeam", "home_team"],
        "away_team": ["AwayTeam", "away_team"],
        "home_goals": ["FTHG", "FullTimeHomeGoals", "home_goals"],
        "away_goals": ["FTAG", "FullTimeAwayGoals", "away_goals"],
        "result": ["FTR", "FullTimeResult", "result"],
        "home_shots": ["HS", "HomeShots", "home_shots"],
        "away_shots": ["AS", "AwayShots", "away_shots"],
        "home_shots_on_target": ["HST", "HomeShotsOnTarget", "home_shots_on_target"],
        "away_shots_on_target": ["AST", "AwayShotsOnTarget", "away_shots_on_target"],
        "home_corners": ["HC", "HomeCorners", "home_corners"],
        "away_corners": ["AC", "AwayCorners", "away_corners"],
        "home_fouls": ["HF", "HomeFouls", "home_fouls"],
        "away_fouls": ["AF", "AwayFouls", "away_fouls"],
        "home_yellows": ["HY", "HomeYellowCards", "home_yellows"],
        "away_yellows": ["AY", "AwayYellowCards", "away_yellows"],
        "home_reds": ["HR", "HomeRedCards", "home_reds"],
        "away_reds": ["AR", "AwayRedCards", "away_reds"],
    }

    def __init__(self, team_map: Optional[Dict[str, str]] = None):
        self.team_map = team_map or DEFAULT_TEAM_MAP

    def load_season_table(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        missing = self.REQUIRED_SEASON_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Season table missing required columns: {missing}")

        out = df.copy()
        out["season_end_year"] = out["season_end_year"].apply(season_string_to_end_year)
        out["team"] = out["team"].apply(lambda x: canonical_team_name(x, self.team_map))

        numeric_cols = ["position", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        return out

    def load_match_table(self, path: str) -> pd.DataFrame:
        raw = pd.read_csv(path)

        rename_map = {}
        for canonical, aliases in self.MATCH_COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in raw.columns:
                    rename_map[alias] = canonical
                    break

        out = raw.rename(columns=rename_map).copy()

        required = {"season", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(out.columns)
        if missing:
            raise ValueError(f"Match table missing columns: {missing}")

        out["season_end_year"] = out["season"].apply(season_string_to_end_year)
        out["home_team"] = out["home_team"].apply(lambda x: canonical_team_name(x, self.team_map))
        out["away_team"] = out["away_team"].apply(lambda x: canonical_team_name(x, self.team_map))

        return out