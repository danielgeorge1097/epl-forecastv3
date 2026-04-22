import yaml
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    # data
    season_table: str
    match_table: str
    # model
    model_type: str
    tune_hyperparams: bool
    # training
    min_train_season: int
    # forecast
    predict_season: int
    promoted_teams: List[str]
    # features
    use_match_features: bool
    rolling_windows: list
    use_home_away_splits: bool
    use_h2h_features: bool
    # simulation
    n_sims: int
    use_poisson: bool
    # output
    output_dir: str
    # system
    random_state: int


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return Config(
        season_table=cfg["data"]["season_table"],
        match_table=cfg["data"]["match_table"],
        model_type=cfg["model"]["type"],
        tune_hyperparams=cfg["model"].get("tune_hyperparams", False),
        min_train_season=cfg["training"]["min_train_season"],
        predict_season=cfg["forecast"]["predict_season_end_year"],
        promoted_teams=cfg["forecast"].get("promoted_teams", []),
        use_match_features=cfg["features"]["use_match_features"],
        rolling_windows=cfg["features"]["rolling_windows"],
        use_home_away_splits=cfg["features"].get("use_home_away_splits", True),
        use_h2h_features=cfg["features"].get("use_h2h_features", True),
        n_sims=cfg["simulation"].get("n_sims", 5000),
        use_poisson=cfg["simulation"].get("use_poisson", True),
        output_dir=cfg["output"]["dir"],
        random_state=cfg["system"]["random_state"],
    )
