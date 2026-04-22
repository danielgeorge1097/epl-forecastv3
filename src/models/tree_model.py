from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("hgbt", "rf", "lgbm", "xgb")


class TreePointsModel:
    """
    Unified model wrapper supporting HGBT, RF, LightGBM, and XGBoost.
    Optionally tunes hyperparameters with Optuna.
    """

    def __init__(
        self,
        model_type: str = "hgbt",
        random_state: int = 42,
        memory: Optional[str] = None,
        tune_hyperparams: bool = False,
        n_trials: int = 50,
    ):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. Supported: {SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self.random_state = random_state
        self.memory = memory
        self.tune_hyperparams = tune_hyperparams
        self.n_trials = n_trials
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns_: List[str] = []
        self.best_params_: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_estimator(self, params: Optional[dict] = None):
        p = params or {}
        if self.model_type == "rf":
            return RandomForestRegressor(
                n_estimators=p.get("n_estimators", 300),
                min_samples_leaf=p.get("min_samples_leaf", 2),
                max_features=p.get("max_features", "sqrt"),
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "hgbt":
            return HistGradientBoostingRegressor(
                max_depth=p.get("max_depth", 6),
                learning_rate=p.get("learning_rate", 0.03),
                max_iter=p.get("max_iter", 300),
                min_samples_leaf=p.get("min_samples_leaf", 5),
                l2_regularization=p.get("l2_regularization", 0.0),
                random_state=self.random_state,
            )
        elif self.model_type == "lgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("lightgbm not installed. Run: pip install lightgbm")
            return lgb.LGBMRegressor(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.03),
                max_depth=p.get("max_depth", 6),
                min_child_samples=p.get("min_child_samples", 10),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                reg_lambda=p.get("reg_lambda", 1.0),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif self.model_type == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("xgboost not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.03),
                max_depth=p.get("max_depth", 6),
                min_child_weight=p.get("min_child_weight", 3),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                reg_lambda=p.get("reg_lambda", 1.0),
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

    def _build_pipeline(self, params: Optional[dict] = None) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ],
                        memory=self.memory,
                    ),
                    self.feature_columns_,
                )
            ],
            remainder="drop",
        )
        return Pipeline(
            [("prep", preprocessor), ("model", self._make_estimator(params))],
            memory=self.memory,
        )

    def _tune_with_optuna(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run Optuna to find best hyperparameters via 5-fold CV."""
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("optuna not installed; skipping tuning.")
            return {}

        def objective(trial):
            if self.model_type == "hgbt":
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                    "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
                }
            elif self.model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
                }
            elif self.model_type == "lgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                }
            elif self.model_type == "xgb":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                }
            else:
                params = {}

            pipeline = self._build_pipeline(params)
            scores = cross_val_score(
                pipeline, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            return -scores.mean()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"Best Optuna RMSE: {study.best_value:.2f} | params: {study.best_params}")
        return study.best_params

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, feature_columns: List[str]):
        self.feature_columns_ = feature_columns
        X = train_df[feature_columns]
        y = train_df["target_points"]

        if self.tune_hyperparams:
            logger.info(f"Tuning {self.model_type} with Optuna ({self.n_trials} trials)…")
            self.best_params_ = self._tune_with_optuna(X, y)

        self.pipeline = self._build_pipeline(self.best_params_ or None)
        self.pipeline.fit(X, y)
        return self

    def predict(self, score_df: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self.pipeline.predict(score_df[self.feature_columns_])

    def feature_importances(self) -> Optional[pd.Series]:
        """Return feature importances if the underlying estimator supports it."""
        if self.pipeline is None:
            return None
        estimator = self.pipeline.named_steps["model"]
        if hasattr(estimator, "feature_importances_"):
            return pd.Series(
                estimator.feature_importances_, index=self.feature_columns_
            ).sort_values(ascending=False)
        return None
