# ⚽ EPL Forecast Dashboard

An end-to-end machine learning pipeline and interactive Streamlit dashboard for forecasting English Premier League season outcomes — predicted table, Monte Carlo simulation probabilities, walk-forward backtesting, and historical team trends.

---

## 🌟 Features

- **Multiple models** — HGBT, Random Forest, LightGBM, XGBoost, switchable from the UI
- **Optuna hyperparameter tuning** — 50-trial TPE search with 5-fold CV (opt-in via config)
- **Poisson goal simulator** — attack/defense ratings per team, goals sampled from Poisson distributions, EPL tiebreaking (pts → GD → GF)
- **Rich feature engineering** — rolling lags, home/away splits, H2H performance vs top-half teams, end-of-season form, squad value, manager change flags
- **Walk-forward backtester** — RMSE, MAE, Spearman ρ, Top 4 accuracy, Relegation accuracy, Brier scores — all in the dashboard
- **Interactive Plotly charts** — title/top4/relegation probabilities, historical points and position trends
- **5-tab Streamlit app** — Forecast · Team Deep Dive · Backtest · Historical · Features

---

## 📁 Project Structure

```
epl-forecasting/
├── app.py                        # Streamlit dashboard (5 tabs)
├── run_pipeline.py               # CLI entry point
├── configs/
│   └── default.yaml              # All config (model, features, simulation, forecast)
├── data/
│   ├── raw/                      # pl-tables-1993-2024.csv, epl_final.csv
│   └── external/                 # squad values, manager change flags
└── src/
    ├── config.py                 # Config dataclass + loader
    ├── pipeline.py               # End-to-end pipeline (backtest + forecast + sim)
    ├── data/
    │   ├── loader.py             # Season table + match table loaders
    │   ├── normalizer.py         # Team name canonicalization
    │   ├── validator.py          # Data quality checks
    │   └── external_loader.py    # Squad value + manager flag loaders
    ├── features/
    │   ├── feature_builder.py    # Rolling lags, home/away splits, H2H features
    │   └── form_features.py      # End-of-season rolling form
    ├── models/
    │   ├── tree_model.py         # HGBT / RF / LightGBM / XGBoost + Optuna tuning
    │   └── baseline.py           # Prev-season points baseline
    ├── forecasting/
    │   ├── forecaster.py         # Trains model, builds forecast frame
    │   └── simulator.py          # Poisson Monte Carlo simulator
    └── evaluation/
        └── backtester.py         # Walk-forward backtest with Brier scores
```

---

## 🚀 Getting Started

### 1. Clone & install

```bash
git clone https://github.com/danielgeorge1097/epl-forecast.git
cd epl-forecast

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the dashboard

```bash
cd epl-forecasting
streamlit run app.py
```

### 3. Run the CLI pipeline

```bash
cd epl-forecasting

# Forecast only
python run_pipeline.py

# Forecast + walk-forward backtest
python run_pipeline.py --backtest

# Skip Monte Carlo simulation
python run_pipeline.py --no-sim
```

---

## ⚙️ Configuration

All behaviour is controlled by `configs/default.yaml`:

```yaml
model:
  type: hgbt              # hgbt | rf | lgbm | xgb
  tune_hyperparams: false # true = run Optuna (slow, ~5 min)

forecast:
  predict_season_end_year: 2025
  promoted_teams:
    - Leicester City
    - Ipswich Town
    - Southampton

simulation:
  n_sims: 5000
  use_poisson: true       # true = Poisson goal model | false = legacy logistic

features:
  rolling_windows: [2, 3]
  use_home_away_splits: true
  use_h2h_features: true
```

---

## 🛠️ Built With

| Library | Purpose |
|---|---|
| scikit-learn | HGBT, Random Forest, preprocessing |
| LightGBM | Gradient boosting |
| XGBoost | Gradient boosting |
| Optuna | Hyperparameter tuning |
| SciPy | Poisson sampling |
| Streamlit | Dashboard |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |

---

## 📝 License

Open-source. Free to use and modify.
