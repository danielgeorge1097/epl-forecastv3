from __future__ import annotations

import pandas as pd


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def load_squad_value(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean_columns(df)

    required = {"team", "squad_value_million"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Squad value CSV missing columns: {missing}. Found: {df.columns.tolist()}"
        )

    df["team"] = df["team"].astype(str).str.strip()
    df["squad_value_million"] = pd.to_numeric(df["squad_value_million"], errors="coerce")

    return df


def load_manager_change_flags(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean_columns(df)

    required = {"team", "manager_change_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Manager change CSV missing columns: {missing}. Found: {df.columns.tolist()}"
        )

    df["team"] = df["team"].astype(str).str.strip()
    df["manager_change_flag"] = pd.to_numeric(df["manager_change_flag"], errors="coerce").fillna(0).astype(int)

    return df