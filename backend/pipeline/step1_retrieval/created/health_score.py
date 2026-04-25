from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    HEALTH_STATUS_MAPPING,
    HEALTH_SCORE_WEIGHTS,
)


def _get_status_series(df: pd.DataFrame) -> pd.Series:
    if "creative_status" not in df.columns:
        return pd.Series("unknown", index=df.index)

    return (
        df["creative_status"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def _status_health_score(df: pd.DataFrame) -> pd.Series:
    status = _get_status_series(df)

    return status.map(HEALTH_STATUS_MAPPING).fillna(
        HEALTH_STATUS_MAPPING["unknown"]
    )


def _normalize_drop_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Converts relative change into a drop severity in [0, 1].

    In this dataset, negative values usually mean:
        last period < first period
        performance dropped

    Example:
        -0.80 -> 0.80 drop severity
         0.20 -> 0.00 drop severity, because performance improved
    """
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)

    change = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if change.abs().quantile(0.95) > 2.0:
        change = change / 100.0

    drop = (-change).clip(lower=0.0, upper=1.0)

    return drop


def _decay_health_score(df: pd.DataFrame, column: str) -> pd.Series:
    drop = _normalize_drop_series(df, column)
    return (1.0 - drop).clip(0.0, 1.0)


def _fatigue_timing_score(df: pd.DataFrame) -> pd.Series:
    """
    Measures whether fatigue happened early or late.

    Non-fatigued creatives get 1.

    Fatigued creatives:
        fatigue_day / total_days_active

    Early fatigue -> low score.
    Late fatigue  -> higher score.
    """
    status = _get_status_series(df)
    fatigued_mask = status.eq("fatigued")

    score = pd.Series(1.0, index=df.index)

    if "fatigue_day" not in df.columns or "total_days_active" not in df.columns:
        score.loc[fatigued_mask] = 0.5
        return score.clip(0.0, 1.0)

    fatigue_day = pd.to_numeric(df["fatigue_day"], errors="coerce")
    total_days = pd.to_numeric(df["total_days_active"], errors="coerce")

    ratio = fatigue_day / total_days.replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.5)

    score.loc[fatigued_mask] = ratio.loc[fatigued_mask].clip(0.0, 1.0)

    return score.clip(0.0, 1.0)


def add_health_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ctr_decay_health_score"] = _decay_health_score(df, "ctr_decay_pct")
    df["cvr_decay_health_score"] = _decay_health_score(df, "cvr_decay_pct")

    total_weight = float(sum(HEALTH_SCORE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("Health score weights must sum to a positive value.")

    df["health_score_final"] = (
        (HEALTH_SCORE_WEIGHTS["ctr_decay"] / total_weight)
        * df["ctr_decay_health_score"]
        + (HEALTH_SCORE_WEIGHTS["cvr_decay"] / total_weight)
        * df["cvr_decay_health_score"]
    ).clip(0.0, 1.0)

    return df