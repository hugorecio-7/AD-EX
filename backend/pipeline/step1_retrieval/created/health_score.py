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


def _normalize_decay_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Converts decay into a penalty-like value in [0, 1].

    Positive decay means performance dropped.
    Negative decay means performance improved, so it is treated as 0 decay.

    If values look like percentages instead of ratios, for example 30 instead
    of 0.30, the function automatically divides by 100.
    """
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)

    decay = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    # If values are likely percentage points, convert to ratio.
    if decay.abs().quantile(0.95) > 2.0:
        decay = decay / 100.0

    decay = decay.clip(lower=0.0, upper=1.0)
    return decay


def _decay_health_score(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Converts decay into health.

    No decay      -> 1
    Full decay    -> 0
    """
    decay = _normalize_decay_series(df, column)
    return (1.0 - decay).clip(0.0, 1.0)


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
    """
    Adds HealthScore columns.

    HealthScore estimates whether the creative is still a good reference
    candidate or whether it is affected by fatigue/decay.

    Output columns:
        creative_status_health_score
        ctr_decay_health_score
        cvr_decay_health_score
        fatigue_timing_score
        health_score_final
    """
    df = df.copy()

    df["creative_status_health_score"] = _status_health_score(df)
    df["ctr_decay_health_score"] = _decay_health_score(df, "ctr_decay_pct")
    df["cvr_decay_health_score"] = _decay_health_score(df, "cvr_decay_pct")
    df["fatigue_timing_score"] = _fatigue_timing_score(df)

    total_weight = float(sum(HEALTH_SCORE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("Health score weights must sum to a positive value.")

    df["health_score_final"] = (
        (HEALTH_SCORE_WEIGHTS["status"] / total_weight)
        * df["creative_status_health_score"]
        + (HEALTH_SCORE_WEIGHTS["ctr_decay"] / total_weight)
        * df["ctr_decay_health_score"]
        + (HEALTH_SCORE_WEIGHTS["cvr_decay"] / total_weight)
        * df["cvr_decay_health_score"]
        + (HEALTH_SCORE_WEIGHTS["fatigue_timing"] / total_weight)
        * df["fatigue_timing_score"]
    ).clip(0.0, 1.0)

    return df