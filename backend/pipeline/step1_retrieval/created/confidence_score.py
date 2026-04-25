from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    CONFIDENCE_SCORE_WEIGHTS,
    CONFIDENCE_QUANTILE_THRESHOLD,
)


def _non_negative_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Returns a non-negative numeric series.
    Missing columns are treated as neutral zeros.
    """
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)

    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return series.clip(lower=0.0)


def _log_saturation_score(
    series: pd.Series,
    quantile_threshold: float = CONFIDENCE_QUANTILE_THRESHOLD,
) -> pd.Series:
    """
    Converts a volume metric into a confidence score in [0, 1].

    Formula:
        score = min(1, log(1 + x) / log(1 + tau))

    where tau is a data-driven threshold, usually the 75th percentile.

    This avoids giving excessive advantage to huge-spend creatives.
    """
    series = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)

    positive_values = series[series > 0]

    if positive_values.empty:
        return pd.Series(0.5, index=series.index)

    tau = positive_values.quantile(quantile_threshold)

    if tau <= 0:
        return pd.Series(0.5, index=series.index)

    score = np.log1p(series) / np.log1p(tau)
    return pd.Series(score, index=series.index).clip(0.0, 1.0)


def add_confidence_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ConfidenceScore columns.

    ConfidenceScore measures how reliable the observed performance is,
    based on volume.

    Output columns:
        total_impressions_confidence_score
        total_spend_usd_confidence_score
        total_clicks_confidence_score
        total_conversions_confidence_score
        total_days_active_confidence_score
        confidence_score_final
    """
    df = df.copy()

    total_weight = float(sum(CONFIDENCE_SCORE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("Confidence score weights must sum to a positive value.")

    df["confidence_score_final"] = 0.0

    for metric, weight in CONFIDENCE_SCORE_WEIGHTS.items():
        raw_series = _non_negative_numeric_series(df, metric)
        confidence_col = f"{metric}_confidence_score"

        df[confidence_col] = _log_saturation_score(raw_series)

        normalized_weight = weight / total_weight
        df["confidence_score_final"] += normalized_weight * df[confidence_col]

    df["confidence_score_final"] = df["confidence_score_final"].clip(0.0, 1.0)

    return df