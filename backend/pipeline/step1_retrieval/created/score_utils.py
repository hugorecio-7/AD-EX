from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CONTEXT_FALLBACK_LEVELS, MIN_CONTEXT_GROUP_SIZE


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Converts selected columns to numeric and fills missing values with the
    column median. If a column is fully missing, it fills it with 0.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Required metric column not found: {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Returns percentile rank in [0, 1].
    Higher raw values receive higher percentile values.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().all():
        return pd.Series(0.5, index=series.index)

    numeric = numeric.fillna(numeric.median())

    if numeric.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=series.index)

    return numeric.rank(method="average", pct=True)


def add_global_percentiles(
    df: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    """
    Adds global percentile columns for each metric.
    Example:
        overall_ctr -> overall_ctr_pct_global
    """
    df = df.copy()

    for metric in metrics:
        df[f"{metric}_pct_global"] = percentile_rank(df[metric])

    return df


def _context_key(df: pd.DataFrame, context_cols: list[str]) -> pd.Series:
    """
    Creates a stable string key for a context level.
    """
    return df[context_cols].astype(str).agg("||".join, axis=1)


def add_contextual_percentiles(
    df: pd.DataFrame,
    metrics: list[str],
    context_fallback_levels: list[list[str]] | None = None,
    min_group_size: int = MIN_CONTEXT_GROUP_SIZE,
) -> pd.DataFrame:
    """
    Adds contextual percentile columns for each metric.

    It tries context levels from less specific to more specific, overwriting
    whenever a row belongs to a sufficiently large group.

    Final column:
        metric_pct_contextual

    Also adds:
        metric_context_level

    Example fallback:
        global
        vertical
        vertical + objective
        vertical + objective + format
    """
    df = df.copy()

    context_fallback_levels = context_fallback_levels or CONTEXT_FALLBACK_LEVELS

    # Start with global percentiles.
    for metric in metrics:
        global_col = f"{metric}_pct_global"
        contextual_col = f"{metric}_pct_contextual"
        level_col = f"{metric}_context_level"

        if global_col not in df.columns:
            df[global_col] = percentile_rank(df[metric])

        df[contextual_col] = df[global_col]
        df[level_col] = "global"

    # Apply from least specific to most specific, so the most specific valid
    # context overwrites previous values.
    for context_cols in reversed(context_fallback_levels):
        available_context_cols = [c for c in context_cols if c in df.columns]

        if not available_context_cols:
            continue

        group_key = _context_key(df, available_context_cols)
        group_sizes = group_key.map(group_key.value_counts())
        valid_group_mask = group_sizes >= min_group_size

        context_name = "+".join(available_context_cols)

        for metric in metrics:
            contextual_col = f"{metric}_pct_contextual"
            level_col = f"{metric}_context_level"

            group_percentiles = (
                df.groupby(available_context_cols, dropna=False)[metric]
                .transform(percentile_rank)
            )

            df.loc[valid_group_mask, contextual_col] = group_percentiles.loc[
                valid_group_mask
            ]
            df.loc[valid_group_mask, level_col] = context_name

    return df


def weighted_score_from_percentiles(
    df: pd.DataFrame,
    metric_weights: dict[str, float],
    suffix: str,
    invert_metrics: list[str] | None = None,
) -> pd.Series:
    """
    Computes a weighted score from percentile columns.

    suffix:
        "contextual" or "global"

    If a metric is inverted, its contribution becomes:
        1 - percentile
    """
    invert_metrics = invert_metrics or []

    total_weight = float(sum(metric_weights.values()))
    if total_weight <= 0:
        raise ValueError("Total metric weight must be positive.")

    score = pd.Series(0.0, index=df.index)

    for metric, weight in metric_weights.items():
        col = f"{metric}_pct_{suffix}"

        if col not in df.columns:
            raise KeyError(f"Percentile column not found: {col}")

        contribution = df[col]

        if metric in invert_metrics:
            contribution = 1.0 - contribution

        score += (weight / total_weight) * contribution

    return score.clip(0.0, 1.0)


def add_score_block(
    df: pd.DataFrame,
    score_spec: dict[str, Any],
) -> pd.DataFrame:
    """
    Generic score builder.

    Given a score spec like:

        {
            "name": "performance",
            "metrics": {"overall_ctr": 0.25, ...},
            "invert_metrics": [],
            "contextual_weight": 0.8,
            "global_weight": 0.2,
        }

    It adds:
        metric_pct_global
        metric_pct_contextual
        score_name_score_global
        score_name_score_contextual
        score_name_score_final
    """
    df = df.copy()

    name = score_spec["name"]
    metric_weights = score_spec["metrics"]
    metrics = list(metric_weights.keys())
    invert_metrics = score_spec.get("invert_metrics", [])

    contextual_weight = float(score_spec.get("contextual_weight", 0.8))
    global_weight = float(score_spec.get("global_weight", 0.2))

    weight_sum = contextual_weight + global_weight
    if weight_sum <= 0:
        raise ValueError("Contextual/global weights must sum to a positive value.")

    contextual_weight = contextual_weight / weight_sum
    global_weight = global_weight / weight_sum

    df = coerce_numeric_columns(df, metrics)
    df = add_global_percentiles(df, metrics)
    df = add_contextual_percentiles(df, metrics)

    contextual_score_col = f"{name}_score_contextual"
    global_score_col = f"{name}_score_global"
    final_score_col = f"{name}_score_final"

    df[contextual_score_col] = weighted_score_from_percentiles(
        df=df,
        metric_weights=metric_weights,
        suffix="contextual",
        invert_metrics=invert_metrics,
    )

    df[global_score_col] = weighted_score_from_percentiles(
        df=df,
        metric_weights=metric_weights,
        suffix="global",
        invert_metrics=invert_metrics,
    )

    df[final_score_col] = (
        contextual_weight * df[contextual_score_col]
        + global_weight * df[global_score_col]
    ).clip(0.0, 1.0)

    return df


def fill_missing_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def assert_score_range(
    df: pd.DataFrame,
    score_columns: list[str],
) -> None:
    """
    Raises an error if any score column is outside [0, 1].
    """
    for col in score_columns:
        if col not in df.columns:
            raise KeyError(f"Score column not found: {col}")

        min_value = df[col].min()
        max_value = df[col].max()

        if min_value < -1e-9 or max_value > 1.0 + 1e-9:
            raise ValueError(
                f"Score column {col} outside [0, 1]. "
                f"min={min_value}, max={max_value}"
            )