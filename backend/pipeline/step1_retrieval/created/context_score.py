from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .config import CONTEXT_SCORE_WEIGHTS


def _normalize_value(value: Any) -> str:
    """
    Normalizes a scalar value for robust comparison.
    """
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _parse_multi_value(value: Any) -> set[str]:
    """
    Parses fields like countries or target_os.

    It supports values such as:
        "US, ES, FR"
        "US|ES|FR"
        "['US', 'ES']"
        "android, ios"
    """
    normalized = _normalize_value(value)

    if not normalized:
        return set()

    normalized = normalized.replace("[", "")
    normalized = normalized.replace("]", "")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace('"', "")

    parts = re.split(r"[,\|;/]+", normalized)

    return {part.strip() for part in parts if part.strip()}


def _exact_match_score(query_value: Any, candidate_value: Any) -> float:
    """
    Returns:
        1.0 if both known and equal
        0.0 if both known and different
        0.5 if one side is unknown
    """
    q = _normalize_value(query_value)
    c = _normalize_value(candidate_value)

    if not q or not c:
        return 0.5

    return 1.0 if q == c else 0.0


def _set_overlap_score(query_value: Any, candidate_value: Any) -> float:
    """
    Jaccard overlap for multi-value fields.

    Returns:
        1.0 if sets are identical and non-empty
        partial overlap if they share some values
        0.0 if both known and no overlap
        0.5 if one side is unknown
    """
    q_set = _parse_multi_value(query_value)
    c_set = _parse_multi_value(candidate_value)

    if not q_set or not c_set:
        return 0.5

    intersection = len(q_set.intersection(c_set))
    union = len(q_set.union(c_set))

    if union == 0:
        return 0.5

    return intersection / union


def _get_query_value(query: dict[str, Any] | pd.Series, column: str) -> Any:
    if isinstance(query, pd.Series):
        return query.get(column, None)

    return query.get(column, None)


def compute_context_score(
    query: dict[str, Any] | pd.Series,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes ContextScore(query, candidate) for every candidate.

    This is an online score: it depends on the query creative.

    Output columns:
        context_vertical_score
        context_objective_score
        context_format_score
        context_language_score
        context_target_os_score
        context_countries_score
        context_target_age_segment_score
        context_score_final
    """
    df = candidates_df.copy()

    total_weight = float(sum(CONTEXT_SCORE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("Context score weights must sum to a positive value.")

    df["context_score_final"] = 0.0

    for column, weight in CONTEXT_SCORE_WEIGHTS.items():
        query_value = _get_query_value(query, column)
        score_col = f"context_{column}_score"

        if column == "countries":
            if column in df.columns:
                df[score_col] = df[column].apply(
                    lambda candidate_value: _set_overlap_score(
                        query_value,
                        candidate_value,
                    )
                )
            else:
                df[score_col] = 0.5

        elif column == "target_os":
            if column in df.columns:
                df[score_col] = df[column].apply(
                    lambda candidate_value: _target_os_score(
                        query_value,
                        candidate_value,
                    )
                )
            else:
                df[score_col] = 0.5

        else:
            if column in df.columns:
                df[score_col] = df[column].apply(
                    lambda candidate_value: _exact_match_score(
                        query_value,
                        candidate_value,
                    )
                )
            else:
                df[score_col] = 0.5

        df["context_score_final"] += (weight / total_weight) * df[score_col]

    df["context_score_final"] = df["context_score_final"].clip(0.0, 1.0)

    return df

def _target_os_score(query_value: Any, candidate_value: Any) -> float:
    """
    Scores OS compatibility.

    Both is treated as compatible with iOS and Android.
    """
    q = _normalize_value(query_value)
    c = _normalize_value(candidate_value)

    if not q or not c:
        return 0.5

    q = q.replace(" ", "")
    c = c.replace(" ", "")

    if q == c:
        return 1.0

    if q in {"both", "all", "ios+android", "android+ios"}:
        return 1.0 if c in {"ios", "android", "both"} else 0.0

    if c in {"both", "all", "ios+android", "android+ios"}:
        return 1.0 if q in {"ios", "android", "both"} else 0.0

    return 0.0