from __future__ import annotations

from typing import Any

import pandas as pd

from .config import (
    FINAL_RETRIEVER_SCORE_WEIGHTS,
    MIN_PERFORMANCE_DELTA_FOR_RETRIEVAL,
)
from .paths import CREATIVE_RETRIEVAL_INDEX_PATH
from .context_score import compute_context_score
from .similarity_score import (
    load_semantic_embeddings,
    compute_semantic_similarity_for_existing_creative,
)


def load_creative_retrieval_index() -> pd.DataFrame:
    """
    Loads the offline retrieval index.

    This index must contain:
        performance_score_final
        creative_quality_score_final
        confidence_score_final
        health_score_final
    """
    if not CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {CREATIVE_RETRIEVAL_INDEX_PATH}. "
            "Run build_score_index first."
        )

    df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)
    df["creative_id"] = df["creative_id"].astype(str)

    return df


def _get_query_row(
    creative_id: str | int,
    retrieval_index_df: pd.DataFrame,
) -> pd.Series:
    creative_id = str(creative_id)

    matches = retrieval_index_df[
        retrieval_index_df["creative_id"].astype(str) == creative_id
    ]

    if matches.empty:
        raise ValueError(f"Query creative_id not found in retrieval index: {creative_id}")

    return matches.iloc[0]


def _safe_numeric_score(
    df: pd.DataFrame,
    column: str,
    default: float = 0.5,
) -> pd.Series:
    """
    Returns a numeric score column.

    If the column is missing, returns a neutral default score.
    """
    if column not in df.columns:
        return pd.Series(default, index=df.index)

    return pd.to_numeric(df[column], errors="coerce").fillna(default).clip(0.0, 1.0)


def add_final_retriever_score(
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds FinalRetrieverScore to already-scored candidates.

    Expected input columns:
        similarity_score_final
        context_score_final
        performance_score_final
        creative_quality_score_final
        confidence_score_final
        health_score_final

    Output:
        final_retriever_score
    """
    df = candidates_df.copy()

    total_weight = float(sum(FINAL_RETRIEVER_SCORE_WEIGHTS.values()))
    if total_weight <= 0:
        raise ValueError("Final retriever score weights must sum to a positive value.")

    df["final_retriever_score"] = 0.0

    for column, weight in FINAL_RETRIEVER_SCORE_WEIGHTS.items():
        score = _safe_numeric_score(df, column)
        df["final_retriever_score"] += (weight / total_weight) * score

    df["final_retriever_score"] = df["final_retriever_score"].clip(0.0, 1.0)

    return df


def compute_final_retriever_score_for_existing_creative(
    query_creative_id: str | int,
    top_k: int = 15,
    exclude_self: bool = True,
    require_better_performance: bool = True,
    min_performance_delta: float = MIN_PERFORMANCE_DELTA_FOR_RETRIEVAL,
) -> pd.DataFrame:
    """
    Computes final retrieval ranking for an existing creative.

    This function combines:
        - SimilarityScore, from semantic embeddings
        - ContextScore, computed online against the query
        - PerformanceScore, precomputed offline
        - CreativeQualityScore, precomputed offline
        - ConfidenceScore, precomputed offline
        - HealthScore, precomputed offline

    By default, candidates must have better PerformanceScore than the query.
    """
    query_creative_id = str(query_creative_id)

    retrieval_index_df = load_creative_retrieval_index()
    query_row = _get_query_row(query_creative_id, retrieval_index_df)

    # 1. SimilarityScore: only candidates with semantic embeddings.
    embeddings = load_semantic_embeddings()

    similarity_df = compute_semantic_similarity_for_existing_creative(
        query_creative_id=query_creative_id,
        embeddings=embeddings,
    )
    similarity_df["creative_id"] = similarity_df["creative_id"].astype(str)

    # 2. ContextScore: computed over all creatives, then merged.
    context_df = compute_context_score(
        query=query_row,
        candidates_df=retrieval_index_df,
    )
    context_df["creative_id"] = context_df["creative_id"].astype(str)

    context_cols = [
        "creative_id",
        "context_score_final",
        "context_vertical_score",
        "context_objective_score",
        "context_format_score",
        "context_language_score",
        "context_target_os_score",
        "context_countries_score",
        "context_target_age_segment_score",
    ]
    context_cols = [c for c in context_cols if c in context_df.columns]

    # 3. Offline scores and metadata.
    offline_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "language",
        "target_os",
        "countries",
        "target_age_segment",
        "asset_file",
        "performance_score_final",
        "creative_quality_score_final",
        "confidence_score_final",
        "health_score_final",
        "creative_status",
        "overall_ctr",
        "overall_cvr",
        "overall_ipm",
        "overall_roas",
        "perf_score",
    ]
    offline_cols = [c for c in offline_cols if c in retrieval_index_df.columns]

    candidates_df = similarity_df.merge(
        context_df[context_cols],
        on="creative_id",
        how="left",
    )

    candidates_df = candidates_df.merge(
        retrieval_index_df[offline_cols],
        on="creative_id",
        how="left",
    )

    # 4. Exclude the query itself.
    if exclude_self:
        candidates_df = candidates_df[
            candidates_df["creative_id"].astype(str) != query_creative_id
        ].copy()

    # 5. Query-relative deltas for interpretability.
    query_performance = float(query_row.get("performance_score_final", 0.0))
    query_creative_quality = float(query_row.get("creative_quality_score_final", 0.0))

    candidates_df["query_performance_score"] = query_performance
    candidates_df["performance_delta_vs_query"] = (
        pd.to_numeric(candidates_df["performance_score_final"], errors="coerce")
        - query_performance
    )

    candidates_df["candidate_is_better_than_query"] = (
        candidates_df["performance_delta_vs_query"] > min_performance_delta
    )

    candidates_df["query_creative_quality_score"] = query_creative_quality
    candidates_df["creative_quality_delta_vs_query"] = (
        pd.to_numeric(candidates_df["creative_quality_score_final"], errors="coerce")
        - query_creative_quality
    )

    # 6. Keep only better-performing candidates by default.
    if require_better_performance:
        candidates_df = candidates_df[
            candidates_df["candidate_is_better_than_query"]
        ].copy()

    # 7. Final weighted score.
    candidates_df = add_final_retriever_score(candidates_df)

    candidates_df = candidates_df.sort_values(
        "final_retriever_score",
        ascending=False,
    )

    return candidates_df.head(top_k)