from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.pipeline.step1_retrieval.created.final_retriever_score import (
    load_creative_retrieval_index,
    compute_final_retriever_score_for_existing_creative,
)


def test_final_retriever_score() -> None:
    retrieval_index_df = load_creative_retrieval_index()

    # Use a creative that exists in the mock semantic JSONs.
    query_creative_id = "500000"

    query_row = retrieval_index_df[
        retrieval_index_df["creative_id"].astype(str) == str(query_creative_id)
    ].iloc[0]

    print("\n=== Query creative ===")
    query_display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "language",
        "target_os",
        "countries",
        "target_age_segment",
        "performance_score_final",
        "creative_quality_score_final",
        "confidence_score_final",
        "health_score_final",
        "asset_file",
    ]
    query_display_cols = [c for c in query_display_cols if c in query_row.index]

    print(query_row[query_display_cols].to_string())

    results = compute_final_retriever_score_for_existing_creative(
        query_creative_id=query_creative_id,
        top_k=15,
        exclude_self=True,
        require_better_performance=True,
    )

    display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "final_retriever_score",
        "similarity_score_final",
        "context_score_final",
        "performance_score_final",
        "performance_delta_vs_query",
        "candidate_is_better_than_query",
        "creative_quality_score_final",
        "confidence_score_final",
        "health_score_final",
        "creative_status",
        "asset_file",
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    print("\n=== Top candidates by FinalRetrieverScore ===")
    print(results[display_cols].to_string(index=False))

    print("\n=== FinalRetrieverScore range in returned top candidates ===")
    print(
        f"min={results['final_retriever_score'].min():.4f}, "
        f"mean={results['final_retriever_score'].mean():.4f}, "
        f"max={results['final_retriever_score'].max():.4f}, "
        f"nan={results['final_retriever_score'].isna().sum()}"
    )

    print("\n=== Better-than-query candidates in returned top candidates ===")
    if "candidate_is_better_than_query" in results.columns:
        print(results["candidate_is_better_than_query"].value_counts().to_string())


if __name__ == "__main__":
    test_final_retriever_score()