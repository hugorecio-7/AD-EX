from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.pipeline.step1_retrieval.created.paths import (
    CREATIVE_RETRIEVAL_INDEX_PATH,
    SEMANTIC_EMBEDDING_INDEX_PATH,
)
from backend.pipeline.step1_retrieval.created.similarity_score import (
    load_semantic_embeddings,
    compute_semantic_similarity_for_existing_creative,
)


def test_semantic_embeddings() -> None:
    embeddings = load_semantic_embeddings()

    creative_ids = embeddings["creative_ids"]
    query_creative_id = creative_ids[0]

    print(f"\n=== Query creative_id: {query_creative_id} ===")

    similarity_df = compute_semantic_similarity_for_existing_creative(
        query_creative_id=query_creative_id,
        embeddings=embeddings,
    )

    # Remove self-match for retrieval-like inspection.
    similarity_df = similarity_df[
        similarity_df["creative_id"].astype(str) != str(query_creative_id)
    ]

    index_df = pd.read_csv(SEMANTIC_EMBEDDING_INDEX_PATH)
    index_df["creative_id"] = index_df["creative_id"].astype(str)

    similarity_df["creative_id"] = similarity_df["creative_id"].astype(str)
    result = similarity_df.merge(index_df, on="creative_id", how="left")

    # Add business/performance info if available.
    if CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        retrieval_df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)
        retrieval_df["creative_id"] = retrieval_df["creative_id"].astype(str)

        useful_cols = [
            "creative_id",
            "app_name",
            "vertical",
            "objective",
            "format",
            "performance_score_final",
            "creative_quality_score_final",
            "confidence_score_final",
            "health_score_final",
            "asset_file",
        ]
        useful_cols = [c for c in useful_cols if c in retrieval_df.columns]

        result = result.merge(
            retrieval_df[useful_cols],
            on="creative_id",
            how="left",
            suffixes=("", "_retrieval"),
        )

    display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "similarity_score_final",
        "global_similarity",
        "elements_similarity",
        "ocr_similarity",
        "layout_similarity",
        "performance_score_final",
        "asset_file",
        "global_text",
    ]
    display_cols = [c for c in display_cols if c in result.columns]

    print("\n=== Top 15 semantic similar creatives ===")
    print(
        result.sort_values("similarity_score_final", ascending=False)
        .head(15)[display_cols]
        .to_string(index=False)
    )

    print("\n=== Similarity score range ===")
    print(
        f"min={similarity_df['similarity_score_final'].min():.4f}, "
        f"mean={similarity_df['similarity_score_final'].mean():.4f}, "
        f"max={similarity_df['similarity_score_final'].max():.4f}, "
        f"nan={similarity_df['similarity_score_final'].isna().sum()}"
    )


if __name__ == "__main__":
    test_semantic_embeddings()