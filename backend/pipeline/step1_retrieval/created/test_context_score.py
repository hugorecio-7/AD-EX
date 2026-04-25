from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.pipeline.step1_retrieval.created.paths import CREATIVE_RETRIEVAL_INDEX_PATH
from backend.pipeline.step1_retrieval.created.context_score import compute_context_score


def test_context_score() -> None:
    df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)

    # Use one existing creative as query for the first test.
    query = df.iloc[0]

    scored_df = compute_context_score(query, df)

    display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "language",
        "target_os",
        "countries",
        "target_age_segment",
        "context_score_final",
    ]
    display_cols = [col for col in display_cols if col in scored_df.columns]

    print("\n=== Query creative ===")
    print(query[display_cols[:-1]].to_string())

    print("\n=== Top 15 by ContextScore ===")
    print(
        scored_df.sort_values("context_score_final", ascending=False)
        .head(15)[display_cols]
        .to_string(index=False)
    )

    print("\n=== ContextScore range ===")
    print(
        f"min={scored_df['context_score_final'].min():.4f}, "
        f"mean={scored_df['context_score_final'].mean():.4f}, "
        f"max={scored_df['context_score_final'].max():.4f}, "
        f"nan={scored_df['context_score_final'].isna().sum()}"
    )


if __name__ == "__main__":
    test_context_score()