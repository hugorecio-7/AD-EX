from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allows running:
#   python backend/retriver/inspect_score_index.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.retriver.paths import CREATIVE_RETRIEVAL_INDEX_PATH


SCORE_COLUMNS = [
    "performance_score_contextual",
    "performance_score_global",
    "performance_score_final",
    "creative_quality_score_contextual",
    "creative_quality_score_global",
    "creative_quality_score_final",
]


def inspect_score_index() -> None:
    if not CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index not found: {CREATIVE_RETRIEVAL_INDEX_PATH}. "
            "Run build_score_index.py first."
        )

    df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)

    print("\n=== Creative Retrieval Index ===")
    print(f"Path: {CREATIVE_RETRIEVAL_INDEX_PATH}")
    print(f"Shape: {df.shape}")

    # ------------------------------------------------------------
    # Check 1: creative_id uniqueness
    # ------------------------------------------------------------
    print("\n=== Creative ID integrity ===")

    if "creative_id" in df.columns:
        total_rows = len(df)
        unique_creatives = df["creative_id"].nunique()
        duplicated_rows = df["creative_id"].duplicated().sum()

        print(f"Rows: {total_rows}")
        print(f"Unique creative_id: {unique_creatives}")
        print(f"Duplicated creative_id rows: {duplicated_rows}")

        if duplicated_rows > 0:
            duplicated_ids = (
                df.loc[df["creative_id"].duplicated(keep=False), "creative_id"]
                .value_counts()
                .head(10)
            )
            print("\nTop duplicated creative_id values:")
            print(duplicated_ids.to_string())
    else:
        print("Column creative_id not found.")

    # ------------------------------------------------------------
    # Check 2: score ranges
    # ------------------------------------------------------------
    print("\n=== Score ranges ===")
    for col in SCORE_COLUMNS:
        if col in df.columns:
            print(
                f"{col}: "
                f"min={df[col].min():.4f}, "
                f"mean={df[col].mean():.4f}, "
                f"max={df[col].max():.4f}, "
                f"nan={df[col].isna().sum()}"
            )
        else:
            print(f"{col}: column not found")

    # ------------------------------------------------------------
    # Check 3: contextual fallback usage
    # ------------------------------------------------------------
    print("\n=== Contextual fallback usage ===")

    context_level_cols = [
        col for col in df.columns
        if col.endswith("_context_level")
    ]

    if not context_level_cols:
        print("No *_context_level columns found.")
    else:
        for col in context_level_cols:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).to_string())

    # ------------------------------------------------------------
    # Check 4: missing values
    # ------------------------------------------------------------
    print("\n=== Columns with most missing values ===")

    missing_summary = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_summary = missing_summary[missing_summary > 0].head(20)

    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary.to_string())

    # ------------------------------------------------------------
    # Check 5: top creatives by PerformanceScore
    # ------------------------------------------------------------
    print("\n=== Top 10 by PerformanceScore ===")
    display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "overall_ctr",
        "overall_cvr",
        "overall_ipm",
        "overall_roas",
        "perf_score",
        "performance_score_final",
        "asset_file",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    if "performance_score_final" in df.columns:
        print(
            df.sort_values("performance_score_final", ascending=False)
            .head(10)[display_cols]
            .to_string(index=False)
        )
    else:
        print("performance_score_final not found.")

    # ------------------------------------------------------------
    # Check 6: top creatives by CreativeQualityScore
    # ------------------------------------------------------------
    print("\n=== Top 10 by CreativeQualityScore ===")
    display_cols = [
        "creative_id",
        "app_name",
        "vertical",
        "objective",
        "format",
        "readability_score",
        "brand_visibility_score",
        "clutter_score",
        "novelty_score",
        "motion_score",
        "creative_quality_score_final",
        "asset_file",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    if "creative_quality_score_final" in df.columns:
        print(
            df.sort_values("creative_quality_score_final", ascending=False)
            .head(10)[display_cols]
            .to_string(index=False)
        )
    else:
        print("creative_quality_score_final not found.")

    # ------------------------------------------------------------
    # Check 7: mean scores by context
    # ------------------------------------------------------------
    if {"vertical", "objective", "format"}.issubset(df.columns):
        print("\n=== Mean scores by vertical/objective/format ===")
        group_summary = (
            df.groupby(["vertical", "objective", "format"], dropna=False)
            .agg(
                n=("creative_id", "count"),
                mean_performance=("performance_score_final", "mean"),
                mean_creative_quality=("creative_quality_score_final", "mean"),
            )
            .reset_index()
            .sort_values("n", ascending=False)
            .head(20)
        )

        print(group_summary.to_string(index=False))

    # ------------------------------------------------------------
    # Check 8: correlation between main scores
    # ------------------------------------------------------------
    print("\n=== Score correlation ===")

    if {
        "performance_score_final",
        "creative_quality_score_final",
    }.issubset(df.columns):
        corr = df["performance_score_final"].corr(
            df["creative_quality_score_final"]
        )
        print(
            "Correlation between performance_score_final "
            f"and creative_quality_score_final: {corr:.4f}"
        )
    else:
        print("Main score columns not found.")


if __name__ == "__main__":
    inspect_score_index()