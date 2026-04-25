"""
Preprocessing Script — Build Retrieval Score Index

Runs the full offline scoring pipeline from the step1_retrieval/created/ package:
  PerformanceScore + CreativeQualityScore + ConfidenceScore + HealthScore

Outputs:
  frontend/public/data/creative_retrieval_index.csv
  frontend/public/data/performance_scores.csv

Run from repo root:
  python backend/scripts/preprocess_retrieval_index.py
"""
import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from pipeline.step1_retrieval.created.data_loader import load_raw_tables, build_master_creative_table
from pipeline.step1_retrieval.created.performance_score import add_performance_score
from pipeline.step1_retrieval.created.creative_quality_score import add_creative_quality_score
from pipeline.step1_retrieval.created.confidence_score import add_confidence_score
from pipeline.step1_retrieval.created.health_score import add_health_score
from pipeline.step1_retrieval.created.score_utils import fill_missing_categoricals, assert_score_range
from pipeline.step1_retrieval.created.paths import CREATIVE_RETRIEVAL_INDEX_PATH, PERFORMANCE_SCORES_PATH
from pipeline.step1_retrieval.created.config import (
    IDENTITY_COLUMNS, RAW_PERFORMANCE_COLUMNS, RAW_CREATIVE_QUALITY_COLUMNS,
    RAW_CONFIDENCE_COLUMNS, RAW_HEALTH_COLUMNS,
)


def _existing_columns(columns, df_columns):
    seen = set()
    result = []
    for col in columns:
        if col in df_columns and col not in seen:
            result.append(col)
            seen.add(col)
    return result


def _build_output_column_order(df_columns):
    preferred = (
        IDENTITY_COLUMNS
        + RAW_PERFORMANCE_COLUMNS
        + ["performance_score_contextual", "performance_score_global", "performance_score_final"]
        + RAW_CREATIVE_QUALITY_COLUMNS
        + ["creative_quality_score_contextual", "creative_quality_score_global", "creative_quality_score_final"]
        + ["confidence_score_final"]
        + RAW_HEALTH_COLUMNS
        + ["health_score_final"]
    )
    existing = _existing_columns(preferred, df_columns)
    remaining = [c for c in df_columns if c not in existing]
    return existing + remaining


def build_creative_retrieval_index():
    print("[Retrieval] Loading raw tables...")
    tables = load_raw_tables()

    print("[Retrieval] Building master creative table...")
    df = build_master_creative_table(tables)
    print(f"[Retrieval] Master table shape: {df.shape}")

    df = fill_missing_categoricals(df, ["vertical", "objective", "format", "language", "app_name", "advertiser_name"])

    print("[Retrieval] Adding PerformanceScore...")
    df = add_performance_score(df)

    print("[Retrieval] Adding CreativeQualityScore...")
    df = add_creative_quality_score(df)

    print("[Retrieval] Adding ConfidenceScore...")
    df = add_confidence_score(df)

    print("[Retrieval] Adding HealthScore...")
    df = add_health_score(df)

    assert_score_range(df, score_columns=[
        "performance_score_final", "creative_quality_score_final",
        "confidence_score_final", "health_score_final",
    ])

    output_columns = _build_output_column_order(list(df.columns))
    df = df[[c for c in output_columns if c in df.columns]]

    CREATIVE_RETRIEVAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CREATIVE_RETRIEVAL_INDEX_PATH, index=False)
    print(f"[Retrieval] ✓ Index saved → {CREATIVE_RETRIEVAL_INDEX_PATH}")

    # Compact performance scores (legacy compat)
    score_cols = _existing_columns(
        ["creative_id", "campaign_id", "advertiser_id", "app_name", "vertical", "objective",
         "format", "asset_file", "overall_ctr", "performance_score_final",
         "creative_quality_score_final", "confidence_score_final", "health_score_final"],
        list(df.columns),
    )
    df[score_cols].to_csv(PERFORMANCE_SCORES_PATH, index=False)
    print(f"[Retrieval] ✓ Scores saved → {PERFORMANCE_SCORES_PATH}")
    print(f"[Retrieval] Done. {len(df)} rows, {len(df.columns)} columns.")


if __name__ == "__main__":
    build_creative_retrieval_index()
