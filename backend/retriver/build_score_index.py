from __future__ import annotations

import sys
from pathlib import Path

# Allows running:
#   python backend/retriver/build_score_index.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.retriver.config import (
    IDENTITY_COLUMNS,
    RAW_PERFORMANCE_COLUMNS,
    RAW_CREATIVE_QUALITY_COLUMNS,
    RAW_CONFIDENCE_COLUMNS,
    RAW_HEALTH_COLUMNS,
)
from backend.retriver.paths import (
    CREATIVE_RETRIEVAL_INDEX_PATH,
    PERFORMANCE_SCORES_PATH,
)
from backend.retriver.data_loader import (
    load_raw_tables,
    build_master_creative_table,
)
from backend.retriver.performance_score import add_performance_score
from backend.retriver.creative_quality_score import add_creative_quality_score
from backend.retriver.confidence_score import add_confidence_score
from backend.retriver.health_score import add_health_score

from backend.retriver.score_utils import (
    fill_missing_categoricals,
    assert_score_range,
)


def _existing_columns(columns: list[str], df_columns: list[str]) -> list[str]:
    result = []
    seen = set()

    for col in columns:
        if col in df_columns and col not in seen:
            result.append(col)
            seen.add(col)

    return result


def _build_output_column_order(df_columns: list[str]) -> list[str]:
    """
    Keeps the output readable and stable.
    Future scores can add their own columns without breaking this function.
    """
    preferred_columns = []

    preferred_columns += IDENTITY_COLUMNS

    preferred_columns += RAW_PERFORMANCE_COLUMNS

    performance_percentile_columns = [
        "overall_ctr_pct_contextual",
        "overall_cvr_pct_contextual",
        "overall_ipm_pct_contextual",
        "overall_roas_pct_contextual",
        "perf_score_pct_contextual",
        "overall_ctr_pct_global",
        "overall_cvr_pct_global",
        "overall_ipm_pct_global",
        "overall_roas_pct_global",
        "perf_score_pct_global",
        "performance_score_contextual",
        "performance_score_global",
        "performance_score_final",
    ]
    preferred_columns += performance_percentile_columns

    preferred_columns += RAW_CREATIVE_QUALITY_COLUMNS

    creative_quality_percentile_columns = [
        "readability_score_pct_contextual",
        "brand_visibility_score_pct_contextual",
        "clutter_score_pct_contextual",
        "novelty_score_pct_contextual",
        "motion_score_pct_contextual",
        "readability_score_pct_global",
        "brand_visibility_score_pct_global",
        "clutter_score_pct_global",
        "novelty_score_pct_global",
        "motion_score_pct_global",
        "creative_quality_score_contextual",
        "creative_quality_score_global",
        "creative_quality_score_final",
    ]
    preferred_columns += creative_quality_percentile_columns

    confidence_score_columns = [
        "total_impressions_confidence_score",
        "total_spend_usd_confidence_score",
        "total_clicks_confidence_score",
        "total_conversions_confidence_score",
        "total_days_active_confidence_score",
        "confidence_score_final",
    ]
    preferred_columns += confidence_score_columns

    preferred_columns += RAW_HEALTH_COLUMNS

    health_score_columns = [
        "creative_status_health_score",
        "ctr_decay_health_score",
        "cvr_decay_health_score",
        "fatigue_timing_score",
        "health_score_final",
    ]
    preferred_columns += health_score_columns

    existing_preferred = _existing_columns(preferred_columns, df_columns)
    remaining = [col for col in df_columns if col not in existing_preferred]

    return existing_preferred + remaining


def build_creative_retrieval_index() -> None:
    """
    Main offline preprocessing script.

    Creates:
        frontend/public/data/creative_retrieval_index.csv
        frontend/public/data/performance_scores.csv
    """
    print("[retriver] Loading raw tables...")
    tables = load_raw_tables()

    print("[retriver] Building master creative table...")
    df = build_master_creative_table(tables)

    print(f"[retriver] Master table shape: {df.shape}")

    df = fill_missing_categoricals(
        df,
        columns=[
            "vertical",
            "objective",
            "format",
            "language",
            "app_name",
            "advertiser_name",
        ],
    )

    print("[retriver] Adding PerformanceScore...")
    df = add_performance_score(df)

    print("[retriver] Adding CreativeQualityScore...")
    df = add_creative_quality_score(df)

    print("[retriver] Adding ConfidenceScore...")
    df = add_confidence_score(df)

    print("[retriver] Adding HealthScore...")
    df = add_health_score(df)

    assert_score_range(
        df,
        score_columns=[
            "performance_score_contextual",
            "performance_score_global",
            "performance_score_final",
            "creative_quality_score_contextual",
            "creative_quality_score_global",
            "creative_quality_score_final",
            "confidence_score_final",
            "health_score_final",
        ],
    )

    output_columns = _build_output_column_order(list(df.columns))
    df = df[output_columns]

    CREATIVE_RETRIEVAL_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[retriver] Saving full retrieval index to: {CREATIVE_RETRIEVAL_INDEX_PATH}")
    df.to_csv(CREATIVE_RETRIEVAL_INDEX_PATH, index=False)

    # Legacy/smaller output for compatibility with the README.
    performance_score_columns = _existing_columns(
        [
            "creative_id",
            "campaign_id",
            "advertiser_id",
            "app_name",
            "vertical",
            "objective",
            "format",
            "asset_file",
            "overall_ctr",
            "overall_cvr",
            "overall_ipm",
            "overall_roas",
            "perf_score",
            "performance_score_contextual",
            "performance_score_global",
            "performance_score_final",
            "creative_quality_score_contextual",
            "creative_quality_score_global",
            "creative_quality_score_final",
            "confidence_score_final",
            "health_score_final",
        ],
        list(df.columns),
    )

    performance_scores_df = df[performance_score_columns]

    print(f"[retriver] Saving compact scores file to: {PERFORMANCE_SCORES_PATH}")
    performance_scores_df.to_csv(PERFORMANCE_SCORES_PATH, index=False)

    print("[retriver] Done.")
    print(f"[retriver] Output rows: {len(df)}")
    print(f"[retriver] Output columns: {len(df.columns)}")


if __name__ == "__main__":
    build_creative_retrieval_index()