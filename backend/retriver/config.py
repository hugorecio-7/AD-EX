"""
Configuration for retrieval scoring.

This file should contain weights, score definitions and column names.
Future scores such as ConfidenceScore, HealthScore, SimilarityScore and
ContextScore should be added here or referenced from here.
"""

# Main contextual grouping used for contextual percentile normalization.
# The score will try the most specific level first, then fall back.
CONTEXT_FALLBACK_LEVELS = [
    ["vertical", "objective", "format"],
    ["vertical", "objective"],
    ["vertical"],
]

MIN_CONTEXT_GROUP_SIZE = 10

CONTEXTUAL_WEIGHT = 0.8
GLOBAL_WEIGHT = 0.2


IDENTITY_COLUMNS = [
    "creative_id",
    "campaign_id",
    "advertiser_id",
    "advertiser_name",
    "app_name",
    "vertical",
    "objective",
    "format",
    "language",
    "asset_file",
]


PERFORMANCE_SCORE_SPEC = {
    "name": "performance",
    "metrics": {
        "overall_ctr": 0.25,
        "overall_cvr": 0.25,
        "overall_ipm": 0.25,
        "overall_roas": 0.15,
        "perf_score": 0.10,
    },
    "invert_metrics": [],
    "contextual_weight": CONTEXTUAL_WEIGHT,
    "global_weight": GLOBAL_WEIGHT,
}


CREATIVE_QUALITY_SCORE_SPEC = {
    "name": "creative_quality",
    "metrics": {
        "readability_score": 0.25,
        "brand_visibility_score": 0.30,
        "clutter_score": 0.20,
        "novelty_score": 0.15,
        "motion_score": 0.10,
    },
    # Higher clutter is worse, so its percentile is inverted.
    "invert_metrics": ["clutter_score"],
    "contextual_weight": CONTEXTUAL_WEIGHT,
    "global_weight": GLOBAL_WEIGHT,
}


RAW_PERFORMANCE_COLUMNS = [
    "overall_ctr",
    "overall_cvr",
    "overall_ipm",
    "overall_roas",
    "perf_score",
    "total_impressions",
    "total_clicks",
    "total_conversions",
    "total_spend_usd",
    "total_revenue_usd",
]


RAW_CREATIVE_QUALITY_COLUMNS = [
    "readability_score",
    "brand_visibility_score",
    "clutter_score",
    "novelty_score",
    "motion_score",
]


FUTURE_SCORE_COLUMNS = [
    "creative_status",
    "fatigue_day",
    "ctr_decay_pct",
    "cvr_decay_pct",
    "total_days_active",
]

# ---------------------------------------------------------------------
# ConfidenceScore
# ---------------------------------------------------------------------

CONFIDENCE_SCORE_WEIGHTS = {
    "total_impressions": 0.40,
    "total_spend_usd": 0.25,
    "total_clicks": 0.15,
    "total_conversions": 0.15,
    "total_days_active": 0.05,
}

# Used as the saturation threshold for log-scaled confidence.
# A creative reaching the 75th percentile of volume gets close to max confidence.
CONFIDENCE_QUANTILE_THRESHOLD = 0.90

RAW_CONFIDENCE_COLUMNS = [
    "total_impressions",
    "total_spend_usd",
    "total_clicks",
    "total_conversions",
    "total_days_active",
]


# ---------------------------------------------------------------------
# HealthScore
# ---------------------------------------------------------------------

HEALTH_STATUS_MAPPING = {
    "top_performer": 1.00,
    "stable": 0.90,
    "underperformer": 0.60,
    "fatigued": 0.30,
    "unknown": 0.50,
}

HEALTH_SCORE_WEIGHTS = {
    "status": 0.45,
    "ctr_decay": 0.25,
    "cvr_decay": 0.20,
    "fatigue_timing": 0.10,
}

RAW_HEALTH_COLUMNS = [
    "creative_status",
    "fatigue_day",
    "ctr_decay_pct",
    "cvr_decay_pct",
]

CONTEXT_SCORE_WEIGHTS = {
    "vertical": 0.30,
    "objective": 0.25,
    "format": 0.15,
    "language": 0.10,
    "target_os": 0.10,
    "countries": 0.05,
    "target_age_segment": 0.05,
}