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