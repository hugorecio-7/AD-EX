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
        "overall_ctr": 0.20,
        "overall_cvr": 0.25,
        "overall_ipm": 0.35,
        "overall_roas": 0.20,
    },
    "invert_metrics": [],
    "contextual_weight": CONTEXTUAL_WEIGHT,
    "global_weight": GLOBAL_WEIGHT,
}


CREATIVE_QUALITY_SCORE_SPEC = {
    "name": "creative_quality",
    "metrics": {
        "brand_visibility_score": 0.35,
        "readability_score": 0.25,
        "clutter_score": 0.25,
        "novelty_score": 0.10,
        "motion_score": 0.05,
    },
    "invert_metrics": ["clutter_score"],
    "contextual_weight": CONTEXTUAL_WEIGHT,
    "global_weight": GLOBAL_WEIGHT,
}


RAW_PERFORMANCE_COLUMNS = [
    "overall_ctr",
    "overall_cvr",
    "overall_ipm",
    "overall_roas",
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
    "total_impressions": 0.45,
    "total_clicks": 0.25,
    "total_conversions": 0.25,
    "total_days_active": 0.05,
}

# Used as the saturation threshold for log-scaled confidence.
# A creative reaching the 75th percentile of volume gets close to max confidence.
CONFIDENCE_QUANTILE_THRESHOLD = 0.90

RAW_CONFIDENCE_COLUMNS = [
    "total_impressions",
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
    "ctr_decay": 0.55,
    "cvr_decay": 0.45,
}

RAW_HEALTH_COLUMNS = [
    "ctr_decay_pct",
    "cvr_decay_pct",
]

CONTEXT_SCORE_WEIGHTS = {
    "vertical": 0.32,
    "objective": 0.28,
    "format": 0.17,
    "target_os": 0.10,
    "countries": 0.07,
    "target_age_segment": 0.03,
    "language": 0.03,
}

# ---------------------------------------------------------------------
# Semantic embeddings
# ---------------------------------------------------------------------

SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_TEXT_FIELDS = [
    "global_text",
    "elements_text",
    "ocr_text",
    "layout_text",
]

SEMANTIC_SIMILARITY_WEIGHTS = {
    "global": 0.20,
    "elements": 0.35,
    "ocr": 0.20,
    "layout": 0.25,
}

# ---------------------------------------------------------------------
# FinalRetrieverScore
# ---------------------------------------------------------------------

FINAL_RETRIEVER_SCORE_WEIGHTS = {
    "similarity_score_final": 0.40,
    "context_score_final": 0.30,
    "performance_score_final": 0.15,
    "creative_quality_score_final": 0.08,
    "confidence_score_final": 0.03,
    "health_score_final": 0.04,
}

# Candidate must perform better than the query by at least this margin.
# 0.0 means strictly better than the query.
MIN_PERFORMANCE_DELTA_FOR_RETRIEVAL = 0.0