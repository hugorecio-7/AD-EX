"""
Step 3 — Performance Evaluator

Wraps the LightGBM-based creative performance predictor from the GEMINI folder.
Provides a clean, synchronous API used by:
    - core.py  (async wrappers that call evaluate_creative / predict_uplift)
    - preprocess_scores.py  (batch scoring at preprocessing time)

Model artifacts (trained externally):
    GEMINI/lgbm_ctr_model.txt
    GEMINI/lgbm_features.pkl
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_STEP3_DIR = Path(__file__).resolve().parent
_GEMINI_DIR = _STEP3_DIR / "GEMINI"
_MODEL_PATH = _GEMINI_DIR / "lgbm_ctr_model.txt"
_FEATURES_PATH = _GEMINI_DIR / "lgbm_features.pkl"

# ── Lazy-loaded model (only imported when actually needed) ────────────────────
_model = None
_features: list[str] | None = None


def _load_model():
    """Load the LightGBM model and feature list. Cached after first call."""
    global _model, _features
    if _model is not None:
        return _model, _features

    try:
        import lightgbm as lgb
        import joblib
    except ImportError as e:
        raise ImportError(
            "lightgbm and joblib are required for the evaluator. "
            "Run: uv add lightgbm joblib"
        ) from e

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"LightGBM model not found: {_MODEL_PATH}")
    if not _FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {_FEATURES_PATH}")

    _model = lgb.Booster(model_file=str(_MODEL_PATH))
    _features = joblib.load(str(_FEATURES_PATH))
    print(f"[Evaluator] Loaded LightGBM model ({len(_features)} features)")
    return _model, _features


def simulate_custom_creative(
    creative_params: dict[str, Any],
    segment_params: dict[str, Any] | None = None,
    target_col: str = "CTR",
    num_days: int = 30,
) -> pd.DataFrame:
    """
    Runs the 30-day recursive CTR simulation for a creative described by
    `creative_params` + optional `segment_params`.

    Returns a DataFrame with columns:
        days_since_launch, predicted_CTR (or other target_col), ...

    This is the core function from GEMINI/evaluation.py, re-imported here
    as a proper importable module (not a script).
    """
    model, model_features = _load_model()
    segment_params = segment_params or {"country": "US", "os": "iOS"}

    # Create a blank timeline dataframe
    df = pd.DataFrame({"days_since_launch": range(1, num_days + 1)})

    # Inject segment and static creative parameters across all 30 days
    for key, value in {**segment_params, **creative_params}.items():
        df[key] = value

    # Cold start flag
    df["is_cold_start"] = (df["days_since_launch"] <= 3).astype(int)

    # Initialise lag/rolling features as NaN (LightGBM handles NaN natively)
    lag_cols = [col for col in model_features if "lag" in col or "rolling" in col]
    for col in lag_cols:
        df[col] = np.nan

    # Cast categorical columns
    categorical_cols = [
        "country", "os", "vertical", "format", "language", "theme",
        "hook_type", "dominant_color", "emotional_tone",
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure column order matches model expectations
    # Add any missing feature columns as NaN
    for col in model_features:
        if col not in df.columns:
            df[col] = np.nan

    df_features = df[model_features].copy()

    # Recursive inference — each prediction feeds the next day's lag features
    predictions: list[float] = []
    for i in range(len(df_features)):
        if i > 0:
            df_features.loc[i, f"{target_col}_lag_1"] = predictions[i - 1]
        if i > 1:
            df_features.loc[i, f"{target_col}_lag_2"] = predictions[i - 2]

        history = predictions[max(0, i - 3): i]
        if history:
            df_features.loc[i, f"{target_col}_rolling_mean_3d"] = float(np.mean(history))
            df_features.loc[i, f"{target_col}_rolling_std_3d"] = (
                float(np.std(history, ddof=1)) if len(history) > 1 else 0.0
            )

        pred = float(model.predict(df_features.loc[[i]])[0])
        predictions.append(pred)

    df[f"predicted_{target_col}"] = predictions
    return df


# ── High-level evaluation helpers ────────────────────────────────────────────

_CREATIVE_PARAM_DEFAULTS: dict[str, Any] = {
    "vertical": "unknown",
    "format": "unknown",
    "language": "EN",
    "theme": "unknown",
    "hook_type": "unknown",
    "dominant_color": "unknown",
    "emotional_tone": "unknown",
    "width": 1080,
    "height": 1920,
    "duration_sec": 15,
    "text_density": 0.12,
    "copy_length_chars": 45,
    "readability_score": 7.0,
    "brand_visibility_score": 0.7,
    "clutter_score": 0.4,
    "novelty_score": 0.6,
    "motion_score": 0.6,
    "faces_count": 0,
    "product_count": 1,
    "has_price": 0,
    "has_discount_badge": 0,
    "has_gameplay": 0,
    "has_ugc_style": 0,
}

_SEGMENT_DEFAULTS: dict[str, Any] = {
    "country": "US",
    "os": "iOS",
}


def evaluate_creative(
    creative_params: dict[str, Any],
    segment_params: dict[str, Any] | None = None,
    old_ctr: float | None = None,
    num_days: int = 30,
) -> dict[str, Any]:
    """
    Full evaluation of a creative using the LightGBM model.

    Returns:
        {
            "performance_score": float  [0..1],
            "predicted_ctr":     float  (mean CTR over num_days),
            "peak_ctr":          float,
            "fatigue_day":       int | None,
            "is_fatigued":       bool,
            "predicted_uplift":  str,
            "logic_version":     str,
        }
    """
    try:
        params = {**_CREATIVE_PARAM_DEFAULTS, **(creative_params or {})}
        seg = {**_SEGMENT_DEFAULTS, **(segment_params or {})}
        sim_df = simulate_custom_creative(params, seg, target_col="CTR", num_days=num_days)
        ctr_series = sim_df["predicted_CTR"]
    except Exception as e:
        print(f"[Evaluator] LightGBM simulation failed ({e}). Falling back to mock.")
        return _mock_evaluate(creative_params, old_ctr)

    mean_ctr = float(ctr_series.mean())
    peak_ctr = float(ctr_series.max())

    # Detect fatigue: day when CTR drops to 20% of the peak CTR
    fatigue_threshold = peak_ctr * 0.2
    
    # We must only look for drops AFTER the peak, to prevent flagging initial warm-up days.
    peak_id = ctr_series.idxmax()
    post_peak_mask = ctr_series.loc[peak_id + 1:] <= fatigue_threshold
    
    fatigue_day: int | None = None
    if post_peak_mask.any():
        fatigue_day = int(sim_df.loc[post_peak_mask.idxmax(), "days_since_launch"])

    # More intelligently made performance metric blending mean, peak, and penalizing early fatigue
    base_score = 0.6 * (mean_ctr / 0.08) + 0.4 * (peak_ctr / 0.12)
    if fatigue_day and fatigue_day <= 7:
        base_score *= 0.85  # Apply penalty for rapid early plateau

    performance_score = round(max(0.0, min(base_score, 0.99)), 4)

    is_fatigued = (fatigue_day is not None and fatigue_day <= 7)

    # Uplift vs old CTR baseline
    if old_ctr is not None and old_ctr > 0:
        uplift_pct = round((mean_ctr - old_ctr) / old_ctr * 100, 1)
    else:
        uplift_pct = 0.0
    uplift_str = f"+{uplift_pct}%" if uplift_pct >= 0 else f"{uplift_pct}%"

    return {
        "performance_score": performance_score,
        "predicted_ctr": round(mean_ctr * 100, 4),   # as percentage
        "peak_ctr": round(peak_ctr * 100, 4),
        "fatigue_day": fatigue_day,
        "is_fatigued": is_fatigued,
        "predicted_uplift": uplift_str,
        "logic_version": "v3-lgbm-simulation",
    }


def _mock_evaluate(creative_params: dict, old_ctr: float | None) -> dict:
    """
    Fallback when the LightGBM model is unavailable (e.g. during dev without GPU).
    Replicates the previous heuristic logic.
    """
    base_score = 0.72
    performance_score = round(min(base_score, 0.99), 3)
    base_ctr = 3.5
    uplift_pct = 0.0
    if old_ctr is not None and old_ctr > 0:
        uplift_pct = round((base_ctr - old_ctr) / max(old_ctr, 0.01) * 100, 1)
    uplift_str = f"+{uplift_pct}%" if uplift_pct >= 0 else f"{uplift_pct}%"
    return {
        "performance_score": performance_score,
        "predicted_ctr": base_ctr,
        "peak_ctr": base_ctr,
        "fatigue_day": None,
        "is_fatigued": performance_score < 0.3,
        "predicted_uplift": uplift_str,
        "logic_version": "v3-mock-fallback",
    }


def evaluate_creative_from_metadata(
    metadata: dict,
    old_ctr: float | None = None,
) -> dict[str, Any]:
    """
    Convenience wrapper: accepts the same metadata dict used everywhere in the
    pipeline (format, theme, hook_type, etc.) and maps it into creative_params.
    """
    creative_params = {
        "format": metadata.get("format", "unknown"),
        "theme": metadata.get("theme", "unknown"),
        "hook_type": metadata.get("hook_type", metadata.get("hook", "unknown")),
        "vertical": metadata.get("vertical", "unknown"),
        "language": metadata.get("language", "EN"),
        "dominant_color": metadata.get("dominant_color", "unknown"),
        "emotional_tone": metadata.get("emotional_tone", "unknown"),
        "readability_score": float(metadata.get("readability_score", 7.0)),
        "brand_visibility_score": float(metadata.get("brand_visibility_score", 0.7)),
        "clutter_score": float(metadata.get("clutter_score", 0.4)),
        "novelty_score": float(metadata.get("novelty_score", 0.6)),
        "motion_score": float(metadata.get("motion_score", 0.6)),
        "has_gameplay": int(bool(metadata.get("has_gameplay", 0))),
        "has_ugc_style": int(bool(metadata.get("has_ugc_style", 0))),
    }
    segment_params = {
        "country": metadata.get("country", "US"),
        "os": metadata.get("os", "iOS"),
    }
    return evaluate_creative(creative_params, segment_params, old_ctr=old_ctr)
