"""
AI Engine — orchestrates the full creative upgrade pipeline:
"""
import time
import asyncio

from pipeline.step1_retrieval.core import get_best_creatives
from pipeline.step2_feature_analysis.core import find_missing_features, explain_missing_features
from pipeline.step3_generation.core import generate_creative_with_flux, predict_performance_uplift
from pipeline.step4_persistence.core import store_new_creative, get_creative_by_id, compute_static_performance_score

async def generate_ai_variant_real(
    creative_id: str,
    format_type: str,
    metadata: dict,
    pipe=None,
) -> dict:
    time_s = time.time()

    # ── 1. Retrieve top cases from the same cluster ──────────────────────────
    top_cases = get_best_creatives(creative_id, format_type, metadata)

    # ── 2. Extract what makes each top case good ─────────────────────────────
    explanations = [c.explain() for c in top_cases]

    # ── 3. Identify features the target creative is missing ──────────────────
    missing_features = find_missing_features(explanations, creative_id)

    # ── 4. Run in parallel: generation + uplift prediction + explanation ──────
    async def _generate_and_predict():
        new_file = await generate_creative_with_flux(
            creative_id=creative_id,
            metadata=metadata,
            missing_features=missing_features,
            pipe=pipe,
        )
        uplift = await predict_performance_uplift(missing_features, creative_id)
        return new_file, uplift

    (new_creative_file, predicted_uplift), missing_features_explained = await asyncio.gather(
        _generate_and_predict(),
        explain_missing_features(missing_features, creative_id),
    )

    # ── 5. Persist the new creative ──────────────────────────────────────────
    original = get_creative_by_id(creative_id) or {}
    new_id = f"{creative_id}_v2"

    base = compute_static_performance_score(creative_id)

    new_entry = {
        **original,
        "id": new_id,
        "image_url": f"/data/assets/creative_{creative_id}_upgraded.png",
        "performance_score": float(predicted_uplift.strip("%+").replace("%", "")) / 100
                             + base["performance_score"]
                             if predicted_uplift.startswith("+") else base["performance_score"],
        "fatigued": False,
        "insights": missing_features_explained,
        "cluster_id": original.get("cluster_id", ""),
    }
    store_new_creative(creative_id, new_entry)

    time_e = time.time()

    return {
        "status": "success",
        "creative_id": new_id,
        "new_image_url": new_entry["image_url"],
        "metadata": {
            "api_latency_s": round(time_e - time_s, 2),
            "model": "smadex-sam-sdxl-v2",
            "top_cases_used": len(top_cases),
            "missing_features": missing_features,
            "missing_features_explained": missing_features_explained,
            "predicted_uplift": predicted_uplift,
        },
    }
