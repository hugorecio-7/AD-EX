"""
AI Engine — orchestrates the full creative upgrade pipeline:
  Step 1: Retrieve top-performing similar creatives
  Step 2: Identify missing visual features (via GPT-4o Vision or fallback)
  Step 3: Generate upgraded creative with SD inpainting + evaluate with LightGBM
  Step 4: Persist and return the result
"""
import time
import asyncio

from pipeline.step1_retrieval.core import get_best_creatives
from pipeline.step2_feature_analysis.core import (
    find_missing_features,
    explain_missing_features,
    enrich_creative_async,
)
from pipeline.step3_generation.core import (
    generate_creative_with_flux,
    predict_performance_uplift,
)
from pipeline.step4_persistence.core import (
    store_new_creative,
    get_creative_by_id,
    compute_static_performance_score,
)


async def generate_ai_variant_real(
    creative_id: str,
    format_type: str,
    metadata: dict,
    pipe=None,
    num_steps: int = 25,
) -> dict:
    time_s = time.time()

    # ── 1. Retrieve top cases from the same cluster ──────────────────────────
    try:
        top_cases = get_best_creatives(creative_id, format_type, metadata)
        print(f"[Engine] Step 1 — Retrieved {len(top_cases)} top cases.")
    except Exception as e:
        print(f"[Engine] Step 1 FAILED: {e}")
        top_cases = []

    # ── 2. Enrich query creative with GPT-4o Vision (if not pre-computed) ────
    from pipeline.step3_generation.helpers import resolve_image_path
    from pathlib import Path
    from pipeline.step2_feature_analysis.helpers import OUTPUT_FEATURES_DIR

    _PROJECT_ROOT_ENGINE = Path(__file__).resolve().parent.parent.parent
    sem_path = OUTPUT_FEATURES_DIR / f"creative_{creative_id}" / "visual_semantic.json"
    # Also check the batch-precomputed location (frontend/public/data/visual_semantic/)
    sem_path_public = _PROJECT_ROOT_ENGINE / "frontend" / "public" / "data" / "visual_semantic" / f"creative_{creative_id}.json"
    if not sem_path.exists() and not sem_path_public.exists():
        try:
            image_path = resolve_image_path(creative_id)
            print(f"[Engine] Step 2 — Running GPT-4o Vision enrichment for {creative_id}...")
            await enrich_creative_async(creative_id, metadata, image_path)
        except Exception as e:
            print(f"[Engine] Step 2 Vision enrichment skipped ({e}). Falling back to explain().")
    else:
        print(f"[Engine] Step 2 — Semantic JSON already exists for {creative_id}. Skipping enrichment.")

    # ── 3. Identify features the target creative is missing ──────────────────
    try:
        missing_features = find_missing_features(
            explanations_or_creatives=top_cases,
            creative_id=creative_id,
            top_creatives=top_cases,
        )
        print(f"[Engine] Step 3 — Missing features: {missing_features}")
    except Exception as e:
        print(f"[Engine] Step 3 FAILED: {e}")
        missing_features = []

    # ── 4. Run in parallel: generation + uplift prediction + explanation ──────
    async def _generate_and_predict():
        try:
            new_file = await generate_creative_with_flux(
                creative_id=creative_id,
                metadata=metadata,
                missing_features=missing_features,
                pipe=pipe,
                num_steps=num_steps,
            )
            print(f"[Engine] Step 4a — Generated image: {new_file}")
        except Exception as e:
            print(f"[Engine] Step 4a Generation FAILED: {e}")
            import traceback; traceback.print_exc()
            raise
        try:
            uplift = await predict_performance_uplift(missing_features, creative_id, metadata=metadata)
        except Exception as e:
            print(f"[Engine] Step 4b Uplift prediction FAILED: {e}")
            uplift = "+0.0%"
        return new_file, uplift

    (new_creative_file, predicted_uplift), missing_features_explained = await asyncio.gather(
        _generate_and_predict(),
        explain_missing_features(missing_features, creative_id),
    )

    # ── 5. Persist the new creative ──────────────────────────────────────────
    from pipeline.step4_persistence.helpers import next_available_id
    original = get_creative_by_id(creative_id) or {}
    # Allocate a fresh numeric ID so preprocess_masks.py can handle it
    new_id = str(next_available_id())

    base = compute_static_performance_score(creative_id)

    try:
        uplift_val = float(predicted_uplift.strip("%+").replace("%", ""))
    except ValueError:
        uplift_val = 0.0

    new_score = (
        base["performance_score"] + uplift_val / 100
        if predicted_uplift.startswith("+")
        else base["performance_score"]
    )

    # Save the generated image under a unique filename in the assets dir
    import shutil as _shutil
    from pathlib import Path as _Path
    src = _Path(new_creative_file)
    _PROJECT_ROOT_ENGINE = _Path(__file__).resolve().parent.parent.parent
    assets_dir = _PROJECT_ROOT_ENGINE / "frontend" / "public" / "data" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    new_asset_filename = f"creative_{new_id}.png"
    dst_asset = assets_dir / new_asset_filename
    if src.exists():
        _shutil.copy2(src, dst_asset)

    new_image_url = f"/data/assets/{new_asset_filename}"

    new_entry = {
        **original,
        "id": new_id,
        "image_url": new_image_url,
        "performance_score": round(new_score, 4),
        "fatigued": False,
        "insights": missing_features_explained,
        "cluster_id": original.get("cluster_id", ""),
        "is_upgraded": True,
    }
    store_new_creative(new_id, new_entry)

    # Trigger Enrichment (Background)
    try:
        from pipeline.post_upgrade_enrichment import enrich_upgraded_creative
        enrich_upgraded_creative(creative_id, new_id, dst_asset)
    except Exception as e:
        print(f"[Engine] Background enrichment trigger failed: {e}")

    # Note: enrichment (SAM mask + visual_semantic.json) is triggered by the frontend
    # AFTER the user clicks 'Replace Image', via POST /enrich — not here.
    # This prevents GPU contention with RNN forecasting.

    time_e = time.time()

    return {
        "status": "success",
        "creative_id": new_id,
        "new_image_url": new_image_url,
        "metadata": {
            "api_latency_s": round(time_e - time_s, 2),
            "model": "smadex-sam-sdxl-v2",
            "top_cases_used": len(top_cases),
            "missing_features": missing_features,
            "missing_features_explained": missing_features_explained,
            "predicted_uplift": predicted_uplift,
        },
    }
