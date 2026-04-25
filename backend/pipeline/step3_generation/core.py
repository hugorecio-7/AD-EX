"""
Step 3 — Generation Core

Orchestrates:
  1. Loading/generating the diffusion inpainting mask (pre-computed or on-the-fly)
  2. Running Stable Diffusion inpainting
  3. Evaluating the upgraded creative with the LightGBM performance predictor

Async-safe: all blocking I/O runs in a thread-pool executor.
"""
import os
import asyncio
import numpy as np
from PIL import Image

from pipeline.step3_generation.helpers import (
    resolve_image_path,
    build_prompt,
    OUTPUT_FEATURES_DIR,
    OUTPUT_ASSETS_DIR,
    _PROJECT_ROOT,
)
from pipeline.step3_generation.evaluator import (
    evaluate_creative_from_metadata,
    evaluate_creative,
)
from generate.mask_generator import generate_diffusion_mask


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

async def generate_creative_with_flux(
    creative_id: str,
    metadata: dict,
    missing_features: list[str],
    pipe=None,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
) -> str:
    """Full inpainting pipeline.

    Returns the path of the saved upgraded creative (or the mask path if pipe
    is not provided).
    """
    loop = asyncio.get_event_loop()

    image_path = resolve_image_path(creative_id)
    output_dir = os.path.join(OUTPUT_FEATURES_DIR, f"creative_{creative_id}")

    precomputed_mask_path = os.path.join(output_dir, f"creative_{creative_id}_diffusion_mask.png")

    if os.path.exists(precomputed_mask_path):
        print(f"[ImageGen] Found pre-computed mask for {creative_id}. Skipping SAM/OCR.")
        mask_pil_orig = Image.open(precomputed_mask_path).convert("L")
        mask_np = np.array(mask_pil_orig)
        mask_path = precomputed_mask_path
    else:
        print(f"[ImageGen] Generating mask on-the-fly for {creative_id}...")
        mask_np, elements, mask_path = await loop.run_in_executor(
            None,
            lambda: generate_diffusion_mask(
                image_path=image_path,
                project_root=_PROJECT_ROOT,
                output_dir=output_dir,
            ),
        )

    if pipe is None:
        print("[ImageGen] No diffusion pipe provided — skipping generation step.")
        return mask_path or image_path

    original_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_pil.size

    target_w = (orig_w // 8) * 8
    target_h = (orig_h // 8) * 8

    sd_image = original_pil.resize((target_w, target_h), Image.LANCZOS)
    mask_pil = Image.fromarray(mask_np).resize((target_w, target_h), Image.NEAREST)

    prompt = build_prompt(metadata, missing_features)
    negative_prompt = "text, watermark, typography, words, letters, blurry, ugly, distorted, low quality"

    print(f"[ImageGen] Running inpainting with prompt: {prompt[:80]}...")
    result_sd = await loop.run_in_executor(
        None,
        lambda: pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=sd_image,
            mask_image=mask_pil,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=target_h,
            width=target_w,
        ).images[0],
    )

    result_native = result_sd.resize((orig_w, orig_h), Image.LANCZOS)
    mask_native = Image.fromarray(mask_np).convert("L")
    inverted_mask = mask_native.point(lambda px: 255 - px)

    final_image = result_native.copy()
    final_image.paste(original_pil, (0, 0), inverted_mask)

    os.makedirs(OUTPUT_ASSETS_DIR, exist_ok=True)
    output_filename = f"creative_{creative_id}_upgraded.png"
    output_path = os.path.join(OUTPUT_ASSETS_DIR, output_filename)
    final_image.save(output_path)

    print(f"[ImageGen] Saved upgraded creative → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (real LightGBM model via evaluator.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dynamic_creative(
    creative_id: str | None,
    features: list[str] | None = None,
    metadata: dict | None = None,
    old_ctr: float | None = None,
) -> dict:
    """
    Synchronous evaluation entry-point.

    If `metadata` is provided it is forwarded directly to the LightGBM model.
    Otherwise a best-effort mapping is built from the feature list strings
    (for backwards compat with calls that only pass format/theme/hook).

    Returns the same dict shape as before so callers don't need to change:
        {
            "performance_score": float,
            "predicted_uplift":  str,
            "predicted_ctr":     float,
            "is_fatigued":       bool,
            "logic_version":     str,
            ...
        }
    """
    if metadata:
        result = evaluate_creative_from_metadata(metadata, old_ctr=old_ctr)
    else:
        # Legacy path: build minimal creative_params from the free-text feature list
        # The evaluator will rely on defaults for fields not inferrable from text.
        theme = None
        fmt = None
        hook = None
        if features:
            # The features list sometimes carries format/theme/hook as plain strings
            for f in features:
                fl = f.lower()
                if any(k in fl for k in ("video", "banner", "interstitial", "rewarded", "playable")):
                    fmt = f
                elif any(k in fl for k in ("gameplay", "tutorial", "story", "challenge")):
                    hook = f
                else:
                    theme = f

        creative_params = {
            "format": fmt or "unknown",
            "theme": theme or "unknown",
            "hook_type": hook or "unknown",
        }
        result = evaluate_creative(creative_params, old_ctr=old_ctr)

    return result


async def evaluate_new_creative(
    format_type: str | None,
    theme: str | None,
    hook: str | None,
    creative_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Async wrapper for the evaluator, called from the API layer."""
    loop = asyncio.get_event_loop()
    features = [f for f in [format_type, theme, hook] if f and f not in ("upgraded", "simulated logic")]
    return await loop.run_in_executor(
        None,
        lambda: evaluate_dynamic_creative(
            creative_id, features=features, metadata=metadata
        ),
    )


async def predict_performance_uplift(
    missing_features: list[str],
    creative_id: str,
    metadata: dict | None = None,
) -> str:
    """Returns only the predicted uplift string (e.g. '+12.3%')."""
    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: evaluate_dynamic_creative(
            creative_id, features=missing_features, metadata=metadata
        ),
    )
    return result["predicted_uplift"]
