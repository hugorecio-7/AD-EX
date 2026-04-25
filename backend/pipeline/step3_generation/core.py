import os
import asyncio
import numpy as np
from PIL import Image

from pipeline.step3_generation.helpers import resolve_image_path, build_prompt, OUTPUT_FEATURES_DIR, OUTPUT_ASSETS_DIR, _PROJECT_ROOT
from generate.mask_generator import generate_diffusion_mask

async def generate_creative_with_flux(
    creative_id: str,
    metadata: dict,
    missing_features: list[str],
    pipe=None,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
) -> str:
    """Full inpainting pipeline."""
    loop = asyncio.get_event_loop()

    image_path = resolve_image_path(creative_id)
    output_dir = os.path.join(OUTPUT_FEATURES_DIR, f"creative_{creative_id}")

    precomputed_mask_path = os.path.join(output_dir, f"creative_{creative_id}_diffusion_mask.png")
    
    if os.path.exists(precomputed_mask_path):
        print(f"[ImageGen] Found pre-computed mask for {creative_id}. Skipping SAM/OCR.")
        mask_pil_orig = Image.open(precomputed_mask_path).convert('L')
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
            )
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
        ).images[0]
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

from pipeline.step4_persistence.helpers import load_data
from pipeline.step4_persistence.core import compute_static_performance_score

def evaluate_dynamic_creative(creative_id: str | None, features: list[str] = None) -> dict:
    data = load_data()

    healthy = [e for e in data if not e.get("fatigued", False)]
    if not healthy:
        base_score = 0.82
        base_ctr = 3.5
    else:
        sorted_scores = sorted(float(e.get("performance_score", 0)) for e in healthy)
        p75_idx = int(len(sorted_scores) * 0.75)
        base_score = sorted_scores[min(p75_idx, len(sorted_scores) - 1)]
        ctrs = [float(e.get("ctr", 0)) for e in healthy if e.get("ctr")]
        base_ctr = sum(ctrs) / len(ctrs) if ctrs else 3.0

    feature_bonus = min(len(features or []) * 0.02, 0.15)
    new_score = min(round(base_score + feature_bonus, 3), 0.99)

    old_score = compute_static_performance_score(creative_id or "")["performance_score"] if creative_id else base_score * 0.7
    uplift_pct = round((new_score - old_score) / max(old_score, 0.01) * 100, 1)
    uplift_str = f"+{uplift_pct}%" if uplift_pct >= 0 else f"{uplift_pct}%"

    return {
        "performance_score": new_score,
        "predicted_uplift": uplift_str,
        "predicted_ctr": round(base_ctr * (1 + feature_bonus), 2),
        "is_fatigued": new_score < 0.3,
        "logic_version": "v2-cluster-uplift",
    }

async def evaluate_new_creative(
    format_type: str | None,
    theme: str | None,
    hook: str | None,
    creative_id: str | None = None,
) -> dict:
    loop = asyncio.get_event_loop()
    features = [f for f in [format_type, theme, hook] if f and f not in ("upgraded", "simulated logic")]
    return await loop.run_in_executor(
        None, lambda: evaluate_dynamic_creative(creative_id, features)
    )

async def predict_performance_uplift(missing_features: list[str], creative_id: str) -> str:
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: evaluate_dynamic_creative(creative_id, missing_features)
    )
    return result["predicted_uplift"]
