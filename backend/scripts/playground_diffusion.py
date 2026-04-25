import os
import sys
import torch
import json
import argparse
from PIL import Image
import numpy as np

# Setup pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir)

from diffusers import StableDiffusionInpaintPipeline
from pipeline.step3_generation.helpers import resolve_image_path, build_prompt
from generate.mask_generator import generate_diffusion_mask
from pipeline.step1_retrieval.core import get_best_creatives
from pipeline.step2_feature_analysis.core import find_missing_features

def load_semantic_data(creative_id, project_root):
    """Try to find the visual_semantic.json in possible locations."""
    paths = [
        os.path.join(project_root, "output", "features", f"creative_{creative_id}", "visual_semantic.json"),
        os.path.join(project_root, "frontend", "public", "data", "visual_semantic", f"creative_{creative_id}.json"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="PixelForge Diffusion Playground - Pipeline Simulator")
    parser.add_argument("--id", type=str, default="500000", help="Creative ID to process")
    parser.add_argument("--features", type=str, help="Manually override missing features")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--auto", action="store_true", help="Automatically run retrieval + feature analysis")
    args = parser.parse_args()

    project_root = os.path.dirname(backend_dir)
    output_dir = os.path.join(project_root, "output", "playground")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[Playground] 🧪 Simulating Pipeline for Creative ID: {args.id}")
    
    # 1. Resolve Image & Semantic Data
    try:
        image_path = resolve_image_path(args.id)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    semantic_data = load_semantic_data(args.id, project_root)
    if semantic_data and not semantic_data.get("mock"):
        global_block = semantic_data.get("global", {})
        print(f"[Playground] Semantic data loaded. Style: {global_block.get('visual_style', '?')} | Tone: {global_block.get('emotional_tone', '?')}")
        metadata = {
            "visual_style": global_block.get("visual_style", ""),
            "emotional_tone": global_block.get("emotional_tone", ""),
            "dominant_colors": global_block.get("dominant_colors", []),
            "main_message": global_block.get("main_message", ""),
            "format": semantic_data.get("format", "mobile_ad"),
            "cluster_id": semantic_data.get("cluster_id"),
        }
    else:
        if semantic_data and semantic_data.get("mock"):
            print("[Playground] ⚠️  Found mock semantic data — no real visual descriptions available. Using generic metadata.")
        else:
            print("[Playground] No semantic data found. Using generic metadata.")
        metadata = {"emotional_tone": "exciting", "dominant_colors": ["blue", "white"]}

    # 2. Simulate Step 1 → 2 (Retrieval → LLM Feature Gap)
    if args.auto:
        from pipeline.step2_feature_analysis.llm_feature_gap import analyze_feature_gap_with_llm

        print("[Playground] Step 1: Running Retrieval...")
        top_cases = get_best_creatives(args.id, metadata.get("format", "banner"), metadata)
        top_ids = [str(c.get("creative_id", c.get("id", ""))) for c in top_cases]

        print(f"[Playground] Step 2: Calling GPT-4o to analyze visual gap vs {top_ids}...")
        result = analyze_feature_gap_with_llm(args.id, top_ids)
        missing_features = result["missing_visual_features"]
        print(f"[Playground] Reasoning: {result['reasoning']}")
    else:
        missing_features = [f.strip() for f in args.features.split(",")] if args.features else [
            "dynamic gradient background",
            "high quality product photography",
            "cinematic lighting",
        ]

    print(f"[Playground] TARGET FEATURES: {missing_features}")

    # 3. Load/Generate Mask
    mask_features_dir = os.path.join(project_root, "output", "features", f"creative_{args.id}")
    mask_path = os.path.join(mask_features_dir, f"creative_{args.id}_diffusion_mask.png")
    if not os.path.exists(mask_path):
        print("[Playground] Generating mask...")
        mask_np, _, _ = generate_diffusion_mask(image_path, project_root, mask_features_dir)
    else:
        mask_np = np.array(Image.open(mask_path).convert("L"))

    # 4. Load Diffusion Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Playground] Loading SDXL Inpainting on {device}...")
    model_id = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # 5. Build Final Prompt
    prompt = build_prompt(metadata, missing_features)
    # STRONGER NEGATIVE PROMPT to avoid text/watermarks
    negative_prompt = (
        "text, words, letters, typography, alphabet, characters, "
        "watermark, signature, logo, brand name, "
        "blurry, distorted, low quality, grainy, "
        "extra limbs, deformed, messy, out of focus"
    )

    print(f"[Playground] 📝 FINAL PROMPT: {prompt}")
    print(f"[Playground] 🚫 NEGATIVE PROMPT: {negative_prompt}")

    # 6. Run Inference
    init_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = init_image.size
    target_w, target_h = (orig_w // 8) * 8, (orig_h // 8) * 8
    
    sd_image = init_image.resize((target_w, target_h), Image.LANCZOS)
    mask_pil = Image.fromarray(mask_np).resize((target_w, target_h), Image.NEAREST)

    print(f"[Playground] Running SD Inpainting...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sd_image,
        mask_image=mask_pil,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        height=target_h,
        width=target_w,
    ).images[0]

    # 7. Composite & Save
    result_native = output.resize((orig_w, orig_h), Image.LANCZOS)
    mask_native = Image.fromarray(mask_np).convert("L")
    inverted_mask = mask_native.point(lambda px: 255 - px)

    final_image = result_native.copy()
    final_image.paste(init_image, (0, 0), inverted_mask)

    out_file = os.path.join(output_dir, f"{args.id}_simulated_pipeline.png")
    final_image.save(out_file)
    print(f"[Playground] Result saved to: {out_file}")

if __name__ == "__main__":
    main()
