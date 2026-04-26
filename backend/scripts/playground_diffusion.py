"""
PixelForge Diffusion Playground
================================
Tests the full pipeline: Retrieval → LLM Feature Gap → SD Inpainting

Usage:
  # Single creative, auto LLM gap analysis:
  python backend/scripts/playground_diffusion.py --id 500000 --auto

  # Single creative, manual features:
  python backend/scripts/playground_diffusion.py --id 500000 --features "deep cobalt gradient, golden bokeh particles"

  # Batch test on first N creatives that have masks + semantic JSON:
  python backend/scripts/playground_diffusion.py --batch 5 --auto

  # Tune inference:
  python backend/scripts/playground_diffusion.py --id 500003 --auto --steps 40 --scale 9.0
"""
import os
import sys
import torch
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
script_dir  = Path(__file__).resolve().parent
backend_dir = script_dir.parent
project_root = backend_dir.parent

sys.path.insert(0, str(backend_dir))

load_dotenv(project_root / ".env")
load_dotenv(backend_dir / ".env", override=True)

# ── Lazy SD pipe (load once, reuse across batch) ───────────────────────────────
_pipe = None

def get_pipe(device: str):
    global _pipe
    if _pipe is None:
        from diffusers import StableDiffusionInpaintPipeline
        model_id = "runwayml/stable-diffusion-inpainting"
        print(f"[Playground] Loading SD Inpainting model on {device}...")
        _pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        if device == "cuda":
            _pipe.enable_attention_slicing()
        print("[Playground] Model loaded.")
    return _pipe


# ── Helpers ────────────────────────────────────────────────────────────────────

ASSETS_DIR     = project_root / "frontend" / "public" / "data" / "assets"
SEMANTIC_DIR   = project_root / "frontend" / "public" / "data" / "visual_semantic"
FEATURES_DIR   = project_root / "output" / "features"
PLAYGROUND_DIR = project_root / "output" / "playground"
PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)


def resolve_image(cid: str) -> Path | None:
    for candidate in [ASSETS_DIR / f"creative_{cid}.png", ASSETS_DIR / f"{cid}.png"]:
        if candidate.exists():
            return candidate
    return None


def load_semantic(cid: str) -> dict | None:
    for p in [
        FEATURES_DIR / f"creative_{cid}" / "visual_semantic.json",
        SEMANTIC_DIR / f"creative_{cid}.json",
    ]:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                d = json.load(f)
            return None if d.get("mock") else d
    return None


def get_mask(cid: str, image_path: Path) -> np.ndarray:
    mask_path = FEATURES_DIR / f"creative_{cid}" / f"creative_{cid}_diffusion_mask.png"
    if mask_path.exists():
        print(f"[Playground] Using pre-generated mask.")
        return np.array(Image.open(mask_path).convert("L"))
    else:
        print(f"[Playground] No mask found — generating on-the-fly (slow)...")
        from generate.mask_generator import generate_diffusion_mask
        mask_np, _, _ = generate_diffusion_mask(
            str(image_path), str(project_root),
            str(FEATURES_DIR / f"creative_{cid}")
        )
        return mask_np


def build_sd_prompt(semantic: dict | None, sd_fragments: list[str]) -> str:
    """
    Build a clean SD inpainting prompt from:
    - semantic global block (visual_style, emotional_tone, colors)
    - sd_fragments: SD-ready background descriptions from LLM gap analysis
    """
    parts = ["high quality professional mobile advertising creative"]

    if semantic:
        g = semantic.get("global", {})
        tone_map = {
            "exciting": "dynamic energetic lighting",
            "calm": "soft ambient lighting",
            "luxurious": "cinematic golden lighting",
            "playful": "bright cheerful colors",
            "serious": "clean professional lighting",
            "adventurous": "dramatic landscape lighting",
            "urgent": "intense dramatic lighting",
        }
        style = (g.get("visual_style") or "").strip()
        tone  = (g.get("emotional_tone") or "").lower().strip()
        colors = g.get("dominant_colors") or []

        if style and "mock" not in style.lower() and "synthetic" not in style.lower():
            parts.append(style)
        if tone in tone_map:
            parts.append(tone_map[tone])
        if colors:
            parts.append(f"{' and '.join(colors[:2])} color palette")

    # Inject SD background fragments (the core upgrade)
    SKIP = ("badge", "button", "text", "cta", "headline", "logo",
            "message", "label", "format", "hook", "themed", "placement")
    for frag in sd_fragments[:5]:
        if frag and not any(k in frag.lower() for k in SKIP):
            parts.append(frag)

    parts.append("no text, no typography, no watermarks, vibrant, sharp focus, 8k")
    return ", ".join(parts)


def get_sd_fragments(cid: str, semantic: dict | None, args) -> list[str]:
    """Return SD prompt fragments from LLM gap analysis or manual --features."""
    if args.features:
        return [f.strip() for f in args.features.split(",")]

    if not args.auto:
        return [
            "deep vibrant gradient background with soft glow",
            "warm bokeh light particles",
            "cinematic depth of field",
        ]

    # Auto: Retrieval → LLM gap analysis
    from pipeline.step1_retrieval.core import get_best_creatives
    from pipeline.step2_feature_analysis.llm_feature_gap import analyze_feature_gap_with_llm

    fmt = (semantic or {}).get("format", "banner") if semantic else "banner"
    top_cases = get_best_creatives(cid, fmt, semantic or {})
    top_ids = [str(c.get("creative_id", c.get("id", ""))) for c in top_cases]

    print(f"[Playground] Step 1 Retrieval: top candidates = {top_ids}")
    print(f"[Playground] Step 2 LLM Gap: asking GPT-4o for SD background fragments...")

    result = analyze_feature_gap_with_llm(cid, top_ids, max_features=5)
    fragments = result.get("missing_visual_features", [])

    print(f"[Playground] ✓ LLM returned {len(fragments)} SD fragments:")
    for f in fragments:
        print(f"   · {f}")
    return fragments


# ── Core processing function ───────────────────────────────────────────────────

def process_one(cid: str, args, pipe) -> str | None:
    """Run the full pipeline for one creative. Returns output path or None."""
    print(f"\n{'='*60}")
    print(f"[Playground] Creative: {cid}")

    img_path = resolve_image(cid)
    if not img_path:
        print(f"[Playground] ✗ Image not found for {cid}")
        return None

    semantic = load_semantic(cid)
    if semantic:
        g = semantic.get("global", {})
        print(f"[Playground] Semantic: style='{g.get('visual_style','?')}' tone='{g.get('emotional_tone','?')}' colors={g.get('dominant_colors',[])}")
    else:
        print(f"[Playground] ⚠ No real semantic data — using generic metadata")

    # Get SD fragments (LLM gap or manual)
    sd_fragments = get_sd_fragments(cid, semantic, args)

    # Build final prompt
    prompt = build_sd_prompt(semantic, sd_fragments)
    negative_prompt = (
        "text, words, letters, numbers, typography, alphabet, watermark, "
        "signature, logo, brand name, UI elements, buttons, "
        "blurry, distorted, low quality, grainy, noisy, pixelated, "
        "deformed, ugly, extra limbs, out of focus, oversaturated"
    )

    print(f"[Playground] 📝 PROMPT: {prompt}")
    print(f"[Playground] 🚫 NEGATIVE: {negative_prompt}")

    # Load mask
    mask_np = get_mask(cid, img_path)

    # Run inference
    init_image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = init_image.size
    target_w = (orig_w // 8) * 8
    target_h = (orig_h // 8) * 8

    sd_image  = init_image.resize((target_w, target_h), Image.LANCZOS)
    mask_pil  = Image.fromarray(mask_np).resize((target_w, target_h), Image.NEAREST)

    print(f"[Playground] Running SD Inpainting ({args.steps} steps, scale={args.scale})...")
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

    # Composite: paste original back over non-masked regions
    result_full  = output.resize((orig_w, orig_h), Image.LANCZOS)
    mask_native  = Image.fromarray(mask_np).convert("L")
    inverted     = mask_native.point(lambda px: 255 - px)
    final        = result_full.copy()
    final.paste(init_image, (0, 0), inverted)

    # Save
    out_path = PLAYGROUND_DIR / f"{cid}_result.png"
    final.save(out_path)

    # Also save a side-by-side comparison
    comparison = Image.new("RGB", (orig_w * 3, orig_h))
    comparison.paste(init_image, (0, 0))
    comparison.paste(Image.fromarray(mask_np).convert("RGB"), (orig_w, 0))
    comparison.paste(final, (orig_w * 2, 0))
    comparison.save(PLAYGROUND_DIR / f"{cid}_comparison.png")

    print(f"[Playground] ✓ Saved: {out_path}")
    print(f"[Playground] ✓ Comparison: {PLAYGROUND_DIR / f'{cid}_comparison.png'}")
    return str(out_path)


# ── Batch helper ───────────────────────────────────────────────────────────────

def find_ready_creatives(n: int) -> list[str]:
    """Find first N creatives that have BOTH a mask AND a visual_semantic.json."""
    ready = []
    mask_dirs = sorted(FEATURES_DIR.iterdir())
    for d in mask_dirs:
        cid = d.name.replace("creative_", "")
        mask_exists = (d / f"creative_{cid}_diffusion_mask.png").exists()
        sem_exists  = (SEMANTIC_DIR / f"creative_{cid}.json").exists() or \
                      (FEATURES_DIR / f"creative_{cid}" / "visual_semantic.json").exists()
        # Skip mocks
        sem_path = SEMANTIC_DIR / f"creative_{cid}.json"
        if sem_path.exists():
            with sem_path.open() as f:
                data = json.load(f)
            if data.get("mock"):
                sem_exists = False
        if mask_exists and sem_exists:
            ready.append(cid)
        if len(ready) >= n:
            break
    return ready


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PixelForge Diffusion Playground")
    parser.add_argument("--id",      type=str, help="Single creative ID to process")
    parser.add_argument("--batch",   type=int, help="Process N ready creatives in batch")
    parser.add_argument("--auto",    action="store_true", help="Use LLM gap analysis for features")
    parser.add_argument("--features",type=str, help="Manual comma-separated SD fragments")
    parser.add_argument("--steps",   type=int, default=35, help="Inference steps (default 35)")
    parser.add_argument("--scale",   type=float, default=8.0, help="Guidance scale (default 8.0)")
    args = parser.parse_args()

    if not args.id and not args.batch:
        parser.error("Specify --id <creative_id> or --batch <N>")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe   = get_pipe(device)

    if args.id:
        process_one(args.id, args, pipe)
    else:
        creatives = find_ready_creatives(args.batch)
        print(f"[Playground] Batch mode: {len(creatives)} creatives ready")
        for cid in creatives:
            process_one(cid, args, pipe)

    print(f"\n[Playground] All done. Results in: {PLAYGROUND_DIR}")


if __name__ == "__main__":
    main()
