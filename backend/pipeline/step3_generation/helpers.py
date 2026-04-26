import os

# Project-root resolution
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_DIR = os.path.dirname(_PIPELINE_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

# Assets live inside frontend/public/data/assets (original creatives)
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "frontend", "public", "data", "assets")
OUTPUT_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "frontend", "public", "data", "assets")

# Output mask/debug folder
OUTPUT_FEATURES_DIR = os.path.join(_PROJECT_ROOT, "output", "features")

def resolve_image_path(creative_id: str) -> str:
    """Find the original image on disk."""
    candidates = [
        os.path.join(ASSETS_DIR, f"creative_{creative_id}.png"),
        os.path.join(ASSETS_DIR, f"{creative_id}.png"),
        # Fallback: backend-local assets
        os.path.join(_BACKEND_DIR, "assets", f"creative_{creative_id}.png"),
        os.path.join(_BACKEND_DIR, "assets", f"{creative_id}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find image for creative '{creative_id}'. Tried: {candidates}")

def build_prompt(metadata: dict, missing_features: list[str]) -> str:
    """
    Build a Stable Diffusion inpainting prompt from the creative's semantic data.

    The prompt focuses on VISUAL descriptions — colors, lighting, composition —
    NOT metadata labels like 'rewarded_video ad format'.
    """
    parts = []

    # Base quality anchor
    parts.append("high quality professional mobile advertising creative")

    # Attempt to load semantic description from visual_semantic.json to anchor the generation
    creative_id = str(metadata.get("id", ""))
    semantic_desc = ""
    if creative_id:
        semantic_path = os.path.join(_PROJECT_ROOT, "frontend", "public", "data", "visual_semantic", f"creative_{creative_id}.json")
        if os.path.exists(semantic_path):
            try:
                import json
                with open(semantic_path, "r", encoding="utf-8") as f:
                    sem_data = json.load(f)
                    semantic_desc = sem_data.get("global", {}).get("description", "").strip()
            except Exception:
                pass
    
    if semantic_desc:
        parts.append(f"base composition: {semantic_desc}")

    # Visual style from semantic JSON (if available and not a mock)
    visual_style = (metadata.get("visual_style") or "").strip()
    if visual_style and "mock" not in visual_style.lower() and "synthetic" not in visual_style.lower():
        parts.append(visual_style)

    # Main message — gives the AI context on what the ad is about
    message = (metadata.get("main_message") or "").strip()
    if message:
        parts.append(f"the creative features a {message} message")

    # OCR and Layout Context — gives the AI background context without drawing text
    ocr_text = (metadata.get("ocr_text") or metadata.get("ocr_summary") or "").strip()
    if ocr_text:
        parts.append(f"contextual background description: the scene relates to {ocr_text}")
    
    layout_text = (metadata.get("layout_text") or metadata.get("layout") or "").strip()
    if layout_text:
        parts.append(f"composition context: {layout_text}")

    # Emotional tone → lighting cue
    tone = (metadata.get("emotional_tone") or "").strip()
    tone_map = {
        "exciting": "dynamic energetic lighting",
        "calm": "soft ambient lighting",
        "luxurious": "cinematic golden lighting",
        "playful": "bright cheerful colors",
        "serious": "clean professional lighting",
        "adventurous": "dramatic landscape lighting",
    }
    if tone and tone.lower() in tone_map:
        parts.append(tone_map[tone.lower()])
    elif tone:
        parts.append(f"{tone} mood")

    # Dominant colors as a visual anchor
    colors = metadata.get("dominant_colors") or []
    if colors:
        color_str = " and ".join(c for c in colors[:2] if c)
        if color_str:
            parts.append(f"{color_str} color palette")

    # Missing visual features (should be descriptions, not labels)
    if missing_features:
        for feat in missing_features[:4]:
            feat = feat.strip()
            # Skip any metadata labels that slipped through
            skip_words = ("format", "themed", "hook style", "ad format", "objective", "vertical")
            if not any(w in feat.lower() for w in skip_words):
                parts.append(feat)

    # Quality tail
    parts.append("vibrant professional render, sharp focus, 8k resolution, no text, no typography")

    return ", ".join(parts)
