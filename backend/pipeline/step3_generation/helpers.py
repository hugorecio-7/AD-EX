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

    Key principles:
      - SHORTER is better — SD follows concise prompts more faithfully
      - DESCRIPTIVE not IMPERATIVE — describe the desired final state, don't give instructions
      - ANCHOR to original — repeat the original's visual identity so SD preserves it
      - NO TEXT CUES — never mention text content, OCR, messages, or copy
    """
    parts = []

    # Strong preservation anchor — this is the most important part
    parts.append(
        "professional mobile ad creative, same composition, same layout, same brand colors"
    )

    # Load semantic description to anchor the visual identity
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
        # Truncate long descriptions — SD attention drops off after ~50 tokens
        if len(semantic_desc) > 120:
            semantic_desc = semantic_desc[:120].rsplit(" ", 1)[0] + "..."
        parts.append(f"original scene: {semantic_desc}")

    # Dominant colors — anchor SD to the existing palette
    colors = metadata.get("dominant_colors") or []
    if colors:
        color_str = " and ".join(c for c in colors[:2] if c)
        if color_str:
            parts.append(f"{color_str} color scheme")

    # Emotional tone → lighting (keep it simple)
    tone = (metadata.get("emotional_tone") or "").strip().lower()
    tone_map = {
        "exciting": "energetic lighting",
        "calm": "soft ambient light",
        "luxurious": "golden cinematic light",
        "playful": "bright cheerful colors",
        "serious": "clean professional light",
    }
    if tone in tone_map:
        parts.append(tone_map[tone])

    # Missing visual features — limited to 2, kept subtle
    if missing_features:
        for feat in missing_features[:2]:
            feat = feat.strip()
            skip_words = ("format", "themed", "hook style", "ad format", "objective", "vertical",
                          "text", "copy", "headline", "message", "cta", "call to action")
            if not any(w in feat.lower() for w in skip_words):
                parts.append(f"subtle {feat}")

    # Quality tail with aggressive anti-text
    parts.append(
        "refined background only, sharp focus, 8k, photographic quality, "
        "absolutely no text, no words, no letters, no writing, no typography"
    )

    return ", ".join(parts)

