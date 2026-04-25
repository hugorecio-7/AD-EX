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
    """Build a diffusion prompt from the creative's metadata and missing features."""
    theme = metadata.get("theme", "")
    format_type = metadata.get("format", "")
    
    base = "high quality digital advertising creative, "
    if theme:
        base += f"{theme} theme, "
    if format_type:
        base += f"{format_type} format, "
    if missing_features:
        base += ", ".join(missing_features) + ", "
    base += "vibrant colors, professional render, 8k, sharp focus, centered composition"
    return base
