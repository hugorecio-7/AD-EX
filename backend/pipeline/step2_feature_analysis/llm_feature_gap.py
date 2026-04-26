"""
Step 2 — LLM Feature Gap Analysis

Given:
  - The query creative's visual_semantic.json  (what it HAS)
  - Top-performing creatives' visual_semantic.json files (what they HAVE)

An LLM (GPT-4o) identifies which VISUAL features are present in the top
performers but missing from the original creative, expressed as
Stable Diffusion-compatible image descriptions.

Output example:
  [
    "warm golden bokeh background with soft light rays",
    "smiling person holding smartphone in foreground",
    "vibrant coral-to-purple gradient overlay",
  ]

These are injected directly into the SD inpainting prompt.
"""
from __future__ import annotations

import json
import os
import base64
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# ── Path resolution ────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent            # step2_feature_analysis
_BACKEND_DIR = _THIS_DIR.parent.parent                 # backend
_PROJECT_ROOT = _BACKEND_DIR.parent                    # repo root

# Load environment variables
env_path = _PROJECT_ROOT / ".env"
backend_env_path = _BACKEND_DIR / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv(dotenv_path=backend_env_path, override=True)

OUTPUT_FEATURES_DIR = _PROJECT_ROOT / "output" / "features"
FRONTEND_SEMANTIC_DIR = _PROJECT_ROOT / "frontend" / "public" / "data" / "visual_semantic"

FRONTEND_ASSETS_DIR = _PROJECT_ROOT / "frontend" / "public" / "data" / "assets"

TEXTUAL_ROLES = {
    "headline",
    "body_text",
    "price",
    "discount_badge",
    "rating",
    "social_proof",
    "logo",
    "cta",
}


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
# ── OpenAI client (lazy) ──────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in backend/.env")
        _client = OpenAI(api_key=api_key)
    return _client


# ── Semantic JSON loader ───────────────────────────────────────────────────────

def _load_semantic(creative_id: str) -> dict | None:
    """Load visual_semantic.json for a creative. Returns None if not found or mock."""
    candidates = [
        OUTPUT_FEATURES_DIR / f"creative_{creative_id}" / "visual_semantic.json",
        FRONTEND_SEMANTIC_DIR / f"creative_{creative_id}.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("mock"):
                print(f"[LLMGap]   ⚠ {path.name} — MOCK data, skipping (no real visual descriptions)")
                return None
            print(f"[LLMGap]   ✓ Loaded real semantic data from {path}")
            return data
    print(f"[LLMGap]   ✗ No visual_semantic.json found for creative {creative_id}")
    return None


def _summarize_creative(data: dict) -> str:
    """Compact but information-rich summary of a creative's semantic data for the LLM."""
    global_block = data.get("global", {})
    elements = data.get("elements", [])
    embedding = data.get("embedding_texts", {})

    lines = [
        f"Creative ID: {data.get('creative_id', 'unknown')}",
        f"Visual Style: {global_block.get('visual_style', 'N/A')}",
        f"Emotional Tone: {global_block.get('emotional_tone', 'N/A')}",
        f"Dominant Colors: {', '.join(global_block.get('dominant_colors', []))}",
        f"Global Description: {global_block.get('description', 'N/A')}",
        "",
        "Visual Elements:",
    ]

    SKIP_ROLES = {"background", "unknown", *TEXTUAL_ROLES}
    for e in elements:
        role = e.get("role", "unknown")
        if role in SKIP_ROLES:
            continue
        desc = e.get("description", "")
        label = e.get("label", "")
        line = f"  [{role}] {label}"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    if embedding.get("layout_text"):
        lines += ["", f"Layout: {embedding['layout_text']}"]

    return "\n".join(lines)


def _remove_textual_information(data: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-textual visual context for LLM comparison."""
    cleaned: dict[str, Any] = {
        "creative_id": data.get("creative_id"),
        "global": dict(data.get("global", {})),
        "elements": [],
        "embedding_texts": {},
    }

    for element in data.get("elements", []):
        if not isinstance(element, dict):
            continue
        role = str(element.get("role", "unknown")).lower()
        if role in TEXTUAL_ROLES:
            continue

        kept = {k: v for k, v in element.items() if k not in {"text_content", "ocr_text"}}
        cleaned["elements"].append(kept)

    if isinstance(data.get("embedding_texts"), dict):
        embedding = dict(data["embedding_texts"])
        embedding.pop("copy_text", None)
        embedding.pop("headline_text", None)
        embedding.pop("cta_text", None)
        cleaned["embedding_texts"] = embedding

    return cleaned


# ── Core LLM call ─────────────────────────────────────────────────────────────

def analyze_feature_gap_with_llm(query_creative_id: str, top_ids: list[str], max_features: int = 5) -> dict:
    """
    Calls GPT-4o Vision to identify what VISUAL BACKGROUND features the top
    performers have that the original is missing.

    Key framing: we are inpainting the BACKGROUND only. Text, CTA, and logo
    regions are masked and will NOT be changed. So we only care about:
      - Background colors, gradients, textures, atmospheres
      - Lighting, glow effects, bokeh, particles
      - Non-text decorative elements (shapes, patterns, overlays)
      - Overall visual mood and depth

    Output: SD-compatible prompt fragments, e.g.:
      "deep cobalt blue radial gradient background"
      "scattered golden hexagon shapes with soft glow"
      "cinematic warm light rays from upper right"
    """
    client = _get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # 1. Load query creative JSON + image
    query_json = _load_semantic(query_creative_id)
    if not query_json:
        print(f"[LLMGap] No semantic JSON for query {query_creative_id}.")
        return {"missing_visual_features": _visual_fallback(), "query_id": query_creative_id, "top_ids_used": []}

    query_json_no_text = _remove_textual_information(query_json)

    query_img_path = FRONTEND_ASSETS_DIR / f"creative_{query_creative_id}.png"

    # Build compact query description (only background + visual style)
    q_global = query_json.get("global", {})
    query_desc = (
        f"Visual style: {q_global.get('visual_style', 'unknown')}\n"
        f"Dominant colors: {', '.join(q_global.get('dominant_colors', []))}\n"
        f"Emotional tone: {q_global.get('emotional_tone', 'unknown')}\n"
        f"Description: {q_global.get('description', '')}"
    )

    # 2. Build user message content
    user_content = [
        {
            "type": "text",
            "text": (
                "You are a Stable Diffusion prompt engineer specializing in mobile ad inpainting.\n\n"
                "TASK: Identify background visual features from top-performing ads that are missing in the original.\n\n"
                "IMPORTANT CONSTRAINT: Text, buttons, and logos are MASKED — Stable Diffusion will NOT touch them.\n"
                "You MUST ONLY describe BACKGROUND and ATMOSPHERIC visual properties:\n"
                "  ✓ Color gradients, solid colors, textures, patterns\n"
                "  ✓ Lighting effects: rays, glow, bokeh, shadows\n"
                "  ✓ Non-text decorative shapes: circles, particles, geometric forms\n"
                "  ✓ Depth effects: blur, vignette, atmospheric haze\n"
                "  ✗ DO NOT describe: text, badges with words, buttons, logos, UI elements\n"
                "  ✗ DO NOT describe: abstract marketing concepts ('better CTA', 'stronger hook')\n\n"
                f"ORIGINAL AD (to improve):\n{query_desc}"
            )
        }
    ]

    if query_img_path.exists():
        b64 = encode_image(query_img_path)
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}})

    # 3. Add top performers
    top_ids_used = []
    user_content.append({"type": "text", "text": "\n\nTOP PERFORMING REFERENCE ADS (study their backgrounds):\n"})

    for tid in top_ids[:3]:  # max 3 references
        t_json = _load_semantic(tid)
        if not t_json:
            continue
        top_ids_used.append(tid)
        t_global = t_json.get("global", {})
        t_desc = (
            f"Visual style: {t_global.get('visual_style', 'unknown')} | "
            f"Colors: {', '.join(t_global.get('dominant_colors', []))} | "
            f"Tone: {t_global.get('emotional_tone', 'unknown')}"
        )
        user_content.append({"type": "text", "text": f"\n[Top ad {tid}]: {t_desc}"})
        t_img = FRONTEND_ASSETS_DIR / f"creative_{tid}.png"
        if t_img.exists():
            b64 = encode_image(t_img)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}})

    # 4. Final instruction with strict output format
    user_content.append({
        "type": "text",
        "text": f"""
Based on what you see, produce {max_features} background enhancement suggestions for the original ad.

Each suggestion must be:
- A 5-12 word visual description usable directly as a Stable Diffusion prompt fragment
- Describing ONLY the background/atmosphere (never text or UI)
- EXACTLY matching the visual style of the original. If the original is flat and simple, suggest flat and simple changes. Do NOT add cinematic effects to simple flat designs.
- Very subtle and graceful. Focus on minor color tweaks, clean layout, and minimal texture.

GOOD examples for simple/flat images:
  "clean minimal background with soft pastel gradient"
  "subtle geometric background pattern with low opacity"
  "flat solid color background with clean edges"
  "minimalist clean layout with subtle drop shadow"

GOOD examples for realistic images:
  "subtle warm lighting adjustments"
  "soft depth of field blur in background"

BAD examples (never output these):
  "Bonus badge" — contains text
  "Centered main message" — text element
  "Better CTA placement" — UI/abstract
  "More vibrant colors" — too vague

Return ONLY this JSON (no extra keys):
{{
  "sd_prompt_fragments": ["fragment 1", "fragment 2", ..., "fragment {max_features}"]
}}"""
    })

    # 5. Call API
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": user_content}],
            temperature=0.25,
            max_tokens=400,
        )
        result = json.loads(response.choices[0].message.content.strip())

        # Accept both key names for robustness
        features = (
            result.get("sd_prompt_fragments")
            or result.get("missing_visual_features")
            or []
        )

        # Filter out any text/UI descriptions that slipped through
        SKIP = ("badge", "button", "text", "cta", "headline", "logo", "message", "label",
                "format", "hook", "themed", "engagement", "placement", "copy")
        features = [f for f in features if not any(k in f.lower() for k in SKIP)]

        print(f"[LLMGap] ✓ {len(features)} SD background fragments:")
        for feat in features:
            print(f"   · {feat}")

        return {
            "missing_visual_features": features[:max_features],
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }

    except Exception as e:
        print(f"[LLMGap] API error: {e}")
        return {
            "missing_visual_features": _visual_fallback(),
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }


def _visual_fallback() -> list[str]:
    """Generic SD background fragments when no semantic data is available."""
    return [
        "deep gradient background with vibrant colors and soft glow",
        "cinematic depth of field with warm bokeh particles",
        "subtle geometric pattern overlay with translucent shapes",
    ]

