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

# ── Path resolution ────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent            # step2_feature_analysis
_BACKEND_DIR = _THIS_DIR.parent.parent                 # backend
_PROJECT_ROOT = _BACKEND_DIR.parent                    # repo root

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
    client = _get_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # 1. Carregar JSON i Imatge de l'Original
    query_json = _load_semantic(query_creative_id)
    if not query_json:
        print(f"[LLMGap] No s'ha trobat el JSON de l'original {query_creative_id}.")
        return {"missing_visual_features": _visual_fallback(), "reasoning": "Missing original JSON", "query_id": query_creative_id, "top_ids_used": []}

    query_json_no_text = _remove_textual_information(query_json)

    query_img_path = FRONTEND_ASSETS_DIR / f"creative_{query_creative_id}.png"
    query_b64 = encode_image(query_img_path) if query_img_path.exists() else None

    # Inicialitzar l'array de missatges
    user_content = [
        {
            "type": "text", 
            "text": "You are an expert creative analyst. Compare the ORIGINAL ad with TOP ads to find missing visual features.\n\n"
                    "IMPORTANT: Ignore all textual content and copywriting. Ignore headlines, body copy, CTA wording, logo text, prices, discounts, and OCR.\n"
                    "Focus only on concrete visual differences that can be drawn/generated in an image.\n\n"
                    "ORIGINAL AD JSON (TEXT FILTERED):\n" + json.dumps(query_json_no_text, indent=2)
        }
    ]
    if query_b64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_b64}"}})

    user_content.append({"type": "text", "text": "\n\nTOP PERFORMING ADS:\n"})

    # 2. Carregar JSON i Imatge dels Anuncis Top
    top_ids_used = []
    for tid in top_ids:
        t_json = _load_semantic(tid)
        if not t_json: 
            continue

        t_json_no_text = _remove_textual_information(t_json)
            
        top_ids_used.append(tid)
        t_img_path = FRONTEND_ASSETS_DIR / f"creative_{tid}.png"
        t_b64 = encode_image(t_img_path) if t_img_path.exists() else None

        user_content.append({"type": "text", "text": f"\n--- TOP AD {tid} ---\nJSON (TEXT FILTERED):\n" + json.dumps(t_json_no_text, indent=2)})
        if t_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{t_b64}"}})

    # 3. Instruccions i prompt final
    user_content.append({
        "type": "text", 
        "text": f"""
Based on the images and JSON data, identify up to {max_features} key VISUAL features present in top-performing ads but MISSING in the original ad.

STRICT RULES:
1. Ignore all text and copywriting. Do not mention headlines, body text, CTA text, logo text, offers, prices, percentages, typography, or messaging.
2. Describe only concrete visual properties (backgrounds, lighting, composition, subject pose, objects, color treatment, depth, textures, non-text icons/shapes).
3. Avoid abstract recommendations (forbidden examples: "make it more engaging", "improve hook", "stronger message").
4. Each feature must be a drawable/generable visual change, specific enough for image generation.
2. Return ONLY a valid JSON object matching this EXACT structure:
{{
  "missing_visual_features": ["feature 1", "feature 2"],
    "reasoning": "Brief visual-only explanation"
}}"""
    })

    # 4. Trucada a OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2
        )
        result = json.loads(response.choices[0].message.content.strip())
        
        # Neteja de paraules clau (codi que ja tenies)
        features = result.get("missing_visual_features", [])
        SKIP_KEYWORDS = (
            "format",
            "themed",
            "hook style",
            "ad format",
            "objective",
            "vertical",
            "cta",
            "headline",
            "body text",
            "copy",
            "message",
            "slogan",
            "price",
            "discount",
            "offer",
            "%",
            "font",
            "typography",
            "text",
            "logo text",
            "more engaging",
            "better conversion",
        )
        features = [
            f for f in features
            if isinstance(f, str) and not any(k in f.lower() for k in SKIP_KEYWORDS)
        ]

        print(f"[LLMGap+Vision] ✓ Got {len(features)} visual features!")
        return {
            "missing_visual_features": features[:max_features],
            "reasoning": result.get("reasoning", ""),
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }

    except Exception as e:
        print(f"[LLMGap+Vision] Vision API fail: {e}")
        return {
            "missing_visual_features": _visual_fallback(),
            "reasoning": str(e),
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }


def _visual_fallback() -> list[str]:
    """Generic visual descriptions when no semantic data is available."""
    return [
        "dynamic gradient background with vibrant colors",
        "high quality product photography with professional lighting",
        "cinematic depth of field effect",
    ]
