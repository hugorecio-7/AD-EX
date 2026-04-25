"""
Step 2 — Feature Analysis Helpers

Two responsibilities:
  A. GPT-4o Vision enrichment — ported from generate/postllm.py, kept in sync.
     Uses the same prompts, post-processing, and output format.
  B. Missing-feature extraction — compares enriched visual_semantic.json roles
     between the query creative and top performers; falls back to explain() parsing.

NOTE: This file is the canonical importable module version of postllm.py.
      Keep both in sync when either changes.
"""
from __future__ import annotations

import os
import json
import base64
from collections import Counter
from pathlib import Path
from typing import Any

# ── Path resolution ───────────────────────────────────────────────────────────
_HELPERS_DIR = Path(__file__).resolve().parent          # pipeline/step2_feature_analysis
_PIPELINE_DIR = _HELPERS_DIR.parent                     # pipeline/
_BACKEND_DIR = _PIPELINE_DIR.parent                     # backend/
_PROJECT_ROOT = _BACKEND_DIR.parent                     # repo root

OUTPUT_FEATURES_DIR = _PROJECT_ROOT / "output" / "features"

# ── Model name (configurable via OPENAI_MODEL env var) ───────────────────────
def _resolve_model_name(raw: str | None) -> str:
    model = (raw or "").strip()
    return model.lower() if model else "gpt-4o"

OPENAI_MODEL_NAME = _resolve_model_name(os.environ.get("OPENAI_MODEL"))

# ── OpenAI client (lazy init) ─────────────────────────────────────────────────
_openai_client = None

def _get_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to backend/.env or export it in your shell."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── Constants ─────────────────────────────────────────────────────────────────
ALLOWED_ROLES = [
    "background", "main_subject", "person", "face", "product", "app_screenshot",
    "gameplay", "cta", "headline", "body_text", "logo", "price",
    "discount_badge", "rating", "social_proof", "icon", "decorative_element", "unknown",
]


# ─────────────────────────────────────────────────────────────────────────────
# A1. Low-level image utilities  (mirrored from postllm.py)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_mime_type(path: str | Path) -> str:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/jpeg"


def _load_elements_data(path: str | Path) -> tuple[dict, list]:
    """Accepts both formats: direct list or dict with 'elements' key."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        elements = data.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError(f"Expected 'elements' to be a list in {path}")
        return data, elements

    if isinstance(data, list):
        return {}, data

    raise ValueError(f"Unsupported JSON structure in {path}: {type(data).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# A2. GPT-4o Vision enrichment  (mirrored from postllm.py)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_element_with_vision(
    full_image_base64: str,
    crop_image_path: str | Path,
    text_ocr: str,
    coords: list,
    advertiser: str = "the brand",
    full_image_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Classify one element crop using GPT-4o Vision.
    Returns {"role": ..., "label": ..., "description": ...}
    """
    client = _get_client()
    crop_b64 = _encode_image(crop_image_path)
    full_mime = _get_image_mime_type(full_image_path) if full_image_path else "image/png"
    crop_mime = _get_image_mime_type(crop_image_path)

    prompt = (
        f"You are analyzing a mobile ad for the company: '{advertiser}'.\n"
        f"Analyze Image 1 (full ad) and Image 2 (specific element).\n"
        f"Bounding box (pixels): {coords}\n\n"
        "Your task is to describe this element LITERALLY.\n"
        "- Mention shapes, colors, and specific visual details (e.g., 'a blue rounded rectangle', 'two dots in the top left corner').\n"
        f"- Describe its exact position and appearance within the context of the '{advertiser}' ad.\n"
        f"- Assign one role: {ALLOWED_ROLES}\n"
        "- If text equals or closely matches the brand/app name, role must be 'logo' (not 'headline').\n"
        "- If it's a generic shape/background/panel, prefer 'decorative_element' over 'unknown'.\n\n"
        'Return ONLY a JSON:\n'
        '{\n'
        '  "role": "...",\n'
        '  "label": "Short literal name (e.g., \'Red CTA Button\')",\n'
        '  "description": "Literal visual description of the element\'s appearance and position."\n'
        '}'
    )
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{full_mime};base64,{full_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{crop_mime};base64,{crop_b64}"}},
                ],
            }],
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[FeatureAnalysis] Vision API error for crop: {e}")
        return {"role": "unknown", "label": "Unknown element", "description": "Failed to analyze."}


def generate_global_and_embeddings(
    creative_data: dict,
    final_elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Ask GPT-4o to produce a global description + embedding_texts for the creative.
    Returns {"global": {...}, "embedding_texts": {...}}
    """
    advertiser = creative_data.get("advertiser_name", "the brand")

    elements_summary = json.dumps([{
        "role": e["role"],
        "label": e["label"],
        "desc": e.get("description", ""),
        "pos": e.get("bbox_normalized", []),
    } for e in final_elements], indent=2)

    prompt = (
        f"Analyze the layout and composition of this mobile ad for '{advertiser}'.\n"
        f"Based on these specific elements detected:\n{elements_summary}\n\n"
        "Create a LONG and DETAILED global description.\n"
        "Explain the ad as if you were describing it to a blind person:\n"
        f"- 'An ad for {advertiser} that features [Background]...'\n"
        "- 'In the center there are [Elements]...'\n"
        "- 'At the bottom, we see [CTA]...'\n"
        "- Be specific about the visual structure and literal layout.\n\n"
        "Return ONLY a valid JSON object in English, matching this EXACT structure:\n"
        "{\n"
        '  "global": {\n'
        '    "description": "A very long, comprehensive literal description of the entire ad layout and composition.",\n'
        '    "visual_style": "Concise visual style (e.g. \'Flat design with high contrast\')",\n'
        '    "main_message": "The core marketing message",\n'
        '    "dominant_colors": ["color1", "color2"],\n'
        '    "emotional_tone": "The psychological feeling of the ad"\n'
        "  },\n"
        '  "embedding_texts": {\n'
        '    "global_text": "Detailed semantic summary for search.",\n'
        '    "elements_text": "background: ... main_subject: ...",\n'
        '    "ocr_text": "All visible text combined",\n'
        '    "layout_text": "Literal structure: [element] at [position], [element] at [position]..."\n'
        "  }\n"
        "}"
    )
    try:
        response = _get_client().chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a JSON-only API."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"[FeatureAnalysis] Global embeddings API error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# A3. Post-processing  (mirrored from postllm.py)
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(value: Any) -> str:
    return " ".join((value or "").strip().split())


def _split_headline_body(text: str) -> tuple[str | None, str | None]:
    cleaned = _clean_text(text)
    if not cleaned:
        return None, None

    lowered = cleaned.lower()
    cue_phrases = [" fast signup", " instant card", " no fees", " terms apply", " learn more"]
    split_at = -1
    for cue in cue_phrases:
        idx = lowered.find(cue)
        if idx > 0 and (split_at == -1 or idx < split_at):
            split_at = idx
    if split_at > 0:
        headline = cleaned[:split_at].strip(" ,")
        body = cleaned[split_at:].strip(" ,")
        if headline and body:
            return headline, body

    for sep in [";", "|", "\n"]:
        if sep in cleaned:
            parts = [p.strip(" ,") for p in cleaned.split(sep) if p.strip(" ,")]
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])

    words = cleaned.replace(",", " ").split()
    if len(words) >= 6:
        return " ".join(words[:3]), " ".join(words[3:])

    return cleaned, None


def _recompute_geometry(element: dict, img_w: int, img_h: int) -> dict:
    x1, y1, x2, y2 = element["bbox_xyxy"]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    element["bbox_normalized"] = [
        round(x1 / img_w, 3), round(y1 / img_h, 3),
        round(x2 / img_w, 3), round(y2 / img_h, 3),
    ]
    element["center_normalized"] = [
        round(((x1 + x2) / 2) / img_w, 3),
        round(((y1 + y2) / 2) / img_h, 3),
    ]
    element["area_percentage"] = round((w * h * 100) / (img_w * img_h), 2)
    return element


def _create_background_element(img_w: int, img_h: int) -> dict:
    return {
        "id": 0,
        "role": "background",
        "label": "Solid background",
        "description": "Solid background covering the full canvas.",
        "text_content": None,
        "bbox_xyxy": [0, 0, img_w, img_h],
        "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "center_normalized": [0.5, 0.5],
        "area_percentage": 100.0,
    }


def postprocess_elements_for_similarity(
    elements: list[dict],
    creative_data: dict,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """
    Post-process the Vision-enriched element list for similarity search quality.
    Mirrors postllm.py's postprocess_elements_for_similarity exactly.
    """
    advertiser = _clean_text(creative_data.get("advertiser_name", "")).lower()
    app_name = _clean_text(creative_data.get("app_name", "")).lower()
    brand_terms = {t for t in [advertiser, app_name] if t}

    processed = []
    for element in elements:
        e = dict(element)
        text = _clean_text(e.get("text_content") or "")
        text_lower = text.lower()
        label_desc = f"{e.get('label', '')} {e.get('description', '')}".lower()

        # 1) Brand text → logo
        if text and any(term and term in text_lower for term in brand_terms):
            e["role"] = "logo"
            e["label"] = f"{text} text logo"

        # 2) Resolve 'unknown' to a safer visual role
        if e.get("role") == "unknown":
            if any(k in label_desc for k in ["rectangle", "shape", "panel", "box", "background"]):
                e["role"] = "decorative_element"
            elif any(k in label_desc for k in ["icon", "glyph"]):
                e["role"] = "icon"
            elif any(k in label_desc for k in ["screen", "screenshot", "ui"]):
                e["role"] = "app_screenshot"
            elif any(k in label_desc for k in ["product", "card"]):
                e["role"] = "product"

        processed.append(e)

    # 3) Split oversized text region into headline + body_text
    split_done = False
    split_elements = []
    for e in processed:
        text = _clean_text(e.get("text_content") or "")
        is_textual = bool(text)
        is_oversized = (e.get("area_percentage") or 0) >= 80
        role = e.get("role", "")

        if not split_done and is_textual and is_oversized and role in {"main_subject", "headline", "body_text", "unknown"}:
            headline_text, body_text = _split_headline_body(text)
            x1, y1, x2, y2 = e["bbox_xyxy"]

            if (x2 - x1) >= int(img_w * 0.9) and (y2 - y1) >= int(img_h * 0.85):
                hx1, hx2 = int(img_w * 0.08), int(img_w * 0.92)
                hy1, hy2 = int(img_h * 0.40), int(img_h * 0.50)
                by1, by2 = int(img_h * 0.50), int(img_h * 0.58)
            else:
                hx1, hx2 = x1, x2
                mid_y = y1 + (y2 - y1) // 2
                hy1, hy2 = y1, mid_y
                by1, by2 = mid_y, y2

            if headline_text:
                h_el = {
                    "id": 0, "role": "headline", "label": "Headline text",
                    "description": "Primary marketing headline text.",
                    "text_content": headline_text,
                    "bbox_xyxy": [hx1, hy1, hx2, hy2],
                }
                split_elements.append(_recompute_geometry(h_el, img_w, img_h))

            if body_text:
                b_el = {
                    "id": 0, "role": "body_text", "label": "Body text",
                    "description": "Supporting descriptive text below the headline.",
                    "text_content": body_text,
                    "bbox_xyxy": [hx1, by1, hx2, by2],
                }
                split_elements.append(_recompute_geometry(b_el, img_w, img_h))

            split_done = True
            continue

        split_elements.append(e)

    processed = split_elements

    # 4) Ensure background exists
    if not any(e.get("role") == "background" for e in processed):
        processed.insert(0, _create_background_element(img_w, img_h))

    # 5) Reassign stable IDs
    for idx, e in enumerate(processed, start=1):
        e["id"] = idx

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# A4. Full enrichment pipeline  (integrates all of the above)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_creative_with_vision(
    creative_id: str,
    creative_metadata: dict,
    image_path: str | Path,
) -> dict[str, Any] | None:
    """
    Full enrichment pipeline for one creative:
      1. Load element crops from output/features/creative_{id}/cropped_elements/
      2. Call GPT-4o Vision per element crop (with advertiser context)
      3. Post-process elements for similarity quality
      4. Generate global description + embedding texts
      5. Save structured JSON to output/features/creative_{id}/visual_semantic.json

    Returns the structured JSON dict, or None on failure.
    """
    import cv2

    creative_dir = OUTPUT_FEATURES_DIR / f"creative_{creative_id}"
    crops_dir = creative_dir / "cropped_elements"
    elements_data_path = creative_dir / "elements_data.json"

    if not crops_dir.exists():
        print(f"[FeatureAnalysis] No cropped_elements folder for {creative_id}. Run preprocess_masks first.")
        return None

    if not elements_data_path.exists():
        print(f"[FeatureAnalysis] elements_data.json missing for {creative_id}.")
        return None

    feature_data, raw_elements = _load_elements_data(elements_data_path)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[FeatureAnalysis] Could not read image: {image_path}")
        return None
    img_h, img_w = img.shape[:2]

    full_b64 = _encode_image(image_path)
    advertiser = creative_metadata.get("advertiser_name") or creative_metadata.get("advertiser", "the brand")

    final_elements: list[dict] = []
    print(f"[FeatureAnalysis] 🧠 Analyzing {len(raw_elements)} elements with GPT-4o Vision...")

    for obj in raw_elements:
        if not isinstance(obj, dict):
            print(f"  -> Skipping invalid element: {obj!r}")
            continue
        if "id" not in obj or "coords" not in obj:
            print(f"  -> Skipping element missing required fields: {obj!r}")
            continue

        crop_path = crops_dir / f"element_{obj['id']}.jpg"
        text_ocr = obj.get("text", "") or ""
        if isinstance(text_ocr, list):
            text_ocr = " ".join(text_ocr)

        x1, y1, x2, y2 = map(float, obj["coords"])
        bbox_norm = [round(x1/img_w, 3), round(y1/img_h, 3), round(x2/img_w, 3), round(y2/img_h, 3)]
        cx_norm = round(((x1+x2)/2)/img_w, 3)
        cy_norm = round(((y1+y2)/2)/img_h, 3)

        if crop_path.exists():
            print(f"  -> Processing element ID:{obj['id']}...")
            ia = analyze_element_with_vision(
                full_b64, crop_path, text_ocr, obj["coords"],
                advertiser=advertiser,
                full_image_path=image_path,
            )
        else:
            ia = {"role": "unknown", "label": obj.get("label", ""), "description": ""}

        final_elements.append({
            "id": obj["id"],
            "role": ia.get("role", "unknown"),
            "label": ia.get("label", ""),
            "description": ia.get("description", ""),
            "text_content": text_ocr.strip() if text_ocr.strip() else None,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_normalized": bbox_norm,
            "center_normalized": [cx_norm, cy_norm],
            "area_percentage": round(obj.get("area_percentage", obj.get("percentatge_area", 0)), 2),
        })

    # Fallback creative_data from feature_data if metadata was empty
    if not creative_metadata and feature_data.get("global_description"):
        creative_metadata = {"global_description": feature_data["global_description"]}

    # Post-process elements for similarity quality
    print(f"[FeatureAnalysis] 📝 Post-processing elements...")
    final_elements = postprocess_elements_for_similarity(final_elements, creative_metadata, img_w, img_h)

    # Generate global description + embedding texts
    print(f"[FeatureAnalysis] 📝 Generating global description and embedding texts...")
    global_data = generate_global_and_embeddings(creative_metadata, final_elements)

    structured_json = {
        "creative_id": str(creative_id),
        "asset_file": f"assets/creative_{creative_id}.png",
        "canvas": {"width": img_w, "height": img_h},
        "global": global_data.get("global", {}),
        "elements": final_elements,
        "embedding_texts": global_data.get("embedding_texts", {}),
    }

    output_path = creative_dir / "visual_semantic.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(structured_json, f, indent=2, ensure_ascii=False)

    print(f"[FeatureAnalysis] ✓ visual_semantic.json saved → {output_path}")
    return structured_json


# ─────────────────────────────────────────────────────────────────────────────
# B. Missing feature extraction for the diffusion prompt
# ─────────────────────────────────────────────────────────────────────────────

def extract_missing_features_from_enriched(
    query_creative_id: str,
    top_creatives: list,
) -> list[str]:
    """
    Compare enriched visual_semantic.json elements between query and top performers.
    Returns VISUAL DESCRIPTIONS suitable for a Stable Diffusion prompt.
    e.g. ["warm golden gradient background", "person smiling holding phone"]
    NOT metadata labels like ["rewarded_video ad format"].
    """
    query_dir = OUTPUT_FEATURES_DIR / f"creative_{query_creative_id}"
    query_sem_path = query_dir / "visual_semantic.json"

    if query_sem_path.exists():
        with query_sem_path.open("r", encoding="utf-8") as f:
            query_data = json.load(f)
        query_roles = {e.get("role") for e in query_data.get("elements", [])}
    else:
        query_roles = set()

    SKIP_ROLES = {"background", "decorative_element", "unknown", "headline", "body_text", "cta"}
    VISUAL_ROLES = {"main_subject", "person", "face", "product", "app_screenshot", "gameplay", "logo", "icon", "rating", "social_proof", "price", "discount_badge"}

    # Collect visual descriptions from top performers for roles the query is missing
    visual_fragments: list[str] = []
    seen_roles: set[str] = set()

    for creative in top_creatives:
        cid = str(creative.get("creative_id", creative.get("id", "")))
        # Search both output/features and frontend mock location
        sem_paths = [
            OUTPUT_FEATURES_DIR / f"creative_{cid}" / "visual_semantic.json",
            _PROJECT_ROOT / "frontend" / "public" / "data" / "visual_semantic" / f"creative_{cid}.json",
        ]
        for sem_path in sem_paths:
            if sem_path.exists():
                with sem_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # Skip mock JSONs — they have no real descriptions
                if data.get("mock"):
                    break
                for e in data.get("elements", []):
                    role = e.get("role", "unknown")
                    desc = (e.get("description") or "").strip()
                    label = (e.get("label") or "").strip()
                    if role in SKIP_ROLES or role in seen_roles:
                        continue
                    if role not in query_roles and role in VISUAL_ROLES and desc:
                        # Use the element's literal visual description
                        visual_fragments.append(desc)
                        seen_roles.add(role)
                # Also pull global visual style as a fragment
                global_block = data.get("global", {})
                vs = (global_block.get("visual_style") or "").strip()
                colors = global_block.get("dominant_colors", [])
                tone = (global_block.get("emotional_tone") or "").strip()
                if vs and "mock" not in vs.lower() and "synthetic" not in vs.lower():
                    visual_fragments.append(vs)
                break

        if len(visual_fragments) >= 5:
            break

    return visual_fragments[:5]


def parse_explanations_to_features(explanations: list[str]) -> list[str]:
    """
    Visual-cue fallback when no enriched semantic data is available.
    Derives SD-compatible visual descriptions from semantic JSON global fields,
    rather than metadata category labels.
    """
    # Try to extract visual cues from semantic global block fields
    visual_cues: list[str] = []

    for explanation in explanations:
        # explanations may be raw strings or dicts (from Creative.explain())
        if isinstance(explanation, dict):
            global_block = explanation.get("global", {})
            style = (global_block.get("visual_style") or "").strip()
            tone = (global_block.get("emotional_tone") or "").strip()
            colors = global_block.get("dominant_colors", [])
            if style and "mock" not in style.lower() and "synthetic" not in style.lower():
                visual_cues.append(style)
            if tone:
                visual_cues.append(f"{tone} mood lighting")
            if colors:
                visual_cues.append(f"{' and '.join(colors[:2])} color palette")
        else:
            # Plain string from explain() — try to extract visual keywords
            lc = explanation.lower()
            if "dark" in lc or "light" in lc or "gradient" in lc:
                visual_cues.append("dynamic gradient background")
            elif "colorful" in lc or "vibrant" in lc:
                visual_cues.append("vibrant colorful composition")
            elif "minimal" in lc or "clean" in lc:
                visual_cues.append("clean minimalist layout")

    if not visual_cues:
        # Hard fallback — better than metadata labels
        visual_cues = [
            "dynamic advertising background",
            "high quality product photography",
            "professional lighting",
        ]

    # Deduplicate while preserving order
    seen = set()
    result = []
    for cue in visual_cues:
        if cue not in seen:
            seen.add(cue)
            result.append(cue)

    return result[:5]


def format_explanation_paragraph(missing_features: list[str], creative_id: str) -> str:
    """Format the features into a dashboard-friendly text block."""
    if not missing_features:
        return "No significant feature gaps found. The creative already matches top-performer patterns."
    lines = [
        f"Analysis for creative {creative_id} identified {len(missing_features)} high-impact features from the top-performing cluster:",
        "",
    ]
    for feat in missing_features:
        lines.append(f"  • {feat.capitalize()}")
    lines += [
        "",
        "These features have been injected into the diffusion prompt to maximise predicted CTR uplift.",
    ]
    return "\n".join(lines)
