import os
import sys
import json
import base64
import cv2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# Paths / setup
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(backend_dir)

load_dotenv(dotenv_path=os.path.join(project_root, ".env"))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def resolve_model_name(raw_model):
    model = (raw_model or "").strip()
    return model.lower() if model else "gpt-4o"


OPENAI_MODEL_NAME = resolve_model_name(os.environ.get("OPENAI_MODEL", "gpt-4o"))
IMG_NMBR = sys.argv[1] if len(sys.argv) > 1 else "500000"

output_dir = os.path.join(project_root, "output", "features", f"creative_{IMG_NMBR}")
cropped_dir = os.path.join(output_dir, "cropped_elements")
elements_json_path = os.path.join(output_dir, "elements_data.json")

data_dir = os.path.join(project_root, "frontend", "public", "data")
creatives_csv = os.path.join(data_dir, "creatives.csv")
visual_semantic_dir = os.path.join(data_dir, "visual_semantic")


def resolve_image_path(creative_id):
    candidates = [
        os.path.join(project_root, "frontend", "public", "data", "assets", f"creative_{creative_id}.png"),
        os.path.join(project_root, "frontend", "public", "data", "assets", f"{creative_id}.png"),
        os.path.join(project_root, "backend", "assets", f"creative_{creative_id}.png"),
        os.path.join(project_root, "backend", "assets", f"{creative_id}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


image_path = resolve_image_path(IMG_NMBR)

ALLOWED_ROLES = [
    "background", "main_subject", "person", "face", "product", "app_screenshot",
    "gameplay", "cta", "headline", "body_text", "logo", "price",
    "discount_badge", "rating", "social_proof", "icon", "decorative_element", "unknown"
]

MAX_ELEMENT_AREA_PCT = 98.0
TEXTUAL_IMPORTANT_ROLES = {"logo", "cta", "headline", "body_text", "price", "discount_badge"}
MIN_DECORATIVE_AREA_PCT = 0.5
MAX_DECORATIVE_ELEMENTS = 4
MAX_TOTAL_ELEMENTS = 12

# ============================================================
# Basic utils
# ============================================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    return "image/jpeg"


def load_elements_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        elements = data.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError(f"Expected elements list in {path}")
        return data, elements
    if isinstance(data, list):
        return {}, data
    raise ValueError(f"Unsupported JSON structure in {path}: {type(data).__name__}")


def clean_text(value):
    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    if text.lower() in {"none", "null", "nan", "n/a", "na", "<none>"}:
        return ""
    return text


def contains_any(text, keywords):
    text = (text or "").lower()
    return any(k in text for k in keywords)


def normalize_role(role):
    role = clean_text(role).lower()
    return role if role in ALLOWED_ROLES else "unknown"


def metadata_value(creative_data, column):
    return clean_text(creative_data.get(column, ""))


def build_brand_terms(creative_data):
    terms = set()
    for col in ["advertiser_name", "app_name"]:
        value = metadata_value(creative_data, col).lower()
        if not value:
            continue
        terms.add(value)
        first_token = value.split()[0]
        if len(first_token) >= 3:
            terms.add(first_token)
    return {t for t in terms if t}

# ============================================================
# Vision classification
# ============================================================
def analyze_element_with_vision(full_image_base64, crop_image_path, text_ocr, coords, advertiser="the brand"):
    crop_base64 = encode_image(crop_image_path)
    full_mime = get_image_mime_type(image_path)
    crop_mime = get_image_mime_type(crop_image_path)

    prompt = f"""You are analyzing a mobile ad for the company/app: '{advertiser}'.

You receive:
- Image 1: the full ad.
- Image 2: one segmented crop from the ad.
- Bounding box in pixels: {coords}
- OCR text detected in this crop: "{text_ocr}"

Classify the crop as ONE useful ad element.

Allowed roles:
{ALLOWED_ROLES}

Role rules:
- Brand/app name => role="logo".
- Button with action text like "Shop now", "Order now", "Install", "Play now", "Book now" => role="cta".
- Promo/discount text like "SALE", "30%", "% OFF", "discount" => role="discount_badge".
- Price like "$4.99", "€9.99" => role="price".
- Main marketing message => role="headline".
- Supporting text below/near headline => role="body_text".
- Arrow, play symbol, glyph => role="icon".
- Generic shape/panel/rectangle/circle/visual decoration => role="decorative_element".
- Large container including multiple smaller elements must NOT be logo/cta/headline; use role="decorative_element".
- If unsure, use role="unknown".

Text extraction:
- If visible text exists, copy it exactly into text_content.
- Prefer OCR text if correct.
- If OCR is missing but text is readable in the crop, fill text_content.
- If no visible text, use null.

Return ONLY valid JSON:
{{
  "role": "...",
  "label": "Short literal name",
  "description": "Short literal visual description of the element and its position.",
  "text_content": "visible text here or null"
}}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{full_mime};base64,{full_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:{crop_mime};base64,{crop_base64}"}},
                ],
            }],
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content.strip())
        return {
            "role": data.get("role", "unknown"),
            "label": data.get("label", ""),
            "description": data.get("description", ""),
            "text_content": data.get("text_content", None),
        }
    except Exception as e:
        print(f"Error analyzing element: {e}")
        return {"role": "unknown", "label": "Unknown element", "description": "Failed to analyze element.", "text_content": None}

# ============================================================
# Geometry
# ============================================================
def recompute_geometry(element, img_w, img_h):
    x1, y1, x2, y2 = element["bbox_xyxy"]
    x1 = max(0, min(int(x1), img_w))
    y1 = max(0, min(int(y1), img_h))
    x2 = max(0, min(int(x2), img_w))
    y2 = max(0, min(int(y2), img_h))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    element["bbox_xyxy"] = [x1, y1, x2, y2]
    element["bbox_normalized"] = [round(x1 / img_w, 3), round(y1 / img_h, 3), round(x2 / img_w, 3), round(y2 / img_h, 3)]
    element["center_normalized"] = [round(((x1 + x2) / 2) / img_w, 3), round(((y1 + y2) / 2) / img_h, 3)]
    element["area_percentage"] = round((w * h * 100) / (img_w * img_h), 2)
    return element


def create_background_element(img_w, img_h):
    return {
        "id": 0,
        "role": "background",
        "label": "Full canvas background",
        "description": "Background covering the full canvas.",
        "text_content": None,
        "bbox_xyxy": [0, 0, img_w, img_h],
        "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "center_normalized": [0.5, 0.5],
        "area_percentage": 100.0,
    }


def is_oversized_element(element):
    area = float(element.get("area_percentage", 0) or 0)
    return area > MAX_ELEMENT_AREA_PCT and element.get("role") != "background"


def bbox_contains(outer, inner, tolerance=0.01):
    if not outer or not inner or len(outer) != 4 or len(inner) != 4:
        return False
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ix1 >= ox1 - tolerance and iy1 >= oy1 - tolerance and ix2 <= ox2 + tolerance and iy2 <= oy2 + tolerance


def bbox_iou(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ============================================================
# Postprocessing rules
# ============================================================
def apply_deterministic_role_rules(element, creative_data):
    e = dict(element)
    role = normalize_role(e.get("role", "unknown"))
    text = clean_text(e.get("text_content") or "")
    text_lower = text.lower()
    label_desc = f"{e.get('label', '')} {e.get('description', '')}".lower()

    brand_terms = build_brand_terms(creative_data)
    metadata_headline = metadata_value(creative_data, "headline").lower()
    metadata_subhead = metadata_value(creative_data, "subhead").lower()
    metadata_cta = metadata_value(creative_data, "cta_text").lower()

    if text_lower and any(term in text_lower for term in brand_terms):
        role = "logo"
        e["label"] = f"{text} text logo"
    elif metadata_cta and text_lower and (text_lower in metadata_cta or metadata_cta in text_lower):
        role = "cta"
    elif metadata_headline and text_lower and (text_lower in metadata_headline or metadata_headline in text_lower):
        role = "headline"
    elif metadata_subhead and text_lower and (text_lower in metadata_subhead or metadata_subhead in text_lower):
        role = "body_text"
    elif text_lower and ("%" in text_lower or "sale" in text_lower or "off" in text_lower or "discount" in text_lower or "promo" in text_lower):
        role = "discount_badge"
    elif text_lower and any(symbol in text_lower for symbol in ["$", "€", "£"]):
        role = "price"
    elif text_lower and contains_any(text_lower, ["shop now", "order now", "install", "install now", "play now", "book now", "sign up", "start now", "get started", "try now", "learn more"]):
        role = "cta"
    elif contains_any(label_desc, ["arrow", "triangle", "play icon", "glyph"]):
        role = "icon"
    elif role == "unknown":
        if contains_any(label_desc, ["rectangle", "shape", "panel", "box", "circle", "background"]):
            role = "decorative_element"
        elif contains_any(label_desc, ["screen", "screenshot", "ui"]):
            role = "app_screenshot"
        elif contains_any(label_desc, ["product", "card"]):
            role = "product"

    e["role"] = role
    return e


def split_headline_body(text):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None
    lowered = cleaned.lower()
    known_pairs = [("limited stock", "free shipping"), ("earn cashback daily", "fast signup"), ("level up today", "play with friends")]
    for first, second in known_pairs:
        if first in lowered and second in lowered:
            i1, i2 = lowered.find(first), lowered.find(second)
            if i1 >= 0 and i2 > i1:
                return cleaned[i1:i2].strip(" ,;.-"), cleaned[i2:].strip(" ,;.-")
    for sep in [";", "|", "\n"]:
        if sep in cleaned:
            parts = [p.strip(" ,") for p in cleaned.split(sep) if p.strip(" ,")]
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])
    cue_phrases = [" fast signup", " instant card", " free shipping", " play with friends", " no fees", " terms apply", " learn more"]
    split_at = -1
    for cue in cue_phrases:
        idx = lowered.find(cue)
        if idx > 0 and (split_at == -1 or idx < split_at):
            split_at = idx
    if split_at > 0:
        return cleaned[:split_at].strip(" ,"), cleaned[split_at:].strip(" ,")
    words = cleaned.replace(",", " ").split()
    if len(words) >= 6:
        return " ".join(words[:3]), " ".join(words[3:])
    return cleaned, None


def remove_bad_containers(elements):
    cleaned = []
    for i, e in enumerate(elements):
        role = e.get("role")
        text = clean_text(e.get("text_content") or "")
        area = float(e.get("area_percentage", 0) or 0)
        bbox = e.get("bbox_normalized", [])
        is_suspicious = role in TEXTUAL_IMPORTANT_ROLES and area >= 8.0
        if not is_suspicious:
            cleaned.append(e)
            continue
        contains_smaller_text_element = False
        for j, other in enumerate(elements):
            if i == j:
                continue
            other_text = clean_text(other.get("text_content") or "")
            other_role = other.get("role")
            other_area = float(other.get("area_percentage", 0) or 0)
            other_bbox = other.get("bbox_normalized", [])
            if not other_text or other_role not in TEXTUAL_IMPORTANT_ROLES or other_area >= area or not bbox_contains(bbox, other_bbox):
                continue
            if not text:
                contains_smaller_text_element = True
                break
            if text.lower() in other_text.lower() or other_text.lower() in text.lower():
                contains_smaller_text_element = True
                break
        if contains_smaller_text_element:
            continue
        if role in {"logo", "cta", "headline", "body_text"} and area >= 12.0:
            e["role"] = "decorative_element"
        cleaned.append(e)
    return cleaned


def deduplicate_overlapping_elements(elements):
    kept = []
    for e in sorted(elements, key=lambda x: float(x.get("area_percentage", 0) or 0)):
        duplicate = False
        e_role = e.get("role")
        e_text = clean_text(e.get("text_content") or "").lower()
        e_bbox = e.get("bbox_normalized", [])
        for existing in kept:
            ex_role = existing.get("role")
            ex_text = clean_text(existing.get("text_content") or "").lower()
            ex_bbox = existing.get("bbox_normalized", [])
            same_role = e_role == ex_role
            same_text = e_text and ex_text and e_text == ex_text
            high_iou = bbox_iou(e_bbox, ex_bbox) >= 0.85
            if same_role and (same_text or high_iou):
                duplicate = True
                break
        if not duplicate:
            kept.append(e)
    return kept


def filter_low_value_decoratives(elements):
    important, decorative = [], []
    for e in elements:
        role = e.get("role")
        area = float(e.get("area_percentage", 0) or 0)
        if role == "decorative_element":
            if area >= MIN_DECORATIVE_AREA_PCT:
                decorative.append(e)
        else:
            important.append(e)
    decorative = sorted(decorative, key=lambda x: float(x.get("area_percentage", 0) or 0), reverse=True)[:MAX_DECORATIVE_ELEMENTS]
    combined = important + decorative
    priority_order = {"background": 0, "logo": 1, "headline": 2, "body_text": 3, "cta": 4, "price": 5, "discount_badge": 6, "product": 7, "main_subject": 8, "app_screenshot": 9, "gameplay": 10, "icon": 11, "decorative_element": 12, "unknown": 13}
    return sorted(combined, key=lambda x: (priority_order.get(x.get("role"), 99), -float(x.get("area_percentage", 0) or 0)))[:MAX_TOTAL_ELEMENTS]


def postprocess_elements_for_similarity(elements, creative_data, img_w, img_h):
    processed = [apply_deterministic_role_rules(e, creative_data) for e in elements]
    split_elements = []
    for e in processed:
        text = clean_text(e.get("text_content") or "")
        role = e.get("role", "")
        if text and role in {"main_subject", "headline", "body_text", "unknown"}:
            headline_text, body_text = split_headline_body(text)
            should_split = bool(body_text) and headline_text and headline_text.lower() != text.lower()
            if should_split:
                x1, y1, x2, y2 = e["bbox_xyxy"]
                if (x2 - x1) >= int(img_w * 0.9) and (y2 - y1) >= int(img_h * 0.85):
                    hx1, hx2 = int(img_w * 0.08), int(img_w * 0.92)
                    hy1, hy2 = int(img_h * 0.62), int(img_h * 0.68)
                    by1, by2 = int(img_h * 0.68), int(img_h * 0.74)
                else:
                    hx1, hx2 = x1, x2
                    mid_y = y1 + (y2 - y1) // 2
                    hy1, hy2 = y1, mid_y
                    by1, by2 = mid_y, y2
                split_elements.append(recompute_geometry({"id": 0, "role": "headline", "label": "Headline text", "description": "Primary marketing headline text.", "text_content": headline_text, "bbox_xyxy": [hx1, hy1, hx2, hy2]}, img_w, img_h))
                split_elements.append(recompute_geometry({"id": 0, "role": "body_text", "label": "Body text", "description": "Supporting descriptive text below the headline.", "text_content": body_text, "bbox_xyxy": [hx1, by1, hx2, by2]}, img_w, img_h))
                continue
        split_elements.append(e)
    processed = [e for e in split_elements if not is_oversized_element(e)]
    processed = remove_bad_containers(processed)
    processed = deduplicate_overlapping_elements(processed)
    if not any(e.get("role") == "background" for e in processed):
        processed.insert(0, create_background_element(img_w, img_h))
    processed = filter_low_value_decoratives(processed)
    for idx, e in enumerate(processed, start=1):
        e["id"] = idx
    return processed

# ============================================================
# Embedding texts
# ============================================================
def position_phrase(center):
    if not center or len(center) != 2:
        return "unknown position"
    x, y = center
    vertical = "top" if y < 0.25 else "bottom" if y > 0.75 else "middle"
    horizontal = "left" if x < 0.33 else "right" if x > 0.67 else "center"
    if vertical == "middle" and horizontal == "center":
        return "center"
    return f"{vertical} {horizontal}"


def size_phrase(area):
    area = float(area or 0)
    if area < 1:
        return "tiny"
    if area < 5:
        return "small"
    if area < 15:
        return "medium"
    if area < 40:
        return "large"
    return "very large"


def build_semantic_elements_text(final_elements):
    parts = []
    for e in final_elements:
        role = e.get("role", "unknown")
        label = clean_text(e.get("label", ""))
        text = clean_text(e.get("text_content") or "")
        area = e.get("area_percentage", 0)
        center = e.get("center_normalized", [])
        if role == "background":
            parts.append("background: full canvas background.")
            continue
        piece = f"{role}: {label}, {size_phrase(area)}, located at {position_phrase(center)}"
        if text:
            piece += f', text: "{text}"'
        parts.append(piece + ".")
    return " ".join(parts)


def build_ocr_text(final_elements, creative_data=None):
    texts = []
    for e in final_elements:
        text = clean_text(e.get("text_content") or "")
        if text and text.lower() not in " ".join(texts).lower():
            texts.append(text)
    if creative_data:
        for col in ["headline", "subhead", "cta_text"]:
            value = clean_text(creative_data.get(col, ""))
            if value and value.lower() not in " ".join(texts).lower():
                texts.append(value)
    return " ".join(texts)


def build_layout_text(final_elements):
    parts = []
    important_roles = {"background", "logo", "headline", "body_text", "cta", "price", "discount_badge", "product", "main_subject", "app_screenshot", "gameplay", "icon"}
    for e in final_elements:
        role = e.get("role", "unknown")
        if role not in important_roles:
            continue
        label = clean_text(e.get("label", "")) or role
        bbox = e.get("bbox_normalized", [])
        center = e.get("center_normalized", [])
        parts.append(f"{role} {label} at {position_phrase(center)}, bbox={bbox}")
    return ". ".join(parts)


def build_precise_elements_text(final_elements):
    lines = []
    for e in final_elements:
        text = clean_text(e.get("text_content", ""))
        text_str = f'text="{text}"' if text else "text=<none>"
        lines.append(
            f"id={e.get('id')} | role={e.get('role', 'unknown')} | label={clean_text(e.get('label', '')) or '-'} | {text_str} | "
            f"bbox_xyxy={e.get('bbox_xyxy', [])} | bbox_norm={e.get('bbox_normalized', [])} | center={e.get('center_normalized', [])} | area_pct={e.get('area_percentage', 0)}"
        )
    return "\n".join(lines)


def build_fallback_global_text(creative_data, final_elements):
    app_name = metadata_value(creative_data, "app_name") or "the app"
    vertical = metadata_value(creative_data, "vertical")
    objective = metadata_value(creative_data, "objective")
    roles = []
    for e in final_elements:
        role = e.get("role")
        if role and role not in roles:
            roles.append(role)
    visible_text = build_ocr_text(final_elements, creative_data)
    return f"Mobile ad for {app_name}. Vertical: {vertical}. Objective: {objective}. Detected elements: {', '.join(roles)}. Visible text: {visible_text}.".strip()

# ============================================================
# Global description
# ============================================================
def generate_global_and_embeddings(creative_data, final_elements):
    advertiser = creative_data.get("advertiser_name", "the brand")
    app_name = creative_data.get("app_name", advertiser)
    elements_summary = json.dumps([
        {"role": e.get("role"), "label": e.get("label"), "text": e.get("text_content"), "bbox": e.get("bbox_normalized"), "position": position_phrase(e.get("center_normalized", [])), "area_pct": e.get("area_percentage")}
        for e in final_elements
    ], indent=2, ensure_ascii=False)
    prompt = f"""Analyze the layout and composition of this mobile ad for '{app_name}'.

Detected structured elements:
{elements_summary}

Create a concise but complete global description.
Do not overdescribe tiny decorative elements.
Focus on the ad structure: background, brand/logo, main subject, headline, body text, CTA, price/discount, and main visual style.

Return ONLY a valid JSON object in English, matching this EXACT structure:
{{
  "global": {{
    "description": "A concise literal description of the entire ad layout and composition.",
    "visual_style": "Concise visual style, for example: Flat design with high contrast",
    "main_message": "The core marketing message",
    "dominant_colors": ["color1", "color2"],
    "emotional_tone": "The psychological feeling of the ad"
  }},
  "embedding_texts": {{
    "global_text": "Concise semantic summary for search.",
    "elements_text": "",
    "ocr_text": "",
    "layout_text": ""
  }}
}}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "You are a JSON-only API."}, {"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generating global description: {e}")
        return {
            "global": {
                "description": build_fallback_global_text(creative_data, final_elements),
                "visual_style": "Unknown visual style",
                "main_message": metadata_value(creative_data, "headline"),
                "dominant_colors": [],
                "emotional_tone": "Unknown tone",
            },
            "embedding_texts": {"global_text": build_fallback_global_text(creative_data, final_elements), "elements_text": "", "ocr_text": "", "layout_text": ""},
        }

# ============================================================
# Main
# ============================================================
def main():
    print("🚀 INICIANT PIPELINE JSON ESTRUCTURAT...")
    if not image_path:
        print(f"❌ Error: No s'ha trobat cap imatge per a {IMG_NMBR}.")
        return
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"❌ Error: No s'ha pogut llegir la imatge original a {image_path}")
        return
    img_h, img_w = img_cv2.shape[:2]
    full_image_base64 = encode_image(image_path)

    df_creatives = pd.read_csv(creatives_csv, skipinitialspace=True)
    try:
        creative_data = df_creatives[df_creatives["creative_id"] == int(IMG_NMBR)].iloc[0].to_dict()
    except IndexError:
        print(f"⚠️ Avís: No hi ha metadades per a {IMG_NMBR} al CSV.")
        creative_data = {}

    if not os.path.exists(elements_json_path):
        print(f"❌ Error: No s'ha trobat {elements_json_path}")
        print("   Abans has de generar 'cropped_elements' i 'elements_data.json' amb mask_generator.py.")
        return

    feature_data, sam_elements = load_elements_data(elements_json_path)
    raw_elements = []
    print("🧠 Analitzant elements individuals amb GPT-4o Vision...")
    for obj in sam_elements:
        if not isinstance(obj, dict):
            print(f"  -> Saltant element invàlid: {obj!r}")
            continue
        if "id" not in obj or "coords" not in obj:
            print(f"  -> Saltant element sense camps mínims: {obj!r}")
            continue
        crop_path = os.path.join(cropped_dir, f"element_{obj['id']}.jpg")
        if not os.path.exists(crop_path):
            print(f"  -> No existeix el retall per element ID:{obj['id']}, saltant.")
            continue
        text_ocr = " ".join(obj.get("text", [])) if isinstance(obj.get("text"), list) else obj.get("text", "")
        x1, y1, x2, y2 = map(float, obj["coords"])
        print(f"  -> Processant element ID:{obj['id']}...")
        ai_data = analyze_element_with_vision(full_image_base64, crop_path, text_ocr, obj["coords"], advertiser=creative_data.get("app_name", creative_data.get("advertiser_name", "unknown")))
        ocr_text = clean_text(text_ocr)
        llm_text = clean_text(ai_data.get("text_content", ""))
        final_text_content = ocr_text or llm_text or None
        candidate = {
            "id": obj["id"],
            "role": ai_data.get("role", "unknown"),
            "label": ai_data.get("label", ""),
            "description": ai_data.get("description", ""),
            "text_content": final_text_content,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        }
        candidate = recompute_geometry(candidate, img_w, img_h)
        if is_oversized_element(candidate):
            print(f"  -> Saltant element ID:{obj['id']} per area_percentage={candidate['area_percentage']} (> {MAX_ELEMENT_AREA_PCT})")
            continue
        raw_elements.append(candidate)

    print("🧹 Postprocessant elements...")
    if not creative_data and feature_data.get("global_description"):
        creative_data = {"global_description": feature_data["global_description"]}
    final_elements = postprocess_elements_for_similarity(raw_elements, creative_data, img_w, img_h)

    print("📝 Generant global description i textos per embeddings...")
    global_data = generate_global_and_embeddings(creative_data, final_elements)
    if not isinstance(global_data, dict):
        global_data = {}
    if not isinstance(global_data.get("global"), dict):
        global_data["global"] = {}
    if not isinstance(global_data.get("embedding_texts"), dict):
        global_data["embedding_texts"] = {}
    if not clean_text(global_data["embedding_texts"].get("global_text", "")):
        global_data["embedding_texts"]["global_text"] = build_fallback_global_text(creative_data, final_elements)

    # Deterministic embedding texts override LLM variability.
    global_data["embedding_texts"]["elements_text"] = build_semantic_elements_text(final_elements)
    global_data["embedding_texts"]["ocr_text"] = build_ocr_text(final_elements, creative_data)
    global_data["embedding_texts"]["layout_text"] = build_layout_text(final_elements)
    global_data["embedding_texts"]["debug_elements_text"] = build_precise_elements_text(final_elements)

    final_json = {
        "creative_id": str(IMG_NMBR),
        "asset_file": f"assets/creative_{IMG_NMBR}.png",
        "canvas": {"width": img_w, "height": img_h},
        "global": global_data.get("global", {}),
        "elements": final_elements,
        "embedding_texts": global_data.get("embedding_texts", {}),
    }

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visual_semantic_dir, exist_ok=True)
    output_final = os.path.join(output_dir, f"creative_{IMG_NMBR}_structured.json")
    semantic_output = os.path.join(visual_semantic_dir, f"creative_{IMG_NMBR}.json")
    with open(output_final, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    with open(semantic_output, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✨ JSON ESTRUCTURAT GENERAT AMB ÈXIT (ID: {IMG_NMBR}) ✨")
    print(f"📄 Guardat a: {output_final}")
    print(f"📄 Copiat també a: {semantic_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
