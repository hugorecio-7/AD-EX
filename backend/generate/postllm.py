import os
import sys
import json
import base64
import cv2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# 1. CONFIGURACIÓ DE RUTES I VARIABLES
# ==========================================
# Com que estàs a backend/generate/postllm.py:
script_dir = os.path.dirname(os.path.abspath(__file__)) # Apunta a 'generate'
backend_dir = os.path.dirname(script_dir)               # Apunta a 'backend'
project_root = os.path.dirname(backend_dir)             # Apunta a l'arrel

env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# Inicialitzar el client de OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def resolve_model_name(raw_model):
    model = (raw_model or "").strip()
    if not model:
        return "gpt-4o"

    # Normalize common casing mistakes from .env (e.g., gpt-4O-mini).
    return model.lower()

OPENAI_MODEL_NAME = resolve_model_name(os.environ.get("OPENAI_MODEL", "gpt-4o"))

# ID de l'anunci
if len(sys.argv) > 1:
    IMG_NMBR = sys.argv[1]
else:
    IMG_NMBR = "500000"  # Valor per defecte si no es proporciona cap argument

# ==========================================
# RUTES D'OUTPUT (Arreglades segons la captura)
# ==========================================
# Ara apunta a: output/features/creative_500000/
output_dir = os.path.join(project_root, "output", "features", f"creative_{IMG_NMBR}")

# Canviat a 'cropped_elements' com ho fa el mask_generator.py
carpeta_retalls = os.path.join(output_dir, "cropped_elements")

# El fitxer es diu 'elements_data.json'
json_path = os.path.join(output_dir, "elements_data.json")

# ==========================================
# RUTES DE DADES I ASSETS
# ==========================================
data_dir = os.path.join(project_root, "frontend", "public", "data") 
creatives_csv = os.path.join(data_dir, "creatives.csv")
summary_csv = os.path.join(data_dir, "creative_summary.csv")

# Ruta de la imatge original (He suposat que tens la carpeta assets dins de backend/data)
# Si les imatges estan a un altre lloc, canvia aquesta línia!
def resolve_image_path(creative_id):
    candidates = [
        os.path.join(project_root, "frontend", "public", "data", "assets", f"creative_{creative_id}.png"),
        os.path.join(project_root, "frontend", "public", "data", "assets", f"{creative_id}.png"),
        os.path.join(project_root, "backend", "assets", f"creative_{creative_id}.png"),
        os.path.join(project_root, "backend", "assets", f"{creative_id}.png"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None

image_path = resolve_image_path(IMG_NMBR)

# Llista tancada de rols permesos
ALLOWED_ROLES = [
    "background", "main_subject", "person", "face", "product", "app_screenshot", 
    "gameplay", "cta", "headline", "body_text", "logo", "price", 
    "discount_badge", "rating", "social_proof", "icon", "decorative_element", "unknown"
]

def encode_image(image_path):
    """Llegeix una imatge i la converteix en Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_mime_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/jpeg"

def load_elements_data(path):
    """Admet els dos formats: llista directa o diccionari amb clau 'elements'."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        elements = data.get("elements", [])
        if not isinstance(elements, list):
            raise ValueError(f"Expected 'elements' to be a list in {path}")
        return data, elements

    if isinstance(data, list):
        return {}, data

    raise ValueError(f"Unsupported JSON structure in {path}: {type(data).__name__}")

def analyze_element_with_vision(full_image_base64, crop_image_path, text_ocr, coords, advertiser="the brand"):
    """Passa la imatge global i el retall a GPT per classificar l'element de forma literal."""
    crop_base64 = encode_image(crop_image_path)
    full_mime = get_image_mime_type(image_path)
    crop_mime = get_image_mime_type(crop_image_path)
    
    prompt = f"""You are analyzing a mobile ad for the company: '{advertiser}'.
Analyze Image 1 (full ad) and Image 2 (specific element). 
Bounding box (pixels): {coords}

Your task is to describe this element LITERALLY.
- Mention shapes, colors, and specific visual details (e.g., 'a blue rounded rectangle', 'two dots in the top left corner').
- Describe its exact position and appearance within the context of the '{advertiser}' ad.
- Assign one role: {ALLOWED_ROLES}
- If text equals or closely matches the brand/app name, role must be 'logo' (not 'headline').
- If it's a generic shape/background/panel, prefer 'decorative_element' over 'unknown'.

Return ONLY a JSON:
{{
  "role": "...",
  "label": "Short literal name (e.g., 'Red CTA Button')",
  "description": "Literal visual description of the element's appearance and position."
}}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{full_mime};base64,{full_image_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:{crop_mime};base64,{crop_base64}"}}
                    ]
                }
            ],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error analitzant l'element: {e}")
        return {"role": "unknown", "label": "Unknown element", "description": "Failed to analyze element."}

def _clean_text(value):
    return " ".join((value or "").strip().split())

def _split_headline_body(text):
    cleaned = _clean_text(text)
    if not cleaned:
        return None, None

    # Prefer splitting at common supporting-text cues.
    lowered = cleaned.lower()
    cue_phrases = [
        " fast signup",
        " instant card",
        " no fees",
        " terms apply",
        " learn more",
    ]
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

    # Prefer explicit separators when available.
    splitters = [";", "|", "\n"]
    for sep in splitters:
        if sep in cleaned:
            parts = [p.strip(" ,") for p in cleaned.split(sep) if p.strip(" ,")]
            if len(parts) >= 2:
                return parts[0], " ".join(parts[1:])

    # Fallback: short first phrase as headline, rest as body.
    words = cleaned.replace(",", " ").split()
    if len(words) >= 6:
        return " ".join(words[:3]), " ".join(words[3:])

    return cleaned, None

def _recompute_geometry(element, img_w, img_h):
    x1, y1, x2, y2 = element["bbox_xyxy"]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    element["bbox_normalized"] = [
        round(x1 / img_w, 3),
        round(y1 / img_h, 3),
        round(x2 / img_w, 3),
        round(y2 / img_h, 3),
    ]
    element["center_normalized"] = [
        round(((x1 + x2) / 2) / img_w, 3),
        round(((y1 + y2) / 2) / img_h, 3),
    ]
    element["area_percentage"] = round((w * h * 100) / (img_w * img_h), 2)
    return element

def _create_background_element(img_w, img_h):
    return {
        "id": 0,
        "role": "background",
        "label": "Solid blue background",
        "description": "Solid blue background covering the full canvas.",
        "text_content": None,
        "bbox_xyxy": [0, 0, img_w, img_h],
        "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "center_normalized": [0.5, 0.5],
        "area_percentage": 100.0,
    }

def postprocess_elements_for_similarity(elements, creative_data, img_w, img_h):
    advertiser = _clean_text(creative_data.get("advertiser_name", "")).lower()
    app_name = _clean_text(creative_data.get("app_name", "")).lower()
    brand_terms = {t for t in [advertiser, app_name] if t}

    processed = []
    for element in elements:
        e = dict(element)
        text = _clean_text(e.get("text_content") or "")
        text_lower = text.lower()
        label_desc = f"{e.get('label', '')} {e.get('description', '')}".lower()

        # 1) Brand text should be logo.
        if text and any(term and term in text_lower for term in brand_terms):
            e["role"] = "logo"
            e["label"] = f"{text} text logo"

        # 2) Reduce 'unknown' when a safer visual role is obvious.
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

    # 3) Split oversized text region into headline/body_text.
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

            # If bbox is effectively full canvas, use a tighter central text region.
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
                    "id": 0,
                    "role": "headline",
                    "label": "Headline text",
                    "description": "Primary marketing headline text.",
                    "text_content": headline_text,
                    "bbox_xyxy": [hx1, hy1, hx2, hy2],
                }
                split_elements.append(_recompute_geometry(h_el, img_w, img_h))

            if body_text:
                b_el = {
                    "id": 0,
                    "role": "body_text",
                    "label": "Body text",
                    "description": "Supporting descriptive text below the headline.",
                    "text_content": body_text,
                    "bbox_xyxy": [hx1, by1, hx2, by2],
                }
                split_elements.append(_recompute_geometry(b_el, img_w, img_h))

            split_done = True
            continue

        split_elements.append(e)

    processed = split_elements

    # 4) Ensure background exists as first-class element.
    if not any(e.get("role") == "background" for e in processed):
        processed.insert(0, _create_background_element(img_w, img_h))

    # 5) Reassign stable IDs.
    for idx, e in enumerate(processed, start=1):
        e["id"] = idx

    return processed

def generate_global_and_embeddings(creative_data, final_elements):
    """Genera la part global fent servir tots els detalls visuals recollits."""
    advertiser = creative_data.get('advertiser_name', 'the brand')
    
    elements_summary = json.dumps([{
        "role": e["role"], 
        "label": e["label"], 
        "desc": e["description"],
        "pos": e["bbox_normalized"]
    } for e in final_elements], indent=2)
    
    prompt = f"""Analyze the layout and composition of this mobile ad for '{advertiser}'.
Based on these specific elements detected:
{elements_summary}

Create a LONG and DETAILED global description. 
Explain the ad as if you were describing it to a blind person: 
- 'An ad for {advertiser} that features [Background]...'
- 'In the center there are [Elements]...'
- 'At the bottom, we see [CTA]...'
- Be specific about the visual structure and literal layout.

Return ONLY a valid JSON object in English, matching this EXACT structure:
{{
  "global": {{
    "description": "A very long, comprehensive literal description of the entire ad layout and composition.",
    "visual_style": "Concise visual style (e.g. 'Flat design with high contrast')",
    "main_message": "The core marketing message",
    "dominant_colors": ["color1", "color2"],
    "emotional_tone": "The psychological feeling of the ad"
  }},
  "embedding_texts": {{
    "global_text": "Detailed semantic summary for search.",
    "elements_text": "background: ... main_subject: ...",
    "ocr_text": "All visible text combined",
    "layout_text": "Literal structure: [element] at [position], [element] at [position]..."
  }}
}}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a JSON-only API."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error generant globals: {e}")
        return {}

def main():
    print("🚀 INICIANT PIPELINE JSON ESTRUCTURAT...")

    if not image_path:
        print(f"❌ Error: No s'ha trobat cap imatge per a {IMG_NMBR}.")
        print("   He provat aquestes rutes:")
        for candidate in [
            os.path.join(project_root, "frontend", "public", "data", "assets", f"creative_{IMG_NMBR}.png"),
            os.path.join(project_root, "frontend", "public", "data", "assets", f"{IMG_NMBR}.png"),
            os.path.join(project_root, "backend", "assets", f"creative_{IMG_NMBR}.png"),
            os.path.join(project_root, "backend", "assets", f"{IMG_NMBR}.png"),
        ]:
            print(f"   - {candidate}")
        return
    
    # 1. Obtenir les dimensions reals del Canvas
    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"❌ Error: No s'ha trobat la imatge original a {image_path}")
        return
    img_h, img_w = img_cv2.shape[:2]
    
    # Carreguem la imatge global en base64 un sol cop per estalviar recursos
    full_image_base64 = encode_image(image_path)

    # 2. Carregar Metadades i dades prèvies
    df_creatives = pd.read_csv(creatives_csv, skipinitialspace=True)
    try:
        creative_data = df_creatives[df_creatives['creative_id'] == int(IMG_NMBR)].iloc[0].to_dict()
    except IndexError:
        print(f"⚠️ Avís: No hi ha metadades per a {IMG_NMBR} al CSV.")
        creative_data = {}

    if not os.path.exists(json_path):
        print(f"❌ Error: No s'ha trobat {json_path}")
        print("   Abans has de generar la carpeta 'cropped_elements' i 'elements_data.json' amb mask_generator.py.")
        return

    feature_data, dades_sam = load_elements_data(json_path)

    # 3. Processar cada element
    final_elements = []
    print("🧠 Analitzant elements individuals amb GPT-4o Vision...")
    
    for obj in dades_sam:
        if not isinstance(obj, dict):
            print(f"  -> Saltant element invàlid: {obj!r}")
            continue

        if "id" not in obj or "coords" not in obj:
            print(f"  -> Saltant element sense camps mínims: {obj!r}")
            continue

        ruta_retall = os.path.join(carpeta_retalls, f"element_{obj['id']}.jpg")
        text_ocr = " ".join(obj.get("text", [])) if isinstance(obj.get("text"), list) else obj.get("text", "")
        
        # Càlcul de coordenades normalitzades
        x1, y1, x2, y2 = map(float, obj["coords"])
        bbox_norm = [round(x1/img_w, 3), round(y1/img_h, 3), round(x2/img_w, 3), round(y2/img_h, 3)]
        cx_norm = round(((x1+x2)/2)/img_w, 3)
        cy_norm = round(((y1+y2)/2)/img_h, 3)
        
        if os.path.exists(ruta_retall):
            print(f"  -> Processant element ID:{obj['id']}...")
            ia_data = analyze_element_with_vision(
                full_image_base64, 
                ruta_retall, 
                text_ocr, 
                obj["coords"],
                advertiser=creative_data.get("advertiser_name", "unknown")
            )
            
            final_elements.append({
                "id": obj["id"],
                "role": ia_data.get("role", "unknown"),
                "label": ia_data.get("label", ""),
                "description": ia_data.get("description", ""),
                "text_content": text_ocr if text_ocr.strip() else None,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_normalized": bbox_norm,
                "center_normalized": [cx_norm, cy_norm],
                "area_percentage": round(obj.get("area_percentage", obj.get("percentatge_area", 0)), 2)
            })

    # 4. Generar dades Globals i Embeddings
    print("📝 Generant descripció global i textos per embeddings...")
    if not creative_data and feature_data.get("global_description"):
        creative_data = {"global_description": feature_data["global_description"]}

    final_elements = postprocess_elements_for_similarity(final_elements, creative_data, img_w, img_h)

    global_data = generate_global_and_embeddings(creative_data, final_elements)

    # 5. Muntar el JSON Final segons l'especificació
    final_json = {
        "creative_id": str(IMG_NMBR),
        "asset_file": f"assets/creative_{IMG_NMBR}.png",
        "canvas": {
            "width": img_w,
            "height": img_h
        },
        "global": global_data.get("global", {}),
        "elements": final_elements,
        "embedding_texts": global_data.get("embedding_texts", {})
    }

    # 6. Guardar i mostrar resultats
    output_final = os.path.join(output_dir, f"creative_{IMG_NMBR}_structured.json")
    with open(output_final, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print(f"✨ JSON ESTRUCTURAT GENERAT AMB ÈXIT (ID: {IMG_NMBR}) ✨")
    print(f"📄 Guardat a: {output_final}")
    print("="*70)

if __name__ == "__main__":
    main()