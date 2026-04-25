import os
import json
import base64
import cv2
import pandas as pd
from openai import OpenAI

# ==========================================
# 1. CONFIGURACIÓ DE RUTES I VARIABLES
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) 

# Rutes dels resultats
output_dir = os.path.join(project_root, "output")
carpeta_retalls = os.path.join(output_dir, "elements_retallats")
json_path = os.path.join(output_dir, "dades_elements.json")

# Rutes de les metadades i assets
data_dir = os.path.join(project_root, "data")
creatives_csv = os.path.join(data_dir, "creatives.csv")
summary_csv = os.path.join(data_dir, "creative_summary.csv")

# ID de l'anunci
IMG_NMBR = "500000"
image_path = os.path.join(project_root, "data", "assets", f"creative_{IMG_NMBR}.png")

# Client d'OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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

def analyze_element_with_vision(full_image_base64, crop_image_path, text_ocr, coords):
    """Passa la imatge global i el retall a GPT-4o per classificar l'element."""
    crop_base64 = encode_image(crop_image_path)
    
    prompt = f"""You are analyzing a mobile ad creative.
Given the full image (Image 1) and a segmented crop of an element (Image 2), its bounding box coords {coords}, and any OCR text ('{text_ocr}'), assign exactly one role from this strict list:
{ALLOWED_ROLES}

If unsure, use "unknown". Do not invent roles.
Write the description and label in English.

Return ONLY a valid JSON object matching this structure exactly:
{{
  "role": "...",
  "label": "...",
  "description": "..."
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_image_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_base64}"}}
                    ]
                }
            ],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Error analitzant l'element: {e}")
        return {"role": "unknown", "label": "Unknown element", "description": "Failed to analyze element."}

def generate_global_and_embeddings(dades_csv, elements_finalitzats):
    """Genera la part global i els embedding_texts analitzant tot el conjunt."""
    
    # Preparem un resum dels elements per passar-li al LLM
    elements_summary = [
        {
            "role": e["role"], 
            "label": e["label"], 
            "text": e["text_content"],
            "area_pct": e["area_percentage"]
        } for e in elements_finalitzats
    ]
    
    prompt = f"""You are an expert creative analyst generating metadata for a similarity search engine.
Analyze these elements detected in a mobile ad and the original metadata.

Original Metadata:
- Theme: {dades_csv.get('theme', 'N/A')}
- Dominant Color: {dades_csv.get('dominant_color', 'N/A')}
- Emotional Tone: {dades_csv.get('emotional_tone', 'N/A')}

Detected Elements:
{json.dumps(elements_summary, indent=2)}

Return ONLY a valid JSON object in English, matching this EXACT structure:
{{
  "global": {{
    "description": "Brief description of the ad.",
    "visual_style": "e.g., 3D game render",
    "main_message": "Core message derived from elements",
    "dominant_colors": ["color1", "color2"],
    "emotional_tone": "e.g., exciting"
  }},
  "embedding_texts": {{
    "global_text": "Semantic text combining description, style, and tone.",
    "elements_text": "background: ... main_subject: ... cta: ...",
    "ocr_text": "All visible text combined",
    "layout_text": "Portrait/Landscape layout. Main subject at X. CTA at Y..."
  }}
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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

    with open(json_path, 'r', encoding='utf-8') as f:
        dades_sam = json.load(f)

    # 3. Processar cada element
    final_elements = []
    print("🧠 Analitzant elements individuals amb GPT-4o Vision...")
    
    for obj in dades_sam:
        ruta_retall = os.path.join(carpeta_retalls, f"element_{obj['id']}.jpg")
        text_ocr = " ".join(obj.get("text", [])) if isinstance(obj.get("text"), list) else obj.get("text", "")
        
        # Càlcul de coordenades normalitzades
        x1, y1, x2, y2 = map(float, obj["coords"])
        bbox_norm = [round(x1/img_w, 3), round(y1/img_h, 3), round(x2/img_w, 3), round(y2/img_h, 3)]
        cx_norm = round(((x1+x2)/2)/img_w, 3)
        cy_norm = round(((y1+y2)/2)/img_h, 3)
        
        if os.path.exists(ruta_retall):
            print(f"  -> Processant element ID:{obj['id']}...")
            ia_data = analyze_element_with_vision(full_image_base64, ruta_retall, text_ocr, obj["coords"])
            
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