import os
import cv2
import json
import easyocr
from ultralytics import SAM

# ==========================================
# 1. CONFIGURACIÓ DE RUTES
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

img_nmbr = "501058" # Canvia-ho segons la imatge que vulguis
image_path = os.path.join(project_root, "data", "assets", f"creative_{img_nmbr}.png")

model_name = "sam2.1_t.pt" # Pots fer servir sam2_s.pt si et va millor
model_path = os.path.join(project_root, "models", model_name)

# ==========================================
# 2. CARREGAR MODELS (SAM + OCR)
# ==========================================
print(f"🔥 Carregant SAM ({model_name})...")
model = SAM(model_path) 

print("📝 Carregant EasyOCR...")
ocr_reader = easyocr.Reader(['en'])

# ==========================================
# 3. INFERÈNCIA (Geometria + Text)
# ==========================================
print("🧠 Analitzant tota la geometria amb SAM...")
resultats_sam = model(image_path)
resultat = resultats_sam[0]

print("📝 Llegint text amb EasyOCR (Agrupant paràgrafs)...")
img_cv2 = cv2.imread(image_path)
# paragraph=True agrupa les paraules d'una mateixa frase
ocr_results = ocr_reader.readtext(img_cv2, paragraph=True) 

# ==========================================
# 4. PROCESSAMENT I FUSIÓ
# ==========================================
alt_img, ample_img = resultat.orig_shape
area_total = alt_img * ample_img

dades_brutes = []
MIN_AREA_PCT = 0.1 # Ignorem brossa minúscula (menys del 0.1% de la pantalla)

# 4.1 Extreure caixes de SAM
if getattr(resultat, 'boxes', None) is not None:
    for box in resultat.boxes:
        coords = box.xyxy[0].tolist()
        coords = [round(c, 2) for c in coords]
        
        amplada_obj = coords[2] - coords[0]
        alcada_obj = coords[3] - coords[1]
        area_obj = amplada_obj * alcada_obj
        pct_pantalla = (area_obj / area_total) * 100
        
        if pct_pantalla >= MIN_AREA_PCT:
            dades_brutes.append({
                "coords": coords,
                "area_abs": area_obj, # Guardem l'àrea absoluta per comparar després
                "area_percentage": round(pct_pantalla, 2),
                "texts": [] # Aquí guardarem els textos
            })

# 4.2 Assignar cada text a l'element més petit que el contingui
for bbox_text, text in ocr_results:
    # Calculem el centre del text
    ox1 = min(p[0] for p in bbox_text)
    ox2 = max(p[0] for p in bbox_text)
    oy1 = min(p[1] for p in bbox_text)
    oy2 = max(p[1] for p in bbox_text)
    
    centre_x = (ox1 + ox2) / 2
    centre_y = (oy1 + oy2) / 2
    
    caixes_candidates = []
    
    # Busquem quines caixes de SAM contenen el centre del text
    for sam_box in dades_brutes:
        sx1, sy1, sx2, sy2 = sam_box["coords"]
        if sx1 <= centre_x <= sx2 and sy1 <= centre_y <= sy2:
            caixes_candidates.append(sam_box)
            
    # Si hem trobat caixes, li assignem a la MÉS PETITA (ex: el botó, no el fons)
    if caixes_candidates:
        caixa_mes_petita = min(caixes_candidates, key=lambda x: x["area_abs"])
        caixa_mes_petita["texts"].append(text)

# 4.3 Ordenar del més gran al més petit i netejar les dades
dades_ordenades = sorted(dades_brutes, key=lambda x: x["area_percentage"], reverse=True)

final_data = []
for index, obj in enumerate(dades_ordenades):
    text_complet = " ".join(obj["texts"])
    etiqueta = f"TEXT: {text_complet}" if text_complet else "VISUAL ELEMENT"
    
    final_data.append({
        "id": index + 1,
        "label": etiqueta,
        "text": text_complet, # El text net (buit si no en té)
        "coords": obj["coords"],
        "area_percentage": obj["area_percentage"]
    })

# ==========================================
# 5. GUARDAR RESULTATS EN CARPETES
# ==========================================
creative_name = f"creative_{img_nmbr}"
output_dir = os.path.join(project_root, "output", "features", creative_name)

os.makedirs(output_dir, exist_ok=True)
crops_folder = os.path.join(output_dir, "cropped_elements")
os.makedirs(crops_folder, exist_ok=True)

alt_max, ample_max = img_cv2.shape[:2]

for obj in final_data:
    x1, y1, x2, y2 = map(int, obj["coords"])
    
    # --- A) GUARDAR EL RETALL (CROP) ---
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(ample_max, x2), min(alt_max, y2)
    
    crop = img_cv2[cy1:cy2, cx1:cx2]
    
    if crop.size > 0:
        crop_path = os.path.join(crops_folder, f"element_{obj['id']}.jpg")
        cv2.imwrite(crop_path, crop)

    # --- B) DIBUIXAR A LA IMATGE FINAL ---
    te_text = obj["text"] != ""
    color = (0, 255, 0) if te_text else (0, 0, 255) # Verd si té text, Vermell si és imatge
    
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
    
    # Text a mostrar a la imatge: ID + Percentatge + (Text si en té)
    text_mostrar = f"ID:{obj['id']} ({obj['area_percentage']}%)"
    if te_text:
        # Tallem el text per no ocupar tota la imatge dibuixant
        text_curt = obj["text"][:15] + "..." if len(obj["text"]) > 15 else obj["text"]
        text_mostrar += f" [{text_curt}]"
        
    cv2.putText(img_cv2, text_mostrar, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Guardem la imatge final
output_path = os.path.join(output_dir, f"{creative_name}_analysis.jpg")
cv2.imwrite(output_path, img_cv2)

# Guardem el diccionari en JSON
json_path = os.path.join(output_dir, "elements_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

# Imprimim resum
print("\n" + "="*50)
print(f"💾 EXTRACCIÓ COMPLETADA PER A '{creative_name}'")
print("="*50)
for obj in final_data:
    if obj["text"]: # Només imprimim a la consola els que tenen text per comprovar-ho ràpid
        print(f"📦 ID: {obj['id']} | Text trobat: '{obj['text']}'")
print("-" * 50)
print(f"📁 Directori Principal: {output_dir}/")
print(f"📄 Dades JSON: {json_path}")