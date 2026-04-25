import os
import cv2
import json
import torch
import easyocr
from ultralytics import SAM

# ==========================================
# 1. CONFIGURACIÓ DE RUTES I PARÀMETRES
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Usando dispositivo: {device}")


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

IMG_NMBR = "501058" # Canvia l'ID segons la imatge
image_path = os.path.join(project_root, "data", "assets", f"creative_{IMG_NMBR}.png")

# Utilitzem el model 'tiny' per a màxima velocitat
model_name = "sam2.1_t.pt" 
model_path = os.path.join(project_root, "models", model_name)

# HEURÍSTICA 1: Filtre de soroll (mida mínima)
# Descarta imperfeccions o petites línies que no són elements reals
MIN_AREA_PCT = 0.5 

# Rutes d'output
creative_name = f"creative_{IMG_NMBR}"
output_dir = os.path.join(project_root, "output", "features", creative_name)
crops_folder = os.path.join(output_dir, "cropped_elements")
os.makedirs(crops_folder, exist_ok=True)

def main():
    print("\n" + "="*50)
    print("🚀 INICIANT PIPELINE VISUAL (RÀPID)")
    print("="*50)

    # ==========================================
    # 2. CÀRREGA DE MODELS (SAM + OCR)
    # ==========================================
    print(f"🔥 Carregant SAM ({model_name})...")
    model_sam = SAM(model_path).to(device)

    print("📝 Carregant EasyOCR...")
    ocr_reader = easyocr.Reader(['en'], gpu=True) # Si tens GPU, posa True per volar

    # ==========================================
    # 3. INFERÈNCIA (Visió + Text)
    # ==========================================
    print("🧠 Analitzant la geometria amb SAM (Mode Ràpid)...")
    # Paràmetres optimitzats per a velocitat (sense màscares d'alta definició)
    resultats_sam = model_sam.predict(
        image_path, 
        conf=0.10, 
        iou=0.9, 
        retina_masks=False, 
        verbose=False
    )[0]

    print("📝 Llegint text amb EasyOCR (Agrupant paràgrafs)...")
    img_cv2 = cv2.imread(image_path)
    alt_max, ample_max = img_cv2.shape[:2]
    area_total = alt_max * ample_max
    
    ocr_results = ocr_reader.readtext(img_cv2, paragraph=True)

    # ==========================================
    # 4. HEURÍSTIQUES DE PROCESSAMENT I FUSIÓ
    # ==========================================
    dades_brutes =[]

    # 4.1. Extreure caixes i aplicar HEURÍSTICA 1 (Filtre de Soroll)
    if getattr(resultats_sam, 'boxes', None) is not None:
        for box in resultats_sam.boxes:
            coords =[round(c, 2) for c in box.xyxy[0].tolist()]
            amplada_obj = coords[2] - coords[0]
            alcada_obj = coords[3] - coords[1]
            area_obj = amplada_obj * alcada_obj
            pct_pantalla = (area_obj / area_total) * 100
            
            if pct_pantalla >= MIN_AREA_PCT:
                dades_brutes.append({
                    "coords": coords,
                    "area_abs": area_obj,
                    "area_percentage": round(pct_pantalla, 2),
                    "texts":[]
                })

    # 4.2. HEURÍSTICA 2: El truc del "Contenidor Més Petit"
    for item in ocr_results:
        bbox_text = item[0]
        text_detectat = item[1]
        
        # Calculem el centre exacte de la capsa del text
        ox1, ox2 = min(p[0] for p in bbox_text), max(p[0] for p in bbox_text)
        oy1, oy2 = min(p[1] for p in bbox_text), max(p[1] for p in bbox_text)
        centre_x, centre_y = (ox1 + ox2) / 2, (oy1 + oy2) / 2
        
        caixes_candidates =[]
        for sam_box in dades_brutes:
            sx1, sy1, sx2, sy2 = sam_box["coords"]
            # El centre del text està dins d'aquesta caixa SAM?
            if sx1 <= centre_x <= sx2 and sy1 <= centre_y <= sy2:
                caixes_candidates.append(sam_box)
                
        if caixes_candidates:
            # Associa el text a la caixa amb menys àrea (ex: el botó vs el fons)
            caixa_mes_petita = min(caixes_candidates, key=lambda x: x["area_abs"])
            caixa_mes_petita["texts"].append(text_detectat)

    # 4.3. HEURÍSTICA 3: Jerarquia de Capes (De fons a detalls)
    dades_ordenades = sorted(dades_brutes, key=lambda x: x["area_percentage"], reverse=True)

    final_data =[]
    for index, obj in enumerate(dades_ordenades):
        text_complet = " ".join(obj["texts"]).strip()
        
        # HEURÍSTICA 5: Classificació Visual Dicotòmica
        etiqueta = f"TEXT/BOTÓ: {text_complet}" if text_complet else "ELEMENT VISUAL"
        
        final_data.append({
            "id": index + 1,
            "label": etiqueta,
            "text": text_complet,
            "coords": obj["coords"],
            "area_percentage": obj["area_percentage"]
        })

    # ==========================================
    # 5. GUARDAT (CROP SEGUR + DIBUIX DICOTÒMIC)
    # ==========================================
    for obj in final_data:
        x1, y1, x2, y2 = map(int, obj["coords"])
        
        # HEURÍSTICA 4: Cropping intel·ligent (Segur)
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(ample_max, x2), min(alt_max, y2)
        
        crop = img_cv2[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            crop_path = os.path.join(crops_folder, f"element_{obj['id']}.jpg")
            cv2.imwrite(crop_path, crop)

        # Dicotomia de colors (Verd = Text/Botó, Vermell = Imatge/Fons)
        te_text = obj["text"] != ""
        color = (0, 255, 0) if te_text else (0, 0, 255) 
        
        # Dibuixar Rectangle
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Text del Rectangle (ID + Pct)
        text_mostrar = f"ID:{obj['id']} ({obj['area_percentage']}%)"
        cv2.putText(img_cv2, text_mostrar, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 6. EXPORTAR RESULTATS
    output_path = os.path.join(output_dir, f"{creative_name}_analysis.jpg")
    cv2.imwrite(output_path, img_cv2)

    json_path = os.path.join(output_dir, "elements_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    # RESUM PER PANTALLA
    print("\n" + "="*50)
    print(f"✅ EXTRACCIÓ RÀPIDA COMPLETADA PER A '{creative_name}'")
    print("="*50)
    for obj in final_data:
        if obj["text"]: 
            print(f"📦 ID: {obj['id']:02d} | Botó/Text: '{obj['text']}'")
    print("-" * 50)
    print(f"🖼️  Imatge processada: {output_path}")
    print(f"📄  Metadades: {json_path}")
    print("Llest per al següent pas (LLM) quan tu vulguis!")

if __name__ == "__main__":
    main()