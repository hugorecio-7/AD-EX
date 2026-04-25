import os
import cv2
import json
import easyocr
from ultralytics import SAM

# 1. PATH CONFIGURATION
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

img_nmbr = "50000" 
image_path = os.path.join(project_root, "data", "assets", f"creative_{img_nmbr}.png")

model_name = "sam2.1_t.pt"
model_path = os.path.join(project_root, "models", model_name)

# 2. MODEL LOADING 
print("🔥 Loading SAM 2.1 (Base Model)... Have patience with the Wi-Fi!")
sam_model = SAM(model_path)

print("Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'])

print("🧠 Analyzing geometry with SAM 2.1 (Fast CPU Mode)...")
sam_results = sam_model.predict(
    image_path, 
    conf=0.10,         # Super baix per intentar caçar línies fines
    iou=0.9,           
    imgsz=736,         # Pugem resolució (múltiple de 32) per donar-li més detall
    retina_masks=True  # Ho activem! Trigarà un pèl més, però detecta millor vores
)
sam_res = sam_results[0]

print("📝 Reading text with EasyOCR (Cleaning Alpha channel)...")
# Read image with OpenCV (cv2.imread ignores the Alpha channel by default)
img_for_ocr = cv2.imread(image_path)
# Pass the clean image matrix directly to EasyOCR
ocr_results = ocr_reader.readtext(img_for_ocr)

def is_text_inside(box_sam, bbox_ocr):
    text_x = (bbox_ocr[0][0] + bbox_ocr[2][0]) / 2
    text_y = (bbox_ocr[0][1] + bbox_ocr[2][1]) / 2
    x1, y1, x2, y2 = box_sam
    return x1 <= text_x <= x2 and y1 <= text_y <= y2

print("\n" + "="*50)
print("🔍 COMBINED RESULTS (SAM 2.1 + TEXT)")
print("="*50)

img_height, img_width = sam_res.orig_shape
total_area = img_height * img_width

# 4.1. Collect all SAM boxes with their area
# 4.1. Collect all SAM boxes with their area
sam_boxes_info = []
MIN_AREA_PCT = 0.5 # ⚠️ EL TRUC: Ignorem tot el que ocupi menys d'un 0.5% de l'anunci

if getattr(sam_res, 'boxes', None) is not None:
    for i, box in enumerate(sam_res.boxes):
        coords = [round(c, 2) for c in box.xyxy[0].tolist()]
        obj_area = (coords[2] - coords[0]) * (coords[3] - coords[1])
        pct = round((obj_area / total_area) * 100, 2)
        
        # Només guardem l'element si supera el filtre de mida
        if pct >= MIN_AREA_PCT:
            sam_boxes_info.append({
                "id": i,
                "coords": coords,
                "area": obj_area,
                "area_percentage": pct,
                "texts": [] 
            })
# 4.2. Assign the text to the SMALLEST possible SAM box
for bbox_text, text, prob in ocr_results:
    if prob > 0.25:
        candidate_boxes = []
        for sam_box in sam_boxes_info:
            if is_text_inside(sam_box["coords"], bbox_text):
                candidate_boxes.append(sam_box)
        
        if candidate_boxes:
            smallest_box = min(candidate_boxes, key=lambda x: x["area"])
            smallest_box["texts"].append(text)

# 4.3. Generate final data
final_data = []
for box in sam_boxes_info:
    full_text = " ".join(box["texts"])
    final_label = f"BUTTON/TEXT: {full_text}" if full_text else "VISUAL ELEMENT (Image/Logo)"
    
    obj_data = {
        "id": box["id"],
        "label": final_label,
        "text": box["texts"],
        "area_percentage": box["area_percentage"],
        "coords": box["coords"]
    }
    final_data.append(obj_data)

# --- 5. SAVE ALL RESULTS ---
# Creem la ruta dinàmica: output/features/creative_XXXXXX
creative_name = f"creative_{img_nmbr}"
output_dir = os.path.join(project_root, "output", "features", creative_name)

# Creem la carpeta (crearà 'output' i 'features' si no existeixen)
os.makedirs(output_dir, exist_ok=True)

# 5.1 Create subfolder for individual crops
crops_folder = os.path.join(output_dir, "cropped_elements")
os.makedirs(crops_folder, exist_ok=True)

# Read original image 
img_cv2 = cv2.imread(image_path)
max_height, max_width = img_cv2.shape[:2]

# 5.2 Process each object: Crop, Draw and Save
for obj in final_data:
    x1, y1, x2, y2 = map(int, obj["coords"])
    
    # --- A) SAVE THE CROP ---
    # Ensure we don't go outside the real image margins
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(max_width, x2), min(max_height, y2)
    
    crop = img_cv2[cy1:cy2, cx1:cx2]
    
    # Only save if the crop has a valid size
    if crop.size > 0:
        crop_path = os.path.join(crops_folder, f"element_{obj['id']}.jpg")
        cv2.imwrite(crop_path, crop)

    # --- B) DRAW ON FINAL IMAGE ---
    is_text = "BUTTON/TEXT" in obj["label"]
    color = (0, 255, 0) if is_text else (0, 0, 255) # Green for text, Red for logo/image
    
    cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
    if is_text:
        cv2.putText(img_cv2, obj["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 5.3 Save the image with all boxes drawn (ara amb el nom del creative)
output_path = os.path.join(output_dir, f"{creative_name}_analysis.jpg")
cv2.imwrite(output_path, img_cv2)

# 5.4 Save data dictionary in a JSON file
json_path = os.path.join(output_dir, "elements_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print("\n" + "="*50)
print(f"💾 EVERYTHING SAVED SUCCESSFULLY FOR '{creative_name}'")
print("="*50)
print(f"📁  Main folder: {output_dir}/")
print(f"🖼️  Annotated image: {output_path}")
print(f"✂️  Individual crops: {crops_folder}/")
print(f"📄  Data JSON: {json_path}")