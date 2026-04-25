"""
Mask Generator — pre-processing step for the creative upgrade pipeline.

Runs SAM 2.1 + EasyOCR once per image and produces TWO outputs:

  OUTPUT A — Diffusion Mask (mask PNG)
    White pixels = AI can redraw (visual background / product area)
    Black pixels = AI must NOT touch (text, buttons, logos)

  OUTPUT B — Feature Data (elements_data.json)
    Per-element: { id, label, text, coords, area_percentage }
    Global:      { global_description } — human-readable summary of the
                 ad's elements, used as an embedding input for the
                 retrieval similarity search (Step 1 of the pipeline).
"""
import os
import cv2
import json
import numpy as np
import easyocr
from ultralytics import SAM
from PIL import Image
import torch

# ─────────────────────────────────────────────
# Lazy-loaded singletons (models are heavy)
# ─────────────────────────────────────────────
_sam_model = None
_ocr_reader = None

def _get_models(project_root: str):
    global _sam_model, _ocr_reader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _sam_model is None:
        model_path = os.path.join(project_root, "models", "sam2.1_l.pt")
        print(f"[MaskGen] Loading SAM from {model_path} on {device}...")
        _sam_model = SAM(model_path).to(device)
    if _ocr_reader is None:
        use_gpu = torch.cuda.is_available()
        print(f"[MaskGen] Loading EasyOCR (GPU={use_gpu})...")
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
    return _sam_model, _ocr_reader


# ─────────────────────────────────────────────────────────────────────────────
# CORE ANALYSIS  (runs SAM + OCR — the expensive part, done once at preprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_image(
    image_path: str,
    project_root: str,
    output_dir: str = None,
    min_area_pct: float = 0.1,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run SAM + EasyOCR on the image and return:
      img_cv2    — the raw BGR image (numpy array)
      elements   — list of detected elements, each with keys:
                     id, label, text, coords, area_percentage
    """
    sam_model, ocr_reader = _get_models(project_root)

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img_cv2.shape[:2]
    total_area = img_h * img_w

    # =========================================================
    # NOU: LÒGICA DINÀMICA DE PERCENTATGE (QUADRADES VS VERTICALS)
    # =========================================================
    aspect_ratio = max(img_w, img_h) / min(img_w, img_h)
    
    if aspect_ratio < 1.3:
        # És una imatge quadrada (ex: 1080x1080, ratio 1.0)
        # Pugem el límit perquè l'àrea total és més petita. 
        # Tallem brossa petita.
        dynamic_min_pct = 0.8  
    else:
        # És una imatge rectangular (ex: 1080x1920, ratio 1.77)
        # Baixem el límit perquè l'àrea total és molt més gran.
        dynamic_min_pct = 0.35 

    print(f"[MaskGen] Format: {img_w}x{img_h} (Ratio {aspect_ratio:.2f}) -> Filtre fixat al {dynamic_min_pct}%")
    # =========================================================

    # ── SAM: segment everything ──────────────────────────────────────────────
    print(f"[MaskGen] Running SAM on {image_path}...")
    sam_results = sam_model.predict(
        image_path,
        conf=0.10,
        iou=0.9,
        retina_masks=False,  # faster; masks not needed, only boxes
        verbose=False,
    )
    sam_res = sam_results[0]

    raw_boxes = []
    if getattr(sam_res, 'boxes', None) is not None:
        for box in sam_res.boxes:
            coords = [round(c, 2) for c in box.xyxy[0].tolist()]
            w = coords[2] - coords[0]
            h = coords[3] - coords[1]
            area = w * h
            pct = (area / total_area) * 100
            
            # ATENCIÓ: Aquí fem servir el nou valor dinàmic en lloc de l'antic
            if pct >= dynamic_min_pct:
                raw_boxes.append({
                    "coords": coords,
                    "area_abs": area,
                    "area_percentage": round(pct, 2),
                    "texts": [],
                })

    # ── EasyOCR: read text ───────────────────────────────────────────────────
    print("[MaskGen] Running EasyOCR...")
    ocr_results = ocr_reader.readtext(img_cv2, paragraph=True)

    # ── Heuristic: assign each OCR text to the SMALLEST SAM box containing it
    for item in ocr_results:
        bbox_text = item[0]
        text = item[1]
        ox1 = min(p[0] for p in bbox_text)
        ox2 = max(p[0] for p in bbox_text)
        oy1 = min(p[1] for p in bbox_text)
        oy2 = max(p[1] for p in bbox_text)
        cx, cy = (ox1 + ox2) / 2, (oy1 + oy2) / 2

        candidates = [
            b for b in raw_boxes
            if b["coords"][0] <= cx <= b["coords"][2]
            and b["coords"][1] <= cy <= b["coords"][3]
        ]
        if candidates:
            min(candidates, key=lambda b: b["area_abs"])["texts"].append(text)

    # Sort largest → smallest (background → details)
    raw_boxes.sort(key=lambda b: b["area_percentage"], reverse=True)

    elements = [
        {
            "id": idx + 1,
            "label": f"TEXT/BUTTON: {' '.join(b['texts'])}" if b["texts"] else "VISUAL ELEMENT",
            "text": " ".join(b["texts"]).strip(),
            "coords": b["coords"],          # [x1, y1, x2, y2] in original px
            "area_percentage": b["area_percentage"],
        }
        for idx, b in enumerate(raw_boxes)
    ]

    # =======================================================
    # NOU: FER ELS RETALLS (CROPS) DURANT LA FASE D'ANÀLISI
    # =======================================================
    if output_dir:
        crops_folder = os.path.join(output_dir, "cropped_elements")
        os.makedirs(crops_folder, exist_ok=True)
        
        for obj in elements:
            x1, y1, x2, y2 = map(int, obj["coords"])
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(img_w, x2), min(img_h, y2)
            
            crop = img_cv2[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cv2.imwrite(os.path.join(crops_folder, f"element_{obj['id']}.jpg"), crop)
        
        print(f"[MaskGen] ✓ {len(elements)} retalls guardats a -> {crops_folder}/")
    # =======================================================

    return img_cv2, elements


def _build_global_description(elements: list[dict], image_path: str) -> str:
    """
    Build a human-readable global description of the ad creative.

    This is stored alongside the per-element data and used as the
    embedding input for similarity search in the retrieval step.
    The richer the description, the better the retrieval quality.
    """
    text_elements = [e for e in elements if e["text"]]
    visual_elements = [e for e in elements if not e["text"]]

    creative_name = os.path.splitext(os.path.basename(image_path))[0]
    parts = [f"Ad creative '{creative_name}'."]

    if text_elements:
        texts = [e["text"] for e in text_elements]
        parts.append(f"Contains {len(text_elements)} text/button regions: {'; '.join(texts[:5])}.")
    if visual_elements:
        parts.append(f"Contains {len(visual_elements)} visual background elements.")

    # Rough layout description based on vertical position of text
    img_heights = {}
    for e in text_elements:
        centre_y = (e["coords"][1] + e["coords"][3]) / 2
        if centre_y < 0.33:
            img_heights["top"] = img_heights.get("top", 0) + 1
        elif centre_y < 0.66:
            img_heights["middle"] = img_heights.get("middle", 0) + 1
        else:
            img_heights["bottom"] = img_heights.get("bottom", 0) + 1

    if img_heights:
        layout = ", ".join(f"{count} at {pos}" for pos, count in img_heights.items())
        parts.append(f"Text layout: {layout}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT A — DIFFUSION MASK
# ─────────────────────────────────────────────────────────────────────────────

def _build_diffusion_mask(
    img_cv2: np.ndarray,
    elements: list[dict],
    text_padding: int = 5,
) -> np.ndarray:
    """
    Build a B&W numpy mask from the element list.
      White (255) = AI can redraw  → VISUAL ELEMENTs (backgrounds, product areas)
      Black  (0)  = AI must protect → TEXT/BUTTON elements

    Returns uint8 numpy array (H×W), same resolution as the input image.
    """
    img_h, img_w = img_cv2.shape[:2]
    mask_np = np.ones((img_h, img_w), dtype=np.uint8) * 255  # start all-white

    for obj in elements:
        if obj["text"]:  # only black-out text/button regions
            x1, y1, x2, y2 = map(int, obj["coords"])
            px1 = max(0, x1 - text_padding)
            py1 = max(0, y1 - text_padding)
            px2 = min(img_w, x2 + text_padding)
            py2 = min(img_h, y2 + text_padding)
            cv2.rectangle(mask_np, (px1, py1), (px2, py2), 0, -1)

    return mask_np


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_diffusion_mask(
    image_path: str,
    project_root: str,
    output_dir: str = None,
    min_area_pct: float = 0.1,
    text_padding: int = 5,
) -> tuple[np.ndarray, list[dict], str]:
    """
    Full pre-processing pipeline for one creative. Called by:
      - preprocess_masks.py  (batch, at preprocessing time)
      - step3_generation/core.py  (on-the-fly fallback if mask missing)

    Saves to output_dir (if provided):
      • {name}_diffusion_mask.png   ← OUTPUT A: used by Stable Diffusion
      • elements_data.json           ← OUTPUT B: used by retrieval similarity
      • {name}_analysis.jpg          ← DEBUG: annotated bounding-box image

    Returns:
      mask_np    — uint8 numpy array (H×W)
      elements   — list of element dicts {id, label, text, coords, area_pct}
      mask_path  — path to the saved mask PNG (or None)
    """
    img_cv2, elements = _analyze_image(image_path, project_root, output_dir, min_area_pct)

    # OUTPUT A: diffusion mask
    mask_np = _build_diffusion_mask(img_cv2, elements, text_padding)

    # OUTPUT B: feature data (for retrieval embedding)
    global_description = _build_global_description(elements, image_path)
    feature_data = {
        "global_description": global_description,
        "elements": elements,
    }

    mask_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        creative_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save OUTPUT A
        mask_path = os.path.join(output_dir, f"{creative_name}_diffusion_mask.png")
        cv2.imwrite(mask_path, mask_np)

        # Save OUTPUT B
        with open(os.path.join(output_dir, "elements_data.json"), "w", encoding="utf-8") as f:
            json.dump(feature_data, f, indent=2, ensure_ascii=False)

        # Save debug annotated image
        debug_img = img_cv2.copy()
        for obj in elements:
            x1, y1, x2, y2 = map(int, obj["coords"])
            color = (0, 255, 0) if obj["text"] else (0, 0, 255)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, obj["label"][:35], (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(os.path.join(output_dir, f"{creative_name}_analysis.jpg"), debug_img)

        print(f"[MaskGen] ✓ Mask  → {mask_path}")
        print(f"[MaskGen] ✓ Features → {output_dir}/elements_data.json")
        print(f"[MaskGen] ✓ Debug → {output_dir}/{creative_name}_analysis.jpg")

    return mask_np, elements, mask_path
