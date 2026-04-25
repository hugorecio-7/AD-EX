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
        model_path = os.path.join(project_root, "models", "sam2.1_b.pt")
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
) -> tuple[np.ndarray, list[dict], list[list[float]]]:
    sam_model, ocr_reader = _get_models(project_root)

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_h, img_w = img_cv2.shape[:2]
    total_area = img_h * img_w

    # ── SAM: segment everything for VISUAL ELEMENTS ──────────────────────────
    print(f"[MaskGen] Running SAM on {image_path}...")
    sam_results = sam_model.predict(
        image_path, conf=0.10, iou=0.9, retina_masks=False, verbose=False,
    )
    sam_res = sam_results[0]

    sam_boxes = []
    if getattr(sam_res, 'boxes', None) is not None:
        for box in sam_res.boxes:
            coords = [round(c, 2) for c in box.xyxy[0].tolist()]
            w = coords[2] - coords[0]
            h = coords[3] - coords[1]
            area = w * h
            pct = (area / total_area) * 100
            if pct >= min_area_pct:
                sam_boxes.append({
                    "coords": coords,
                    "area_abs": area,
                    "area_percentage": round(pct, 2),
                })

    # ── EasyOCR: read text — paragraph=False gives individual word boxes ─────
    print("[MaskGen] Running EasyOCR...")
    # Run twice: paragraph mode for readable text, word mode for mask coverage
    ocr_paragraphs = ocr_reader.readtext(img_cv2, paragraph=True)
    ocr_words      = ocr_reader.readtext(img_cv2, paragraph=False)

    def bbox_to_xyxy(bbox_pts) -> list[float]:
        """Convert EasyOCR [[x,y],...] polygon to [x1,y1,x2,y2]."""
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        return [min(xs), min(ys), max(xs), max(ys)]

    # Build text elements directly from OCR bounding boxes (NOT SAM boxes)
    # Use paragraph results for readable label text, word results for mask coverage
    text_elements = []
    for item in ocr_paragraphs:
        # NOTE: EasyOCR returns 2 values in paragraph mode (bbox, text)
        bbox_pts, text = item
        coords = bbox_to_xyxy(bbox_pts)
        w = coords[2] - coords[0]
        h = coords[3] - coords[1]
        area = w * h
        pct = (area / total_area) * 100
        if pct < 0.05:          # skip tiny noise detections
            continue
        text_elements.append({
            "coords": coords,
            "area_abs": area,
            "area_percentage": round(pct, 2),
            "text": text.strip(),
            "source": "ocr_paragraph",
        })

    # Also collect individual word boxes — used to EXPAND mask coverage
    # (catches words the paragraph merger missed)
    word_boxes = []
    for item in ocr_words:
        # NOTE: EasyOCR returns 3 values in standard mode (bbox, text, conf)
        bbox_pts, text, conf = item
        if conf < 0.3 or not text.strip():
            continue
        coords = bbox_to_xyxy(bbox_pts)
        word_boxes.append(coords)

    # ── Determine which SAM boxes overlap significantly with OCR text regions
    # → those are TEXT/BUTTON, the rest are VISUAL ELEMENTs
    def iou(a, b) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def centre_inside(box_coords, region) -> bool:
        """True if the centre of box_coords falls inside region."""
        cx = (box_coords[0] + box_coords[2]) / 2
        cy = (box_coords[1] + box_coords[3]) / 2
        return (region[0] <= cx <= region[2]) and (region[1] <= cy <= region[3])

    text_coords_list = [e["coords"] for e in text_elements]

    visual_sam_boxes = []
    for box in sam_boxes:
        # Filter 1: IoU against paragraph-level OCR (catches big overlaps)
        overlaps_paragraph = any(iou(box["coords"], tc) > 0.15 for tc in text_coords_list)
        
        # Filter 2: centre falls inside any individual word box (catches letter-shaped segments)
        overlaps_word = any(centre_inside(box["coords"], wb) for wb in word_boxes)
        
        if not overlaps_paragraph and not overlaps_word:
            visual_sam_boxes.append(box)

    # ── Build unified elements list ───────────────────────────────────────────
    # Sort visual elements largest → smallest
    visual_sam_boxes.sort(key=lambda b: b["area_percentage"], reverse=True)
    text_elements.sort(key=lambda e: e["area_percentage"], reverse=True)

    elements = []
    idx = 1

    for b in visual_sam_boxes:
        elements.append({
            "id": idx,
            "label": "VISUAL ELEMENT",
            "text": "",
            "coords": b["coords"],
            "area_percentage": b["area_percentage"],
        })
        idx += 1

    for e in text_elements:
        elements.append({
            "id": idx,
            "label": f"TEXT/BUTTON: {e['text']}",
            "text": e["text"],
            "coords": e["coords"],
            "area_percentage": e["area_percentage"],
        })
        idx += 1

    # Store word_boxes separately for the mask builder to use
    # (attach to img_cv2 isn't possible, so we return it via a side-channel on elements)
    # Simpler: just return word_boxes alongside

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

    return img_cv2, elements, word_boxes


def _build_global_description(elements: list[dict], image_path: str, img_h: int) -> str:
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
        norm_y = centre_y / img_h          # ← normalize
        if norm_y < 0.33:
            img_heights["top"] = img_heights.get("top", 0) + 1
        elif norm_y < 0.66:
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
    word_boxes: list[list[float]],   # ← NEW: raw OCR word bboxes
    text_padding: int = 15,          # increased from 5
) -> np.ndarray:
    """
    White = AI can redraw (visual elements)
    Black = protected (text/buttons)

    Uses OCR bounding boxes directly — not SAM boxes — for text regions.
    Also blacks out individual word boxes for maximum coverage.
    """
    img_h, img_w = img_cv2.shape[:2]
    mask_np = np.ones((img_h, img_w), dtype=np.uint8) * 255

    def blackout(coords, pad):
        x1, y1, x2, y2 = map(int, coords)
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(img_w, x2 + pad)
        py2 = min(img_h, y2 + pad)
        cv2.rectangle(mask_np, (px1, py1), (px2, py2), 0, -1)

    # Black out paragraph-level text elements (from elements list)
    for obj in elements:
        if obj["text"]:
            blackout(obj["coords"], text_padding)

    # Also black out individual word boxes (catches anything paragraph missed)
    for wbox in word_boxes:
        blackout(wbox, text_padding)

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
    img_cv2, elements, word_boxes = _analyze_image(image_path, project_root, output_dir, min_area_pct)

    # OUTPUT A: diffusion mask
    mask_np = _build_diffusion_mask(img_cv2, elements, word_boxes, text_padding)
    img_h = img_cv2.shape[0]
    img_w = img_cv2.shape[1]
    # OUTPUT B: feature data (for retrieval embedding)
    global_description = _build_global_description(elements, image_path, img_h)
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
