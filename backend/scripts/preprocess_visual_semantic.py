"""
Batch Visual Semantic JSON Generator — Single Call Per Creative
==============================================================
Generates frontend/public/data/visual_semantic/creative_{id}.json for all creatives.

Strategy: ONE API call per creative (not per element).
  - Sends all element bboxes + OCR text in a single prompt
  - Gets back complete structured JSON (elements + global + embeddings)
  - Model: gpt-4o-mini (vision-capable, cheap, fast)
  - Cost: ~$1-3 for all 1080 creatives

Usage:
    python backend/scripts/preprocess_visual_semantic.py
    python backend/scripts/preprocess_visual_semantic.py --resume   (skip existing)
    python backend/scripts/preprocess_visual_semantic.py --id 500000  (single creative)
"""
import os
import sys
import json
import time
import re
import argparse
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir   = Path(__file__).resolve().parent
backend_dir  = script_dir.parent
project_root = backend_dir.parent

load_dotenv(project_root / ".env")
load_dotenv(backend_dir / ".env", override=True)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set in backend/.env")

client = OpenAI(api_key=api_key)
MODEL  = "gpt-4o-mini"   # Vision capable, cheap, fast. NOT o4-mini (that's a reasoning model).

DATA_JSON    = project_root / "frontend" / "src" / "mocks" / "data.json"
ASSETS_DIR   = project_root / "frontend" / "public" / "data" / "assets"
OUTPUT_DIR   = project_root / "frontend" / "public" / "data" / "visual_semantic"
FEATURES_DIR = project_root / "output" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a mobile advertising creative analyst.
Given element data (bounding boxes, OCR text, area) and creative metadata,
you produce a structured visual analysis JSON used for ad performance optimization.
You ALWAYS respond with valid JSON only — no markdown, no explanation."""

def build_user_prompt(creative: dict, raw_elements: list, img_w: int, img_h: int) -> str:
    advertiser = creative.get("advertiser_name", creative.get("subject", "unknown"))
    vertical   = creative.get("vertical", "unknown")
    objective  = creative.get("objective", "unknown")
    fmt        = creative.get("format", "unknown")

    # Describe each element compactly
    elem_lines = []
    for e in raw_elements:
        eid   = e.get("id", "?")
        text  = e.get("text") or ""
        if isinstance(text, list): text = " ".join(text)
        text  = text.strip()
        x1,y1,x2,y2 = map(float, e.get("coords", [0,0,img_w,img_h]))
        area  = round(e.get("area_percentage", 0), 1)
        # Describe vertical position
        cy_norm = ((y1+y2)/2) / img_h if img_h else 0.5
        pos = "top" if cy_norm < 0.3 else "bottom" if cy_norm > 0.7 else "center"
        bbox = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
        line = f"ID={eid} | area={area}% | pos={pos} | bbox={bbox}"
        if text:
            line += f' | text="{text}"'
        elem_lines.append(line)

    elements_block = "\n".join(elem_lines)
    canvas = f"{img_w}x{img_h}px"

    valid_roles = "background, main_subject, person, face, product, app_screenshot, gameplay, logo, icon, headline, body_text, cta, rating, price, discount_badge, social_proof, decorative_element, unknown"

    return f"""Analyze this mobile advertising creative:
Advertiser: {advertiser}
Vertical: {vertical} | Objective: {objective} | Format: {fmt}
Canvas: {canvas}

Elements detected by SAM segmentation + OCR:
{elements_block}

For each element, assign a role, a short label, and a visual description suitable for Stable Diffusion image generation.
Valid roles: {valid_roles}

Return ONLY this JSON structure:
{{
  "elements": [
    {{
      "id": <same id as input>,
      "role": "<role>",
      "label": "<3-5 word label>",
      "description": "<one sentence: what does this element look like visually? Be specific enough for an image model to reproduce it.>"
    }}
  ],
  "global": {{
    "description": "<One factual sentence describing the overall ad composition>",
    "visual_style": "<3-6 word visual style descriptor e.g. 'dark cinematic with neon glow'>",
    "main_message": "<What is this ad communicating?>",
    "dominant_colors": ["<color1>", "<color2>"],
    "emotional_tone": "<one word: exciting/calm/luxurious/playful/serious/adventurous/urgent>"
  }},
  "embedding_texts": {{
    "global_text": "<1-2 sentences for semantic search: advertiser, vertical, format, visual style, key elements>",
    "layout_text": "<Spatial layout description: where headline, subject, and CTA are positioned>",
    "ocr_text": "<All visible text comma-separated>"
  }}
}}"""


# ── Single-call API function ───────────────────────────────────────────────────

def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_llm_single(creative: dict, raw_elements: list, img_w: int, img_h: int, image_path: Path | None) -> dict:
    """
    One API call → complete structured JSON for the creative.
    If image_path is provided, includes the image for visual context (costs slightly more).
    """
    user_text = build_user_prompt(creative, raw_elements, img_w, img_h)

    # Build message: optionally include image thumbnail for better accuracy
    if image_path and image_path.exists():
        b64 = encode_image(image_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "low",   # low detail = cheaper, still enough for layout understanding
                }},
            ]},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    # Retry loop for malformed JSON (e.g. unescaped quotes in OCR text)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Increase temperature slightly on retries to avoid getting stuck producing the same bad JSON
            temperature = 0.1 if attempt == 0 else 0.4

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to parse JSON after {max_retries} attempts. Last error: {e}")
            print(f"      [Retry {attempt+1}/{max_retries}] JSON parse error: {e}. Retrying...")


# ── Element geometry builder ───────────────────────────────────────────────────

def build_final_element(raw_elem: dict, llm_elem: dict, img_w: int, img_h: int) -> dict:
    x1,y1,x2,y2 = map(float, raw_elem.get("coords", [0,0,img_w,img_h]))
    bbox_norm = [round(x1/img_w,3), round(y1/img_h,3), round(x2/img_w,3), round(y2/img_h,3)]
    cx = round(((x1+x2)/2)/img_w, 3)
    cy = round(((y1+y2)/2)/img_h, 3)
    text = raw_elem.get("text") or ""
    if isinstance(text, list): text = " ".join(text)
    text = text.strip() or None

    return {
        "id": llm_elem.get("id", raw_elem.get("id")),
        "role": llm_elem.get("role", "unknown"),
        "label": llm_elem.get("label", "Element"),
        "description": llm_elem.get("description", ""),
        "text_content": text,
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_normalized": bbox_norm,
        "center_normalized": [cx, cy],
        "area_percentage": round(raw_elem.get("area_percentage", 0), 2),
    }


# ── Per-creative processor ─────────────────────────────────────────────────────

def get_image_path(cid: str) -> Path | None:
    for candidate in [ASSETS_DIR / f"creative_{cid}.png", ASSETS_DIR / f"{cid}.png"]:
        if candidate.exists():
            return candidate
    return None


def get_image_size(img_path: Path | None) -> tuple:
    if img_path and img_path.exists():
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            return img.shape[1], img.shape[0]
    return 1080, 1920


def load_elements(cid: str) -> list:
    path = FEATURES_DIR / f"creative_{cid}" / "elements_data.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("elements", data) if isinstance(data, dict) else data


def process_creative(creative: dict, use_vision: bool = True) -> tuple[str, str]:
    """Returns (cid, status_message)"""
    cid = str(creative.get("id", ""))
    if cid.endswith("_v2"):
        return cid, "SKIP (generated image, not a source asset)"
        
    out_path = OUTPUT_DIR / f"creative_{cid}.json"

    raw_elements = load_elements(cid)
    if not raw_elements:
        return cid, "SKIP (no elements_data.json)"

    img_path = get_image_path(cid)
    img_w, img_h = get_image_size(img_path)

    llm_result = call_llm_single(
        creative, raw_elements, img_w, img_h,
        img_path if use_vision else None,
    )

    # Merge LLM element results with raw geometry
    llm_elements_by_id = {str(e["id"]): e for e in llm_result.get("elements", [])}
    final_elements = []
    for raw_e in raw_elements:
        eid = str(raw_e.get("id", ""))
        llm_e = llm_elements_by_id.get(eid, {"role": "unknown", "label": "Element", "description": ""})
        final_elements.append(build_final_element(raw_e, llm_e, img_w, img_h))

    # Ensure background exists
    if not any(e["role"] == "background" for e in final_elements):
        final_elements.insert(0, {
            "id": 0, "role": "background", "label": "Ad background",
            "description": "Full canvas background of the mobile advertisement.",
            "text_content": None,
            "bbox_xyxy": [0,0,img_w,img_h],
            "bbox_normalized": [0.0,0.0,1.0,1.0],
            "center_normalized": [0.5,0.5],
            "area_percentage": 100.0,
        })

    # Build precise elements_text from deterministic fields
    elems_text = " | ".join(
        f"{e['role']}: {e['description']}" + (f" text='{e['text_content']}'" if e["text_content"] else "")
        for e in final_elements if e["role"] != "background"
    )

    emb = llm_result.get("embedding_texts", {})
    emb["elements_text"] = elems_text

    final_json = {
        "creative_id": cid,
        "asset_file": f"assets/creative_{cid}.png",
        "canvas": {"width": img_w, "height": img_h},
        "global": llm_result.get("global", {}),
        "elements": final_elements,
        "embedding_texts": emb,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    n = len(final_elements)
    style = final_json["global"].get("visual_style", "")
    return cid, f"✓ {n} elements | {style}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    action="store_true", help="Skip already-processed creatives")
    parser.add_argument("--id",        type=str, help="Process a single creative ID")
    parser.add_argument("--ids",       type=str, help="Comma-separated list of IDs to process (e.g. 501080,501081)")
    parser.add_argument("--no-vision", action="store_true", help="Text-only mode (no image upload, cheaper)")
    parser.add_argument("--workers",   type=int, default=4, help="Parallel workers (default 4)")
    args = parser.parse_args()

    use_vision = not args.no_vision

    with DATA_JSON.open("r", encoding="utf-8") as f:
        all_creatives = json.load(f)

    if args.id:
        # Single-creative mode
        creative = next((c for c in all_creatives if str(c.get("id")) == args.id), None)
        if not creative:
            print(f"Creative {args.id} not found in data.json")
            return
        cid, msg = process_creative(creative, use_vision)
        print(f"[{cid}] {msg}")
        return

    # --ids filter (explicit set)
    id_filter = None
    if args.ids:
        id_filter = {s.strip() for s in args.ids.split(",") if s.strip()}

    # Filter if resuming
    to_process = []
    skipped = 0
    for c in all_creatives:
        cid = str(c.get("id", ""))
        if id_filter is not None and cid not in id_filter:
            skipped += 1
            continue
        if args.resume and (OUTPUT_DIR / f"creative_{cid}.json").exists():
            skipped += 1
            continue
        to_process.append(c)


    total = len(to_process)
    mode = "vision" if use_vision else "text-only"
    print(f"[Batch] {total} creatives to process ({skipped} skipped) | model={MODEL} | mode={mode} | workers={args.workers}")
    print(f"[Batch] Est. cost: ${total * 0.003:.2f} (vision) or ${total * 0.0008:.2f} (text-only)")
    print()

    done = failed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_creative, c, use_vision): c for c in to_process}
        for i, future in enumerate(as_completed(futures), start=1):
            cid = str(futures[future].get("id", "?"))
            try:
                cid, msg = future.result()
                done += 1
                elapsed = time.time() - start
                rate = done / elapsed
                eta  = (total - done) / rate if rate > 0 else 0
                print(f"[{i}/{total}] {cid} — {msg} | {rate:.1f}/s | ETA {eta/60:.1f}m")
            except Exception as e:
                failed += 1
                print(f"[{i}/{total}] {cid} — ERROR: {e}")

    print()
    print("=" * 60)
    print(f"[Batch] DONE: {done} generated | {failed} failed")
    print(f"[Batch] Time: {(time.time()-start)/60:.1f} min")
    print(f"[Batch] Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
