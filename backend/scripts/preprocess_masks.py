import os
import sys
import json

# Add backend directory to path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir)

from generate.mask_generator import generate_diffusion_mask

PROJECT_ROOT = os.path.dirname(backend_dir)
DATA_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "mocks", "data.json")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "data", "assets")
OUTPUT_FEATURES_DIR = os.path.join(PROJECT_ROOT, "output", "features")

def load_creatives():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_image_path(creative_id):
    candidates = [
        os.path.join(ASSETS_DIR, f"creative_{creative_id}.png"),
        os.path.join(ASSETS_DIR, f"{creative_id}.png"),
        os.path.join(backend_dir, "assets", f"creative_{creative_id}.png"),
        os.path.join(backend_dir, "assets", f"{creative_id}.png"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def main():
    creatives = load_creatives()
    print(f"Loaded {len(creatives)} creatives for preprocessing.")
    
    success = 0
    skipped = 0
    failed = 0
    
    for c in creatives:
        cid = c.get("id")
        if not cid:
            continue
            
        output_dir = os.path.join(OUTPUT_FEATURES_DIR, f"creative_{cid}")
        expected_mask = os.path.join(output_dir, f"creative_{cid}_diffusion_mask.png")
        
        if os.path.exists(expected_mask):
            print(f"[{cid}] Mask already exists. Skipping.")
            skipped += 1
            continue
            
        img_path = resolve_image_path(cid)
        if not img_path:
            print(f"[{cid}] Image not found.")
            failed += 1
            continue
            
        print(f"[{cid}] Processing: {img_path}")
        try:
            generate_diffusion_mask(
                image_path=img_path,
                project_root=PROJECT_ROOT,
                output_dir=output_dir
            )
            success += 1
        except Exception as e:
            print(f"[{cid}] Error processing: {e}")
            failed += 1
            
    print("\n--- Preprocessing Complete ---")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main()
