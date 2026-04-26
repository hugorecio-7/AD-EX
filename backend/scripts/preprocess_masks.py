import os
import sys
import json
import argparse

# Add backend directory to path so we can import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir)

from generate.mask_generator import generate_diffusion_mask

PROJECT_ROOT = os.path.dirname(backend_dir)
DATA_PATH = os.path.join(PROJECT_ROOT, "frontend", "src", "mocks", "data.json")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "frontend", "public", "data", "assets")
OUTPUT_FEATURES_DIR = os.path.join(PROJECT_ROOT, "output", "features")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess diffusion masks with flexible filters."
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start position in the creatives list (0-based, inclusive).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="End position in the creatives list (0-based, exclusive).",
    )
    parser.add_argument(
        "--min-id",
        type=int,
        default=None,
        help="Minimum creative id to process (inclusive).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=None,
        help="Maximum creative id to process (inclusive).",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated list of ids to process, e.g. 500751,500900,500999.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without generating masks.",
    )
    return parser.parse_args()


def parse_id_set(ids_arg):
    if not ids_arg:
        return None

    out = set()
    for token in ids_arg.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(token))
        except ValueError:
            raise ValueError(f"Invalid id in --ids: '{token}'")
    return out

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
    args = parse_args()
    creatives = load_creatives()
    total = len(creatives)

    start = max(0, args.start_index)
    end = total if args.end_index is None else min(total, args.end_index)

    if start >= total:
        print(f"Start index {start} is out of range. Total creatives: {total}")
        return

    if end <= start:
        print(f"Invalid range: start-index={start}, end-index={end}")
        return

    id_set = parse_id_set(args.ids)
    creatives = creatives[start:end]

    print(f"Loaded {total} creatives for preprocessing.")
    print(f"Index filter: [{start}, {end}) -> {len(creatives)} candidates")
    if args.min_id is not None or args.max_id is not None:
        print(f"ID range filter: min_id={args.min_id}, max_id={args.max_id}")
    if id_set is not None:
        print(f"Explicit ID filter: {len(id_set)} ids")
    if args.dry_run:
        print("Dry-run enabled: masks will not be generated")
    
    success = 0
    skipped = 0
    failed = 0
    filtered = 0
    
    for c in creatives:
        cid_raw = c.get("id")
        if cid_raw is None:
            skipped += 1
            continue

        try:
            cid_num = int(str(cid_raw))
        except (TypeError, ValueError):
            print(f"[{cid_raw}] Invalid creative id. Skipping.")
            skipped += 1
            continue

        if args.min_id is not None and cid_num < args.min_id:
            filtered += 1
            continue

        if args.max_id is not None and cid_num > args.max_id:
            filtered += 1
            continue

        if id_set is not None and cid_num not in id_set:
            filtered += 1
            continue

        cid = str(cid_num)
            
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
        if args.dry_run:
            skipped += 1
            continue

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
    print(f"Filtered out by args: {filtered}")

if __name__ == "__main__":
    main()
