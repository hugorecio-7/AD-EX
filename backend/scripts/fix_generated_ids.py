"""
fix_generated_ids.py — one-shot cleanup script.

What it does:
  1. Finds all entries in data.json whose IDs are non-numeric (e.g. 500864_v2, 500864_impl1745...)
  2. Reassigns each one a fresh plain integer ID (next_available_id logic)
  3. Renames their asset image file to match the new numeric ID
  4. Saves the cleaned data.json
  5. Runs preprocess_masks.py --ids <new_ids> to generate SAM masks
  6. Runs preprocess_visual_semantic.py --ids <new_ids> to generate visual_semantic.json

Usage:
    python backend/scripts/fix_generated_ids.py [--dry-run]
"""
import sys
import os
import json
import shutil
import subprocess
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _BACKEND_DIR)

DATA_PATH  = os.path.join(_PROJECT_ROOT, "frontend", "src", "mocks", "data.json")
ASSETS_DIR = os.path.join(_PROJECT_ROOT, "frontend", "public", "data", "assets")
VS_DIR     = os.path.join(_PROJECT_ROOT, "frontend", "public", "data", "visual_semantic")
FEAT_DIR   = os.path.join(_PROJECT_ROOT, "output", "features")

VENV_PYTHON = os.path.join(_PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable   # fallback to current python


def _next_id(existing: set) -> int:
    candidate = max(existing) + 1 if existing else 600000
    while candidate in existing:
        candidate += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Fix non-numeric creative IDs in data.json")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    args = parser.parse_args()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_int_ids = set()
    for entry in data:
        try:
            existing_int_ids.add(int(str(entry["id"])))
        except (ValueError, TypeError, KeyError):
            pass

    bad_entries = [e for e in data if not str(e.get("id", "")).isdigit()]
    if not bad_entries:
        print("✓ No non-numeric IDs found. Nothing to fix.")
        return

    print(f"Found {len(bad_entries)} entries with non-numeric IDs:")
    for e in bad_entries:
        print(f"  {e['id']}")

    remap = {}   # old_id → new_id (str)
    for entry in bad_entries:
        old_id = str(entry["id"])
        new_num = _next_id(existing_int_ids)
        existing_int_ids.add(new_num)
        remap[old_id] = str(new_num)
        print(f"  {old_id}  →  {new_num}")

    if args.dry_run:
        print("\n[dry-run] No changes applied.")
        return

    # ── 1. Rename asset files ─────────────────────────────────────────────────
    for old_id, new_id in remap.items():
        for ext in (".png", ".jpg", ".jpeg"):
            old_file = os.path.join(ASSETS_DIR, f"creative_{old_id}{ext}")
            new_file = os.path.join(ASSETS_DIR, f"creative_{new_id}{ext}")
            if os.path.exists(old_file) and not os.path.exists(new_file):
                shutil.copy2(old_file, new_file)
                print(f"[Asset] {old_id}{ext}  →  {new_id}{ext}")

        # Also rename visual_semantic if it exists
        old_vs = os.path.join(VS_DIR, f"creative_{old_id}.json")
        new_vs = os.path.join(VS_DIR, f"creative_{new_id}.json")
        if os.path.exists(old_vs) and not os.path.exists(new_vs):
            shutil.copy2(old_vs, new_vs)
            # Patch the creative_id field inside
            with open(new_vs, "r", encoding="utf-8") as f:
                vs_data = json.load(f)
            vs_data["creative_id"] = new_id
            vs_data["asset_file"] = f"assets/creative_{new_id}.png"
            with open(new_vs, "w", encoding="utf-8") as f:
                json.dump(vs_data, f, indent=2, ensure_ascii=False)
            print(f"[VS JSON] patched  →  creative_{new_id}.json")

        # Rename/copy features dir
        old_feat = os.path.join(FEAT_DIR, f"creative_{old_id}")
        new_feat = os.path.join(FEAT_DIR, f"creative_{new_id}")
        if os.path.exists(old_feat) and not os.path.exists(new_feat):
            shutil.copytree(old_feat, new_feat)
            print(f"[Features] copied  →  creative_{new_id}/")

    # ── 2. Update data.json ───────────────────────────────────────────────────
    for entry in data:
        old_id = str(entry.get("id", ""))
        if old_id in remap:
            new_id = remap[old_id]
            entry["id"] = new_id
            # Fix image_url if it embeds the old ID
            for key in ("image_url", "asset_file"):
                if key in entry and old_id in str(entry[key]):
                    entry[key] = str(entry[key]).replace(old_id, new_id)

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ data.json updated with {len(remap)} remapped IDs.")

    new_ids_csv = ",".join(remap.values())

    # ── 3. Run preprocess_masks.py for new IDs ────────────────────────────────
    print(f"\n[MaskGen] Running preprocess_masks.py --ids {new_ids_csv} ...")
    masks_script = os.path.join(_SCRIPT_DIR, "preprocess_masks.py")
    subprocess.run([VENV_PYTHON, masks_script, "--ids", new_ids_csv], check=False)

    # ── 4. Run preprocess_visual_semantic.py for new IDs ─────────────────────
    print(f"\n[Semantic] Running preprocess_visual_semantic.py --ids {new_ids_csv} ...")
    sem_script = os.path.join(_SCRIPT_DIR, "preprocess_visual_semantic.py")
    # Check if the script supports --ids
    result = subprocess.run(
        [VENV_PYTHON, sem_script, "--ids", new_ids_csv],
        check=False,
    )
    if result.returncode != 0:
        print("[Semantic] Note: script may not support --ids. Run --resume manually to pick them up.")

    print("\n✓ Done! You can now run preprocess_masks.py without skips for these IDs.")


if __name__ == "__main__":
    main()
