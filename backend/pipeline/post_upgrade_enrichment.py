"""
Post-upgrade enrichment — runs after generate_ai_variant_real() succeeds.

Two steps:
  1. generate_diffusion_mask() → elements_data.json + mask PNG
  2. process_creative()        → visual_semantic.json for the new _v2 creative

Both steps run in a background thread so they don't block the API response.
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

_THIS_DIR   = Path(__file__).resolve().parent
_BACKEND    = _THIS_DIR.parent
_PROJECT    = _BACKEND.parent

ASSETS_DIR      = _PROJECT / "frontend" / "public" / "data" / "assets"
VISUAL_SEM_DIR  = _PROJECT / "frontend" / "public" / "data" / "visual_semantic"
FEATURES_DIR    = _PROJECT / "output" / "features"


def _run(original_id: str, new_id: str, new_image_path: str | Path) -> None:
    """Called in a background thread — does NOT raise to the caller."""
    new_image_path = Path(new_image_path)
    print(f"[PostUpgrade] Starting enrichment for {new_id} ...")

    # ── 0. Ensure the image exists in the public assets directory ─────────────
    dst_asset = ASSETS_DIR / f"creative_{new_id}.png"
    if not dst_asset.exists() and new_image_path.exists():
        try:
            shutil.copy2(new_image_path, dst_asset)
            print(f"[PostUpgrade] Copied image → {dst_asset}")
        except Exception as e:
            print(f"[PostUpgrade] Warning: could not copy image: {e}")

    # Use whichever exists
    image_to_use = dst_asset if dst_asset.exists() else new_image_path
    if not image_to_use.exists():
        print(f"[PostUpgrade] ERROR: No image found at {image_to_use}, aborting enrichment.")
        return

    # ── 1. Run SAM + EasyOCR mask generation ─────────────────────────────────
    output_dir = FEATURES_DIR / f"creative_{new_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from generate.mask_generator import generate_diffusion_mask
        generate_diffusion_mask(
            image_path=str(image_to_use),
            project_root=str(_PROJECT),
            output_dir=str(output_dir),
        )
        print(f"[PostUpgrade] ✓ Mask + elements_data.json → {output_dir}")
    except Exception as e:
        print(f"[PostUpgrade] Mask generation failed (non-fatal): {e}")
        # Create a minimal elements_data.json so the semantic step can still run
        _write_fallback_elements(output_dir, original_id)

    # ── 2. Generate visual_semantic.json ──────────────────────────────────────
    try:
        from scripts.preprocess_visual_semantic import process_creative as _process
        # Build a minimal creative dict (same fields as data.json entries)
        original_meta = _load_original_meta(original_id)
        creative_dict = {**original_meta, "id": new_id, "is_upgraded": True}
        cid, msg = _process(creative_dict, use_vision=True)
        print(f"[PostUpgrade] ✓ visual_semantic.json: {msg}")
    except Exception as e:
        print(f"[PostUpgrade] Semantic enrichment failed: {e}")
        # Last-resort: copy the original's semantic JSON and patch the ID
        _copy_and_patch_semantic(original_id, new_id, image_to_use)

    print(f"[PostUpgrade] Done enriching {new_id}.")


def _load_original_meta(original_id: str) -> dict:
    """Try to read the original creative's metadata from data.json."""
    data_json = _PROJECT / "frontend" / "src" / "mocks" / "data.json"
    if not data_json.exists():
        return {"id": original_id}
    with data_json.open("r", encoding="utf-8") as f:
        all_creatives = json.load(f)
    for c in all_creatives:
        if str(c.get("id", "")) == str(original_id):
            return c
    return {"id": original_id}


def _write_fallback_elements(output_dir: Path, original_id: str) -> None:
    """Copy the original's elements_data.json as a fallback."""
    src = FEATURES_DIR / f"creative_{original_id}" / "elements_data.json"
    dst = output_dir / "elements_data.json"
    if src.exists() and not dst.exists():
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass


def _copy_and_patch_semantic(original_id: str, new_id: str, image_path: Path) -> None:
    """Copy original's visual_semantic.json and patch the creative_id field."""
    src = VISUAL_SEM_DIR / f"creative_{original_id}.json"
    dst = VISUAL_SEM_DIR / f"creative_{new_id}.json"
    if not src.exists() or dst.exists():
        return
    try:
        with src.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["creative_id"] = new_id
        data["asset_file"]  = f"assets/creative_{new_id}.png"
        with dst.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[PostUpgrade] ✓ Patched semantic JSON copy → {dst}")
    except Exception as e:
        print(f"[PostUpgrade] Could not copy semantic JSON: {e}")


def enrich_upgraded_creative(original_id: str, new_id: str, new_image_path: str | Path) -> None:
    """
    Kick off enrichment in a background daemon thread.
    Returns immediately — the caller (upgrade endpoint) is not blocked.
    """
    t = threading.Thread(
        target=_run,
        args=(original_id, new_id, new_image_path),
        daemon=True,
        name=f"enrich-{new_id}",
    )
    t.start()
