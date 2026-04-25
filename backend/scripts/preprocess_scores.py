"""
Preprocessing Script — Generate performance_scores.csv and data.json

Scores every creative in creatives.csv using the real LightGBM CTR predictor
(GEMINI/lgbm_ctr_model.txt) and writes:
  frontend/src/mocks/data.json          ← used by the frontend mock layer
  frontend/public/data/performance_scores.csv

Run from repo root:
  python backend/scripts/preprocess_scores.py
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.insert(0, backend_dir)

import csv
import json

from pipeline.step3_generation.evaluator import evaluate_creative_from_metadata


def clean_str(val):
    if val is None:
        return ""
    return str(val).strip()


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def csv_to_json():
    root_dir = os.path.dirname(backend_dir)

    creatives_path = os.path.join(root_dir, "frontend", "public", "data", "creatives.csv")
    summary_path   = os.path.join(root_dir, "frontend", "public", "data", "creative_summary.csv")
    output_path    = os.path.join(root_dir, "frontend", "src", "mocks", "data.json")
    scores_csv_path = os.path.join(root_dir, "frontend", "public", "data", "performance_scores.csv")
    assets_dir     = os.path.join(root_dir, "frontend", "public", "data", "assets")

    # Map summary data by creative_id
    summary_map: dict[str, dict] = {}
    with open(summary_path, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = clean_str(row.get("creative_id"))
            summary_map[cid] = row

    data = []
    with open(creatives_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[Scores] Evaluating {len(rows)} creatives with LightGBM model...")

    for i, row in enumerate(rows):
        cid = clean_str(row.get("creative_id"))
        summary = summary_map.get(cid, {})

        # ── Raw CTR from summary (used as baseline for uplift calc) ──────────
        raw_ctr = safe_float(summary.get("overall_ctr", "0"))
        ctr_pct = round(raw_ctr * 100, 4)   # convert to percentage

        # ── Build metadata dict for evaluator ────────────────────────────────
        metadata = {
            "format":                 clean_str(row.get("format")),
            "theme":                  clean_str(row.get("theme")),
            "hook_type":              clean_str(row.get("hook_type")),
            "vertical":               clean_str(row.get("vertical")),
            "language":               clean_str(row.get("language")),
            "dominant_color":         clean_str(row.get("dominant_color")),
            "emotional_tone":         clean_str(row.get("emotional_tone")),
            "readability_score":      safe_float(row.get("readability_score"), 7.0),
            "brand_visibility_score": safe_float(row.get("brand_visibility_score"), 0.7),
            "clutter_score":          safe_float(row.get("clutter_score"), 0.4),
            "novelty_score":          safe_float(row.get("novelty_score"), 0.6),
            "motion_score":           safe_float(row.get("motion_score"), 0.6),
            "has_gameplay":           int(safe_float(row.get("has_gameplay"), 0)),
            "has_ugc_style":          int(safe_float(row.get("has_ugc_style"), 0)),
        }

        # ── Run real LightGBM evaluation ──────────────────────────────────────
        analysis = evaluate_creative_from_metadata(metadata, old_ctr=raw_ctr)

        perf_score = analysis["performance_score"]
        is_fatigued = analysis["is_fatigued"]

        # ── Asset resolution ──────────────────────────────────────────────────
        asset_filename = f"creative_{cid}.png"
        asset_path = os.path.join(assets_dir, asset_filename)
        image_url = (
            f"/data/assets/{asset_filename}"
            if os.path.exists(asset_path)
            else "https://images.unsplash.com/photo-1518770660439-4636190af475"
        )

        headline = clean_str(row.get("headline"))

        entry = {
            "id":                cid,
            "campaign":          clean_str(row.get("campaign_id")),
            "advertiser":        clean_str(row.get("advertiser_name")),
            "format":            metadata["format"],
            "theme":             metadata["theme"],
            "hook_type":         metadata["hook_type"],
            "performance_score": round(perf_score, 4),
            "image_url":         image_url,
            "ctr":               ctr_pct,
            "cpi":               round(1.0 + (float(cid) % 100) / 100, 2),
            "fatigued":          is_fatigued,
            "fatigue_day":       analysis.get("fatigue_day"),
            "predicted_ctr":     analysis.get("predicted_ctr"),
            "predicted_uplift":  analysis.get("predicted_uplift"),
            "insights":          headline or f"AI Performance analysis: {perf_score:.3f}.",
            "cluster_id":        f"{metadata['format']}-{metadata['theme']}-{metadata['hook_type']}",
            "logic_version":     analysis.get("logic_version"),
        }
        data.append(entry)

        if (i + 1) % 100 == 0:
            print(f"[Scores]  ... {i + 1}/{len(rows)} done")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # ── Save scores CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(scores_csv_path), exist_ok=True)
    with open(scores_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "creative_id", "performance_score", "is_fatigued",
            "fatigue_day", "predicted_ctr", "predicted_uplift", "logic_version",
        ])
        for entry in data:
            writer.writerow([
                entry["id"],
                entry["performance_score"],
                entry["fatigued"],
                entry.get("fatigue_day", ""),
                entry.get("predicted_ctr", ""),
                entry.get("predicted_uplift", ""),
                entry.get("logic_version", ""),
            ])

    print(f"[SUCCESS] Mock data      → {output_path}")
    print(f"[SUCCESS] Scores CSV     → {scores_csv_path}")
    print(f"[SUCCESS] Total creatives evaluated: {len(data)}")


if __name__ == "__main__":
    csv_to_json()
