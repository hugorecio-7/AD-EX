"""
Step 2 — Mock Semantic JSON Generator

Creates mock visual_semantic.json files under output/features/creative_{id}/
for pipeline testing when GPT-4o Vision enrichment has not run yet.

These mocks have the same schema as the real output from
helpers.py → enrich_creative_with_vision().

Run from repo root:
    python backend/scripts/create_mock_semantic_jsons.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent.parent

sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

OUTPUT_FEATURES_DIR = _PROJECT_ROOT / "output" / "features"
CREATIVE_RETRIEVAL_INDEX_PATH = _PROJECT_ROOT / "output" / "creative_retrieval_index.csv"

PREFERRED_IDS = ["500000", "500003", "500761", "500494", "500849"]


def _make_mock_json(row: pd.Series) -> dict:
    creative_id = str(row["creative_id"])
    asset_file = str(row.get("asset_file", f"assets/creative_{creative_id}.png"))
    vertical = str(row.get("vertical", "unknown"))
    objective = str(row.get("objective", "unknown"))
    fmt = str(row.get("format", "unknown"))
    app_name = str(row.get("app_name", "unknown app"))

    global_text = (
        f"Mobile ad for {app_name}. Vertical: {vertical}. "
        f"Objective: {objective}. Format: {fmt}. "
        "The creative shows a central subject, a visible headline, "
        "and a call-to-action button."
    )
    elements_text = (
        "background: full canvas advertising background. "
        "main_subject: central visual subject in the middle of the ad. "
        "headline: promotional headline near the top. "
        "cta: call-to-action button near the bottom."
    )
    ocr_text = "PLAY NOW. INSTALL NOW."
    layout_text = (
        "Portrait mobile ad layout. Main subject centered. "
        "Headline at the top. CTA button at the bottom center."
    )

    return {
        "creative_id": creative_id,
        "asset_file": asset_file,
        "canvas": {"width": 1080, "height": 1920},
        "global": {
            "description": global_text,
            "visual_style": "synthetic mock visual style",
            "main_message": f"Promote {app_name} for {objective}.",
            "dominant_colors": ["blue", "yellow"],
            "emotional_tone": "exciting",
        },
        "elements": [
            {
                "id": 1, "role": "background", "label": "Ad background",
                "description": "Full canvas mobile advertising background.",
                "text_content": None,
                "bbox_xyxy": [0, 0, 1080, 1920],
                "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                "center_normalized": [0.5, 0.5], "area_percentage": 100.0,
            },
            {
                "id": 2, "role": "main_subject", "label": "Central subject",
                "description": "Main visual subject placed in the center of the creative.",
                "text_content": None,
                "bbox_xyxy": [180, 450, 900, 1450],
                "bbox_normalized": [0.167, 0.234, 0.833, 0.755],
                "center_normalized": [0.5, 0.495], "area_percentage": 35.0,
            },
            {
                "id": 3, "role": "headline", "label": "Headline text",
                "description": "Promotional headline placed near the top of the ad.",
                "text_content": "INSTALL NOW",
                "bbox_xyxy": [120, 120, 960, 300],
                "bbox_normalized": [0.111, 0.063, 0.889, 0.156],
                "center_normalized": [0.5, 0.109], "area_percentage": 7.2,
            },
            {
                "id": 4, "role": "cta", "label": "CTA button",
                "description": "Call-to-action button placed at the bottom center.",
                "text_content": "PLAY NOW",
                "bbox_xyxy": [320, 1600, 760, 1760],
                "bbox_normalized": [0.296, 0.833, 0.704, 0.917],
                "center_normalized": [0.5, 0.875], "area_percentage": 5.8,
            },
        ],
        "embedding_texts": {
            "global_text": global_text,
            "elements_text": elements_text,
            "ocr_text": ocr_text,
            "layout_text": layout_text,
        },
        "mock": True,
        "mock_note": "Mock file for pipeline testing. Replace with real GPT-4o enrichment.",
    }


def create_mock_semantic_jsons(n: int = 5) -> None:
    if not CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {CREATIVE_RETRIEVAL_INDEX_PATH}. "
            "Run preprocess_retrieval_index.py first."
        )

    df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)
    df["creative_id_str"] = df["creative_id"].astype(str)

    sample_df = df[df["creative_id_str"].isin(PREFERRED_IDS)].copy()
    if len(sample_df) < n:
        extra = df[~df["creative_id_str"].isin(sample_df["creative_id_str"])].head(n - len(sample_df))
        sample_df = pd.concat([sample_df, extra], ignore_index=True)
    sample_df = sample_df.head(n)

    for _, row in sample_df.iterrows():
        creative_id = str(row["creative_id"])
        creative_dir = OUTPUT_FEATURES_DIR / f"creative_{creative_id}"
        creative_dir.mkdir(parents=True, exist_ok=True)
        output_path = creative_dir / "visual_semantic.json"

        data = _make_mock_json(row)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[MockSemantic] Created: {output_path}")

    print(f"[MockSemantic] Done. Created {len(sample_df)} mock visual_semantic.json files.")


if __name__ == "__main__":
    create_mock_semantic_jsons(n=5)
