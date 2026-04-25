from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.pipeline.step1_retrieval.created.paths import (
    CREATIVE_RETRIEVAL_INDEX_PATH,
    VISUAL_SEMANTIC_DIR,
)


def _make_mock_json(row: pd.Series) -> dict:
    creative_id = str(row["creative_id"])
    asset_file = str(row.get("asset_file", f"assets/creative_{creative_id}.png"))

    vertical = str(row.get("vertical", "unknown"))
    objective = str(row.get("objective", "unknown"))
    fmt = str(row.get("format", "unknown"))
    app_name = str(row.get("app_name", "unknown app"))
    language = str(row.get("language", "en"))

    global_text = (
        f"Mobile ad for {app_name}. Vertical: {vertical}. "
        f"Objective: {objective}. Format: {fmt}. "
        f"The creative shows a central subject, a visible headline, "
        f"and a call-to-action button."
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
        "canvas": {
            "width": 1080,
            "height": 1920,
        },
        "global": {
            "description": global_text,
            "visual_style": "synthetic mock visual style",
            "main_message": f"Promote {app_name} for {objective}.",
            "dominant_colors": ["blue", "yellow"],
            "emotional_tone": "exciting",
        },
        "elements": [
            {
                "id": 1,
                "role": "background",
                "label": "Ad background",
                "description": "Full canvas mobile advertising background.",
                "text_content": None,
                "bbox_xyxy": [0, 0, 1080, 1920],
                "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                "center_normalized": [0.5, 0.5],
                "area_percentage": 100.0,
            },
            {
                "id": 2,
                "role": "main_subject",
                "label": "Central subject",
                "description": "Main visual subject placed in the center of the creative.",
                "text_content": None,
                "bbox_xyxy": [180, 450, 900, 1450],
                "bbox_normalized": [0.167, 0.234, 0.833, 0.755],
                "center_normalized": [0.5, 0.495],
                "area_percentage": 35.0,
            },
            {
                "id": 3,
                "role": "headline",
                "label": "Headline text",
                "description": "Promotional headline placed near the top of the ad.",
                "text_content": "INSTALL NOW",
                "bbox_xyxy": [120, 120, 960, 300],
                "bbox_normalized": [0.111, 0.063, 0.889, 0.156],
                "center_normalized": [0.5, 0.109],
                "area_percentage": 7.2,
            },
            {
                "id": 4,
                "role": "cta",
                "label": "CTA button",
                "description": "Call-to-action button placed at the bottom center.",
                "text_content": "PLAY NOW",
                "bbox_xyxy": [320, 1600, 760, 1760],
                "bbox_normalized": [0.296, 0.833, 0.704, 0.917],
                "center_normalized": [0.5, 0.875],
                "area_percentage": 5.8,
            },
        ],
        "embedding_texts": {
            "global_text": global_text,
            "elements_text": elements_text,
            "ocr_text": ocr_text,
            "layout_text": layout_text,
        },
        "mock": True,
        "mock_note": "This JSON is only for pipeline testing, not for real similarity evaluation.",
    }


def create_mock_semantic_jsons(n: int = 5) -> None:
    if not CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {CREATIVE_RETRIEVAL_INDEX_PATH}. "
            "Run build_score_index first."
        )

    df = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH)

    # Prefer a small varied sample if available.
    preferred_ids = ["500000", "500003", "500761", "500494", "500849"]
    df["creative_id_str"] = df["creative_id"].astype(str)

    sample_df = df[df["creative_id_str"].isin(preferred_ids)].copy()

    if len(sample_df) < n:
        extra_df = df[~df["creative_id_str"].isin(sample_df["creative_id_str"])].head(
            n - len(sample_df)
        )
        sample_df = pd.concat([sample_df, extra_df], ignore_index=True)

    sample_df = sample_df.head(n)

    VISUAL_SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in sample_df.iterrows():
        creative_id = str(row["creative_id"])
        output_path = VISUAL_SEMANTIC_DIR / f"creative_{creative_id}.json"

        data = _make_mock_json(row)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Created mock semantic JSON: {output_path}")

    print(f"Done. Created {len(sample_df)} mock JSON files.")


if __name__ == "__main__":
    create_mock_semantic_jsons(n=5)