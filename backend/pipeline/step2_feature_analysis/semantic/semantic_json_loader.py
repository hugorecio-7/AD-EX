"""
Step 2 — Semantic JSON Loader

Loads visual_semantic.json files produced by the GPT-4o Vision enrichment
(step2_feature_analysis/helpers.py → enrich_creative_with_vision) into a
flat DataFrame ready for sentence-transformer embedding.

These JSONs live at:
  output/features/creative_{id}/visual_semantic.json

and are the primary output of Step 2 preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_SEMANTIC_DIR = Path(__file__).resolve().parent          # step2/semantic
_STEP2_DIR = _SEMANTIC_DIR.parent                        # step2
_PIPELINE_DIR = _STEP2_DIR.parent                        # pipeline
_BACKEND_DIR = _PIPELINE_DIR.parent                      # backend
_PROJECT_ROOT = _BACKEND_DIR.parent                      # repo root

# Default source: output/features/**/visual_semantic.json
OUTPUT_FEATURES_DIR = _PROJECT_ROOT / "output" / "features"

# ── Validation ────────────────────────────────────────────────────────────────
REQUIRED_TOP_LEVEL_FIELDS = ["creative_id", "asset_file", "canvas", "global", "elements", "embedding_texts"]
REQUIRED_EMBEDDING_TEXT_FIELDS = ["global_text", "elements_text", "ocr_text", "layout_text"]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validate_semantic_json(data: dict[str, Any], path: Path) -> None:
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {path}")
    embedding_texts = data.get("embedding_texts", {})
    if not isinstance(embedding_texts, dict):
        raise ValueError(f"'embedding_texts' must be a dict in {path}")
    for field in REQUIRED_EMBEDDING_TEXT_FIELDS:
        if field not in embedding_texts:
            raise ValueError(f"Missing embedding_texts['{field}'] in {path}")


def _build_elements_summary(elements: list[dict[str, Any]]) -> str:
    parts = []
    for element in elements:
        role = _safe_str(element.get("role"))
        label = _safe_str(element.get("label"))
        description = _safe_str(element.get("description"))
        text_content = _safe_str(element.get("text_content"))
        piece = f"{role}: {label}. {description}"
        if text_content:
            piece += f" Text: {text_content}."
        parts.append(piece)
    return " ".join(parts).strip()


def semantic_json_to_record(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """Converts one visual_semantic.json into a flat record for indexing."""
    _validate_semantic_json(data, path)
    embedding_texts = data.get("embedding_texts", {})
    elements = data.get("elements", [])
    global_text = _safe_str(embedding_texts.get("global_text"))
    elements_text = _safe_str(embedding_texts.get("elements_text"))
    ocr_text = _safe_str(embedding_texts.get("ocr_text"))
    layout_text = _safe_str(embedding_texts.get("layout_text"))

    if not elements_text:
        elements_text = _build_elements_summary(elements)

    global_block = data.get("global", {})
    return {
        "creative_id": str(data.get("creative_id")),
        "asset_file": _safe_str(data.get("asset_file")),
        "json_path": str(path),
        "global_description": _safe_str(global_block.get("description")),
        "visual_style": _safe_str(global_block.get("visual_style")),
        "main_message": _safe_str(global_block.get("main_message")),
        "emotional_tone": _safe_str(global_block.get("emotional_tone")),
        "global_text": global_text,
        "elements_text": elements_text,
        "ocr_text": ocr_text,
        "layout_text": layout_text,
        "num_elements": len(elements),
    }


def load_semantic_json_records(
    features_dir: Path = OUTPUT_FEATURES_DIR,
) -> pd.DataFrame:
    """
    Discovers all visual_semantic.json files under output/features/**/
    and loads them into a flat DataFrame.
    """
    json_paths = sorted(features_dir.glob("*/visual_semantic.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No visual_semantic.json files found under: {features_dir}\n"
            "Run backend/scripts/preprocess_semantic_enrichment.py first."
        )

    records = []
    for path in json_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            record = semantic_json_to_record(data, path)
            records.append(record)
        except Exception as e:
            print(f"[SemanticLoader] Skipping {path}: {e}")

    df = pd.DataFrame(records)

    if df["creative_id"].duplicated().any():
        dups = df.loc[df["creative_id"].duplicated(keep=False), "creative_id"].value_counts()
        print(f"[SemanticLoader] WARNING: Duplicated creative_ids: {dups.to_string()}")
        df = df.drop_duplicates(subset="creative_id", keep="first")

    return df
