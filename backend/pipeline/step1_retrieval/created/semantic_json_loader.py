from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import VISUAL_SEMANTIC_DIR


REQUIRED_TOP_LEVEL_FIELDS = [
    "creative_id",
    "asset_file",
    "canvas",
    "global",
    "elements",
    "embedding_texts",
]

REQUIRED_EMBEDDING_TEXT_FIELDS = [
    "global_text",
    "elements_text",
    "ocr_text",
    "layout_text",
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_semantic_json(data: dict[str, Any], path: Path) -> None:
    """
    Minimal validation for one semantic creative JSON.

    We keep it intentionally permissive because these files are generated
    by an AI pipeline and we do not want one imperfect field to break
    the whole process.
    """
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            raise ValueError(f"Missing required field '{field}' in {path}")

    embedding_texts = data.get("embedding_texts", {})
    if not isinstance(embedding_texts, dict):
        raise ValueError(f"'embedding_texts' must be a dict in {path}")

    for field in REQUIRED_EMBEDDING_TEXT_FIELDS:
        if field not in embedding_texts:
            raise ValueError(
                f"Missing embedding_texts['{field}'] in {path}"
            )


def _build_elements_summary(elements: list[dict[str, Any]]) -> str:
    """
    Fallback text if elements_text is missing or empty.
    """
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
    """
    Converts one semantic JSON into a flat record for indexing.
    """
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
    semantic_dir: Path = VISUAL_SEMANTIC_DIR,
) -> pd.DataFrame:
    """
    Loads all semantic JSON files into a DataFrame.

    Expected input:
        frontend/public/data/visual_semantic/*.json
    """
    if not semantic_dir.exists():
        raise FileNotFoundError(
            f"Semantic JSON directory not found: {semantic_dir}"
        )

    json_paths = sorted(semantic_dir.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No .json files found in semantic directory: {semantic_dir}"
        )

    records = []

    for path in json_paths:
        data = _load_json_file(path)
        record = semantic_json_to_record(data, path)
        records.append(record)

    df = pd.DataFrame(records)

    if df["creative_id"].duplicated().any():
        duplicated = df.loc[
            df["creative_id"].duplicated(keep=False),
            "creative_id",
        ].value_counts()
        raise ValueError(
            "Duplicated creative_id values in semantic JSONs:\n"
            f"{duplicated.to_string()}"
        )

    return df