"""
Step 2 — Semantic Embedding Builder

Reads all visual_semantic.json files produced by the GPT-4o Vision enrichment,
encodes the text fields with a sentence-transformer, and saves the embeddings
to output/semantic_embeddings.pkl.

These embeddings are consumed by step1_retrieval/created/similarity_score.py
to rank retrieved creatives by semantic similarity.

Run from repo root:
    python backend/scripts/preprocess_semantic_embeddings.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer

from pipeline.step2_feature_analysis.semantic.semantic_json_loader import load_semantic_json_records

# ── Config ────────────────────────────────────────────────────────────────────
_STEP2_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _STEP2_DIR.parent.parent.parent.parent   # repo root

SENTENCE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SEMANTIC_TEXT_FIELDS = ["global_text", "elements_text", "ocr_text", "layout_text"]

SEMANTIC_EMBEDDINGS_DIR = _PROJECT_ROOT / "output" / "semantic"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_EMBEDDINGS_DIR / "semantic_embeddings.pkl"
SEMANTIC_EMBEDDING_INDEX_PATH = SEMANTIC_EMBEDDINGS_DIR / "semantic_embedding_index.csv"


def _encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_semantic_embeddings() -> None:
    print("[Semantic] Loading visual_semantic.json records from output/features/...")
    df = load_semantic_json_records()
    print(f"[Semantic] Loaded {len(df)} records.")

    print(f"[Semantic] Loading sentence-transformer: {SENTENCE_EMBEDDING_MODEL}")
    model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

    embeddings: dict = {
        "creative_ids": df["creative_id"].astype(str).tolist(),
        "asset_files": df["asset_file"].astype(str).tolist(),
        "model_name": SENTENCE_EMBEDDING_MODEL,
        "text_fields": SEMANTIC_TEXT_FIELDS,
    }

    for field in SEMANTIC_TEXT_FIELDS:
        if field not in df.columns:
            raise KeyError(f"Missing semantic text field in DataFrame: {field}")
        print(f"[Semantic] Encoding field: {field}")
        texts = df[field].fillna("").astype(str).tolist()
        embeddings[f"{field}_embeddings"] = _encode_texts(model, texts)

    SEMANTIC_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Semantic] Saving embeddings → {SEMANTIC_EMBEDDINGS_PATH}")
    with SEMANTIC_EMBEDDINGS_PATH.open("wb") as f:
        pickle.dump(embeddings, f)

    print(f"[Semantic] Saving index CSV → {SEMANTIC_EMBEDDING_INDEX_PATH}")
    df.to_csv(SEMANTIC_EMBEDDING_INDEX_PATH, index=False)

    print("[Semantic] Done.")


if __name__ == "__main__":
    build_semantic_embeddings()
