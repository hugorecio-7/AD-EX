from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from sentence_transformers import SentenceTransformer

from backend.retriver.config import (
    SENTENCE_EMBEDDING_MODEL,
    SEMANTIC_TEXT_FIELDS,
)
from backend.retriver.paths import (
    SEMANTIC_EMBEDDINGS_DIR,
    SEMANTIC_EMBEDDINGS_PATH,
    SEMANTIC_EMBEDDING_INDEX_PATH,
)
from backend.retriver.semantic_json_loader import load_semantic_json_records


def _encode_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    """
    Encodes texts into normalized embeddings.

    normalize_embeddings=True makes cosine similarity equivalent to dot product.
    """
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_semantic_embeddings() -> None:
    print("[semantic] Loading semantic JSON records...")
    df = load_semantic_json_records()

    print(f"[semantic] Loaded records: {len(df)}")

    print(f"[semantic] Loading model: {SENTENCE_EMBEDDING_MODEL}")
    model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

    embeddings = {
        "creative_ids": df["creative_id"].astype(str).tolist(),
        "asset_files": df["asset_file"].astype(str).tolist(),
        "model_name": SENTENCE_EMBEDDING_MODEL,
        "text_fields": SEMANTIC_TEXT_FIELDS,
    }

    for field in SEMANTIC_TEXT_FIELDS:
        if field not in df.columns:
            raise KeyError(f"Missing semantic text field: {field}")

        print(f"[semantic] Encoding field: {field}")
        texts = df[field].fillna("").astype(str).tolist()
        embeddings[f"{field}_embeddings"] = _encode_texts(model, texts)

    SEMANTIC_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[semantic] Saving embeddings to: {SEMANTIC_EMBEDDINGS_PATH}")
    with SEMANTIC_EMBEDDINGS_PATH.open("wb") as f:
        pickle.dump(embeddings, f)

    print(f"[semantic] Saving embedding index to: {SEMANTIC_EMBEDDING_INDEX_PATH}")
    df.to_csv(SEMANTIC_EMBEDDING_INDEX_PATH, index=False)

    print("[semantic] Done.")


if __name__ == "__main__":
    build_semantic_embeddings()