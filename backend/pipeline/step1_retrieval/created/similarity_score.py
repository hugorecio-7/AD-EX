from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SEMANTIC_SIMILARITY_WEIGHTS
from .paths import SEMANTIC_EMBEDDINGS_PATH


def load_semantic_embeddings(path: Path = SEMANTIC_EMBEDDINGS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Semantic embeddings not found: {path}. "
            "Run build_semantic_embeddings.py first."
        )

    with path.open("rb") as f:
        return pickle.load(f)


def cosine_scores_against_query(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Assumes embeddings are already normalized.
    Then cosine similarity is dot product.
    """
    return candidate_embeddings @ query_embedding


def compute_semantic_similarity_for_existing_creative(
    query_creative_id: str | int,
    embeddings: dict[str, Any],
) -> pd.DataFrame:
    """
    Computes semantic similarity between one existing creative and all others.

    Output columns:
        creative_id
        global_similarity
        elements_similarity
        ocr_similarity
        layout_similarity
        similarity_score_final
    """
    creative_ids = [str(x) for x in embeddings["creative_ids"]]
    query_creative_id = str(query_creative_id)

    if query_creative_id not in creative_ids:
        raise ValueError(f"Query creative_id not found: {query_creative_id}")

    query_idx = creative_ids.index(query_creative_id)

    result = pd.DataFrame({"creative_id": creative_ids})

    field_to_score_name = {
        "global_text": "global_similarity",
        "elements_text": "elements_similarity",
        "ocr_text": "ocr_similarity",
        "layout_text": "layout_similarity",
    }

    for field, score_col in field_to_score_name.items():
        emb_key = f"{field}_embeddings"

        if emb_key not in embeddings:
            raise KeyError(f"Missing embeddings key: {emb_key}")

        matrix = embeddings[emb_key]
        query_embedding = matrix[query_idx]

        scores = cosine_scores_against_query(query_embedding, matrix)

        # sentence-transformers cosine can be [-1, 1].
        # We map it to [0, 1] to combine with the other scores.
        result[score_col] = ((scores + 1.0) / 2.0).clip(0.0, 1.0)

    weights = SEMANTIC_SIMILARITY_WEIGHTS
    total_weight = float(sum(weights.values()))

    result["similarity_score_final"] = (
        (weights["global"] / total_weight) * result["global_similarity"]
        + (weights["elements"] / total_weight) * result["elements_similarity"]
        + (weights["ocr"] / total_weight) * result["ocr_similarity"]
        + (weights["layout"] / total_weight) * result["layout_similarity"]
    ).clip(0.0, 1.0)

    return result