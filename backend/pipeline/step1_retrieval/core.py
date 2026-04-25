"""
Step 1 — Retrieval Core

This is the public interface of the retrieval step.
It delegates to the `created/` sub-package which contains the full scoring
system (PerformanceScore, ConfidenceScore, HealthScore, ContextScore,
SimilarityScore).

Usage from the AI engine:
    from pipeline.step1_retrieval.core import get_best_creatives
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .created.paths import CREATIVE_RETRIEVAL_INDEX_PATH, SEMANTIC_EMBEDDINGS_PATH
from .created.context_score import compute_context_score
from .helpers import Creative

# ─── constants ───────────────────────────────────────────────────────────────
FINAL_SCORE_COL = "retrieval_score_final"
HEALTH_COL = "health_score_final"
DEFAULT_TOP_N = 5


# ─── index loader (cached in memory for the lifetime of the process) ─────────
_retrieval_index: pd.DataFrame | None = None

def _load_index() -> pd.DataFrame:
    global _retrieval_index
    if _retrieval_index is not None:
        return _retrieval_index

    if not CREATIVE_RETRIEVAL_INDEX_PATH.exists():
        print(
            f"[Retrieval] WARNING: retrieval index not found at "
            f"{CREATIVE_RETRIEVAL_INDEX_PATH}. "
            "Run: python backend/scripts/preprocess_retrieval_index.py"
        )
        return pd.DataFrame()

    _retrieval_index = pd.read_csv(CREATIVE_RETRIEVAL_INDEX_PATH, dtype={"creative_id": str})
    print(f"[Retrieval] Loaded index with {len(_retrieval_index)} rows.")
    return _retrieval_index


# ─── optional: semantic similarity loader ───────────────────────────────────
_semantic_embeddings: dict | None = None

def _load_semantic_embeddings() -> dict | None:
    global _semantic_embeddings
    if _semantic_embeddings is not None:
        return _semantic_embeddings

    if not SEMANTIC_EMBEDDINGS_PATH.exists():
        # Semantic embeddings are optional — graceful fallback
        return None

    import pickle
    try:
        with SEMANTIC_EMBEDDINGS_PATH.open("rb") as f:
            _semantic_embeddings = pickle.load(f)
        print("[Retrieval] Loaded semantic embeddings.")
    except Exception as e:
        print(f"[Retrieval] Warning: Failed to load semantic embeddings: {e}")
        return None
    return _semantic_embeddings


# ─── main retrieval function ─────────────────────────────────────────────────

def get_best_creatives(
    creative_id: str,
    format_type: str,
    metadata: dict,
    top_n: int = DEFAULT_TOP_N,
) -> list[Creative]:
    """
    Return the top_n creatives most likely to improve the target creative.
    """
    print(f"[Retrieval] Querying best creatives for {creative_id} (Format: {format_type})")
    
    df = _load_index()
    if df.empty:
        # Graceful fallback to old JSON-based retrieval
        from .helpers import load_data
        data = load_data()
        return _json_fallback(data, creative_id, format_type, metadata, top_n)

    creative_id_str = str(creative_id)

    # Exclude the query itself and remove dead/unknown creatives
    candidates = df[df["creative_id"] != creative_id_str].copy()

    if HEALTH_COL in candidates.columns:
        # Prefer healthy creatives (health_score_final > 0.3 means not fully fatigued)
        candidates = candidates[candidates[HEALTH_COL] > 0.3]

    # ── Online: ContextScore ─────────────────────────────────────────────────
    query_row = df[df["creative_id"] == creative_id_str]
    query = query_row.iloc[0].to_dict() if not query_row.empty else metadata

    candidates = compute_context_score(query, candidates)

    # ── Online: SimilarityScore (optional) ──────────────────────────────────
    candidates["similarity_score_final"] = 0.5 # Default
    embeddings = _load_semantic_embeddings()
    if embeddings is not None:
        try:
            from .created.similarity_score import compute_semantic_similarity_for_existing_creative
            sim_df = compute_semantic_similarity_for_existing_creative(creative_id_str, embeddings)
            candidates = candidates.merge(
                sim_df[["creative_id", "similarity_score_final"]],
                on="creative_id",
                how="left",
                suffixes=("", "_new")
            )
            if "similarity_score_final_new" in candidates.columns:
                candidates["similarity_score_final"] = candidates["similarity_score_final_new"].fillna(0.5)
                candidates = candidates.drop(columns=["similarity_score_final_new"])
        except Exception as e:
            print(f"[Retrieval] Warning: Similarity calculation failed: {e}")

    # ── Final blended score ──────────────────────────────────────────────────
    # performance_score_final (offline) × context × similarity
    perf_col = "performance_score_final" if "performance_score_final" in candidates.columns else None
    conf_col = "confidence_score_final" if "confidence_score_final" in candidates.columns else None

    candidates[FINAL_SCORE_COL] = 0.0
    total_w = 0.0

    if perf_col:
        candidates[FINAL_SCORE_COL] += 0.45 * candidates[perf_col].fillna(0.5)
        total_w += 0.45
    if conf_col:
        candidates[FINAL_SCORE_COL] += 0.15 * candidates[conf_col].fillna(0.5)
        total_w += 0.15

    candidates[FINAL_SCORE_COL] += 0.25 * candidates["context_score_final"].fillna(0.5)
    candidates[FINAL_SCORE_COL] += 0.15 * candidates["similarity_score_final"]
    total_w += 0.40

    if total_w > 0:
        candidates[FINAL_SCORE_COL] = (candidates[FINAL_SCORE_COL] / (total_w + 0.40)).clip(0.0, 1.0)

    top = candidates.nlargest(top_n, FINAL_SCORE_COL)
    
    print(f"[Retrieval] Found {len(top)} top candidates for creative {creative_id_str}:")
    for idx, row in top.iterrows():
        print(f"  - ID: {row['creative_id']} | Final Score: {row[FINAL_SCORE_COL]:.4f} (Perf: {row.get(perf_col, 0):.2f}, Sim: {row['similarity_score_final']:.2f})")

    return [Creative(row.to_dict()) for _, row in top.iterrows()]


def _json_fallback(
    data: list[dict],
    creative_id: str,
    format_type: str,
    metadata: dict,
    top_n: int,
) -> list[Creative]:
    """Old JSON-based fallback when the retrieval index is not built yet."""
    target_cluster = (
        metadata.get("cluster_id")
        or f"{format_type}-{metadata.get('theme', '')}-{metadata.get('hook_type', '')}"
    )

    same_cluster = [
        d for d in data
        if d.get("cluster_id") == target_cluster
        and str(d.get("id")) != str(creative_id)
        and not d.get("fatigued", False)
    ]
    same_cluster.sort(key=lambda d: d.get("performance_score", 0), reverse=True)
    top = same_cluster[:top_n]

    if len(top) < top_n:
        global_top = [
            d for d in data
            if str(d.get("id")) != str(creative_id)
            and not d.get("fatigued", False)
            and d not in same_cluster
        ]
        global_top.sort(key=lambda d: d.get("performance_score", 0), reverse=True)
        top += global_top[: top_n - len(top)]

    print(f"[Retrieval] (JSON fallback) Returning {len(top)} top creatives for cluster '{target_cluster}'.")
    return [Creative(d) for d in top]
