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


_PROJECT_ROOT_RETRIEVAL = Path(__file__).resolve().parent.parent.parent.parent

def _load_visual_semantic_for(creative_id: str) -> dict | None:
    """Load visual_semantic.json for a creative, checking both output and public dirs."""
    import json
    candidates_paths = [
        _PROJECT_ROOT_RETRIEVAL / "output" / "features" / f"creative_{creative_id}" / "visual_semantic.json",
        _PROJECT_ROOT_RETRIEVAL / "frontend" / "public" / "data" / "visual_semantic" / f"creative_{creative_id}.json",
    ]
    for p in candidates_paths:
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[Retrieval] Sim: loaded visual_semantic from {p}")
                return data
            except Exception as e:
                print(f"[Retrieval] Sim: failed to read {p}: {e}")
    return None


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

    # ── Online: SimilarityScore ──────────────────────────────────────────────
    candidates["similarity_score_final"] = 0.5  # Default neutral fallback
    embeddings = _load_semantic_embeddings()
    if embeddings is not None:
        creative_ids_in_index = [str(x) for x in embeddings.get("creative_ids", [])]
        if creative_id_str in creative_ids_in_index:
            # ── Fast path: query is in the precomputed pickle ──
            print(f"[Retrieval] Sim: found {creative_id_str} in precomputed embeddings — using fast path.")
            try:
                from .created.similarity_score import compute_semantic_similarity_for_existing_creative
                sim_df = compute_semantic_similarity_for_existing_creative(creative_id_str, embeddings)
                candidates = candidates.merge(
                    sim_df[["creative_id", "similarity_score_final"]],
                    on="creative_id", how="left", suffixes=("", "_new")
                )
                if "similarity_score_final_new" in candidates.columns:
                    candidates["similarity_score_final"] = candidates["similarity_score_final_new"].fillna(0.5)
                    candidates = candidates.drop(columns=["similarity_score_final_new"])
                print(f"[Retrieval] Sim: fast-path done. Sample Sim={candidates['similarity_score_final'].mean():.3f} (avg)")
            except Exception as e:
                print(f"[Retrieval] Sim: fast-path failed unexpectedly: {e}")
        else:
            # ── Slow path: compute embedding on-the-fly from visual_semantic.json ──
            print(f"[Retrieval] Sim: {creative_id_str} NOT in precomputed index ({len(creative_ids_in_index)} entries). Trying on-the-fly embedding...")
            _sem_json = _load_visual_semantic_for(creative_id_str)
            if _sem_json is not None:
                try:
                    from .created.similarity_score import cosine_scores_against_query
                    from .created.config import SENTENCE_EMBEDDING_MODEL, SEMANTIC_SIMILARITY_WEIGHTS
                    from sentence_transformers import SentenceTransformer
                    import numpy as np

                    # ✔️ Use embedding_texts sub-object — this is what build_semantic_embeddings uses
                    emb_texts = _sem_json.get("embedding_texts", {})
                    query_texts = {
                        "global_text":   emb_texts.get("global_text", ""),
                        "elements_text": emb_texts.get("elements_text", ""),
                        "ocr_text":      emb_texts.get("ocr_text", ""),
                        "layout_text":   emb_texts.get("layout_text", ""),
                    }
                    print(f"[Retrieval] Sim: embedding_texts found: " +
                          " | ".join(f"{k}={len(v)} chars" for k, v in query_texts.items()))

                    # If embedding_texts is missing/empty, fall back to global fields
                    if not any(query_texts.values()):
                        print(f"[Retrieval] Sim: WARNING — embedding_texts all empty, falling back to global fields")
                        g = _sem_json.get("global", {})
                        query_texts = {
                            "global_text":   g.get("description", "") + " " + " ".join(g.get("tags", [])),
                            "elements_text": " ".join(e.get("description", "") for e in _sem_json.get("elements", [])),
                            "ocr_text":      g.get("ocr_summary", ""),
                            "layout_text":   g.get("layout", ""),
                        }
                        print(f"[Retrieval] Sim: fallback texts: " +
                              " | ".join(f"{k}={len(v)} chars" for k, v in query_texts.items()))

                    print(f"[Retrieval] Sim: loading sentence-transformer '{SENTENCE_EMBEDDING_MODEL}'...")
                    _st_model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

                    total_w = float(sum(SEMANTIC_SIMILARITY_WEIGHTS.values()))
                    sim_scores = pd.Series(0.0, index=candidates.index)

                    for field, emb_key in [
                        ("global_text",   "global_text_embeddings"),
                        ("elements_text", "elements_text_embeddings"),
                        ("ocr_text",      "ocr_text_embeddings"),
                        ("layout_text",   "layout_text_embeddings"),
                    ]:
                        if emb_key not in embeddings:
                            print(f"[Retrieval] Sim: key {emb_key} missing from pickle, skipping")
                            continue
                        text = query_texts.get(field, "")
                        if not text.strip():
                            print(f"[Retrieval] Sim: {field} is EMPTY — this will produce 0.5 for this component")
                        matrix = embeddings[emb_key]  # (N, D)
                        q_vec = _st_model.encode([text], normalize_embeddings=True)[0]
                        raw = cosine_scores_against_query(q_vec, matrix)  # (N,)
                        scores_01 = ((raw + 1.0) / 2.0).clip(0.0, 1.0)
                        score_series = pd.Series(dict(zip(creative_ids_in_index, scores_01)))
                        weight_key = field.replace("_text", "")
                        weight = SEMANTIC_SIMILARITY_WEIGHTS.get(weight_key, 0)
                        sim_scores_raw = candidates["creative_id"].map(score_series).fillna(0.5)
                        print(f"[Retrieval] Sim: {field} → avg={sim_scores_raw.mean():.3f} (weight={weight})")
                        sim_scores = sim_scores + (weight / total_w) * sim_scores_raw

                    candidates["similarity_score_final"] = sim_scores.clip(0.0, 1.0)
                    print(f"[Retrieval] Sim: on-the-fly done. Final avg Sim={candidates['similarity_score_final'].mean():.3f}")
                except Exception as e:
                    print(f"[Retrieval] Sim: on-the-fly embedding FAILED: {e} — keeping 0.5 fallback")
                    import traceback; traceback.print_exc()
            else:
                print(f"[Retrieval] Sim: no visual_semantic.json found for {creative_id_str} — using 0.5 fallback (run post-upgrade enrichment!)")

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
