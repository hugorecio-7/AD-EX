"""
Step 2 — Feature Analysis Core

Given the top-performing creatives from Step 1, this step:
  1. Calls GPT-4o to analyze the visual gap between the query creative
     and the top performers (LLM Feature Gap Analysis).
  2. Falls back to heuristic role comparison if no enriched data exists.
  3. Falls back further to explain()-based extraction if nothing is available.
  4. Provides an async wrapper for on-demand vision enrichment (called during upgrade).
"""
import asyncio
from pathlib import Path

from pipeline.step2_feature_analysis.helpers import (
    extract_missing_features_from_enriched,
    parse_explanations_to_features,
    format_explanation_paragraph,
    enrich_creative_with_vision,
)
from pipeline.step2_feature_analysis.llm_feature_gap import (
    analyze_feature_gap_with_llm,
    _load_semantic,
)


def find_missing_features(
    explanations_or_creatives,
    creative_id: str,
    top_creatives: list | None = None,
) -> list[str]:
    """
    Identify VISUAL features the target creative is missing vs top performers.

    Priority order:
      1. LLM gap analysis (GPT-4o reads both JSONs, returns SD-ready descriptions)
      2. Heuristic role comparison (extract_missing_features_from_enriched)
      3. explain() string parsing (parse_explanations_to_features)
    """
    if top_creatives is not None and len(top_creatives) > 0:

        # ── Path 1: LLM Gap Analysis (best quality) ─────────────────────────
        top_ids = [
            str(c.get("creative_id", c.get("id", "")))
            for c in top_creatives
        ]
        top_ids = [cid for cid in top_ids if cid]

        # Only attempt if the query creative has real semantic data
        query_has_semantic = _load_semantic(creative_id) is not None

        if query_has_semantic and top_ids:
            result = analyze_feature_gap_with_llm(creative_id, top_ids)
            features = result.get("missing_visual_features", [])
            if features:
                print(f"[FeatureAnalysis] LLM gap features for {creative_id}: {features}")
                return features

        # ── Path 2: Heuristic role comparison ───────────────────────────────
        features = extract_missing_features_from_enriched(creative_id, top_creatives)
        if features:
            print(f"[FeatureAnalysis] Heuristic visual features for {creative_id}: {features}")
            return features

    # ── Path 3: explain() string parsing (legacy fallback) ──────────────────
    explanations = (
        explanations_or_creatives
        if explanations_or_creatives and isinstance(explanations_or_creatives[0], str)
        else [c.explain() for c in explanations_or_creatives]
    ) if explanations_or_creatives else []

    prompt_fragments = parse_explanations_to_features(explanations)
    print(f"[FeatureAnalysis] Fallback visual features for {creative_id}: {prompt_fragments}")
    return prompt_fragments


async def explain_missing_features(missing_features: list[str], creative_id: str) -> str:
    """
    Produce a human-friendly explanation for the dashboard.
    Runs async so it can be gathered alongside the generation step.
    """
    await asyncio.sleep(0)   # yield to event loop
    return format_explanation_paragraph(missing_features, creative_id)


async def enrich_creative_async(
    creative_id: str,
    creative_metadata: dict,
    image_path: str,
) -> dict | None:
    """
    Async wrapper for GPT-4o Vision enrichment.
    Called during the upgrade pipeline when visual_semantic.json is not pre-computed.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: enrich_creative_with_vision(creative_id, creative_metadata, image_path),
    )
