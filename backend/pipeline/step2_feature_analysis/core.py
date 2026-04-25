"""
Step 2 — Feature Analysis Core

Given the top-performing creatives from Step 1, this step:
  1. Tries to use pre-enriched visual_semantic.json data (from GPT-4o Vision)
     to identify what high-performing creatives have that the query creative doesn't.
  2. Falls back to explain()-based keyword extraction if vision data is absent.
  3. Provides an async wrapper for on-demand vision enrichment (called during upgrade).
"""
import asyncio
from pathlib import Path

from pipeline.step2_feature_analysis.helpers import (
    extract_missing_features_from_enriched,
    parse_explanations_to_features,
    format_explanation_paragraph,
    enrich_creative_with_vision,
)


def find_missing_features(
    explanations_or_creatives,
    creative_id: str,
    top_creatives: list | None = None,
) -> list[str]:
    """
    Identify features the target creative is missing vs top performers.

    If top_creatives list is provided (and enriched visual_semantic.json files
    exist on disk), uses the vision-enriched role comparison.

    Otherwise, falls back to parsing the explain() strings.
    """
    if top_creatives is not None:
        features = extract_missing_features_from_enriched(creative_id, top_creatives)
        if features:
            print(f"[FeatureAnalysis] Vision-enriched missing features for {creative_id}: {features}")
            return features

    # Fallback path: use .explain() string parsing
    explanations = (
        explanations_or_creatives
        if isinstance(explanations_or_creatives[0], str)
        else [c.explain() for c in explanations_or_creatives]
    ) if explanations_or_creatives else []

    prompt_fragments = parse_explanations_to_features(explanations)
    print(f"[FeatureAnalysis] Fallback missing features for {creative_id}: {prompt_fragments}")
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
