import asyncio
from pipeline.step2_feature_analysis.helpers import parse_explanations_to_features, format_explanation_paragraph

def find_missing_features(explanations: list[str], creative_id: str) -> list[str]:
    """
    Parse the explanation strings from the top-performing creatives and identify
    the features that appear most often among them.
    """
    prompt_fragments = parse_explanations_to_features(explanations)
    print(f"[FeatureAnalysis] Missing/recommended features for {creative_id}: {prompt_fragments}")
    return prompt_fragments

async def explain_missing_features(missing_features: list[str], creative_id: str) -> str:
    """
    Produce a human-friendly explanation for the dashboard.
    Runs async so it can be gathered alongside the generation step.
    """
    await asyncio.sleep(0)   # yield to event loop
    return format_explanation_paragraph(missing_features, creative_id)
