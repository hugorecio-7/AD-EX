import asyncio

def find_missing_features(explanations, creative_id):
    """
    Run the missing feature finder.
    """
    return ["feature_A", "feature_B"]

async def explain_missing_features(missing_features, creative_id):
    """
    Explain the missing features to the user in a human-friendly way.
    """
    await asyncio.sleep(0.1)
    return "These features are missing: " + ", ".join(missing_features)
