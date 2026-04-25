import asyncio
from scoring_engine import evaluate_dynamic_creative

async def evaluate_new_creative(format_type, theme, hook, missing_features):
    """
    Evaluate a new AI-generated creative and return its performance metrics.
    Delegate to the centralized scoring engine.
    """
    await asyncio.sleep(0.4) 
    return evaluate_dynamic_creative(None, missing_features)

async def predict_performance_uplift(missing_features, creative_id):
    # This is for the 'prediction' phase in existing pipelines
    res = await evaluate_new_creative(None, None, None, missing_features)
    return res["predicted_uplift"]
