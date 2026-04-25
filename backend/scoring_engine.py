import random

def compute_static_performance_score(creative_id):
    """
    Backend scoring logic for existing/static images.
    Deterministic based on ID for consistency.
    """
    random.seed(int(creative_id))
    score = round(random.uniform(0, 1), 3)
    is_fatigued = score < 0.3
    
    return {
        "performance_score": score,
        "is_fatigued": is_fatigued,
        "logic_version": "v1-static-deterministic"
    }

def evaluate_dynamic_creative(creative_id, features=None):
    """
    Evaluation logic for new generated images.
    """
    # Simulate a more complex calculation
    new_score = round(random.uniform(0.7, 0.98), 3)
    uplift = f"+{round(random.uniform(15, 35), 1)}%"
    
    return {
        "performance_score": new_score,
        "predicted_uplift": uplift,
        "is_fatigued": new_score < 0.3,
        "logic_version": "v1-dynamic-neural"
    }
