import sys
import json
import time
import asyncio

from creative_retrieval import get_best_creatives
from feature_analysis import find_missing_features, explain_missing_features
from image_generation import generate_creative_with_flux
from performance_prediction import predict_performance_uplift
from db_storage import store_new_creative

def generate_ai_variant_real(creative_id, format_type, metadata):
    time_s = time.time()

    # 1. Retrieve top cases
    top_cases = get_best_creatives(creative_id, format_type, metadata)
    
    # 2. Get the explanations 
    explanations = [n.explain() for n in top_cases]
    
    # 3. Run the missing feature finder
    missing_features = find_missing_features(explanations, creative_id)

    async def async_pipeline():
        async def generation_and_prediction():
            # 4. Generate the new creative using the missing features as a prompt
            new_file = await generate_creative_with_flux(creative_id, metadata, missing_features)
            
            # 5. Predict the performance uplift based on historical data of similar cases
            uplift = await predict_performance_uplift(missing_features, creative_id)
            
            return new_file, uplift
            
        return await asyncio.gather(
            generation_and_prediction(),
            # 6. Explain the missing features to the user in a human-friendly way
            explain_missing_features(missing_features, creative_id)
        )
        
    (new_creative_file, predicted_uplift), missing_features_explained = asyncio.run(async_pipeline())

    # 7. Store the new creative and update the database
    new_creative_id = store_new_creative(creative_id, new_creative_file)

    time_e = time.time()

    result = {
        "status": "success",
        "creative_id": new_creative_id,
        "new_image_url": f"/data/assets/creative_{new_creative_id}.png",
        "metadata": {
            "api_latency": time_e - time_s,
            "model": "smadex-custom-flux-v1-intelligence",
            "missing_features_detected": missing_features_explained,
            "predicted_uplift": predicted_uplift
        }
    }
    return result

def generate_ai_variant(creative_id, format_type, metadata):
    """
    Mock AI Generation script.
    In a real scenario, this would call Flux/Stable Diffusion/DALL-E
    """
    print(f"DEBUG: Initializing PixelForge AI for {creative_id}...")
    time.sleep(1)
    
    # Simulate processing logic based on tags
    prompt = f"A high-quality {format_type} advertisement image. Theme: {theme}. Hook: {hook}."
    
    # Mock result
    result = {
        "status": "success",
        "creative_id": creative_id,
        "new_image_url": f"/data/assets/creative_{creative_id}_upgraded.png",
        "metadata": {
            "prompt_used": prompt,
            "api_latency": "2.4s",
            "model": "smadex-custom-flux-v1"
        }
    }
    return result

if __name__ == "__main__":
    # For CLI testing
    if len(sys.argv) > 1:
        cid = sys.argv[1]
        # Use the real logic pipeline for testing
        print(json.dumps(generate_ai_variant_real(cid, "rewarded_video", "gameplay", "free rewards")))
