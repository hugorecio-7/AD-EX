import sys
import os
import asyncio
import torch
from diffusers import StableDiffusionInpaintPipeline

# Setup paths so we can import from backend root
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
sys.path.append(backend_dir)

from pipeline.step3_generation.core import generate_creative_with_flux

async def test_generation():
    print("🚀 Starting Diffusion Model Test...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 Using device: {device}")
    
    print("🔥 Loading Stable Diffusion Inpainting Pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    pipe.safety_checker = None
    
    # Test parameters
    creative_id = "500000"
    metadata = {
        "format": "Interstitial",
        "theme": "Action RPG",
        "hook_type": "Gameplay"
    }
    missing_features = ["High-quality lighting", "Dynamic characters", "Vibrant colors"]
    
    print(f"🎨 Generating upgraded creative for ID: {creative_id}...")
    output_path = await generate_creative_with_flux(
        creative_id=creative_id,
        metadata=metadata,
        missing_features=missing_features,
        pipe=pipe,
        # num_steps=20 # Faster for testing
    )
    
    print(f"✅ Success! Upgraded image saved at: {output_path}")

if __name__ == "__main__":
    asyncio.run(test_generation())
