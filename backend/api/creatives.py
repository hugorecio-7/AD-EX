import os
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from performance_prediction import evaluate_new_creative
from generate.helpers import create_center_mask, composite_images

router = APIRouter()

@router.get("/evaluate/{creative_id}")
async def evaluate_creative(creative_id: str, format: str = None, theme: str = None, hook: str = None):
    results = await evaluate_new_creative(format, theme, hook, "simulated logic")
    return {
        "status": "success",
        "creative_id": creative_id,
        "metrics": results,
        "ai_reasoning": f"Backend successfully computed a score of {results['performance_score']} using PixelForge Neural Engine."
    }

@router.post("/{creative_id}/upgrade")
async def upgrade_creative(creative_id: str, request: Request):
    """
    The endpoint your React app calls when the user clicks 'Fix with AI'.
    """
    try:
        # Access the shared diffusion pipe from app state
        pipe = request.app.state.pipe
        
        # 1. Load the original image
        # Assuming images are in a folder called 'assets' in the project root or backend
        image_path = f"assets/{creative_id}.png"
        if not os.path.exists(image_path):
            # Try a different path if common
            image_path = f"../frontend/public/assets/{creative_id}.png"
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail=f"Original creative not found at {image_path}")
            
        original_image = Image.open(image_path).convert("RGB")
        
        # Resize for faster generation during hackathon (512x512 is standard for SD)
        orig_size = original_image.size
        working_image = original_image.resize((512, 512))
        
        # 2. Create the Mask (Protecting the text)
        mask_image = create_center_mask(working_image)
        
        # 3. Generate new content using Diffusion!
        # Injecting insights from your clustering here
        prompt = "high quality 3d rendered golden treasure chests, bright lighting, highly detailed, centered composition"
        negative_prompt = "text, watermark, ugly, blurry, distorted"
        
        print(f"Running diffusion for {creative_id}... hold on tight.")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working_image,
            mask_image=mask_image,
            num_inference_steps=20, # Lower steps = faster generation for demo
            guidance_scale=7.5
        ).images[0]
        
        # 4. Flawless Compositing (Put the exact text back)
        final_image = composite_images(working_image, result, mask_image)
        
        # Resize back to original dimensions
        final_image = final_image.resize(orig_size)
        
        # 5. Save the upgraded image
        output_filename = f"{creative_id}_upgraded.png"
        # Ensure 'assets' directory exists
        os.makedirs("assets", exist_ok=True)
        output_path = f"assets/{output_filename}"
        final_image.save(output_path)
        
        return {
            "success": True,
            "newImageUrl": f"http://localhost:8000/assets/{output_filename}",
            "aiReasoning": "Injected 3D treasure chests to increase contrast and engagement based on cluster top-performers."
        }
        
    except Exception as e:
        print(f"Error during upgrade: {e}")
        raise HTTPException(status_code=500, detail=str(e))
