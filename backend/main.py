import uvicorn
import torch
import os
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from diffusers import StableDiffusionInpaintPipeline
from api.creatives import router as creatives_router

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure assets directory exists for serving generated images
os.makedirs("assets", exist_ok=True)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Keep API responsive while model loads in background.
app.state.pipe = None


def _load_diffusion_pipe_in_background() -> None:
    try:
        print("Loading Diffusion Model... (This may take time on first run)")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

        # Optional: Disable safety checker to speed up hackathon dev (use responsibly)
        pipe.safety_checker = None

        app.state.pipe = pipe
        print("Model loaded successfully!")
    except Exception as exc:
        print(f"[SYSTEM] Warning: Could not load diffusion model yet: {exc}")
        app.state.pipe = None


threading.Thread(target=_load_diffusion_pipe_in_background, daemon=True).start()

# Include our organized routers
app.include_router(creatives_router, prefix="/api/creatives", tags=["creatives"])

@app.get("/")
async def root():
    return {"message": "PixelForge Backend API is running"}


@app.get("/evaluate/{creative_id}")
async def evaluate_root(creative_id: str, format: str = None, theme: str = None, hook: str = None):
    """Root-level alias so /evaluate/{id} works without the /api/creatives prefix."""
    from pipeline.step3_generation.core import evaluate_new_creative
    results = await evaluate_new_creative(format, theme, hook, creative_id)
    return {
        "status": "success",
        "creative_id": creative_id,
        "metrics": results,
        "ai_reasoning": (
            f"PixelForge Neural Engine computed a score of "
            f"{results['performance_score']} with predicted uplift {results['predicted_uplift']}."
        ),
    }

if __name__ == "__main__":
    print("[SYSTEM] Starting Backend API Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
