import uvicorn
import torch
import os
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

print("Loading Diffusion Model... (This takes a minute on first run)")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Optional: Disable safety checker to speed up hackathon dev (use responsibly)
pipe.safety_checker = None

# Store the pipe in app state so it can be accessed by routers
app.state.pipe = pipe
print("Model loaded successfully!")

# Include our organized routers
app.include_router(creatives_router, prefix="/api/creatives", tags=["creatives"])

@app.get("/")
async def root():
    return {"message": "PixelForge Backend API is running"}

if __name__ == "__main__":
    print("[SYSTEM] Starting Backend API Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
