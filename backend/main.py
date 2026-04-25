import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from performance_prediction import evaluate_new_creative

app = FastAPI()

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/evaluate/{creative_id}")
async def evaluate_creative(creative_id: str, format: str = None, theme: str = None, hook: str = None):
    results = await evaluate_new_creative(format, theme, hook, "simulated logic")
    return {
        "status": "success",
        "creative_id": creative_id,
        "metrics": results,
        "ai_reasoning": f"Backend successfully computed a score of {results['performance_score']} using PixelForge Neural Engine."
    }

if __name__ == "__main__":
    print("[SYSTEM] Starting Backend API Server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
