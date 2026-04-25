"""
API Router — /api/creatives/*

  GET  /api/creatives/evaluate/{creative_id}  — score an existing creative
  POST /api/creatives/{creative_id}/upgrade   — run full AI upgrade pipeline
"""
import os
from fastapi import APIRouter, HTTPException, Request
from pipeline.step3_generation.core import evaluate_new_creative
from pipeline.step4_persistence.core import compute_static_performance_score, get_creative_by_id
from pipeline.ai_engine import generate_ai_variant_real

router = APIRouter()


@router.get("/evaluate/{creative_id}")
async def evaluate_creative(
    creative_id: str,
    format: str = None,
    theme: str = None,
    hook: str = None,
):
    """Return performance metrics for an existing creative."""
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


@router.post("/{creative_id}/upgrade")
async def upgrade_creative(creative_id: str, request: Request):
    """
    Full AI upgrade pipeline:
      SAM mask → SD inpainting → composite → persist → return result.
    """
    # Fetch metadata from the database so the pipeline has context
    metadata = get_creative_by_id(creative_id)
    if metadata is None:
        # Graceful fallback — pipeline can still run with empty metadata
        metadata = {"id": creative_id}

    # Grab the shared SD pipe from app state (set in main.py)
    pipe = getattr(request.app.state, "pipe", None)

    try:
        result = await generate_ai_variant_real(
            creative_id=creative_id,
            format_type=metadata.get("format", ""),
            metadata=metadata,
            pipe=pipe,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        print(f"[API] Error upgrading {creative_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "success": True,
        **result,
    }
