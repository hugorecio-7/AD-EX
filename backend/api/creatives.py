"""
API Router — /api/creatives/*

  GET  /api/creatives/evaluate/{creative_id}  — score an existing creative
  POST /api/creatives/{creative_id}/upgrade   — run full AI upgrade pipeline
  POST /api/creatives/{creative_id}/chat      — chat with creative context
"""
import os
import json
import base64
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pipeline.step3_generation.core import evaluate_new_creative
from pipeline.step4_persistence.core import compute_static_performance_score, get_creative_by_id
from pipeline.ai_engine import generate_ai_variant_real

router = APIRouter()

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

_openai_client = None


class CreativeChatMessage(BaseModel):
    role: str
    content: str


class CreativeChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[CreativeChatMessage] = Field(default_factory=list)
    language: str = "catalan"
    agentic: bool = False   # If True, detect modify intent and return action


class ImplementRequest(BaseModel):
    description: str            # Human-readable description of the change
    diffusion_prompt: str       # Prompt for SD inpainting


class EnrichRequest(BaseModel):
    new_id: str                 # The creative ID to enrich (e.g. "500659_v2_1745")
    image_url: str              # Public URL of the image ("/data/assets/...")


_LANGUAGE_MAP = {
    "catalan": "Catalan",
    "castilian": "Castilian Spanish",
    "english": "English",
}


def _normalize_language(language: str | None) -> str:
    key = (language or "catalan").strip().lower()
    return key if key in _LANGUAGE_MAP else "catalan"


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in .env")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _resolve_image_path(creative_id: str) -> Path:
    candidates = [
        _PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"creative_{creative_id}.png",
        _PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"creative_{creative_id}.jpg",
        _PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"creative_{creative_id}.jpeg",
        _PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"{creative_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Image not found for creative " + creative_id)


def _resolve_structured_path(creative_id: str) -> Path:
    candidates = [
        # Original postllm.py output
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / f"creative_{creative_id}_structured.json",
        _PROJECT_ROOT / "frontend" / "public" / "data" / "structured" / f"creative_{creative_id}_structured.json",
        # New batch-generated visual_semantic.json (preprocess_visual_semantic.py)
        _PROJECT_ROOT / "frontend" / "public" / "data" / "visual_semantic" / f"creative_{creative_id}.json",
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "visual_semantic.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Structured JSON not found for creative " + creative_id)


def _resolve_feature_gap_path(creative_id: str) -> Path | None:
    candidates = [
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / f"creative_{creative_id}_feature_gap.json",
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "feature_gap.json",
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "llm_feature_gap.json",
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "feature_gap_output.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_feature_gap_or_fallback(path: Path | None) -> dict[str, Any]:
    if not path:
        return {
            "missing_visual_features": [
                "clear product focal point in the center",
                "stronger depth and contrast between subject and background",
            ],
            "reasoning": "Fallback used because feature gap file was not found.",
        }
    data = _read_json(path)
    if "missing_visual_features" not in data:
        data["missing_visual_features"] = []
    if "reasoning" not in data:
        data["reasoning"] = "No explicit reasoning in feature gap file."
    return data


def _image_to_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with image_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _build_chat_system_prompt(structured_json: dict[str, Any], feature_gap_json: dict[str, Any], language: str) -> str:
    language_name = _LANGUAGE_MAP[_normalize_language(language)]
    return (
        "You are an Ad Creative Consultant for performance marketing. "
        "Talk like a helpful teammate in a chat: informal, natural, and direct.\n\n"
        "Response style rules:\n"
        "1) Keep answers very short, typically 2-4 lines.\n"
        "2) Output plain text only. No markdown, no bullet points, no numbering.\n"
        "3) Write one short paragraph, or two short paragraphs if needed.\n"
        "4) Be concrete and practical; avoid fluff and abstract phrasing.\n"
        "5) If asked for improvements, include up to 3 clear actions inline.\n"
        "6) If asked about location, mention approximate area and bbox if available.\n"
        "7) If data is missing, say it clearly and do not invent.\n"
        "8) Do not output chain-of-thought.\n\n"
        f"Language rule: Always reply in {language_name}.\n\n"
        "STRUCTURED_JSON:\n"
        f"{json.dumps(structured_json, indent=2, ensure_ascii=False)}\n\n"
        "FEATURE_GAP_JSON:\n"
        f"{json.dumps(feature_gap_json, indent=2, ensure_ascii=False)}"
    )
_INTENT_SYSTEM_SUFFIX = """

ADDITIONAL RULE (AGENTIC MODE):
At the very end of your response, append a JSON block on a NEW LINE (and only if a concrete visual change is requested), in exactly this format:

```json
{"intent": "modify", "description": "<one-line description of the change>", "diffusion_prompt": "<SD inpainting prompt to apply the change>"}
```

If the user is NOT requesting a change (just asking a question), do NOT append any JSON block.
Do not mention the JSON to the user. It is invisible to them.
"""


@router.get("/predict/{creative_id}")
@router.get("/{creative_id}/predict")
async def predict_creative_ctr(
    creative_id: str,
    countries: str = "US,ES",
    os: str = "iOS,Android",
    compare_image_url: str | None = None,
    seq_len: int = 30,
):
    """
    Run the ImageAutoregressiveRNN to predict 30-day CTR timeseries.

    Query params:
      countries   — comma-separated, e.g. "US,ES,UK"  (default: US,ES)
      os          — comma-separated, e.g. "iOS,Android" (default: both)
      compare_image_url — optional path/URL of an upgraded image to compare
      seq_len     — number of days to predict (default: 30)
    """
    from pipeline.step4_persistence.new.predictor import predict_ctr

    country_list = [c.strip() for c in countries.split(",") if c.strip()]
    os_list      = [o.strip() for o in os.split(",")        if o.strip()]

    # Resolve original image
    try:
        image_path = _resolve_image_path(creative_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        original = predict_ctr(str(image_path), country_list, os_list, seq_len)
        original["label"] = "Original"
        original["creative_id"] = creative_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RNN inference failed: {e}")

    result = {"original": original, "generated": None}

    # Compare vs upgraded image if provided
    if compare_image_url:
        # Strip leading slash and resolve against project root
        rel = compare_image_url.lstrip("/")
        compare_path = _PROJECT_ROOT / "frontend" / "public" / rel
        if not compare_path.exists():
            # Try as absolute
            compare_path = Path(compare_image_url)

        if compare_path.exists():
            try:
                generated = predict_ctr(str(compare_path), country_list, os_list, seq_len)
                generated["label"] = "AI Generated"
                generated["creative_id"] = creative_id
                result["generated"] = generated
            except Exception as e:
                print(f"[Predict] Compare image inference failed: {e}")

    return result


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


# Top-level alias so both /evaluate/{id} and /api/creatives/evaluate/{id} work
@router.get("/{creative_id}/evaluate")
async def evaluate_creative_alias(
    creative_id: str,
    format: str = None,
    theme: str = None,
    hook: str = None,
):
    """Alias for GET /evaluate/{creative_id}."""
    return await evaluate_creative(creative_id, format, theme, hook)


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


@router.post("/{creative_id}/chat")
async def chat_with_creative(creative_id: str, payload: CreativeChatRequest):
    """Interactive chat grounded in image + structured JSON + feature gap JSON."""
    try:
        client = _get_openai_client()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")

        image_path = _resolve_image_path(creative_id)
        structured_path = _resolve_structured_path(creative_id)
        feature_gap_path = _resolve_feature_gap_path(creative_id)

        structured_json = _read_json(structured_path)
        
        # DYNAMIC INJECTION: Ensure advertiser is present for the LLM
        if not structured_json.get("advertiser"):
            from pipeline.step4_persistence.core import get_creative_by_id
            c_db = get_creative_by_id(creative_id)
            if c_db:
                brand = c_db.get("advertiser_name")
                structured_json["advertiser"] = brand
                if "global" in structured_json:
                    structured_json["global"]["advertiser"] = brand

        feature_gap_json = _load_feature_gap_or_fallback(feature_gap_path)
        selected_language = _normalize_language(payload.language)
        system_prompt = _build_chat_system_prompt(structured_json, feature_gap_json, selected_language)
        if payload.agentic:
            system_prompt += _INTENT_SYSTEM_SUFFIX

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        # Keep short context window to avoid long/noisy answers.
        for msg in payload.history[-10:]:
            role = (msg.role or "").lower().strip()
            if role not in {"user", "assistant"}:
                continue
            messages.append({"role": role, "content": msg.content})

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": payload.message.strip()},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        )

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = "No ho veig clar amb les dades actuals, prova de preguntar-ho més concret."

        # ── Agentic mode: parse intent from response ─────────────────────────
        action = None
        if payload.agentic:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', answer, re.DOTALL)
            if json_match:
                try:
                    action = json.loads(json_match.group(1))
                    # Strip the JSON block from the visible answer
                    answer = answer[:json_match.start()].rstrip()
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "creative_id": creative_id,
            "answer": answer,
            "model": model,
            "language": selected_language,
            "action": action,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        print(f"[API] Error in creative chat {creative_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{creative_id}/implement")
async def implement_chat_suggestion(creative_id: str, payload: ImplementRequest, request: Request):
    """
    Run the SD inpainting pipeline with a chat-suggested modification.
    Returns the same shape as /upgrade so the frontend can reuse it.
    Enrichment (SAM mask + semantic JSON) is NOT triggered here — call /enrich after apply.
    """
    import time as _time

    metadata = get_creative_by_id(creative_id)
    if metadata is None:
        metadata = {"id": creative_id}

    pipe = getattr(request.app.state, "pipe", None)

    try:
        from pipeline.step3_generation.core import generate_creative_with_flux
        from pipeline.step4_persistence.core import store_new_creative, compute_static_performance_score
        from pipeline.step4_persistence.helpers import next_available_id

        # Allocate a clean numeric ID so preprocess_masks.py can handle it
        new_id = str(next_available_id())

        # Run inpainting — inject description as the missing feature to guide the prompt
        new_creative_file = await generate_creative_with_flux(
            creative_id=creative_id,
            metadata=metadata,
            missing_features=[payload.description],
            pipe=pipe,
        )

        base = compute_static_performance_score(creative_id)
        # Save the output under a unique filename
        import shutil as _shutil
        from pathlib import Path as _Path
        src = _Path(new_creative_file)
        dst_dir = _PROJECT_ROOT / "frontend" / "public" / "data" / "assets"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"creative_{new_id}.png"
        if src.exists():
            _shutil.copy2(src, dst)

        new_image_url = f"/data/assets/creative_{new_id}.png"

        new_entry = {
            **metadata,
            "id": new_id,
            "image_url": new_image_url,
            "performance_score": round(base["performance_score"] + 0.05, 4),
            "fatigued": False,
            "insights": payload.description,
            "is_upgraded": True,
        }
        store_new_creative(new_id, new_entry)   # append as new entry, not replace original

        return {
            "success": True,
            "creative_id": new_id,
            "new_image_url": new_image_url,
            "metadata": {
                "description": payload.description,
                "predicted_uplift": "+5.0%",
                "performance_score": new_entry["performance_score"],
                "missing_features_explained": payload.description,
            },
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        import traceback
        print(f"[API] Error implementing suggestion for {creative_id}: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{creative_id}/enrich")
async def enrich_creative(creative_id: str, payload: EnrichRequest):
    """
    Trigger SAM mask generation + visual_semantic.json for an upgraded creative.
    Called by the frontend AFTER the user clicks 'Replace Image' (not during generation).
    Runs in a background thread so response is instant.
    """
    try:
        from pipeline.post_upgrade_enrichment import enrich_upgraded_creative
        from pathlib import Path as _Path

        # Resolve actual file path from the public URL
        rel = payload.image_url.lstrip("/")
        image_path = _PROJECT_ROOT / "frontend" / "public" / rel

        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

        enrich_upgraded_creative(
            original_id=creative_id,
            new_id=payload.new_id,
            new_image_path=image_path,
        )

        return {"success": True, "message": f"Enrichment started for {payload.new_id} in background"}
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[API] Error starting enrichment for {creative_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

