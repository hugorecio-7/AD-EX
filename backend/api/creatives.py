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
        _PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / f"creative_{creative_id}_structured.json",
        _PROJECT_ROOT / "frontend" / "public" / "data" / "structured" / f"creative_{creative_id}_structured.json",
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
        feature_gap_json = _load_feature_gap_or_fallback(feature_gap_path)
        selected_language = _normalize_language(payload.language)
        system_prompt = _build_chat_system_prompt(structured_json, feature_gap_json, selected_language)

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

        return {
            "success": True,
            "creative_id": creative_id,
            "answer": answer,
            "model": model,
            "language": selected_language,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        print(f"[API] Error in creative chat {creative_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
