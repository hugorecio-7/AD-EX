"""Interactive Ad QA Bot for a specific creative.

Usage examples:
  python backend/scripts/ad_qa_bot.py --creative-id 500593
  python backend/scripts/ad_qa_bot.py --creative-id 500593 --feature-gap path/to/feature_gap.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


THIS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = THIS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

LANGUAGE_MAP = {
    "catalan": "Catalan",
    "castilian": "Castilian Spanish",
    "english": "English",
}


def normalize_language(language: str | None) -> str:
    value = (language or "catalan").strip().lower()
    return value if value in LANGUAGE_MAP else "catalan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chatbot for ad creative diagnosis.")
    parser.add_argument("--creative-id", default="500000", help="Creative ID (example: 500000)")
    parser.add_argument("--image", default=None, help="Optional custom image path (.png/.jpg)")
    parser.add_argument("--structured", default=None, help="Optional custom structured JSON path")
    parser.add_argument("--feature-gap", default=None, help="Optional custom feature gap JSON path")
    parser.add_argument("--model", default=None, help="Override model (defaults to OPENAI_MODEL or gpt-4o)")
    parser.add_argument("--language", default="catalan", choices=["catalan", "castilian", "english"], help="Response language")
    return parser.parse_args()


def resolve_image_path(creative_id: str, override: str | None) -> Path:
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(f"Image override not found: {path}")

    candidates = [
        PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"creative_{creative_id}.png",
        PROJECT_ROOT / "frontend" / "public" / "data" / "assets" / f"{creative_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No image found. Tried: " + ", ".join(str(p) for p in candidates)
    )


def resolve_structured_path(creative_id: str, override: str | None) -> Path:
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(f"Structured JSON override not found: {path}")

    candidates = [
        PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / f"creative_{creative_id}_structured.json",
        PROJECT_ROOT / "frontend" / "public" / "data" / "structured" / f"creative_{creative_id}_structured.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    available = sorted((PROJECT_ROOT / "output" / "features").glob("creative_*/creative_*_structured.json"))
    available_hint = ", ".join(p.parent.name.replace("creative_", "") for p in available[:8])
    if len(available) > 8:
        available_hint += ", ..."

    raise FileNotFoundError(
        "No structured JSON found for creative "
        f"{creative_id}. Tried: "
        + ", ".join(str(p) for p in candidates)
        + (f". Available structured creative IDs: {available_hint}" if available_hint else "")
    )


def resolve_feature_gap_path(creative_id: str, override: str | None) -> Path | None:
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise FileNotFoundError(f"Feature gap override not found: {path}")

    candidates = [
        PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / f"creative_{creative_id}_feature_gap.json",
        PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "feature_gap.json",
        PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "llm_feature_gap.json",
        PROJECT_ROOT / "output" / "features" / f"creative_{creative_id}" / "feature_gap_output.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def get_image_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def encode_image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def load_feature_gap(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "missing_visual_features": [
                "clear high-contrast CTA button in lower-right area",
                "strong visual hierarchy with one dominant headline",
                "human face near product to improve trust and engagement",
            ],
            "reasoning": "Mock fallback because no feature gap JSON file was found.",
        }

    try:
        data = load_json_file(path)
    except Exception as exc:
        print(f"[Warning] Could not load feature gap file ({path}): {exc}")
        return {
            "missing_visual_features": [
                "clear high-contrast CTA button in lower-right area",
                "strong visual hierarchy with one dominant headline",
            ],
            "reasoning": "Fallback used because feature gap JSON was invalid.",
        }

    if "missing_visual_features" not in data:
        data["missing_visual_features"] = []
    if "reasoning" not in data:
        data["reasoning"] = "No explicit reasoning provided in feature gap JSON."
    return data


def build_system_prompt(structured_json: dict[str, Any], feature_gap_json: dict[str, Any], language: str) -> str:
    language_name = LANGUAGE_MAP[normalize_language(language)]
    return (
        "You are an Ad Creative Consultant for performance marketing. "
        "Talk like a helpful teammate in a chat: informal, natural, and direct.\n\n"
        "Response style rules:\n"
        "1) Keep answers very short (normally 2-4 lines total).\n"
        "2) Output plain text only, no markdown, no bullet points, no numbering.\n"
        "3) Write as one short paragraph, or two very short paragraphs when needed.\n"
        "4) Be clear and concrete: no fluff, no long intros, no going off-topic.\n"
        "5) If user asks for improvements, mention up to 3 actions max, inline in the paragraph.\n"
        "6) If user asks location, include approximate position (top-left, center, bottom-right) and bbox if available.\n"
        "7) If data is missing, say it clearly instead of inventing.\n"
        "8) Do not output chain-of-thought.\n\n"
        f"Language rule: Always reply in {language_name}.\n\n"
        "STRUCTURED_JSON:\n"
        f"{json.dumps(structured_json, indent=2, ensure_ascii=False)}\n\n"
        "FEATURE_GAP_JSON:\n"
        f"{json.dumps(feature_gap_json, indent=2, ensure_ascii=False)}"
    )


def chat_loop(
    client: OpenAI,
    model: str,
    system_prompt: str,
    image_base64: str,
    image_mime_type: str,
) -> None:
    print("\nAd QA Bot ready. Ask questions about the creative.")
    print("Type 'exit' or 'quit' to stop.\n")

    conversation_history: list[dict[str, Any]] = []

    while True:
        try:
            user_question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Ad QA Bot.")
            break

        if not user_question:
            continue

        if user_question.lower() in {"exit", "quit", "q"}:
            print("Exiting Ad QA Bot.")
            break

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_mime_type};base64,{image_base64}",
                        },
                    },
                ],
            }
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                answer = "I could not generate a response. Please try rephrasing your question."
            print(f"\nBot: {answer}\n")
        except Exception as exc:
            print(f"\n[Error] OpenAI request failed: {exc}\n")
            continue

        conversation_history.append({"role": "user", "content": user_question})
        conversation_history.append({"role": "assistant", "content": answer})


def main() -> None:
    args = parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Add it to .env at project root.")

    model = (args.model or os.environ.get("OPENAI_MODEL") or "gpt-4o").strip()
    language = normalize_language(args.language)

    image_path = resolve_image_path(args.creative_id, args.image)
    structured_path = resolve_structured_path(args.creative_id, args.structured)
    feature_gap_path = resolve_feature_gap_path(args.creative_id, args.feature_gap)

    image_base64 = encode_image_to_base64(image_path)
    image_mime_type = get_image_mime_type(image_path)
    structured_json = load_json_file(structured_path)
    feature_gap_json = load_feature_gap(feature_gap_path)

    print("[Loaded] Image:", image_path)
    print("[Loaded] Structured JSON:", structured_path)
    print("[Loaded] Feature Gap JSON:", feature_gap_path if feature_gap_path else "<mock fallback>")
    print("[Model]", model)
    print("[Language]", language)

    system_prompt = build_system_prompt(structured_json, feature_gap_json, language)
    client = OpenAI(api_key=api_key)

    chat_loop(
        client=client,
        model=model,
        system_prompt=system_prompt,
        image_base64=image_base64,
        image_mime_type=image_mime_type,
    )


if __name__ == "__main__":
    main()
