"""
Step 2 — LLM Feature Gap Analysis

Given:
  - The query creative's visual_semantic.json  (what it HAS)
  - Top-performing creatives' visual_semantic.json files (what they HAVE)

An LLM (GPT-4o) identifies which VISUAL features are present in the top
performers but missing from the original creative, expressed as
Stable Diffusion-compatible image descriptions.

Output example:
  [
    "warm golden bokeh background with soft light rays",
    "smiling person holding smartphone in foreground",
    "vibrant coral-to-purple gradient overlay",
  ]

These are injected directly into the SD inpainting prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# ── Path resolution ────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent            # step2_feature_analysis
_BACKEND_DIR = _THIS_DIR.parent.parent                 # backend
_PROJECT_ROOT = _BACKEND_DIR.parent                    # repo root

OUTPUT_FEATURES_DIR = _PROJECT_ROOT / "output" / "features"
FRONTEND_SEMANTIC_DIR = _PROJECT_ROOT / "frontend" / "public" / "data" / "visual_semantic"

# ── OpenAI client (lazy) ──────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in backend/.env")
        _client = OpenAI(api_key=api_key)
    return _client


# ── Semantic JSON loader ───────────────────────────────────────────────────────

def _load_semantic(creative_id: str) -> dict | None:
    """Load visual_semantic.json for a creative. Returns None if not found or mock."""
    candidates = [
        OUTPUT_FEATURES_DIR / f"creative_{creative_id}" / "visual_semantic.json",
        FRONTEND_SEMANTIC_DIR / f"creative_{creative_id}.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("mock"):
                print(f"[LLMGap]   ⚠ {path.name} — MOCK data, skipping (no real visual descriptions)")
                return None
            print(f"[LLMGap]   ✓ Loaded real semantic data from {path}")
            return data
    print(f"[LLMGap]   ✗ No visual_semantic.json found for creative {creative_id}")
    return None


def _summarize_creative(data: dict) -> str:
    """Compact but information-rich summary of a creative's semantic data for the LLM."""
    global_block = data.get("global", {})
    elements = data.get("elements", [])
    embedding = data.get("embedding_texts", {})

    lines = [
        f"Creative ID: {data.get('creative_id', 'unknown')}",
        f"Visual Style: {global_block.get('visual_style', 'N/A')}",
        f"Emotional Tone: {global_block.get('emotional_tone', 'N/A')}",
        f"Dominant Colors: {', '.join(global_block.get('dominant_colors', []))}",
        f"Global Description: {global_block.get('description', 'N/A')}",
        "",
        "Visual Elements:",
    ]

    SKIP_ROLES = {"background", "unknown"}
    for e in elements:
        role = e.get("role", "unknown")
        if role in SKIP_ROLES:
            continue
        desc = e.get("description", "")
        label = e.get("label", "")
        text = e.get("text_content", "")
        line = f"  [{role}] {label}"
        if desc:
            line += f" — {desc}"
        if text:
            line += f" (text: '{text}')"
        lines.append(line)

    if embedding.get("layout_text"):
        lines += ["", f"Layout: {embedding['layout_text']}"]

    return "\n".join(lines)


# ── Core LLM call ─────────────────────────────────────────────────────────────

def analyze_feature_gap_with_llm(
    query_creative_id: str,
    top_creative_ids: list[str],
    max_features: int = 5,
) -> dict[str, Any]:
    """
    Call GPT-4o to identify visual features present in top performers
    that are missing from the query creative.

    Returns:
        {
            "missing_visual_features": ["description1", "description2", ...],
            "reasoning": "brief explanation of why these features matter",
            "query_id": str,
            "top_ids_used": [str, ...],
        }
    """
    # Load query semantic data
    print(f"[LLMGap] ─── Feature Gap Analysis ───")
    print(f"[LLMGap] Query creative : {query_creative_id}")
    print(f"[LLMGap] Top candidates : {top_creative_ids}")
    print(f"[LLMGap] Checking query semantic data...")
    query_data = _load_semantic(query_creative_id)
    if query_data is None:
        print(f"[LLMGap] ✗ SKIPPING LLM — query creative has no real semantic data.")
        print(f"[LLMGap]   → Run GPT-4o Vision enrichment first: enrich_creative_with_vision('{query_creative_id}', ...)")
        print(f"[LLMGap]   → Using generic visual fallback instead.")
        return {
            "missing_visual_features": _visual_fallback(),
            "reasoning": "No semantic data available for LLM analysis.",
            "query_id": query_creative_id,
            "top_ids_used": [],
        }

    # Load top performers' semantic data (skip mocks and missing)
    print(f"[LLMGap] Checking top performers' semantic data...")
    top_data = []
    top_ids_used = []
    for cid in top_creative_ids:
        data = _load_semantic(cid)
        if data:
            top_data.append(data)
            top_ids_used.append(cid)
        if len(top_data) >= 3:
            break

    if not top_data:
        print(f"[LLMGap] ✗ SKIPPING LLM — no real semantic data for any top performer.")
        print(f"[LLMGap]   → Checked: {top_creative_ids}")
        print(f"[LLMGap]   → Run preprocess_semantic_enrichment.py to generate visual_semantic.json files.")
        print(f"[LLMGap]   → Using generic visual fallback instead.")
        return {
            "missing_visual_features": _visual_fallback(),
            "reasoning": "No enriched top-performer data available.",
            "query_id": query_creative_id,
            "top_ids_used": [],
        }

    print(f"[LLMGap] ✓ Using {len(top_data)} real semantic files for LLM analysis.")

    # Build the LLM prompt
    query_summary = _summarize_creative(query_data)
    top_summaries = "\n\n---\n\n".join(
        f"TOP PERFORMER #{i+1}:\n{_summarize_creative(d)}"
        for i, d in enumerate(top_data)
    )

    system_prompt = (
        "You are an expert mobile advertising creative analyst specializing in performance optimization. "
        "You analyze visual compositions to identify what makes high-performing ads successful. "
        "You ONLY output valid JSON."
    )

    user_prompt = f"""You are comparing a low-performing mobile ad against {len(top_data)} top-performing ads 
in the same campaign segment. Your job is to identify what VISUAL features the top performers have 
that the original creative is MISSING.

ORIGINAL CREATIVE (to be improved):
{query_summary}

TOP PERFORMING CREATIVES (reference):
{top_summaries}

Identify up to {max_features} visual features that appear in the top performers but NOT in the original.

STRICT RULES:
1. Only describe VISUAL properties that an image generation model (Stable Diffusion) can reproduce
2. Be SPECIFIC and LITERAL — describe what you actually SEE in the elements, not abstract concepts
3. Focus on: backgrounds, lighting, subjects, colors, textures, composition, mood, visual effects
4. DO NOT include: text content, button labels, marketing copy, metadata labels, format names
5. DO NOT include abstract marketing terms like "better hook", "stronger CTA", "more engaging"

GOOD examples:
  "warm golden-hour sunlight streaming from upper left"
  "smiling young woman looking directly at camera, holding product"
  "deep navy-to-purple gradient background with subtle particle effects"
  "product centered with dramatic drop shadow on clean white surface"

BAD examples (do NOT output these):
  "rewarded_video ad format"
  "discount themed"
  "stronger CTA"
  "better headline"

Return ONLY this JSON:
{{
  "missing_visual_features": ["feature 1", "feature 2", ...],
  "reasoning": "2-3 sentences explaining what the top performers do visually that makes them more impactful"
}}"""

    print(f"[LLMGap] Calling GPT-4o to analyze gap: {query_creative_id} vs {top_ids_used}...")
    print(f"[LLMGap] ── SYSTEM PROMPT ──────────────────────────────────")
    print(system_prompt)
    print(f"[LLMGap] ── USER PROMPT ────────────────────────────────────")
    print(user_prompt)
    print(f"[LLMGap] ─────────────────────────────────────────────────")
    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        result = json.loads(response.choices[0].message.content.strip())
        features = result.get("missing_visual_features", [])
        reasoning = result.get("reasoning", "")
        
        # Filter out any metadata labels that slipped through
        SKIP_KEYWORDS = ("format", "themed", "hook style", "ad format", "objective", "vertical", "cta", "headline")
        features = [
            f for f in features
            if not any(k in f.lower() for k in SKIP_KEYWORDS)
        ]

        print(f"[LLMGap] ✓ Got {len(features)} visual features:")
        for f in features:
            print(f"   · {f}")
        print(f"[LLMGap] Reasoning: {reasoning}")

        return {
            "missing_visual_features": features[:max_features],
            "reasoning": reasoning,
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }

    except Exception as e:
        print(f"[LLMGap] GPT-4o call failed: {e}. Using visual fallback.")
        return {
            "missing_visual_features": _visual_fallback(),
            "reasoning": f"LLM analysis failed: {e}",
            "query_id": query_creative_id,
            "top_ids_used": top_ids_used,
        }


def _visual_fallback() -> list[str]:
    """Generic visual descriptions when no semantic data is available."""
    return [
        "dynamic gradient background with vibrant colors",
        "high quality product photography with professional lighting",
        "cinematic depth of field effect",
    ]
