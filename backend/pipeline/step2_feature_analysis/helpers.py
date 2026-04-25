from collections import Counter

def parse_explanations_to_features(explanations: list[str]) -> list[str]:
    """Parse explanation strings and return the top 6 human-readable prompt fragments."""
    feature_counts: Counter = Counter()
    
    for explanation in explanations:
        for part in explanation.split(","):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key in ("theme", "format", "hook"):
                    canonical = f"{key}:{val}"
                    feature_counts[canonical] += 1

    top_features = [feat for feat, _ in feature_counts.most_common(6)]
    
    prompt_fragments = []
    for feat in top_features:
        key, val = feat.split(":", 1)
        if key == "theme":
            prompt_fragments.append(f"{val} themed")
        elif key == "format":
            prompt_fragments.append(f"{val} ad format")
        elif key == "hook":
            prompt_fragments.append(f"{val} hook style")
            
    return prompt_fragments

def format_explanation_paragraph(missing_features: list[str], creative_id: str) -> str:
    """Format the features into a dashboard-friendly text block."""
    if not missing_features:
        return "No significant feature gaps found. The creative already matches top-performer patterns."

    lines = [
        f"Analysis for creative {creative_id} identified {len(missing_features)} high-impact features from the top-performing cluster:",
        "",
    ]
    for feat in missing_features:
        lines.append(f"  • {feat.capitalize()}")

    lines += [
        "",
        "These features have been injected into the diffusion prompt to maximise predicted CTR uplift.",
    ]
    return "\n".join(lines)
