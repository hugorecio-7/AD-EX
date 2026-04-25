from pipeline.step1_retrieval.helpers import load_data, Creative

def get_best_creatives(
    creative_id: str,
    format_type: str,
    metadata: dict,
    top_n: int = 5,
) -> list[Creative]:
    """
    Return the top_n creatives from the same cluster that are NOT fatigued
    and have the highest performance_score.

    Falls back to best across all formats if the cluster has fewer than top_n entries.
    """
    data = load_data()
    if not data:
        return []

    target_cluster = metadata.get("cluster_id") or f"{format_type}-{metadata.get('theme', '')}-{metadata.get('hook_type', '')}"

    # Primary: same cluster, non-fatigued, sorted by score desc
    same_cluster = [
        d for d in data
        if d.get("cluster_id") == target_cluster
        and str(d.get("id")) != str(creative_id)
        and not d.get("fatigued", False)
    ]
    same_cluster.sort(key=lambda d: d.get("performance_score", 0), reverse=True)

    top = same_cluster[:top_n]

    # If we don't have enough from the cluster, pad with global top performers
    if len(top) < top_n:
        global_top = [
            d for d in data
            if str(d.get("id")) != str(creative_id)
            and not d.get("fatigued", False)
            and d not in same_cluster
        ]
        global_top.sort(key=lambda d: d.get("performance_score", 0), reverse=True)
        top += global_top[: top_n - len(top)]

    print(f"[CreativeRetrieval] Found {len(top)} top cases for cluster '{target_cluster}'")
    return [Creative(d) for d in top]
