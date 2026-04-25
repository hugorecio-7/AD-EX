from pipeline.step4_persistence.helpers import load_data, save_data

def store_new_creative(original_id, new_entry):
    """
    Store the new creative and update the database.
    If original_id is found, it replaces that entry (preserving its position in the list).
    Otherwise, it appends the new entry.
    """
    data = load_data()
    replaced = False
    for i, entry in enumerate(data):
        if str(entry.get('id')) == str(original_id):
            data[i] = new_entry
            replaced = True
            break
            
    if not replaced:
        data.append(new_entry)
        
    save_data(data)
    return new_entry.get('id')

def get_creative_by_id(creative_id):
    """Retrieve a specific creative's metadata."""
    data = load_data()
    for entry in data:
        if str(entry.get('id')) == str(creative_id):
            return entry
    return None

def compute_static_performance_score(creative_id: str) -> dict:
    """
    Return the stored performance_score for an existing creative.
    Falls back to a cluster-median estimate if the entry is not found.
    """
    data = load_data()
    for entry in data:
        if str(entry.get("id")) == str(creative_id):
            score = float(entry.get("performance_score", 0))
            return {
                "performance_score": round(score, 3),
                "is_fatigued": entry.get("fatigued", score < 0.3),
                "ctr": entry.get("ctr", 0),
                "logic_version": "v2-real-data",
            }

    if data:
        scores = [float(e.get("performance_score", 0)) for e in data]
        median = sorted(scores)[len(scores) // 2]
    else:
        median = 0.5

    return {
        "performance_score": round(median, 3),
        "is_fatigued": median < 0.3,
        "ctr": 0,
        "logic_version": "v2-cluster-median-fallback",
    }
