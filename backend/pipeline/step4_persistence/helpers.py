import json
import os

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_DIR = os.path.dirname(_PIPELINE_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

DATA_PATH = os.path.join(_PROJECT_ROOT, "frontend", "src", "mocks", "data.json")

def load_data():
    """Load current creatives from the JSON database."""
    if not os.path.exists(DATA_PATH):
        print(f"[WARNING] Data path {DATA_PATH} not found.")
        return []
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return []

def save_data(data):
    """Save creatives back to the JSON database."""
    try:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save data: {e}")
        return False
