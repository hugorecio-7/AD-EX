import os
import json

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
DATA_PATH = os.path.join(_PROJECT_ROOT, "frontend", "src", "mocks", "data.json")

class Creative:
    """
    Lightweight wrapper around a creative metadata dict.
    Exposes `.explain()` which summarises what makes this creative a top performer.
    """
    def __init__(self, data: dict):
        self._data = data

    def explain(self) -> str:
        d = self._data
        parts = []
        if d.get("theme"):
            parts.append(f"theme={d['theme']}")
        if d.get("format"):
            parts.append(f"format={d['format']}")
        if d.get("hook_type"):
            parts.append(f"hook={d['hook_type']}")
        score = d.get("performance_score", 0)
        ctr = d.get("ctr", 0)
        parts.append(f"score={score:.3f}")
        parts.append(f"ctr={ctr:.2f}%")
        return ", ".join(parts)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __repr__(self):
        return f"<Creative id={self._data.get('id')} score={self._data.get('performance_score')}>"

def load_data() -> list[dict]:
    if not os.path.exists(DATA_PATH):
        print(f"[CreativeRetrieval] WARNING: data.json not found at {DATA_PATH}")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
