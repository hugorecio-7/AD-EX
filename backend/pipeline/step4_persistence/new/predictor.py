"""
RNN CTR Predictor — Singleton wrapper around ImageAutoregressiveRNN.

Usage:
    from pipeline.step4_persistence.new.predictor import predict_ctr

    result = predict_ctr(
        image_path="path/to/creative.png",
        countries=["US", "ES"],
        os_types=["iOS", "Android"],
        seq_len=30,
    )
"""
from __future__ import annotations

import os
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

_THIS_DIR = Path(__file__).resolve().parent

COUNTRY2IDX = {
    'CA': 0, 'US': 1, 'ES': 2, 'JP': 3, 'UK': 4,
    'MX': 5, 'IT': 6, 'BR': 7, 'DE': 8, 'FR': 9,
}
OS2IDX = {'iOS': 0, 'Android': 1}

DEFAULT_COUNTRIES = ['US', 'ES']
DEFAULT_OS = ['iOS', 'Android']

_model = None
_device = None

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _get_model():
    global _model, _device
    if _model is None:
        from pipeline.step4_persistence.new.image_rnn_model import ImageAutoregressiveRNN

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _THIS_DIR / "image_rnn_model.pth"

        _model = ImageAutoregressiveRNN(
            latent_dim=128,
            hidden_dim=64,
            num_countries=len(COUNTRY2IDX),
            num_os=len(OS2IDX),
        ).to(_device)

        if model_path.exists():
            _model.load_state_dict(
                torch.load(str(model_path), map_location=_device, weights_only=True)
            )
        else:
            print(f"[RNN] ⚠ Model weights not found at {model_path} — using untrained weights")

        _model.eval()
        print(f"[RNN] Model loaded on {_device}")

    return _model, _device


def _run_inference(image_tensor: torch.Tensor, country: str, os_type: str, seq_len: int) -> list[float]:
    model, device = _get_model()
    image_tensor = image_tensor.to(device)

    country_idx = torch.tensor([COUNTRY2IDX.get(country, 1)], dtype=torch.long).to(device)
    os_idx      = torch.tensor([OS2IDX.get(os_type, 0)],   dtype=torch.long).to(device)

    with torch.no_grad():
        preds = model(image_tensor, country_idx, os_idx, targets=None, seq_len=seq_len)

    # Model outputs are scaled ×100; bring back to [0, 1] range
    values = preds.squeeze().cpu().numpy() / 100.0
    # Ensure it's iterable (single step edge case)
    if values.ndim == 0:
        values = [float(values)]
    return [max(0.0, float(v)) for v in values]


def _compute_fatigue_day(series: list[float]) -> int | None:
    """First day CTR drops permanently below 50% of the peak."""
    if not series:
        return None
    peak = max(series)
    threshold = peak * 0.5
    for i, v in enumerate(series):
        if v < threshold and i > 0:
            return i + 1  # 1-indexed day
    return None


def predict_ctr(
    image_path: str,
    countries: list[str] | None = None,
    os_types: list[str] | None = None,
    seq_len: int = 30,
) -> dict:
    """
    Run the RNN model on a single image for all country × OS combinations.

    Returns:
    {
        "image_path": str,
        "seq_len": int,
        "predictions": [
            {
                "country": "US",
                "os": "iOS",
                "ctr_timeseries": [0.021, ...],   # seq_len floats
                "peak_ctr": 0.025,
                "avg_ctr": 0.019,
                "fatigue_day": 14 | null,
            },
            ...
        ],
        "summary": {
            "best_segment": "US iOS",
            "avg_ctr_all": 0.019,
        }
    }
    """
    countries = [c for c in (countries or DEFAULT_COUNTRIES) if c in COUNTRY2IDX]
    os_types  = [o for o in (os_types  or DEFAULT_OS)       if o in OS2IDX]

    if not countries:
        countries = DEFAULT_COUNTRIES
    if not os_types:
        os_types = DEFAULT_OS

    # Load image once
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise FileNotFoundError(f"Cannot open image {image_path}: {e}")

    image_tensor = _TRANSFORM(img).unsqueeze(0)

    predictions = []
    for country in countries:
        for os_type in os_types:
            series = _run_inference(image_tensor, country, os_type, seq_len)
            peak   = max(series) if series else 0.0
            avg    = sum(series) / len(series) if series else 0.0
            predictions.append({
                "country": country,
                "os": os_type,
                "ctr_timeseries": [round(v, 6) for v in series],
                "peak_ctr": round(peak, 6),
                "avg_ctr": round(avg, 6),
                "fatigue_day": _compute_fatigue_day(series),
            })

    best = max(predictions, key=lambda p: p["avg_ctr"]) if predictions else None
    all_avg = sum(p["avg_ctr"] for p in predictions) / len(predictions) if predictions else 0.0

    return {
        "image_path": str(image_path),
        "seq_len": seq_len,
        "predictions": predictions,
        "summary": {
            "best_segment": f"{best['country']} {best['os']}" if best else "N/A",
            "avg_ctr_all": round(all_avg, 6),
        },
    }
