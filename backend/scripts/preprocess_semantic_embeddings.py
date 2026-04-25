"""
Preprocessing Script — Build Semantic Embeddings

Reads the visual_semantic JSON files produced by the mask preprocessing step
(elements_data.json → converted to semantic format) and encodes them into
vector embeddings using sentence-transformers.

Outputs:
  frontend/public/data/semantic_embeddings/semantic_embeddings.pkl
  frontend/public/data/semantic_embeddings/semantic_embedding_index.csv

Run from repo root:
  python backend/scripts/preprocess_semantic_embeddings.py
"""
import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
backend_dir = script_dir.parent
repo_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

from pipeline.step1_retrieval.created.build_semantic_embeddings import build_semantic_embeddings

if __name__ == "__main__":
    build_semantic_embeddings()
