from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

FRONTEND_DATA_DIR = REPO_ROOT / "frontend" / "public" / "data"

CREATIVE_SUMMARY_PATH = FRONTEND_DATA_DIR / "creative_summary.csv"
CREATIVES_PATH = FRONTEND_DATA_DIR / "creatives.csv"
CAMPAIGNS_PATH = FRONTEND_DATA_DIR / "campaigns.csv"
ADVERTISERS_PATH = FRONTEND_DATA_DIR / "advertisers.csv"

CREATIVE_RETRIEVAL_INDEX_PATH = FRONTEND_DATA_DIR / "creative_retrieval_index.csv"

# Kept because your README already mentions this file.
PERFORMANCE_SCORES_PATH = FRONTEND_DATA_DIR / "performance_scores.csv"

# Final semantic JSONs used by the retriever.
VISUAL_SEMANTIC_DIR = FRONTEND_DATA_DIR / "visual_semantic"

# Final semantic embeddings used by the retriever.
SEMANTIC_EMBEDDINGS_DIR = FRONTEND_DATA_DIR / "semantic_embeddings"
SEMANTIC_EMBEDDINGS_PATH = SEMANTIC_EMBEDDINGS_DIR / "semantic_embeddings.pkl"
SEMANTIC_EMBEDDING_INDEX_PATH = SEMANTIC_EMBEDDINGS_DIR / "semantic_embedding_index.csv"
