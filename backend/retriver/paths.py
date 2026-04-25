from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DATA_DIR = REPO_ROOT / "frontend" / "public" / "data"

CREATIVE_SUMMARY_PATH = FRONTEND_DATA_DIR / "creative_summary.csv"
CREATIVES_PATH = FRONTEND_DATA_DIR / "creatives.csv"
CAMPAIGNS_PATH = FRONTEND_DATA_DIR / "campaigns.csv"
ADVERTISERS_PATH = FRONTEND_DATA_DIR / "advertisers.csv"

CREATIVE_RETRIEVAL_INDEX_PATH = FRONTEND_DATA_DIR / "creative_retrieval_index.csv"

# Kept because your README already mentions this file.
PERFORMANCE_SCORES_PATH = FRONTEND_DATA_DIR / "performance_scores.csv"