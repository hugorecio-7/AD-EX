from __future__ import annotations

import pandas as pd

from .config import PERFORMANCE_SCORE_SPEC
from .score_utils import add_score_block


def add_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds PerformanceScore columns to the creative index.

    Output columns:
        performance_score_contextual
        performance_score_global
        performance_score_final
    """
    return add_score_block(df, PERFORMANCE_SCORE_SPEC)