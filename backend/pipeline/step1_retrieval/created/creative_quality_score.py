from __future__ import annotations

import pandas as pd

from .config import CREATIVE_QUALITY_SCORE_SPEC
from .score_utils import add_score_block


def add_creative_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds CreativeQualityScore columns to the creative index.

    Output columns:
        creative_quality_score_contextual
        creative_quality_score_global
        creative_quality_score_final

    Note:
        clutter_score is inverted in the config, because higher clutter is worse.
    """
    return add_score_block(df, CREATIVE_QUALITY_SCORE_SPEC)