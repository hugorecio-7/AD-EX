from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import (
    CREATIVE_SUMMARY_PATH,
    CREATIVES_PATH,
    CAMPAIGNS_PATH,
    ADVERTISERS_PATH,
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """
    Loads the raw CSV files needed for the retrieval scoring index.
    """
    tables = {
        "creative_summary": _read_csv(CREATIVE_SUMMARY_PATH),
        "creatives": _read_csv(CREATIVES_PATH),
        "campaigns": _read_csv(CAMPAIGNS_PATH),
    }

    if ADVERTISERS_PATH.exists():
        tables["advertisers"] = _read_csv(ADVERTISERS_PATH)

    return tables


def _merge_missing_columns(
    base: pd.DataFrame,
    other: pd.DataFrame,
    on: str,
    candidate_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Merge only columns that are missing from the base dataframe.

    This avoids duplicate columns like vertical_x / vertical_y and keeps the
    master table clean.
    """
    if on not in base.columns or on not in other.columns:
        return base

    columns_to_add = [
        col for col in candidate_columns
        if col in other.columns and col not in base.columns
    ]

    if not columns_to_add:
        return base

    right = other[[on] + columns_to_add].drop_duplicates(subset=[on])
    return base.merge(right, on=on, how="left")


def build_master_creative_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds a one-row-per-creative table.

    Base table:
        creative_summary.csv

    Enriched with missing columns from:
        creatives.csv
        campaigns.csv
        advertisers.csv, if available
    """
    creative_summary = tables["creative_summary"].copy()
    creatives = tables["creatives"].copy()
    campaigns = tables["campaigns"].copy()
    advertisers = tables.get("advertisers")

    df = creative_summary.copy()

    # Add creative-level columns if they are not already present.
    creative_candidate_columns = [
        "advertiser_name",
        "app_name",
        "vertical",
        "format",
        "width",
        "height",
        "language",
        "creative_launch_date",
        "theme",
        "hook_type",
        "cta_text",
        "headline",
        "subhead",
        "dominant_color",
        "emotional_tone",
        "duration_sec",
        "text_density",
        "copy_length_chars",
        "readability_score",
        "brand_visibility_score",
        "clutter_score",
        "novelty_score",
        "motion_score",
        "faces_count",
        "product_count",
        "has_price",
        "has_discount_badge",
        "has_gameplay",
        "has_ugc_style",
        "asset_file",
    ]

    df = _merge_missing_columns(
        base=df,
        other=creatives,
        on="creative_id",
        candidate_columns=creative_candidate_columns,
    )

    # Add campaign-level columns.
    campaign_candidate_columns = [
        "advertiser_id",
        "advertiser_name",
        "app_name",
        "vertical",
        "objective",
        "primary_theme",
        "target_age_segment",
        "target_os",
        "countries",
        "start_date",
        "end_date",
        "daily_budget_usd",
        "kpi_goal",
    ]

    df = _merge_missing_columns(
        base=df,
        other=campaigns,
        on="campaign_id",
        candidate_columns=campaign_candidate_columns,
    )

    # Add advertiser metadata if available.
    if advertisers is not None and "advertiser_id" in df.columns:
        advertiser_candidate_columns = [
            "advertiser_name",
            "vertical",
            "hq_region",
        ]

        df = _merge_missing_columns(
            base=df,
            other=advertisers,
            on="advertiser_id",
            candidate_columns=advertiser_candidate_columns,
        )

    return df