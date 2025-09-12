from typing import Dict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.preprocessing import (
    extract_bed_type,
    extract_bedroom_count,
    extract_beds,
    extract_room_type,
    normalize,
    tokenize,
)


def eq_or_unknown(series: pd.Series, value) -> pd.Series:
    """
    Return a boolean mask where the row matches `value`,
    OR the candidate-side value is missing (NA), OR the `value` itself is NA.
    This way, unknowns don't block candidates.
    """
    if pd.isna(value):
        return pd.Series(True, index=series.index)
    return (series == value) | series.isna()


def score_against_column(target: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a RapidFuzz token_set_ratio similarity score between `target`
    and every row in df["normalized"].
    Adds a `similarity` column and returns df sorted descending by score.
    """
    scores = df["normalized"].apply(lambda x: fuzz.token_set_ratio(target, x))
    out = df.copy()
    out["similarity"] = scores
    return out.sort_values("similarity", ascending=False)


def _sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame safe to serialize to JSON:
      - Convert Pandas NA/NaN to Python None
      - Convert NumPy scalars to native Python types
      - Ensure 'similarity' is a plain int when present
    """
    out = df.copy().astype(object)

    # similarity as int (when present and not missing)
    if "similarity" in out.columns:
        out["similarity"] = out["similarity"].apply(
            lambda x: int(x) if (x is not None and pd.notna(x)) else None
        )

    for col in out.columns:
        # NaN / <NA> -> None
        out[col] = out[col].where(pd.notnull(out[col]), None)

        # NumPy scalars -> Python scalars
        out[col] = out[col].apply(
            lambda x: x.item() if isinstance(x, np.generic) else x
        )

    return out


def _extract_supplier_features(room_name: str) -> Dict:
    """Build the same features for a supplier room name using shared preprocessing."""
    norm = normalize(room_name)
    toks = tokenize(room_name)
    return {
        "normalized": norm,
        "tokens": toks,
        "room_type": extract_room_type(norm),
        "bed_type": extract_bed_type(toks),
        "num_beds": extract_beds(toks),
        "num_bedrooms": extract_bedroom_count(toks),
    }


def _prefilter_candidates(
    internal_df: pd.DataFrame,
    hotel_id: str,
    supplier_feats: Dict,
) -> pd.DataFrame:
    """
    Filter internal_df down to the same hotel and compatible features.
    Uses the internal column 'lp_id' for hotel code filtering.
    """
    if "lp_id" not in internal_df.columns:
        raise KeyError("internal_df must contain 'lp_id' column for hotel filtering")

    pool = internal_df.loc[internal_df["lp_id"] == hotel_id].copy()
    if pool.empty:
        return pool

    mask = pd.Series(True, index=pool.index)
    for col in ("room_type", "bed_type", "num_beds", "num_bedrooms"):
        mask &= eq_or_unknown(pool[col], supplier_feats.get(col, pd.NA))
    return pool.loc[mask]


def match_supplier_room(
    internal_df: pd.DataFrame,
    hotel_id: str,
    supplier_room_name: str,
) -> pd.DataFrame:
    """
    Match a supplier room name against internal catalog entries for the same hotel.

    Steps:
      - Extract features for the supplier room name
      - Prefilter internal candidates for the same hotel (by 'lp_id') and compatible features
      - Score similarity using RapidFuzz token_set_ratio on 'normalized'
      - Return ALL matches sorted by similarity (descending)
    """
    feats = _extract_supplier_features(supplier_room_name)
    pool = _prefilter_candidates(internal_df, hotel_id, feats)

    if pool.empty:
        cols = [
            "lp_id",
            "room_name",
            "normalized",
            "room_type",
            "bed_type",
            "num_beds",
            "num_bedrooms",
        ]
        empty = pd.DataFrame(columns=cols + ["similarity"])
        return _sanitize_for_json(empty)

    scored = score_against_column(feats["normalized"], pool)
    return _sanitize_for_json(scored)
