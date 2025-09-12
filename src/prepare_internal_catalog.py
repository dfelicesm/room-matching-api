import pandas as pd

from src.config import PRECOMPUTED_PATH
from src.load_data import load_internal_raw
from src.preprocessing import (
    extract_bed_type,
    extract_bedroom_count,
    extract_beds,
    extract_room_type,
    normalize,
    tokenize,
)


def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized text, tokens, and room features to the raw internal
    catalog.

    Expects columns:
      - hotel_id
      - room_name
    """
    df = raw_df.copy()

    df["normalized"] = df["room_name"].astype(str).apply(normalize)

    df["tokens"] = df["room_name"].astype(str).apply(tokenize)

    # extract features
    df["room_type"] = df["normalized"].apply(extract_room_type)
    df["bed_type"] = df["tokens"].apply(extract_bed_type)
    df["num_beds"] = df["tokens"].apply(extract_beds)
    df["num_bedrooms"] = df["tokens"].apply(extract_bedroom_count)

    # make bed counts nullable integers
    df["num_beds"] = df["num_beds"].astype("Int64")
    df["num_bedrooms"] = df["num_bedrooms"].astype("Int64")

    return df


def save_precomputed(df: pd.DataFrame, path: str = PRECOMPUTED_PATH) -> None:
    """Persist the precomputed internal catalog with features."""
    df.to_csv(path, index=False)


def build_precomputed_catalog(path: str = PRECOMPUTED_PATH) -> None:
    """Load raw data, compute features, and save precomputed catalog."""
    raw = load_internal_raw()
    pre = compute_features(raw)
    save_precomputed(pre, path)


if __name__ == "__main__":
    build_precomputed_catalog()
