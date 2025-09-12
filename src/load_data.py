from typing import Optional

import pandas as pd

from src.config import PRECOMPUTED_PATH, RAW_PATH


def load_internal_raw(path: Optional[str] = RAW_PATH) -> pd.DataFrame:
    """
    Load the raw internal hotel catalog (no features yet).

    Args:
        path: Path to the raw CSV file. Defaults to RAW_PATH from config.py.

    Returns:
        pd.DataFrame with raw room data.
    """
    return pd.read_csv(path)


def load_internal_precomputed(path: Optional[str] = PRECOMPUTED_PATH) -> pd.DataFrame:
    """
    Load the precomputed internal catalog with features (ready for matching).

    Args:
        path: Path to the precomputed CSV file. Defaults to PRECOMPUTED_PATH from config.py.

    Returns:
        pd.DataFrame with preprocessed room data.
    """
    return pd.read_csv(path)
