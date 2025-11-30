# src/data_loader.py

import pandas as pd
from typing import Optional


PHOTO_COLS = [
    "photo_neg_companies",
    "photo_neg_industries",
    "photo_neg_policies",
    "photo_neg_technology",
]

TEXT_COLS = [
    "text_neg_companies",
    "text_neg_industries",
    "text_neg_policies",
    "text_neg_technology",
]

IGNORED_COLS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def load_raw_data(
    csv_path: str,
    parse_dates: bool = True,
    drop_ignored: bool = True,
) -> pd.DataFrame:
    """
    Load the raw dataset from CSV.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    parse_dates : bool
        Whether to parse the 'date' column as datetime.
    drop_ignored : bool
        Whether to drop Fama-French style columns:
        Mkt-RF, SMB, HML, RMW, CMA, RF

    Returns
    -------
    df : pd.DataFrame
        Cleaned raw dataframe sorted by date.
    """
    if parse_dates:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        df = pd.read_csv(csv_path)

    # Sort by date and drop duplicates
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Optionally drop the factor columns you said to ignore
    if drop_ignored:
        existing_ignored = [c for c in IGNORED_COLS if c in df.columns]
        df = df.drop(columns=existing_ignored)

    # Basic NA handling: we usually do NOT forward-fill prices/returns here.
    # For now, just leave NA handling for a later stage or explicit function.
    return df


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that numeric columns (close + sentiment) are floats.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    df : pd.DataFrame
        DataFrame with numeric conversions applied.
    """
    numeric_cols = (
        ["composite_close"]
        + PHOTO_COLS
        + TEXT_COLS
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_and_prepare_base(csv_path: str) -> pd.DataFrame:
    """
    Convenience function: load the CSV and enforce numeric cols.

    This is what you'll typically call first.

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    df : pd.DataFrame
    """
    df = load_raw_data(csv_path)
    df = ensure_numeric_columns(df)
    return df
