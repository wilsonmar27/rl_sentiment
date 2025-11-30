# src/features.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict

from data_loader import PHOTO_COLS, TEXT_COLS


# Which columns we'll use as state features (v1)
STATE_FEATURE_COLS = [
    "market_ret",      # previous day's market return
    "market_vol",      # rolling volatility
    "photo_neg_mean",  # aggregated photo sentiment
    "text_neg_mean",   # aggregated text sentiment
]


def add_aggregated_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2D sentiment features to the dataframe:
    - photo_neg_mean: mean of 4 photo_neg_* columns
    - text_neg_mean : mean of 4 text_neg_* columns

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with original sentiment columns.

    Returns
    -------
    df : pd.DataFrame
        Same DataFrame with new sentiment columns added.
    """
    missing_photo = [c for c in PHOTO_COLS if c not in df.columns]
    missing_text = [c for c in TEXT_COLS if c not in df.columns]

    if missing_photo:
        raise ValueError(f"Missing photo sentiment columns: {missing_photo}")
    if missing_text:
        raise ValueError(f"Missing text sentiment columns: {missing_text}")

    df = df.copy()

    df["photo_neg_mean"] = df[PHOTO_COLS].mean(axis=1)
    df["text_neg_mean"] = df[TEXT_COLS].mean(axis=1)

    return df


def build_base_timeseries(csv_path: str) -> pd.DataFrame:
    """
    High-level helper:
    - Load raw data
    - Ensure numeric
    - Add aggregated sentiment features

    Parameters
    ----------
    csv_path : str

    Returns
    -------
    df : pd.DataFrame
        DataFrame with:
        - date
        - composite_close
        - photo_neg_* and text_neg_* original columns
        - photo_neg_mean
        - text_neg_mean
    """
    from data_loader import load_and_prepare_base

    df = load_and_prepare_base(csv_path)
    df = add_aggregated_sentiment_features(df)
    return df


def add_market_features(
    df: pd.DataFrame,
    price_col: str = "composite_close",
    ret_col: str = "market_ret",
    vol_col: str = "market_vol",
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Add market return and rolling volatility features.

    - market_ret: percentage return based on composite_close
    - market_vol: rolling std of market_ret over a given window

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with `price_col` and sentiment columns already present.
    price_col : str
        Column name for the overall market close (composite_close).
    ret_col : str
        Name of the new return column.
    vol_col : str
        Name of the new volatility column.
    vol_window : int
        Rolling window size (in days) for volatility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new columns `ret_col` and `vol_col`.
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    # Daily percentage return (e.g., 0.01 = +1%)
    df[ret_col] = df[price_col].pct_change()

    # Rolling volatility of returns
    df[vol_col] = df[ret_col].rolling(window=vol_window).std()

    return df


def build_state_and_target_returns(
    df: pd.DataFrame,
    vol_window: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    From a raw dataframe (with composite_close + sentiment), build:

    - market_ret (daily returns)
    - market_vol (rolling volatility)
    - state matrix (for RL/teacher)
    - next-day market return as target

    The alignment is:
        state[t]  -> uses features at day t
        target[t] -> uses market_ret at day t+1

    So length of arrays = N-1 after dropping initial NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least:
        - 'date'
        - 'composite_close'
        - sentiment columns (photo_neg_* and text_neg_*)
    vol_window : int
        Rolling window size for volatility computation.

    Returns
    -------
    states : np.ndarray, shape (T, state_dim)
        State matrix with columns:
        [market_ret_t, market_vol_t, photo_neg_mean_t, text_neg_mean_t].
    next_returns : np.ndarray, shape (T,)
        Next-day market returns aligned with states.
    dates : np.ndarray, shape (T,)
        Dates corresponding to each state row.
    df_trimmed : pd.DataFrame
        DataFrame trimmed to the rows actually used to build states/targets.
    """
    # 1) Ensure aggregated sentiment exists
    df = add_aggregated_sentiment_features(df)

    # 2) Add market_ret and market_vol
    df = add_market_features(df, vol_window=vol_window)

    # 3) Drop rows with missing values in any required feature
    required_cols = ["date"] + STATE_FEATURE_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for state: {missing_cols}")

    df = df.dropna(subset=STATE_FEATURE_COLS).reset_index(drop=True)

    # 4) Build state at t and next_return at t+1
    # We need at least 2 rows after dropping NaNs
    if len(df) < 2:
        raise ValueError("Not enough data points after NaN filtering to build states.")

    # States use rows 0 .. N-2
    # Next returns use market_ret from rows 1 .. N-1
    state_df = df.iloc[:-1].copy()
    next_ret_series = df["market_ret"].iloc[1:].copy()

    # Convert to numpy arrays
    states = state_df[STATE_FEATURE_COLS].to_numpy(dtype=float)
    next_returns = next_ret_series.to_numpy(dtype=float)
    dates = state_df["date"].to_numpy()

    # df_trimmed is the aligned subset used for states
    df_trimmed = state_df.copy()

    return states, next_returns, dates, df_trimmed


def split_train_val_test(
    states: np.ndarray,
    next_returns: np.ndarray,
    dates: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Chronologically split (states, next_returns, dates) into
    train / validation / test sets.

    Parameters
    ----------
    states : np.ndarray, shape (T, state_dim)
    next_returns : np.ndarray, shape (T,)
    dates : np.ndarray, shape (T,)
    train_frac : float
        Fraction of data to use for training.
    val_frac : float
        Fraction of data to use for validation (from the remainder after train).

    Returns
    -------
    splits : dict
        {
          "train": {"states": ..., "returns": ..., "dates": ...},
          "val":   {"states": ..., "returns": ..., "dates": ...},
          "test":  {"states": ..., "returns": ..., "dates": ...},
        }
    """
    assert states.shape[0] == next_returns.shape[0] == dates.shape[0], \
        "states, next_returns, and dates must have the same length."

    T = states.shape[0]
    train_end = int(T * train_frac)
    val_end = train_end + int(T * val_frac)

    train_slice = slice(0, train_end)
    val_slice = slice(train_end, val_end)
    test_slice = slice(val_end, T)

    splits = {
        "train": {
            "states": states[train_slice],
            "returns": next_returns[train_slice],
            "dates": dates[train_slice],
        },
        "val": {
            "states": states[val_slice],
            "returns": next_returns[val_slice],
            "dates": dates[val_slice],
        },
        "test": {
            "states": states[test_slice],
            "returns": next_returns[test_slice],
            "dates": dates[test_slice],
        },
    }
    return splits
