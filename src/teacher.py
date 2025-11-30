# src/teacher.py

import numpy as np
from typing import Tuple, Dict


def compute_teacher_weights(
    states: np.ndarray,
    lambda_risk: float = 10.0,
    ret_idx: int = 0,
    vol_idx: int = 1,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute the Markowitz-style teacher allocation for a 2-asset portfolio:
    - Risky asset: market ETF
    - Risk-free asset: cash/bonds (0% daily return)

    The teacher uses ONLY:
    - market_ret_t  (as a proxy for expected return μ_t)
    - market_vol_t  (as proxy for σ_t, the std of returns)

    and applies a 1D mean-variance utility:
        U_t(w) = μ_t * w - λ * σ_t^2 * w^2

    The maximizer is:
        w*_t = μ_t / (2 * λ * σ_t^2)
    clipped into [0, 1].

    Parameters
    ----------
    states : np.ndarray, shape (T, state_dim)
        State matrix as returned by build_state_and_target_returns in features.py.
        By construction, we expect:
            states[:, 0] = market_ret_t
            states[:, 1] = market_vol_t
        but you can override indices via ret_idx, vol_idx.
    lambda_risk : float
        Risk aversion parameter λ. Higher λ -> more risk-averse teacher
        (smaller allocations to the risky asset).
    ret_idx : int
        Index of the market return feature within the state vector.
    vol_idx : int
        Index of the market volatility feature within the state vector.
    eps : float
        Small number to avoid division by zero when σ_t^2 ~ 0.

    Returns
    -------
    weights : np.ndarray, shape (T,)
        Teacher's fraction allocated to the market ETF on each time step.
        The cash weight is (1 - weights[t]).
    """
    if states.ndim != 2:
        raise ValueError(f"'states' must be 2D, got shape {states.shape}")

    market_ret = states[:, ret_idx]         # μ_t proxy
    market_vol = states[:, vol_idx]         # σ_t proxy (std dev)

    sigma2 = market_vol ** 2                # variance
    weights = np.zeros_like(market_ret)

    # For days where variance is effectively zero, fall back to:
    #   if μ_t > 0 -> go all-in on market
    #   else       -> stay all in cash
    small_var_mask = sigma2 < eps
    normal_mask = ~small_var_mask

    # Normal days: apply Markowitz rule
    weights_normal = np.zeros_like(market_ret[normal_mask])
    if np.any(normal_mask):
        mu_normal = market_ret[normal_mask]
        sigma2_normal = sigma2[normal_mask]

        raw_w = mu_normal / (2.0 * lambda_risk * sigma2_normal)
        weights_normal = np.clip(raw_w, 0.0, 1.0)

    weights[normal_mask] = weights_normal

    # Edge cases: near-zero variance
    if np.any(small_var_mask):
        mu_small = market_ret[small_var_mask]
        # If expected return positive -> invest fully in market, else cash
        weights_small = np.where(mu_small > 0.0, 1.0, 0.0)
        weights[small_var_mask] = weights_small

    return weights


def simulate_teacher_equity(
    teacher_weights: np.ndarray,
    next_returns: np.ndarray,
    initial_capital: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the teacher's portfolio given allocations and next-day returns.

    At each time step t:
        - teacher chooses weight w_t in the market
        - cash weight is (1 - w_t) with 0% return
        - realized portfolio return r_p,t+1 = w_t * market_ret_{t+1}

    Parameters
    ----------
    teacher_weights : np.ndarray, shape (T,)
        Fraction of capital invested in the market at each step.
    next_returns : np.ndarray, shape (T,)
        Next-day market returns aligned with teacher_weights.
        This should be the same array produced by build_state_and_target_returns.
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    portfolio_returns : np.ndarray, shape (T,)
        Daily portfolio returns under the teacher policy.
    equity_curve : np.ndarray, shape (T+1,)
        Portfolio value over time. equity_curve[0] = initial_capital.
        equity_curve[t+1] = equity_curve[t] * (1 + portfolio_returns[t]).
    """
    if teacher_weights.shape != next_returns.shape:
        raise ValueError(
            f"Shape mismatch: weights {teacher_weights.shape}, returns {next_returns.shape}"
        )

    # Portfolio daily returns: risky weight * market return
    portfolio_returns = teacher_weights * next_returns

    equity_curve = np.empty(len(portfolio_returns) + 1, dtype=float)
    equity_curve[0] = initial_capital
    for t in range(len(portfolio_returns)):
        equity_curve[t + 1] = equity_curve[t] * (1.0 + portfolio_returns[t])

    return portfolio_returns, equity_curve


def compute_simple_metrics(
    portfolio_returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute simple performance metrics for a return series:
    - total_return
    - annualized_return
    - volatility
    - sharpe_ratio (assuming risk-free ~ 0 for now)

    This is mainly for quick sanity checks of the teacher vs baselines.

    Parameters
    ----------
    portfolio_returns : np.ndarray, shape (T,)
        Daily portfolio returns.
    trading_days_per_year : int
        Number of trading days per year for annualization.

    Returns
    -------
    metrics : dict
        Dictionary of basic performance statistics.
    """
    r = np.asarray(portfolio_returns, dtype=float)
    mean_daily = r.mean()
    std_daily = r.std(ddof=1) if len(r) > 1 else 0.0

    total_return = (1.0 + r).prod() - 1.0

    annualized_return = (1.0 + mean_daily) ** trading_days_per_year - 1.0 \
        if mean_daily > -1.0 else -1.0

    annualized_vol = std_daily * np.sqrt(trading_days_per_year)

    sharpe = mean_daily / std_daily * np.sqrt(trading_days_per_year) \
        if std_daily > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
    }
