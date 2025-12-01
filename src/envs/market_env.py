# src/envs/market_env.py

import numpy as np
from typing import Optional, Tuple, Dict, Any


class MarketEnv:
    """
    Simple 2-asset trading environment:

    - Asset 1: Market ETF (risky), with given daily returns.
    - Asset 2: Cash/Bonds (risk-free), assumed 0% daily return.

    State at time t: provided externally (e.g. [ret_t, vol_t, photo_mean_t, text_mean_t])
    Action at time t: scalar a_t in [0, 1] = fraction of portfolio in market.
      -> cash weight = 1 - a_t

    Reward (v1): daily portfolio return r_p,t+1 = a_t * market_ret_{t+1}
    """

    def __init__(
        self,
        states: np.ndarray,
        next_returns: np.ndarray,
        start_index: int = 0,
        end_index: Optional[int] = None,
        initial_capital: float = 1.0,
        reward_mode: str = "return",
        vol_idx: int = 1,
        risk_penalty: float = 0.0,
    ):
        """
        Parameters
        ----------
        states : np.ndarray, shape (T, state_dim)
            State matrix (e.g., from build_state_and_target_returns).
        next_returns : np.ndarray, shape (T,)
            Next-day market returns aligned with states.
        start_index : int
            Starting index within states/returns for this environment.
        end_index : int or None
            End index (exclusive). If None, uses len(states).
        initial_capital : float
            Starting portfolio value at reset.
        reward_mode : str
            Currently only "return" supported (reward = daily portfolio return).
            Hook for future risk-adjusted reward.
        """
        assert states.shape[0] == next_returns.shape[0], \
            f"states and next_returns must have same length, got {states.shape[0]} and {next_returns.shape[0]}"

        self.states = states
        self.next_returns = next_returns
        self.start_index = start_index
        self.end_index = end_index if end_index is not None else len(states)
        self.initial_capital = initial_capital
        self.reward_mode = reward_mode
        self.vol_idx = vol_idx
        self.risk_penalty = risk_penalty

        if self.end_index - self.start_index < 2:
            raise ValueError("Not enough timesteps in the selected range to run an episode.")

        self.current_index: int = self.start_index
        self.portfolio_value: float = self.initial_capital

    def reset(self) -> np.ndarray:
        """
        Reset environment to the beginning of the episode.

        Returns
        -------
        state : np.ndarray
            Initial state at time t = start_index.
        """
        self.current_index = self.start_index
        self.portfolio_value = self.initial_capital
        return self.states[self.current_index]

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : float
            Fraction of portfolio allocated to market. Will be clipped to [0, 1].

        Returns
        -------
        next_state : np.ndarray
        reward : float
        done : bool
        info : dict
            Contains debug info, e.g., 'portfolio_value', 'daily_return', 'weight_market'.
        """
        # Clip action into [0, 1] to ensure valid allocation
        w_market = float(np.clip(action, 0.0, 1.0))

        # Use next_returns aligned with current_index
        # portfolio return r_p = w_market * market_return_next_day
        daily_market_ret = float(self.next_returns[self.current_index])
        daily_portfolio_ret = w_market * daily_market_ret

        # Update portfolio value
        self.portfolio_value *= (1.0 + daily_portfolio_ret)

        # Compute reward
        if self.reward_mode == "return":
            reward = daily_portfolio_ret
        elif self.reward_mode == "sharpe_proxy":
            # Get current volatility from state (e.g. index 1 = market_vol)
            sigma_t = float(self.states[self.current_index][self.vol_idx])
            sigma2_t = sigma_t * sigma_t
            reward = daily_portfolio_ret - self.risk_penalty * (w_market ** 2) * sigma2_t
        else:
            raise NotImplementedError(f"Unknown reward_mode: {self.reward_mode}")

        # Move time forward
        self.current_index += 1

        # Episode termination: no next state beyond end_index-1
        done = self.current_index >= (self.end_index - 1)

        if not done:
            next_state = self.states[self.current_index]
        else:
            # For terminal step, we can still return the last valid state
            next_state = self.states[self.end_index - 1]

        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return": daily_portfolio_ret,
            "weight_market": w_market,
            "market_return": daily_market_ret,
            "t_index": self.current_index,
        }

        return next_state, reward, done, info
