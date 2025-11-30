# scripts/demo_env.py

import numpy as np
import matplotlib.pyplot as plt

from features import (
    build_base_timeseries,
    build_state_and_target_returns,
    split_train_val_test,
)
from teacher import compute_teacher_weights, simulate_teacher_equity, compute_simple_metrics
from envs.market_env import MarketEnv


def run_random_policy(env: MarketEnv) -> np.ndarray:
    """
    Run a random policy in the environment.
    At each step: action ~ Uniform(0, 1).
    Returns the equity curve over the episode.
    """
    state = env.reset()
    equity_curve = [env.portfolio_value]

    done = False
    while not done:
        action = np.random.rand()  # random weight in [0,1]
        next_state, reward, done, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        state = next_state

    return np.array(equity_curve)


def run_teacher_policy(env: MarketEnv, teacher_weights: np.ndarray) -> np.ndarray:
    """
    Run the teacher policy inside the environment.
    teacher_weights[t] must align with env's internal timestep indices.
    """
    state = env.reset()
    equity_curve = [env.portfolio_value]

    done = False
    t = 0
    while not done:
        action = float(teacher_weights[env.current_index])  # aligned with current_index
        next_state, reward, done, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        state = next_state
        t += 1

    return np.array(equity_curve)


def main():
    csv_path = "data/dataset_wilso_v1.csv"  # adjust path as needed

    # 1) Build dataframe with sentiment, returns, vol
    df_base = build_base_timeseries(csv_path)
    states, next_returns, dates, df_trimmed = build_state_and_target_returns(
        df_base,
        vol_window=20,
    )

    # 2) Chronological split
    splits = split_train_val_test(states, next_returns, dates, train_frac=0.7, val_frac=0.15)
    train_states = splits["train"]["states"]
    train_returns = splits["train"]["returns"]

    # 3) Compute teacher weights on train split
    teacher_w_train = compute_teacher_weights(train_states, lambda_risk=10.0)

    # 4) Instantiate environment on train data
    env_random = MarketEnv(
        states=train_states,
        next_returns=train_returns,
        start_index=0,
        end_index=None,
        initial_capital=1.0,
        reward_mode="return",
    )

    env_teacher = MarketEnv(
        states=train_states,
        next_returns=train_returns,
        start_index=0,
        end_index=None,
        initial_capital=1.0,
        reward_mode="return",
    )

    # 5) Run random policy
    equity_random = run_random_policy(env_random)

    # 6) Run teacher policy
    equity_teacher = run_teacher_policy(env_teacher, teacher_w_train)

    # 7) Quick metrics for teacher on train split
    #    (using direct simulation utility as a cross-check)
    teacher_ret_direct, teacher_equity_direct = simulate_teacher_equity(
        teacher_w_train, train_returns, initial_capital=1.0
    )
    metrics_teacher = compute_simple_metrics(teacher_ret_direct)
    print("Teacher metrics on train split:", metrics_teacher)
    
    # Buy and hold: always 100% in market
    w_buyhold = np.ones_like(train_returns)
    ret_bh, eq_bh = simulate_teacher_equity(w_buyhold, train_returns)
    metrics_bh = compute_simple_metrics(ret_bh)
    print("Buy & Hold metrics:", metrics_bh)

    # Teacher
    teacher_ret, teacher_equity = simulate_teacher_equity(teacher_w_train, train_returns)
    metrics_teacher = compute_simple_metrics(teacher_ret)
    print("Teacher metrics:", metrics_teacher)
    
    print("Mean daily market return on train:", train_returns.mean())

    # 8) Plot equity curves
    plt.figure(figsize=(10, 5))
    plt.plot(equity_random, label="Random Policy")
    plt.plot(equity_teacher, label="Teacher Policy (env)")
    plt.plot(teacher_equity_direct, "--", label="Teacher (direct sim check)", alpha=0.7)
    plt.xlabel("Time step")
    plt.ylabel("Portfolio value")
    plt.title("Random vs Teacher Policy on Train Split")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    
