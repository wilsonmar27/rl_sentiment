# src/eval.py

import os
import argparse
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from features import (
    build_base_timeseries,
    build_state_and_target_returns,
    split_train_val_test,
    STATE_FEATURE_COLS,
)
from teacher import (
    compute_teacher_weights,
    simulate_teacher_equity,
)
from models.actor import ActorNet


# Metrics utilities
def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from a sequence of returns.
    Returns a negative number (e.g., -0.25 for -25% max drawdown).
    """
    r = np.asarray(returns, dtype=float)
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    return float(max_dd)


def compute_metrics(
    portfolio_returns: np.ndarray,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute:
    - total_return
    - annualized_return
    - annualized_vol
    - sharpe_ratio (RF ~ 0)
    - max_drawdown
    """
    r = np.asarray(portfolio_returns, dtype=float)
    if len(r) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    mean_daily = r.mean()
    std_daily = r.std(ddof=1) if len(r) > 1 else 0.0

    total_return = (1.0 + r).prod() - 1.0
    if mean_daily > -1.0:
        annualized_return = (1.0 + mean_daily) ** trading_days_per_year - 1.0
    else:
        annualized_return = -1.0

    annualized_vol = std_daily * np.sqrt(trading_days_per_year)
    sharpe = mean_daily / std_daily * np.sqrt(trading_days_per_year) if std_daily > 0 else 0.0
    max_dd = compute_max_drawdown(r)

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_vol": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }


# Actor loading & evaluation helpers
def load_actor(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[ActorNet, int]:
    """
    Load an ActorNet from a checkpoint saved in train_supervised.py or train_rl.py.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dim = ckpt["state_dim"]
    hidden_sizes = ckpt.get("hidden_sizes", [64, 64])

    actor = ActorNet(state_dim=state_dim, hidden_sizes=hidden_sizes).to(device)
    actor.load_state_dict(ckpt["model_state_dict"])
    actor.eval()
    return actor, state_dim


def actor_weights_for_states(
    actor: ActorNet,
    states: np.ndarray,
    device: torch.device,
    w_max: float = 1.0,
) -> np.ndarray:
    """
    Get continuous allocation weights from an actor over a whole dataset slice.
    """
    with torch.no_grad():
        s_torch = torch.from_numpy(states.astype(np.float32)).to(device)
        a_raw = actor(s_torch).cpu().numpy().flatten()
        a = np.clip(a_raw, 0.0, 1.0) * w_max               
    return a


def evaluate_policy_from_weights(
    weights: np.ndarray,
    next_returns: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Given precomputed weights and market returns, compute:
    - portfolio_returns
    - equity_curve
    - metrics dict
    """
    portfolio_returns, equity_curve = simulate_teacher_equity(weights, next_returns)
    metrics = compute_metrics(portfolio_returns)
    return portfolio_returns, equity_curve, metrics


# Sentiment sensitivity analysis
def sentiment_sensitivity_analysis(
    df_trimmed,
    dates_test: np.ndarray,
    returns_test: np.ndarray,
    teacher_weights_test: np.ndarray,
    sup_weights_test: np.ndarray,
    rl_weights_test: np.ndarray,
    output_dir: str,
):
    """
    Slice test period into high-pessimism vs normal days (based on average
    of photo_neg_mean + text_neg_mean). Compare metrics of each policy
    in each regime.

    This is a first-pass / simple version:
    - High pessimism: top 20% of sentiment index (more negative)
    - Normal: remaining 80%
    """
    if not {"photo_neg_mean", "text_neg_mean"}.issubset(df_trimmed.columns):
        print("Sentiment columns not found in df_trimmed; skipping sentiment analysis.")
        return

    # Align df_trimmed to test dates
    df_indexed = df_trimmed.set_index("date")
    df_test = df_indexed.loc[dates_test].copy()

    # Combined sentiment index (you can tweak this)
    df_test["sentiment_combined"] = 0.5 * (
        df_test["photo_neg_mean"] + df_test["text_neg_mean"]
    )

    sent = df_test["sentiment_combined"].values
    # Top 20% = high pessimism regime
    thresh_high = np.quantile(sent, 0.8)
    mask_high = sent >= thresh_high
    mask_normal = ~mask_high

    # Compute returns per policy on full test first
    teacher_ret_test, _ = simulate_teacher_equity(teacher_weights_test, returns_test)
    sup_ret_test, _ = simulate_teacher_equity(sup_weights_test, returns_test)
    rl_ret_test, _ = simulate_teacher_equity(rl_weights_test, returns_test)

    def metrics_for_mask(mask, name: str):
        print(f"\n=== Sentiment regime: {name} ===")
        for label, ret in [
            ("Teacher", teacher_ret_test),
            ("Supervised", sup_ret_test),
            ("RL", rl_ret_test),
        ]:
            m = compute_metrics(ret[mask])
            print(
                f"{label:10s} | "
                f"total={m['total_return']:.4f}, "
                f"ann_ret={m['annualized_return']:.4f}, "
                f"vol={m['annualized_vol']:.4f}, "
                f"sharpe={m['sharpe_ratio']:.3f}, "
                f"maxDD={m['max_drawdown']:.3f}"
            )

    metrics_for_mask(mask_high, "High pessimism (top 20%)")
    metrics_for_mask(mask_normal, "Normal sentiment (bottom 80%)")

    # Optionally, save regime mask for further analysis
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "test_sentiment_mask_high.npy"), mask_high)
    np.save(os.path.join(output_dir, "test_sentiment_mask_normal.npy"), mask_normal)


# Main evaluation script
def main():
    parser = argparse.ArgumentParser(description="Evaluate teacher vs students (supervised & RL).")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/dataset_wilso_v1.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--supervised_actor",
        type=str,
        default="models/supervised_actor.pth",
        help="Path to the supervised actor checkpoint.",
    )
    parser.add_argument(
        "--rl_actor",
        type=str,
        default="models/rl_actor.pth",
        help="Path to the RL-trained actor checkpoint.",
    )
    parser.add_argument(
        "--lambda_risk",
        type=float,
        default=10.0,
        help="Risk aversion parameter for the teacher.",
    )
    parser.add_argument(
        "--vol_window",
        type=int,
        default=20,
        help="Rolling window size for volatility.",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.74,
        help="Fraction of data for training.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.15,
        help="Fraction of data for validation.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for plots and artifacts.",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build data & states
    print("Loading data and building features...")
    df_base = build_base_timeseries(args.csv)
    states_all, next_returns_all, dates_all, df_trimmed = build_state_and_target_returns(
        df_base,
        vol_window=args.vol_window,
    )

    print(f"States shape: {states_all.shape}, returns shape: {next_returns_all.shape}")

    # 2) Split into train/val/test
    splits = split_train_val_test(
        states_all,
        next_returns_all,
        dates_all,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    train_states = splits["train"]["states"]
    val_states = splits["val"]["states"]
    test_states = splits["test"]["states"]

    train_returns = splits["train"]["returns"]
    val_returns = splits["val"]["returns"]
    test_returns = splits["test"]["returns"]

    train_dates = splits["train"]["dates"]
    val_dates = splits["val"]["dates"]
    test_dates = splits["test"]["dates"]

    # Dev = train + val
    dev_states = np.concatenate([train_states, val_states], axis=0)
    dev_returns = np.concatenate([train_returns, val_returns], axis=0)
    dev_dates = np.concatenate([train_dates, val_dates], axis=0)

    print("Split sizes:")
    print(f"  Train: {train_states.shape[0]}")
    print(f"  Val  : {val_states.shape[0]}")
    print(f"  Dev  : {dev_states.shape[0]} (train+val)")
    print(f"  Test : {test_states.shape[0]}")
    print(f"State feature columns: {STATE_FEATURE_COLS}")

    state_dim = states_all.shape[1]

    # 3) Load actors
    print(f"\nLoading supervised actor from {args.supervised_actor}")
    supervised_actor, sup_state_dim = load_actor(args.supervised_actor, device)
    if sup_state_dim != state_dim:
        raise ValueError(f"Supervised actor state_dim={sup_state_dim}, but data state_dim={state_dim}")

    print(f"Loading RL actor from {args.rl_actor}")
    rl_actor, rl_state_dim = load_actor(args.rl_actor, device)
    if rl_state_dim != state_dim:
        raise ValueError(f"RL actor state_dim={rl_state_dim}, but data state_dim={state_dim}")

    # 4) Compute weights for each policy on dev & test

    # Teacher weights
    teacher_w_dev = compute_teacher_weights(dev_states, lambda_risk=args.lambda_risk)
    teacher_w_test = compute_teacher_weights(test_states, lambda_risk=args.lambda_risk)

    # Supervised student
    sup_w_dev = actor_weights_for_states(supervised_actor, dev_states, device)
    sup_w_test = actor_weights_for_states(supervised_actor, test_states, device)

    # RL student
    rl_w_dev = actor_weights_for_states(rl_actor, dev_states, device)
    rl_w_test = actor_weights_for_states(rl_actor, test_states, device)

    # 5) Evaluate each policy: dev
    teacher_ret_dev, teacher_eq_dev, teacher_metrics_dev = evaluate_policy_from_weights(
        teacher_w_dev, dev_returns
    )
    sup_ret_dev, sup_eq_dev, sup_metrics_dev = evaluate_policy_from_weights(
        sup_w_dev, dev_returns
    )
    rl_ret_dev, rl_eq_dev, rl_metrics_dev = evaluate_policy_from_weights(
        rl_w_dev, dev_returns
    )

    # 6) Evaluate each policy: test
    teacher_ret_test, teacher_eq_test, teacher_metrics_test = evaluate_policy_from_weights(
        teacher_w_test, test_returns
    )
    sup_ret_test, sup_eq_test, sup_metrics_test = evaluate_policy_from_weights(
        sup_w_test, test_returns
    )
    rl_ret_test, rl_eq_test, rl_metrics_test = evaluate_policy_from_weights(
        rl_w_test, test_returns
    )
    
    # 7a) Save test equity curves to CSV (aligned with dates)
    # equity arrays are length T+1, dates are length T -> use equity[1:]
    if len(test_dates) != len(teacher_eq_test) - 1:
        raise ValueError(
            f"Length mismatch: test_dates={len(test_dates)}, "
            f"teacher_eq_test={len(teacher_eq_test)} (expected len(equity) = len(dates)+1)"
        )

    eq_df = pd.DataFrame({
        "date": test_dates,
        "equity_teacher": teacher_eq_test[1:],     # end-of-day equity
        "equity_supervised": sup_eq_test[1:],
        "equity_rl": rl_eq_test[1:],
    })

    os.makedirs(args.out_dir, exist_ok=True)
    eq_csv_path = os.path.join(args.out_dir, "equity_test_timeseries.csv")
    eq_df.to_csv(eq_csv_path, index=False)
    print(f"Saved test equity timeseries CSV to {eq_csv_path}")

    # 7) Print metrics tables
    def print_table(split_name: str, teacher_m: Dict, sup_m: Dict, rl_m: Dict):
        print(f"\n=== {split_name} metrics ===")
        header = (
            f"{'Policy':12s} | "
            f"{'Total':>8s} | {'AnnRet':>8s} | {'Vol':>8s} | {'Sharpe':>8s} | {'MaxDD':>8s}"
        )
        print(header)
        print("-" * len(header))

        def row(name, m):
            print(
                f"{name:12s} | "
                f"{m['total_return']:8.4f} | "
                f"{m['annualized_return']:8.4f} | "
                f"{m['annualized_vol']:8.4f} | "
                f"{m['sharpe_ratio']:8.4f} | "
                f"{m['max_drawdown']:8.4f}"
            )

        row("Teacher", teacher_m)
        row("Supervised", sup_m)
        row("RL", rl_m)

    print_table("DEV (Train+Val)", teacher_metrics_dev, sup_metrics_dev, rl_metrics_dev)
    print_table("TEST", teacher_metrics_test, sup_metrics_test, rl_metrics_test)

    # 8) Plot equity curves
    # Dev
    plt.figure(figsize=(10, 5))
    plt.plot(teacher_eq_dev, label="Teacher")
    plt.plot(sup_eq_dev, label="Supervised")
    plt.plot(rl_eq_dev, label="RL")
    plt.title("Equity Curves - DEV (Train+Val)")
    plt.xlabel("Time step")
    plt.ylabel("Portfolio value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    dev_plot_path = os.path.join(args.out_dir, "equity_dev.png")
    plt.savefig(dev_plot_path)
    plt.close()
    print(f"Saved DEV equity plot to {dev_plot_path}")

    # Test
    plt.figure(figsize=(10, 5))
    plt.plot(teacher_eq_test, label="Teacher")
    plt.plot(sup_eq_test, label="Supervised")
    plt.plot(rl_eq_test, label="RL")
    plt.title("Equity Curves - TEST")
    plt.xlabel("Time step")
    plt.ylabel("Portfolio value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    test_plot_path = os.path.join(args.out_dir, "equity_test.png")
    plt.savefig(test_plot_path)
    plt.close()
    print(f"Saved TEST equity plot to {test_plot_path}")

    # 9) Sentiment-sensitivity analysis on test period
    print("\nRunning sentiment-sensitivity analysis on TEST period...")
    sentiment_sensitivity_analysis(
        df_trimmed=df_trimmed,
        dates_test=test_dates,
        returns_test=test_returns,
        teacher_weights_test=teacher_w_test,
        sup_weights_test=sup_w_test,
        rl_weights_test=rl_w_test,
        output_dir=args.out_dir,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
