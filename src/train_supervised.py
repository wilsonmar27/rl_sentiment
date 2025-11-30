# train_supervised.py

import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from features import (
    build_base_timeseries,
    build_state_and_target_returns,
    split_train_val_test,
    STATE_FEATURE_COLS,
)
from teacher import (
    compute_teacher_weights,
    simulate_teacher_equity,
    compute_simple_metrics,
)
from models.actor import ActorNet


class StateTeacherDataset(Dataset):
    """
    Simple dataset holding states and teacher allocations.
    """

    def __init__(self, states: np.ndarray, teacher_weights: np.ndarray):
        assert states.shape[0] == teacher_weights.shape[0], \
            "States and teacher_weights must have same length."
        self.states = states.astype(np.float32)
        self.teacher_weights = teacher_weights.astype(np.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(self.states[idx])
        w = torch.from_numpy(self.teacher_weights[idx])
        return s, w


def train_supervised_actor(
    csv_path: str,
    output_path: str = "models/supervised_actor.pth",
    lambda_risk: float = 10.0,
    vol_window: int = 20,
    hidden_sizes = [64, 64],
    batch_size: int = 64,
    lr: float = 1e-3,
    num_epochs: int = 30,
    train_frac: float = 0.74,
    val_frac: float = 0.15,
    device: str = "auto",
):
    """
    Train an ActorNet to mimic the teacher's allocations.

    Steps:
    - Build states & next_returns from CSV.
    - Split into train/val/test (chronologically).
    - Compute teacher weights on train & val.
    - Train actor with MSE loss on teacher weights.
    - Save trained model to output_path.
    """

    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1) Build base DF and states/returns
    print("Loading data and building features...")
    df_base = build_base_timeseries(csv_path)
    states, next_returns, dates, df_trimmed = build_state_and_target_returns(
        df_base,
        vol_window=vol_window,
    )

    print(f"States shape: {states.shape}, next_returns shape: {next_returns.shape}")

    # 2) Split into train/val/test
    splits = split_train_val_test(states, next_returns, dates,
                                  train_frac=train_frac, val_frac=val_frac)
    train_states = splits["train"]["states"]
    val_states = splits["val"]["states"]
    test_states = splits["test"]["states"]

    train_returns = splits["train"]["returns"]
    val_returns = splits["val"]["returns"]
    test_returns = splits["test"]["returns"]

    print("Split sizes:")
    print("  Train:", train_states.shape[0])
    print("  Val  :", val_states.shape[0])
    print("  Test :", test_states.shape[0])

    # 3) Compute teacher weights
    print("Computing teacher weights...")
    teacher_w_train = compute_teacher_weights(train_states, lambda_risk=lambda_risk)
    teacher_w_val = compute_teacher_weights(val_states, lambda_risk=lambda_risk)

    # 4) Create datasets & dataloaders
    train_dataset = StateTeacherDataset(train_states, teacher_w_train)
    val_dataset = StateTeacherDataset(val_states, teacher_w_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    state_dim = train_states.shape[1]
    print(f"State dimension: {state_dim} (features: {STATE_FEATURE_COLS})")

    # 5) Initialize actor network
    actor = ActorNet(state_dim=state_dim, hidden_sizes=hidden_sizes).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 6) Training loop
    best_val_loss = float("inf")
    best_state_dict = None

    print("Starting supervised training...")
    for epoch in range(1, num_epochs + 1):
        actor.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch_states, batch_targets in train_loader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = actor(batch_states)           # shape (batch_size, 1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        # Validation
        actor.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_states, batch_targets in val_loader:
                batch_states = batch_states.to(device)
                batch_targets = batch_targets.to(device)

                outputs = actor(batch_states)
                loss = criterion(outputs, batch_targets)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}"
        )

        # Track best model by val_loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = actor.state_dict()

    # 7) Save best model
    if best_state_dict is not None:
        actor.load_state_dict(best_state_dict)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "state_dim": state_dim,
            "hidden_sizes": hidden_sizes,
            "feature_cols": STATE_FEATURE_COLS,
        },
        output_path,
    )
    print(f"Saved supervised actor to: {output_path}")

    # 8) Quick sanity check: compare teacher vs student on val & test

    actor.eval()
    with torch.no_grad():
        val_states_t = torch.from_numpy(val_states.astype(np.float32)).to(device)
        pred_val_w = actor(val_states_t).cpu().numpy().flatten()
        test_states_t = torch.from_numpy(test_states.astype(np.float32)).to(device)
        pred_test_w = actor(test_states_t).cpu().numpy().flatten()

    # Teacher weights for test set
    teacher_w_test = compute_teacher_weights(test_states, lambda_risk=lambda_risk)

    # Simple correlation metrics
    corr_val = np.corrcoef(pred_val_w, teacher_w_val)[0, 1] if len(pred_val_w) > 1 else np.nan
    corr_test = np.corrcoef(pred_test_w, teacher_w_test)[0, 1] if len(pred_test_w) > 1 else np.nan
    print(f"Correlation (student vs teacher) on VAL:  {corr_val:.4f}")
    print(f"Correlation (student vs teacher) on TEST: {corr_test:.4f}")

    # 9) Portfolio performance comparison on validation set
    #    (teacher vs student, as a sanity check)
    student_ret_val, student_equity_val = simulate_teacher_equity(pred_val_w, val_returns)
    teacher_ret_val, teacher_equity_val = simulate_teacher_equity(teacher_w_val, val_returns)

    metrics_teacher_val = compute_simple_metrics(teacher_ret_val)
    metrics_student_val = compute_simple_metrics(student_ret_val)

    print("Validation performance (Teacher):", metrics_teacher_val)
    print("Validation performance (Student):", metrics_student_val)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised training of ActorNet to mimic teacher.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/dataset_wilso_v1.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/supervised_actor.pth",
        help="Path to save the trained actor model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lambda_risk",
        type=float,
        default=10.0,
        help="Teacher risk-aversion parameter.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_supervised_actor(
        csv_path=args.csv,
        output_path=args.out,
        num_epochs=args.epochs,
        lambda_risk=args.lambda_risk,
    )
