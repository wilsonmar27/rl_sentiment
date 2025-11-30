# train_rl.py

import os
import argparse
import copy
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

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
from envs.market_env import MarketEnv
from models.actor import ActorNet
from models.critic import CriticNet


class ReplayBuffer:
    """
    Simple replay buffer for off-policy RL.
    Stores transitions: (s, a, r, s', done).
    """

    def __init__(self, state_dim: int, action_dim: int, size: int = 100000):
        self.state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        idx = self.ptr
        self.state_buf[idx] = state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = next_state
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            state=self.state_buf[idxs],
            action=self.action_buf[idxs],
            reward=self.reward_buf[idxs],
            next_state=self.next_state_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_policy(
    actor: ActorNet,
    states: np.ndarray,
    next_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Run deterministic actor over a dataset slice (no env needed),
    compute portfolio returns and basic metrics.
    """
    actor.eval()

    # Infer device from actor parameters
    device = next(actor.parameters()).device

    with torch.no_grad():
        s_torch = torch.from_numpy(states.astype(np.float32)).to(device)
        a = actor(s_torch).cpu().numpy().flatten()

    # Simulate portfolio using same helper as teacher
    portfolio_ret, equity = simulate_teacher_equity(a, next_returns)
    metrics = compute_simple_metrics(portfolio_ret)
    return metrics



def train_ddpg(
    csv_path: str,
    supervised_actor_path: str = "models/supervised_actor.pth",
    rl_actor_out_path: str = "models/rl_actor.pth",
    vol_window: int = 20,
    lambda_risk: float = 10.0,  # teacher param, just for baseline eval
    total_steps: int = 50000,
    start_steps: int = 1000,
    update_after: int = 1000,
    update_every: int = 50,
    batch_size: int = 64,
    replay_size: int = 100000,
    gamma: float = 0.99,
    polyak: float = 0.995,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    act_noise_std: float = 0.1,
    train_frac: float = 0.74,
    val_frac: float = 0.15,
    device: str = "auto",
    seed: int = 0,
):
    """
    DDPG training:

    - Loads data & supervised actor
    - Trains in MarketEnv on train split
    - Evaluates best actor on validation set vs teacher baseline
    """

    set_seed(seed)

    # Device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # 1) Build features
    print("Loading data...")
    df_base = build_base_timeseries(csv_path)
    states_all, next_returns_all, dates_all, df_trimmed = build_state_and_target_returns(
        df_base,
        vol_window=vol_window,
    )

    # 2) Split into train/val/test
    splits = split_train_val_test(
        states_all, next_returns_all, dates_all,
        train_frac=train_frac, val_frac=val_frac
    )
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

    state_dim = train_states.shape[1]
    action_dim = 1
    print(f"State dim: {state_dim}, action dim: {action_dim}")
    print(f"State feature columns: {STATE_FEATURE_COLS}")

    # 3) Load supervised actor as initialization
    print(f"Loading supervised actor from {supervised_actor_path}")
    checkpoint = torch.load(supervised_actor_path, map_location=device)
    supervised_state_dim = checkpoint["state_dim"]
    if supervised_state_dim != state_dim:
        raise ValueError(
            f"State dim mismatch: supervised actor expects {supervised_state_dim}, "
            f"but current state_dim is {state_dim}"
        )
    hidden_sizes = checkpoint.get("hidden_sizes", [64, 64])

    actor = ActorNet(state_dim=state_dim, hidden_sizes=hidden_sizes).to(device)
    actor.load_state_dict(checkpoint["model_state_dict"])

    # Target actor
    actor_target = ActorNet(state_dim=state_dim, hidden_sizes=hidden_sizes).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # 4) Critic + target critic
    critic = CriticNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[64, 64]).to(device)
    critic_target = CriticNet(state_dim=state_dim, action_dim=action_dim, hidden_sizes=[64, 64]).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    mse_loss = nn.MSELoss()

    # 5) Replay buffer
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, size=replay_size)

    # 6) Environment over train split
    env = MarketEnv(
        states=train_states,
        next_returns=train_returns,
        start_index=0,
        end_index=None,
        initial_capital=1.0,
        reward_mode="return",
    )

    # 7) DDPG training loop
    state = env.reset()
    episode_return = 0.0
    episode_length = 0

    best_val_sharpe = -1e9
    best_actor_state_dict = copy.deepcopy(actor.state_dict())

    print("Starting DDPG training...")
    for t in range(1, total_steps + 1):
        # Select action
        if t < start_steps:
            # Exploration: random uniform action
            action = np.random.rand()
        else:
            # Actor + exploration noise
            actor.eval()
            with torch.no_grad():
                s_t = torch.from_numpy(state.astype(np.float32)).to(device).unsqueeze(0)
                a_t = actor(s_t).cpu().numpy().flatten()[0]
            noise = np.random.normal(0.0, act_noise_std)
            action = np.clip(a_t + noise, 0.0, 1.0)

        # Step env
        next_state, reward, done, info = env.step(action)
        episode_return += reward
        episode_length += 1

        # Store in replay buffer
        replay_buffer.store(
            state=state,
            action=np.array([action], dtype=np.float32),
            reward=reward,
            next_state=next_state,
            done=done,
        )

        state = next_state

        # If episode ends, reset
        if done:
            # Optional: print episode info occasionally
            # print(f"Episode finished: return={episode_return:.4f}, length={episode_length}")
            state = env.reset()
            episode_return = 0.0
            episode_length = 0

        # Start updating after enough data is collected
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                s = torch.from_numpy(batch["state"]).to(device)
                a = torch.from_numpy(batch["action"]).to(device)
                r = torch.from_numpy(batch["reward"]).to(device).unsqueeze(-1)
                s2 = torch.from_numpy(batch["next_state"]).to(device)
                d = torch.from_numpy(batch["done"]).to(device).unsqueeze(-1)

                # Critic update
                with torch.no_grad():
                    a2 = actor_target(s2)
                    q_target = critic_target(s2, a2)
                    # Bellman backup: r + gamma * (1-d) * Q_target
                    backup = r + gamma * (1.0 - d) * q_target

                q_val = critic(s, a)
                critic_loss = mse_loss(q_val, backup)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update (policy gradient)
                actor_optimizer.zero_grad()
                # We want to maximize Q(s, actor(s)) -> minimize -Q
                a_pi = actor(s)
                actor_loss = -critic(s, a_pi).mean()
                actor_loss.backward()
                actor_optimizer.step()

                # Update target networks with polyak averaging
                with torch.no_grad():
                    for p, p_targ in zip(actor.parameters(), actor_target.parameters()):
                        p_targ.data.mul_(polyak).add_((1 - polyak) * p.data)
                    for p, p_targ in zip(critic.parameters(), critic_target.parameters()):
                        p_targ.data.mul_(polyak).add_((1 - polyak) * p.data)

        # Periodically evaluate on validation set
        if t % 5000 == 0 or t == total_steps:
            print(f"\nStep {t}/{total_steps}: evaluating on validation set...")
            val_metrics = evaluate_policy(actor, val_states, val_returns)
            print("Student (RL) validation metrics:", val_metrics)

            # Teacher baseline on val
            teacher_w_val = compute_teacher_weights(val_states, lambda_risk=lambda_risk)
            teacher_ret_val, _ = simulate_teacher_equity(teacher_w_val, val_returns)
            teacher_metrics = compute_simple_metrics(teacher_ret_val)
            print("Teacher validation metrics    :", teacher_metrics)

            # Track best by Sharpe
            val_sharpe = val_metrics.get("sharpe_ratio", 0.0)
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                best_actor_state_dict = copy.deepcopy(actor.state_dict())
                print(f"New best validation Sharpe: {best_val_sharpe:.4f} (step {t})")

    # Load best actor
    actor.load_state_dict(best_actor_state_dict)

    # 8) Save trained RL actor
    os.makedirs(os.path.dirname(rl_actor_out_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": actor.state_dict(),
            "state_dim": state_dim,
            "hidden_sizes": hidden_sizes,
            "feature_cols": STATE_FEATURE_COLS,
        },
        rl_actor_out_path,
    )
    print(f"\nSaved RL-trained actor to: {rl_actor_out_path}")

    # 9) Final evaluation on validation & test
    print("\nFinal evaluation (best RL actor):")
    val_metrics = evaluate_policy(actor, val_states, val_returns)
    teacher_w_val = compute_teacher_weights(val_states, lambda_risk=lambda_risk)
    teacher_ret_val, _ = simulate_teacher_equity(teacher_w_val, val_returns)
    teacher_val_metrics = compute_simple_metrics(teacher_ret_val)

    print("Validation - Teacher:", teacher_val_metrics)
    print("Validation - Student (RL):", val_metrics)

    test_metrics = evaluate_policy(actor, test_states, test_returns)
    teacher_w_test = compute_teacher_weights(test_states, lambda_risk=lambda_risk)
    teacher_ret_test, _ = simulate_teacher_equity(teacher_w_test, test_returns)
    teacher_test_metrics = compute_simple_metrics(teacher_ret_test)

    print("\nTest - Teacher:", teacher_test_metrics)
    print("Test - Student (RL):", test_metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="DDPG RL fine-tuning of actor policy.")
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
        help="Path to the pretrained supervised actor checkpoint.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/rl_actor.pth",
        help="Path to save the RL-trained actor.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total DDPG environment steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Random seed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ddpg(
        csv_path=args.csv,
        supervised_actor_path=args.supervised_actor,
        rl_actor_out_path=args.out,
        total_steps=args.steps,
        seed=args.seed,
    )
