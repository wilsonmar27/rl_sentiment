# src/models/critic.py

import torch
import torch.nn as nn
from typing import List


class CriticNet(nn.Module):
    """
    Q-network: estimates Q(s, a) for continuous actions.

    Input:  state s_t (batch, state_dim), action a_t (batch, action_dim)
    Output: Q-value (batch, 1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_sizes: List[int] = [64, 64],
    ):
        super().__init__()

        in_dim = state_dim + action_dim
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor, shape (batch_size, state_dim)
        action : torch.Tensor, shape (batch_size, action_dim)

        Returns
        -------
        q_value : torch.Tensor, shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
