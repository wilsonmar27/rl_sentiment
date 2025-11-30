# src/models/actor.py

import torch
import torch.nn as nn
from typing import List


class ActorNet(nn.Module):
    """
    Simple feed-forward actor network for 2-asset allocation.

    Input:  state vector s_t (e.g. [ret_t, vol_t, photo_mean_t, text_mean_t])
    Output: scalar a_t in [0, 1] = weight in market ETF
            cash weight = 1 - a_t
    """

    def __init__(
        self,
        state_dim: int,
        hidden_sizes: List[int] = [64, 64],
    ):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        # Single output neuron -> sigmoid to squash into [0,1]
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor, shape (batch_size, state_dim)

        Returns
        -------
        action : torch.Tensor, shape (batch_size, 1)
            Fraction of portfolio allocated to market in [0,1].
        """
        return self.model(state)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for acting in an environment.

        Parameters
        ----------
        state : torch.Tensor
            Shape (state_dim,) or (1, state_dim)

        Returns
        -------
        action : float
            Scalar in [0,1].
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            action = self.forward(state)
        return action.squeeze().item()
