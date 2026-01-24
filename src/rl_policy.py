from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        action_dim: int = 2,
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            u = dist.rsample()
            action = torch.tanh(u)
        else:
            u = atanh(action)

        log_prob_u = dist.log_prob(u).sum(-1)
        log_det = torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob_u - log_det

        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mu = self.actor(obs)
        return torch.tanh(mu)

