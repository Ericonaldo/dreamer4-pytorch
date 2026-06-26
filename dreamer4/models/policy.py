from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.models.dynamics import DynamicsModel
from dreamer4.models.task_embedder import TaskEmbedder


def _cfg_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


class _MLPHead(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class AgentOutputs:
    action: torch.Tensor
    reward: torch.Tensor
    value: torch.Tensor


class AgentHeads(nn.Module):
    """BC heads on agent readout h_t: L-step action/reward + scalar value."""

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        *,
        action_horizon: int = 8,
        policy_hidden: int = 256,
        reward_hidden: int = 256,
        value_hidden: int = 256,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.policy = _MLPHead(d_model, policy_hidden, self.action_horizon * self.action_dim)
        self.reward = _MLPHead(d_model, reward_hidden, self.action_horizon)
        self.value = _MLPHead(d_model, value_hidden, 1)

    def forward(self, h_t: torch.Tensor) -> AgentOutputs:
        """h_t: (B, T, D) pooled agent readout."""
        B, T, D = h_t.shape
        action = self.policy(h_t).view(B, T, self.action_horizon, self.action_dim)
        reward = self.reward(h_t).view(B, T, self.action_horizon)
        value = self.value(h_t).squeeze(-1)
        return AgentOutputs(action=action, reward=reward, value=value)


class BCModel(nn.Module):
    """Dynamics backbone (wm_agent) + task agent tokens + BC heads."""

    def __init__(
        self,
        dynamics_cfg: Mapping[str, Any] | DictConfig,
        *,
        n_latents: int,
        latent_dim: int,
        heads_cfg: Mapping[str, Any] | DictConfig,
    ):
        super().__init__()
        raw = _cfg_dict(dynamics_cfg)
        self.action_dim = int(raw["action_dim"])
        self.n_agent = int(raw.get("n_agent", 1))
        self.d_model = int(raw["embed_dim"])

        self.dynamics = DynamicsModel(raw, n_latents=n_latents, latent_dim=latent_dim)
        self.task_embedder = TaskEmbedder(
            self.d_model,
            self.n_agent,
            use_ids=bool(raw.get("use_task_ids", True)),
            n_tasks=int(raw.get("n_tasks", 1)),
        )
        heads = _cfg_dict(heads_cfg)
        self.heads = AgentHeads(
            self.d_model,
            self.action_dim,
            action_horizon=int(heads.get("action_horizon", 8)),
            policy_hidden=int(heads.get("policy_hidden", 256)),
            reward_hidden=int(heads.get("reward_hidden", 256)),
            value_hidden=int(heads.get("value_hidden", 256)),
        )

    def forward(
        self,
        packed_z: torch.Tensor,
        actions: torch.Tensor,
        task_id: torch.Tensor,
    ) -> AgentOutputs:
        B, T = packed_z.shape[:2]
        agent_tokens = self.task_embedder(task_id, B=B, T=T)
        sigma = torch.zeros(B, T, device=packed_z.device, dtype=torch.float32)
        _, h_agent = self.dynamics(actions, sigma, packed_z, agent_tokens=agent_tokens)
        if h_agent is None:
            raise ValueError("BCModel requires n_agent > 0 on dynamics backbone")
        h_t = h_agent.mean(dim=2)
        return self.heads(h_t)


def _future_targets(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Build (B, T, L, ...) targets where slot l at time t is x[:, t+l]."""
    B, T = x.shape[:2]
    trailing = x.shape[2:]
    out = []
    for lag in range(horizon):
        shifted = x[:, lag:]
        if lag > 0:
            pad = x.new_zeros(B, lag, *trailing)
            shifted = torch.cat([shifted, pad], dim=1)
        out.append(shifted)
    return torch.stack(out, dim=2)


def _valid_future_mask(T: int, horizon: int, device: torch.device) -> torch.Tensor:
    t_idx = torch.arange(T, device=device)[None, :, None]
    l_idx = torch.arange(horizon, device=device)[None, None, :]
    return (t_idx + l_idx) < T


def bc_loss(
    outputs: AgentOutputs,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    *,
    action_horizon: int,
    action_weight: float = 1.0,
    reward_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, T, _ = actions.shape
    device = actions.device
    valid = _valid_future_mask(T, action_horizon, device).float()

    target_a = _future_targets(actions, action_horizon)
    action_err = (outputs.action.float() - target_a.float()).pow(2).mean(dim=-1)
    action_mse = (action_err * valid).sum() / valid.sum().clamp_min(1.0)

    target_r = _future_targets(rewards.unsqueeze(-1), action_horizon).squeeze(-1)
    reward_err = (outputs.reward.float() - target_r.float()).pow(2)
    reward_mse = (reward_err * valid).sum() / valid.sum().clamp_min(1.0)

    value_mse = (outputs.value.float() - rewards.float()).pow(2).mean()

    loss = action_weight * action_mse + reward_weight * reward_mse + value_weight * value_mse
    metrics = {
        "action_mse": float(action_mse.detach()),
        "reward_mse": float(reward_mse.detach()),
        "value_mse": float(value_mse.detach()),
    }
    return loss, metrics
