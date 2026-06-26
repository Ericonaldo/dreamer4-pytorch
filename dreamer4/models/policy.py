from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf


def _cfg_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


class ActionEncoder(nn.Module):
    """Continuous actions (B,T,A) -> single token (B,T,1,D)."""

    def __init__(self, d_model: int, action_dim: int, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)
        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: torch.Tensor,
        *,
        batch_time_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            out = self.fc2(F.silu(self.fc1(actions))) + self.base.view(1, 1, -1)
        return out[:, :, None, :]


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
    action_raw: torch.Tensor
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
        raw = self.policy(h_t).view(B, T, self.action_horizon, self.action_dim)
        action = torch.tanh(raw)
        reward = self.reward(h_t).view(B, T, self.action_horizon)
        value = self.value(h_t).squeeze(-1)
        return AgentOutputs(action=action, action_raw=raw, reward=reward, value=value)


class BCModel(nn.Module):
    """Dynamics backbone (wm_agent) + learned agent tokens + BC heads."""

    def __init__(
        self,
        dynamics_cfg: Mapping[str, Any] | DictConfig,
        *,
        n_latents: int,
        latent_dim: int,
        heads_cfg: Mapping[str, Any] | DictConfig,
    ):
        super().__init__()
        from dreamer4.models.dynamics import DynamicsModel

        raw = _cfg_dict(dynamics_cfg)
        self.action_dim = int(raw["action_dim"])
        self.n_agent = int(raw.get("n_agent", 1))
        self.d_model = int(raw["embed_dim"])

        self.dynamics = DynamicsModel(raw, n_latents=n_latents, latent_dim=latent_dim)
        self.agent_tokens = nn.Parameter(torch.empty(self.n_agent, self.d_model))
        nn.init.normal_(self.agent_tokens, std=0.02)
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
    ) -> AgentOutputs:
        B, T = packed_z.shape[:2]
        agent_tokens = self.agent_tokens.view(1, 1, self.n_agent, self.d_model).expand(B, T, -1, -1)
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
    denom = (valid.sum() * B).clamp_min(1.0)

    target_a = _future_targets(actions, action_horizon).clamp(-1.0, 1.0)
    # Train in pre-tanh space; tanh is only for bounded env actions at inference.
    action_err = (outputs.action_raw.float() - target_a.float()).pow(2).mean(dim=-1)
    action_mse = (action_err * valid).sum() / denom

    tanh_err = (outputs.action.float() - target_a.float()).pow(2).mean(dim=-1)
    action_mse_tanh = (tanh_err * valid).sum() / denom

    target_r = _future_targets(rewards.unsqueeze(-1), action_horizon).squeeze(-1)
    reward_err = (outputs.reward.float() - target_r.float()).pow(2)
    reward_mse = (reward_err * valid).sum() / denom

    value_mse = (outputs.value.float() - rewards.float()).pow(2).mean()

    loss = action_weight * action_mse + reward_weight * reward_mse + value_weight * value_mse
    pred = outputs.action.float()
    metrics = {
        "action_mse": float(action_mse.detach()),
        "action_mse_tanh": float(action_mse_tanh.detach()),
        "action_out_mean": float(pred.mean().detach()),
        "action_out_abs_mean": float(pred.abs().mean().detach()),
        "action_raw_mean": float(outputs.action_raw.float().mean().detach()),
        "action_target_mean": float(target_a.mean().detach()),
        "action_target_abs_mean": float(target_a.abs().mean().detach()),
        "reward_mse": float(reward_mse.detach()),
        "value_mse": float(value_mse.detach()),
    }
    return loss, metrics
