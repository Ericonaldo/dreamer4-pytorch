"""BC policy, action encoder, and readout heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Normal


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


def build_symexp_twohot_bins(num_bins: int, symexp_span: float = 20.0) -> torch.Tensor:
    """Bin centers matching DreamerV3 ``Head.symexp_twohot`` (embodied/jax/heads.py)."""
    if num_bins % 2 == 1:
        half = torch.linspace(-symexp_span, 0, (num_bins - 1) // 2 + 1)
        half = symexp(half)
        return torch.cat([half, -half[:-1].flip(0)])
    half = torch.linspace(-symexp_span, 0, num_bins // 2)
    half = symexp(half)
    return torch.cat([half, -half.flip(0)])


class SymExpTwoHotEncoder(nn.Module):
    """Two-hot targets on symexp-spaced bins (DreamerV3 ``outs.TwoHot``)."""

    def __init__(self, num_bins: int = 255, symexp_span: float = 20.0):
        super().__init__()
        self.num_bins = int(num_bins)
        bins = build_symexp_twohot_bins(self.num_bins, symexp_span)
        self.register_buffer("bins", bins)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Scalar values -> two-hot targets (..., num_bins)."""
        values = values.float()
        flat = values.reshape(-1)
        bins = self.bins
        k = bins.numel()
        below = (bins.unsqueeze(0) <= flat.unsqueeze(1)).sum(dim=1) - 1
        above = k - (bins.unsqueeze(0) > flat.unsqueeze(1)).sum(dim=1)
        below = below.clamp(0, k - 1)
        above = above.clamp(0, k - 1)
        equal = below == above
        b_below = bins[below]
        b_above = bins[above]
        dist_below = torch.where(equal, torch.ones_like(flat), (b_below - flat).abs())
        dist_above = torch.where(equal, torch.ones_like(flat), (b_above - flat).abs())
        total = dist_below + dist_above
        w_below = dist_above / total
        w_above = dist_below / total
        target = F.one_hot(below, k).float() * w_below.unsqueeze(-1)
        target = target + F.one_hot(above, k).float() * w_above.unsqueeze(-1)
        return target.view(*values.shape, k)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Symmetric expectation over bins (DreamerV3 ``TwoHot.pred``)."""
        probs = F.softmax(logits.float(), dim=-1)
        bins = self.bins.to(dtype=probs.dtype, device=probs.device)
        n = probs.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = probs[..., :m], probs[..., m : m + 1], probs[..., m + 1 :]
            b1, b2, b3 = bins[:m], bins[m : m + 1], bins[m + 1 :]
            return (p2 * b2).sum(dim=-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(dim=-1)
        p1, p2 = probs[..., : n // 2], probs[..., n // 2 :]
        b1, b2 = bins[: n // 2], bins[n // 2 :]
        return ((p1 * b1).flip(-1) + (p2 * b2)).sum(dim=-1)

    def cross_entropy(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        target = self.encode(values)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -(target * log_probs).sum(dim=-1)


class SymExpTwoHotHead(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden: int,
        out_shape: tuple[int, ...],
        encoder: SymExpTwoHotEncoder,
        *,
        layers: int = 1,
    ):
        super().__init__()
        self.out_shape = tuple(out_shape)
        out_dim = int(math.prod(self.out_shape))
        self.encoder = encoder
        blocks: list[nn.Module] = []
        dim = d_in
        for _ in range(max(1, layers)):
            blocks.extend([nn.Linear(dim, hidden), nn.SiLU()])
            dim = hidden
        blocks.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        logits = self.net(h_t)
        return logits.view(*h_t.shape[:2], *self.out_shape)

    def loss(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return self.encoder.cross_entropy(logits, values)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(logits)


def _cfg_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


class SquashedGaussianHead(nn.Module):
    """Diagonal Gaussian in pre-tanh space, squashed to [-1, 1] (ref squashed_gaussian)."""

    def __init__(
        self,
        d_in: int,
        hidden: int,
        action_dim: int,
        action_horizon: int,
        *,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.eps = float(eps)
        out = self.action_horizon * self.action_dim * 2
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns squashed mean action, Gaussian mean u, log_std (B, T, L, A)."""
        B, T, _ = h_t.shape
        raw = self.net(h_t).view(B, T, self.action_horizon, self.action_dim, 2)
        mean_u = raw[..., 0]
        log_std = raw[..., 1].clamp(self.log_std_min, self.log_std_max)
        action = torch.tanh(mean_u)
        return action, mean_u, log_std

    def log_prob(
        self,
        mean_u: torch.Tensor,
        log_std: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Log prob of expert actions under squashed Gaussian; sum over action dims -> (...,)."""
        actions = actions.clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(actions)
        std = log_std.exp()
        normal = Normal(mean_u, std)
        log_prob = normal.log_prob(pre_tanh).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - actions.pow(2) + self.eps).sum(dim=-1)
        return log_prob


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


@dataclass
class AgentOutputs:
    action: torch.Tensor
    action_mean_u: torch.Tensor
    action_log_std: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor


class AgentHeads(nn.Module):
    """BC heads: squashed Gaussian policy + symexp twohot reward MTP (DreamerV3 rewhead)."""

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        *,
        action_horizon: int = 8,
        policy_hidden: int = 256,
        reward_hidden: int = 256,
        reward_bins: int = 255,
        reward_symexp_span: float = 20.0,
        reward_layers: int = 1,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)

        self.policy = SquashedGaussianHead(
            d_model,
            policy_hidden,
            self.action_dim,
            self.action_horizon,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        reward_enc = SymExpTwoHotEncoder(num_bins=reward_bins, symexp_span=reward_symexp_span)
        self.reward_head = SymExpTwoHotHead(
            d_model,
            reward_hidden,
            (self.action_horizon, reward_enc.num_bins),
            reward_enc,
            layers=reward_layers,
        )

    def forward(self, h_t: torch.Tensor) -> AgentOutputs:
        action, mean_u, log_std = self.policy(h_t)
        reward_logits = self.reward_head(h_t)
        reward = self.reward_head.decode(reward_logits)
        return AgentOutputs(
            action=action,
            action_mean_u=mean_u,
            action_log_std=log_std,
            reward_logits=reward_logits,
            reward=reward,
        )


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
        self.bc_space_mode = str(raw.get("bc_space_mode", "wm_agent"))
        self.dynamics_space_mode = str(raw.get("dynamics_space_mode", raw.get("space_mode", "wm_dynamics")))
        self.agent_tokens = nn.Parameter(torch.empty(self.n_agent, self.d_model))
        nn.init.normal_(self.agent_tokens, std=0.02)
        heads = _cfg_dict(heads_cfg)
        self.heads = AgentHeads(
            self.d_model,
            self.action_dim,
            action_horizon=int(heads.get("action_horizon", 8)),
            policy_hidden=int(heads.get("policy_hidden", 256)),
            reward_hidden=int(heads.get("reward_hidden", 256)),
            reward_bins=int(heads.get("reward_bins", 255)),
            reward_symexp_span=float(heads.get("reward_symexp_span", 20.0)),
            reward_layers=int(heads.get("reward_layers", 1)),
            log_std_min=float(heads.get("log_std_min", -5.0)),
            log_std_max=float(heads.get("log_std_max", 2.0)),
        )

    def forward(
        self,
        packed_z: torch.Tensor,
        actions: torch.Tensor,
        *,
        space_mode: Optional[str] = None,
    ) -> AgentOutputs:
        B, T = packed_z.shape[:2]
        agent_tokens = self.agent_tokens.view(1, 1, self.n_agent, self.d_model).expand(B, T, -1, -1)
        sigma = torch.zeros(B, T, device=packed_z.device, dtype=torch.float32)
        _, h_agent = self.dynamics(
            actions,
            sigma,
            packed_z,
            agent_tokens=agent_tokens,
            space_mode=space_mode or self.bc_space_mode,
        )
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
    heads: AgentHeads,
    *,
    action_horizon: int,
    action_weight: float = 1.0,
    reward_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    B, T, _ = actions.shape
    device = actions.device
    valid = _valid_future_mask(T, action_horizon, device).float()
    denom = (valid.sum() * B).clamp_min(1.0)

    target_a = _future_targets(actions, action_horizon)
    action_log_prob = heads.policy.log_prob(
        outputs.action_mean_u.float(),
        outputs.action_log_std.float(),
        target_a.float(),
    )
    action_nll = -(action_log_prob * valid).sum() / denom

    target_r = _future_targets(rewards.unsqueeze(-1), action_horizon).squeeze(-1)
    reward_ce = heads.reward_head.loss(outputs.reward_logits.float(), target_r.float())
    reward_ce = (reward_ce * valid).sum() / denom
    reward_mse = (outputs.reward.float() - target_r.float()).pow(2)
    reward_mse = (reward_mse * valid).sum() / denom

    action_mse = (outputs.action.float() - target_a.float()).pow(2).mean(dim=-1)
    action_mse = (action_mse * valid).sum() / denom

    loss = action_weight * action_nll + reward_weight * reward_ce
    metrics = {
        "action_nll": float(action_nll.detach()),
        "action_mse": float(action_mse.detach()),
        "reward_ce": float(reward_ce.detach()),
        "reward_mse": float(reward_mse.detach()),
        "action_out_mean": float(outputs.action.float().mean().detach()),
        "action_out_abs_mean": float(outputs.action.abs().float().mean().detach()),
    }
    return loss, metrics
