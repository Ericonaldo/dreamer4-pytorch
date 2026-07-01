"""BC policy and readout heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributions import Normal

from dreamer4.config import config_to_dict
from dreamer4.models.dynamics import DynamicsModel


# MTP slot l at aligned time t predicts action a_{t+l}; from state s_t execute a_{t+1} (slot 1).
POLICY_ENV_ACTION_SLOT = 1


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
        self._init_weights()

    def _init_weights(self) -> None:
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        logits = self.net(h_t)
        return logits.view(*h_t.shape[:2], *self.out_shape)

    def loss(self, logits: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return self.encoder.cross_entropy(logits, values)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(logits)


def init_value_head_from_reward_head(
    value_head: SymExpTwoHotHead,
    reward_head: SymExpTwoHotHead,
) -> None:
    """Warm-start value predictions from BC reward head MTP slot 0."""
    with torch.no_grad():
        rh_net = reward_head.net
        vh_net = value_head.net
        for i in range(len(vh_net) - 1):
            vh_net[i].load_state_dict(rh_net[i].state_dict())
        n_bins = value_head.encoder.num_bins
        vh_net[-1].weight.copy_(rh_net[-1].weight[:n_bins])
        vh_net[-1].bias.copy_(rh_net[-1].bias[:n_bins])


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
        # Linear may run in bf16 under mixed precision; distribution math stays fp32.
        raw = self.net(h_t).view(B, T, self.action_horizon, self.action_dim, 2).float()
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
        mean_u = mean_u.float()
        log_std = log_std.float()
        actions = actions.float().clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = torch.atanh(actions)
        std = log_std.exp()
        normal = Normal(mean_u, std)
        log_prob = normal.log_prob(pre_tanh).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - actions.pow(2) + self.eps).sum(dim=-1)
        return log_prob

    def sample(
        self, h_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stochastic squashed action + log prob per MTP slot (B, T, L, A)."""
        _, mean_u, log_std = self.forward(h_t)
        std = log_std.exp()
        u = mean_u + std * torch.randn_like(mean_u)
        action = torch.tanh(u)
        log_prob = self.log_prob(mean_u, log_std, action)
        return action, log_prob, mean_u, log_std

    def gaussian_kl(
        self,
        mean_u: torch.Tensor,
        log_std: torch.Tensor,
        mean_u_prior: torch.Tensor,
        log_std_prior: torch.Tensor,
    ) -> torch.Tensor:
        """KL between diagonal Gaussians in pre-tanh space; sum over action dims."""
        mean_u = mean_u.float()
        log_std = log_std.float()
        mean_u_prior = mean_u_prior.float()
        log_std_prior = log_std_prior.float()
        var = (log_std.exp()) ** 2
        var_prior = (log_std_prior.exp()) ** 2
        return 0.5 * (
            2 * (log_std_prior - log_std)
            + (var + (mean_u - mean_u_prior).pow(2)) / var_prior
            - 1
        ).sum(dim=-1)


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


class PolicyModel(nn.Module):
    """Dynamics backbone (wm_agent) + learned agent tokens + agent heads."""

    def __init__(
        self,
        dynamics_cfg: Mapping[str, Any] | DictConfig,
        *,
        n_latents: int,
        latent_dim: int,
        heads_cfg: Mapping[str, Any] | DictConfig,
    ):
        super().__init__()

        raw = config_to_dict(dynamics_cfg)
        self.action_dim = int(raw["action_dim"])
        self.n_agent = int(raw.get("n_agent", 1))
        self.d_model = int(raw["embed_dim"])

        self.dynamics = DynamicsModel(raw, n_latents=n_latents, latent_dim=latent_dim)

        self.dynamics_space_mode = str(
            raw.get("dynamics_space_mode", self.dynamics.space_mode)
        )
        self.bc_space_mode = str(raw.get("bc_space_mode", "wm_agent"))
        self.agent_tokens = nn.Parameter(torch.empty(self.n_agent, self.d_model))
        heads = config_to_dict(heads_cfg)
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
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.agent_tokens, std=0.02)

    def agent_hidden(
        self,
        packed_z: torch.Tensor,
        actions: torch.Tensor,
        *,
        space_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Pooled agent hidden states (B, T, d_model) from clean latents and actions."""
        B, T = packed_z.shape[:2]
        agent_tokens = self.agent_tokens.view(1, 1, self.n_agent, self.d_model).expand(B, T, -1, -1)
        step_idx, signal_idx = self.dynamics.clean_conditioning((B, T), packed_z.device)
        _, h_agent = self.dynamics(
            actions,
            step_idx,
            signal_idx,
            packed_z,
            agent_tokens=agent_tokens,
            space_mode=space_mode or self.bc_space_mode,
        )
        if h_agent is None:
            raise ValueError("PolicyModel requires n_agent > 0 on dynamics backbone")
        return h_agent.mean(dim=2) # mean over n_agent tokens

    def forward(
        self,
        packed_z: torch.Tensor,
        actions: torch.Tensor,
        *,
        space_mode: Optional[str] = None,
    ) -> AgentOutputs:
        h_t = self.agent_hidden(packed_z, actions, space_mode=space_mode)
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


def td_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """
    TD-λ returns on imagined rewards (Walker: no terminal mask, c_t = 1).

    rewards: (B, H) predicted r_1..r_H
    values:  (B, H+1) value estimates v_0..v_H
    """
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    g_next = values[:, -1].detach()
    for t in reversed(range(H)):
        v_next = values[:, t + 1].detach()
        g_next = rewards[:, t] + gamma * ((1.0 - lambda_) * v_next + lambda_ * g_next.detach())
        returns[:, t] = g_next
    return returns


def pmpo_policy_loss(
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """PMPO: sign-only advantages; alpha balances positive vs negative action sets.

    Ref: Dreamerv4 paper, https://github.com/edwhu/dreamer4-jax
    """
    flat_lp = log_prob.reshape(-1)
    flat_adv = advantages.reshape(-1)
    mask_pos = flat_adv >= 0
    mask_neg = flat_adv < 0
    n_pos = int(mask_pos.sum().item())
    n_neg = int(mask_neg.sum().item())
    loss_neg = (1.0 - alpha) * (flat_lp * mask_neg).sum() / max(n_neg, 1)
    loss_pos = -alpha * (flat_lp * mask_pos).sum() / max(n_pos, 1)
    return loss_neg + loss_pos


def imagination_rl_value_loss(
    hidden: torch.Tensor,
    imagined_actions: torch.Tensor,
    heads: AgentHeads,
    value_head: SymExpTwoHotHead,
    *,
    gamma: float,
    lambda_: float,
    normalize_advantages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Value CE on TD-λ targets; returns (val_loss, advantages, value metrics)."""
    h = hidden.detach()
    H = imagined_actions.shape[1]

    reward_logits = heads.reward_head(h[:, 1:])
    reward_slot0 = reward_logits[:, :, 0]
    rewards = heads.reward_head.decode(reward_slot0)

    val_logits = value_head(h)
    values = value_head.decode(val_logits)

    td_returns = td_lambda_returns(rewards, values, gamma, lambda_)
    val_loss = value_head.loss(val_logits[:, :-1], td_returns).mean()

    advantages = (td_returns - values[:, :-1]).detach().float()
    if normalize_advantages:
        adv_std = advantages.std().clamp_min(1e-8)
        advantages = (advantages - advantages.mean()) / adv_std

    metrics = {
        "val_loss": float(val_loss.detach()),
        "mean_advantage": float(advantages.mean().detach()),
        "mean_td_return": float(td_returns.mean().detach()),
        "mean_reward_pred": float(rewards.mean().detach()),
        "mean_value": float(values.mean().detach()),
    }
    return val_loss, advantages, metrics


def imagination_rl_policy_loss(
    h_pi: torch.Tensor,
    imagined_actions: torch.Tensor,
    advantages: torch.Tensor,
    heads: AgentHeads,
    policy_prior: SquashedGaussianHead,
    *,
    beta: float,
    alpha: float = 0.5,
    train_policy: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """PMPO policy + KL loss on imagined trajectories; returns (pi_loss, kl_loss, metrics)."""
    slot = POLICY_ENV_ACTION_SLOT
    imagined_actions_f = imagined_actions.detach().float()

    policy_ctx = torch.enable_grad() if train_policy else torch.no_grad()
    with policy_ctx:
        _, mean_u, log_std = heads.policy(h_pi)
        if not torch.isfinite(mean_u).all():
            raise RuntimeError("non-finite policy mean_u in imagination_rl_loss")

        log_prob = heads.policy.log_prob(
            mean_u[:, :, slot],
            log_std[:, :, slot],
            imagined_actions_f,
        )
        pi_loss = pmpo_policy_loss(log_prob, advantages, alpha)
        _, mean_u_bc, log_std_bc = policy_prior(h_pi)
        kl = heads.policy.gaussian_kl(
            mean_u[:, :, slot],
            log_std[:, :, slot],
            mean_u_bc[:, :, slot],
            log_std_bc[:, :, slot],
        ).mean()
        kl_loss = beta * kl

    pi_loss_f = float(pi_loss.detach())
    kl_loss_f = float(kl_loss.detach())
    metrics: dict[str, float] = {
        "pi_loss": pi_loss_f if train_policy else 0.0,
        "pi_kl_loss": kl_loss_f if train_policy else 0.0,
    }
    return pi_loss, kl_loss, metrics


def imagination_rl_loss(
    hidden: torch.Tensor,
    imagined_actions: torch.Tensor,
    heads: AgentHeads,
    policy_prior: SquashedGaussianHead,
    value_head: SymExpTwoHotHead,
    *,
    gamma: float,
    lambda_: float,
    beta: float,
    alpha: float = 0.5,
    normalize_advantages: bool = False,
    train_policy: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Value CE on TD-λ targets + PMPO policy loss + KL(π || π_BC) on imagined trajectories.

    hidden: (B, H+1, D) agent states s_0..s_H (s_0 = last context state)
    imagined_actions: (B, H, A) policy actions a_1..a_H (fixed; log_prob recomputed for PMPO)
    """
    h = hidden.detach()
    H = imagined_actions.shape[1]
    h_pi = h[:, :H]

    val_loss, advantages, metrics = imagination_rl_value_loss(
        hidden,
        imagined_actions,
        heads,
        value_head,
        gamma=gamma,
        lambda_=lambda_,
        normalize_advantages=normalize_advantages,
    )
    pi_loss, kl_loss, policy_metrics = imagination_rl_policy_loss(
        h_pi,
        imagined_actions,
        advantages,
        heads,
        policy_prior,
        beta=beta,
        alpha=alpha,
        train_policy=train_policy,
    )
    metrics.update(policy_metrics)

    total = val_loss + (pi_loss + kl_loss if train_policy else 0.0)
    return total, metrics
