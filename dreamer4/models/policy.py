"""BC policy and readout heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributions import Normal

from dreamer4.checkpoint import load_state
from dreamer4.config import config_to_dict
from dreamer4.models.dynamics import DynamicsModel, sample_one_timestep_packed
from dreamer4.models.tokenizer import Tokenizer, build_tokenizer


# MTP slot l at aligned time t predicts action a_{t+l}; from state s_t execute a_{t+1} (slot 1).
POLICY_ENV_ACTION_SLOT = 1


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class SymlogHead(nn.Module):
    """MLP predicting symlog targets; decode with symexp."""

    def __init__(
        self,
        d_in: int,
        hidden: int,
        out_shape: tuple[int, ...] = (),
        *,
        layers: int = 1,
    ):
        super().__init__()
        self.out_shape = tuple(out_shape)
        out_dim = int(math.prod(self.out_shape)) if self.out_shape else 1
        blocks: list[nn.Module] = []
        dim = d_in
        for _ in range(max(1, layers)):
            blocks.extend([nn.Linear(dim, hidden), nn.SiLU()])
            dim = hidden
        blocks.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*blocks)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.net(h)
        if not self.out_shape:
            return out.squeeze(-1)
        return out.view(*h.shape[:2], *self.out_shape)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return symexp(x.float())

    def loss(self, pred: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return (pred.float() - symlog(values.float())).pow(2)


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
    reward_symlog: torch.Tensor
    reward: torch.Tensor


class AgentHeads(nn.Module):
    """BC heads: squashed Gaussian policy + symlog reward MTP."""

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        *,
        action_horizon: int = 8,
        policy_hidden: int = 256,
        reward_hidden: int = 256,
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
        self.reward_head = SymlogHead(
            d_model,
            reward_hidden,
            (self.action_horizon,),
            layers=reward_layers,
        )

    def forward(self, h_t: torch.Tensor) -> AgentOutputs:
        action, mean_u, log_std = self.policy(h_t)
        reward_symlog = self.reward_head(h_t)
        reward = self.reward_head.decode(reward_symlog)
        return AgentOutputs(
            action=action,
            action_mean_u=mean_u,
            action_log_std=log_std,
            reward_symlog=reward_symlog,
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

        self.agent_tokens = nn.Parameter(torch.empty(self.n_agent, self.d_model))
        heads = config_to_dict(heads_cfg)
        self.heads = AgentHeads(
            self.d_model,
            self.action_dim,
            action_horizon=int(heads.get("action_horizon", 8)),
            policy_hidden=int(heads.get("policy_hidden", 256)),
            reward_hidden=int(heads.get("reward_hidden", 256)),
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
            space_mode=space_mode or "wm_agent",
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


def build_policy(
    cfg: DictConfig,
    *,
    tokenizer: Tokenizer | None = None,
    tokenizer_ckpt: str | None = None,
    ckpt: str | None = None,
) -> tuple[Tokenizer, PolicyModel, int, int]:
    """Frozen tokenizer + PolicyModel; returns (tokenizer, model, n_spatial, packing_factor)."""
    if tokenizer is None:
        tokenizer = build_tokenizer(cfg.model.tokenizer, ckpt=tokenizer_ckpt)
        for p in tokenizer.parameters():
            p.requires_grad_(False)

    n_latents = tokenizer.encoder.n_latents
    latent_dim = tokenizer.encoder.bottleneck_proj.out_features
    packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
    model = PolicyModel(
        cfg.model.dynamics,
        n_latents=n_latents,
        latent_dim=latent_dim,
        heads_cfg=cfg.model,
    )
    if ckpt:
        load_state(model, ckpt, prefix="model.")
    return tokenizer, model, n_latents // packing_factor, packing_factor


@dataclass
class ImaginationRollout:
    """Latent imagination rollout from a context window."""

    latents: torch.Tensor  # (B, H, n_spatial, d_spatial)
    hidden: torch.Tensor  # (B, H+1, D) agent states s_0..s_H
    actions: torch.Tensor  # (B, H, A)
    log_prob: torch.Tensor  # (B, H)


def imagine_latent_rollout(
    policy_model: PolicyModel,
    dynamics: nn.Module,
    packed_z_ctx: torch.Tensor,
    actions_ctx: torch.Tensor,
    policy: SquashedGaussianHead,
    horizon: int,
    flow_steps: int,
    *,
    ctx_len: int | None = None,
) -> ImaginationRollout:
    """Roll out H imagined steps in latent space with policy-sampled actions."""
    ctx = packed_z_ctx.shape[1]
    if ctx_len is not None and ctx_len < ctx:
        z_sliding = packed_z_ctx[:, -ctx_len:]
        a_sliding = actions_ctx[:, -ctx_len:]
    else:
        z_sliding = packed_z_ctx
        a_sliding = actions_ctx
        ctx_len = ctx

    z_sliding = z_sliding.float()
    a_sliding = a_sliding.float()

    with torch.no_grad():
        h_seq = policy_model.agent_hidden(z_sliding, a_sliding)
    h = h_seq[:, -1]

    imagined_latents: list[torch.Tensor] = []
    imagined_actions: list[torch.Tensor] = []
    imagined_log_prob: list[torch.Tensor] = []
    imagined_hidden: list[torch.Tensor] = [h]

    for _ in range(horizon):
        if not torch.isfinite(h).all():
            raise RuntimeError("non-finite agent hidden before policy sample in imagination rollout")
        h_in = h.unsqueeze(1)
        # Sampling under no_grad; PMPO re-evaluates log_prob on fixed actions in imagination_rl_policy_loss.
        with torch.no_grad():
            action_mtp, log_p_mtp, _, _ = policy.sample(h_in)
        slot = POLICY_ENV_ACTION_SLOT
        action = action_mtp[:, 0, slot]
        log_p = log_p_mtp[:, 0, slot]
        imagined_actions.append(action)
        imagined_log_prob.append(log_p)

        actions_step = torch.cat([a_sliding, action.unsqueeze(1)], dim=1)
        with torch.no_grad():
            z_next = sample_one_timestep_packed(
                dynamics,
                z_sliding,
                actions_step,
                flow_steps,
            )
        imagined_latents.append(z_next)
        z_sliding = torch.cat([z_sliding, z_next.unsqueeze(1)], dim=1)
        if z_sliding.shape[1] > ctx_len:
            z_sliding = z_sliding[:, -ctx_len:]
        a_sliding = torch.cat([a_sliding, action.unsqueeze(1)], dim=1)
        if a_sliding.shape[1] > ctx_len:
            a_sliding = a_sliding[:, -ctx_len:]
        with torch.no_grad():
            h = policy_model.agent_hidden(z_sliding, a_sliding)[:, -1]

        imagined_hidden.append(h)

    return ImaginationRollout(
        latents=torch.stack(imagined_latents, dim=1),
        hidden=torch.stack(imagined_hidden, dim=1),
        actions=torch.stack(imagined_actions, dim=1),
        log_prob=torch.stack(imagined_log_prob, dim=1),
    )


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
    reward_symlog_mse = heads.reward_head.loss(outputs.reward_symlog.float(), target_r.float())
    reward_symlog_mse = (reward_symlog_mse * valid).sum() / denom
    reward_dec_mse = (outputs.reward.float() - target_r.float()).pow(2)
    reward_dec_mse = (reward_dec_mse * valid).sum() / denom

    action_mse = (outputs.action.float() - target_a.float()).pow(2).mean(dim=-1)
    action_mse = (action_mse * valid).sum() / denom

    loss = action_weight * action_nll + reward_weight * reward_symlog_mse
    metrics = {
        "action_nll": float(action_nll.detach()),
        "action_mse": float(action_mse.detach()),
        "reward_symlog_mse": float(reward_symlog_mse.detach()),
        "reward_dec_mse": float(reward_dec_mse.detach()),
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
    value_head: SymlogHead,
    *,
    gamma: float,
    lambda_: float,
    normalize_advantages: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Value symlog MSE on TD-λ targets; returns (val_loss, advantages, value metrics)."""
    h = hidden.detach()
    H = imagined_actions.shape[1]

    reward_symlog = heads.reward_head(h[:, 1:])
    rewards = heads.reward_head.decode(reward_symlog[:, :, 0])

    val_symlog = value_head(h)
    values = value_head.decode(val_symlog)

    td_returns = td_lambda_returns(rewards, values, gamma, lambda_)
    val_loss = value_head.loss(val_symlog[:, :-1], td_returns).mean()

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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """PMPO policy + KL loss on imagined trajectories; returns (pi_loss, kl_loss, metrics)."""
    slot = POLICY_ENV_ACTION_SLOT
    imagined_actions_f = imagined_actions.detach().float()

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

    metrics: dict[str, float] = {
        "pi_loss": float(pi_loss.detach()),
        "pi_kl_loss": float(kl_loss.detach()),
    }
    return pi_loss, kl_loss, metrics


def imagination_rl_loss(
    hidden: torch.Tensor,
    imagined_actions: torch.Tensor,
    heads: AgentHeads,
    policy_prior: SquashedGaussianHead,
    value_head: SymlogHead,
    *,
    gamma: float,
    lambda_: float,
    beta: float,
    alpha: float = 0.5,
    normalize_advantages: bool = False,
    policy_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Value symlog MSE on TD-λ targets + PMPO policy loss + KL(π || π_BC) on imagined trajectories.

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
    )
    metrics.update(policy_metrics)

    total = val_loss + policy_weight * (pi_loss + kl_loss)
    return total, metrics
