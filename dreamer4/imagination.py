"""Latent imagination rollouts for policy RL (frozen dynamics, learned policy actions)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dreamer4.models.dynamics import sample_one_timestep_packed
from dreamer4.models.policy import PolicyModel, POLICY_ENV_ACTION_SLOT, SquashedGaussianHead


@dataclass
class ImaginationRollout:
    hidden: torch.Tensor
    actions: torch.Tensor
    log_prob: torch.Tensor


def imagine_latent_rollout(
    policy_model: PolicyModel,
    dynamics: nn.Module,
    packed_z_ctx: torch.Tensor,
    actions_ctx: torch.Tensor,
    policy: SquashedGaussianHead,
    horizon: int,
    flow_steps: int,
    *,
    bc_space_mode: str,
    ctx_len: int | None = None,
) -> ImaginationRollout:
    """
    Roll out H imagined steps in latent space with policy-sampled actions.

    Context: packed_z_ctx (B, ctx, n_spatial, d_spatial), actions_ctx (B, ctx, A).
    Returns hidden (B, H+1, D), actions (B, H, A), log_prob (B, H).
    """
    ctx = packed_z_ctx.shape[1]
    if ctx_len is not None and ctx_len < ctx:
        z_sliding = packed_z_ctx[:, -ctx_len:]
        a_sliding = actions_ctx[:, -ctx_len:]
    else:
        z_sliding = packed_z_ctx
        a_sliding = actions_ctx
        ctx_len = ctx

    with torch.no_grad():
        h_seq = policy_model.agent_hidden(z_sliding, a_sliding, space_mode=bc_space_mode)
    h = h_seq[:, -1]

    imagined_actions: list[torch.Tensor] = []
    imagined_log_prob: list[torch.Tensor] = []
    imagined_hidden: list[torch.Tensor] = [h]

    for _ in range(horizon):
        h_in = h.detach().unsqueeze(1)
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
            z_sliding = torch.cat([z_sliding, z_next.unsqueeze(1)], dim=1)
            if z_sliding.shape[1] > ctx_len:
                z_sliding = z_sliding[:, -ctx_len:]
            a_sliding = torch.cat([a_sliding, action.unsqueeze(1)], dim=1)
            if a_sliding.shape[1] > ctx_len:
                a_sliding = a_sliding[:, -ctx_len:]
            h = policy_model.agent_hidden(z_sliding, a_sliding, space_mode=bc_space_mode)[:, -1]

        imagined_hidden.append(h)

    return ImaginationRollout(
        hidden=torch.stack(imagined_hidden, dim=1),
        actions=torch.stack(imagined_actions, dim=1),
        log_prob=torch.stack(imagined_log_prob, dim=1),
    )
