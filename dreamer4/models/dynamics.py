from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dreamer4.config import config_to_dict

from dreamer4.models.tokenizer import temporal_unpatchify
from dreamer4.models.transformer_blocks import (
    BlockCausalTransformer,
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
)


def pack_bottleneck_to_spatial(z_btld: torch.Tensor, n_spatial: int, k: int) -> torch.Tensor:
    """(B,T,L,D) with L = n_spatial * k -> (B,T,n_spatial,k*D)."""
    B, T, L, D = z_btld.shape
    assert L == n_spatial * k
    return z_btld.view(B, T, n_spatial, k * D)


def unpack_spatial_to_bottleneck(z_btsd: torch.Tensor, k: int) -> torch.Tensor:
    """(B,T,n_spatial,k*D) -> (B,T,n_spatial*k,D)."""
    B, T, S, DK = z_btsd.shape
    D = DK // k
    return z_btsd.view(B, T, S * k, D)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be a power of two"
    return emax


def _sample_step_excluding_dmin(
    device: torch.device, B: int, T: int, k_max: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coarser shortcut step sizes (excludes finest d_min)."""
    emax = _emax_from_kmax(k_max)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(B, T), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(
    device: torch.device, B: int, T: int, k_max: int, step_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample signal level tau and discrete signal index on the flow grid for step_idx."""
    K = (1 << step_idx).to(torch.long)
    u = torch.rand((B, T), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * K.to(torch.float32)).to(torch.long) # index of tau on the corresponding coarse grid
    tau = j_idx.to(torch.float32) / K.to(torch.float32) # tau ∈ [0, 1]
    scale = torch.div(torch.tensor(k_max, device=device), K, rounding_mode="floor")
    tau_idx = j_idx * scale # compute the index of tau on the corresponding fine grid
    return tau, tau_idx


def make_tau_schedule(*, k_max: int, flow_steps: int) -> dict[str, Any]:
    """
    Integration grid for one generated frame.
    flow_steps is the number of forward passes (K); must divide k_max evenly.
    """
    assert _is_pow2(k_max), "k_max must be a power of two"
    K = int(flow_steps)
    assert K > 0 and k_max % K == 0, f"k_max={k_max} must be divisible by flow_steps={K}"
    e = int(round(math.log2(K)))
    assert (1 << e) == K, "flow_steps must be a power of two"
    scale = k_max // K
    tau = [i / K for i in range(K)]
    tau_idx = [i * scale for i in range(K)]
    return dict(K=K, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / K)


def shortcut_forcing_loss(
    model: nn.Module,
    z1: torch.Tensor,
    actions: torch.Tensor,
    *,
    k_max: int,
    B_self: int = 0,
    global_step: int = 0,
    bootstrap_start: int = 0,
    space_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    Paper-style dynamics pretrain loss: finest-step flow matching + bootstrap self-consistency.

    First B - B_self rows use d_min (flow branch); last B_self rows use coarser steps (bootstrap).
    """
    device = z1.device
    B, T = z1.shape[:2]
    if global_step < bootstrap_start:
        B_self = 0
    else:
        B_self = max(0, min(int(B_self), B - 1))
    B_emp = B - B_self
    emax = _emax_from_kmax(k_max)

    step_idx_emp = torch.full((B_emp, T), emax, device=device, dtype=torch.long) # B_emp uses finest step
    d_self = torch.zeros((0, T), device=device, dtype=torch.float32) # d_self ∈ [0, 1]
    step_idx_self = torch.zeros((0, T), device=device, dtype=torch.long) # step_idx_self ∈ [0, emax-1]
    if B_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, B_self, T, k_max) # step_idx_self ∈ [0, emax-1]
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        step_idx_full = step_idx_emp

    sigma_full, sigma_idx_full = _sample_tau_for_step(device, B, T, k_max, step_idx_full)
    sigma_emp = sigma_full[:B_emp]
    sigma_self = sigma_full[B_emp:]
    sigma_idx_self = sigma_idx_full[B_emp:]

    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[B_emp:]

    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    z1_hat_full, _ = model(
        actions, step_idx_full, sigma_idx_full, z_tilde_full, space_mode=space_mode
    ) # directly predict the next step (full step, no half)
    z1_hat_emp = z1_hat_full[:B_emp]
    z1_hat_self = z1_hat_full[B_emp:]

    flow_per = (z1_hat_emp.float() - z1[:B_emp].float()).pow(2).mean(dim=(2, 3))
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)

    if B_self > 0:
        d_half = d_self / 2.0 # half step model
        step_idx_half = step_idx_self + 1 # d = 1 / 2^(step_idx_half)
        sigma_plus = sigma_self + d_half # progress towards the next step
        sigma_idx_plus = sigma_idx_self + (k_max * d_half).to(torch.long)

        actions_self = actions[B_emp:]
        z1_hat_half1, _ = model(
            actions_self, step_idx_half, sigma_idx_self, z_tilde_self, space_mode=space_mode
        ) # current step predict half step forward
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None] # velocity of the current step, v = (x1_hat - z_tilde) / (1 - sigma)
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None] # prediction of the next step

        z1_hat_half2, _ = model(
            actions_self,
            step_idx_half,
            sigma_idx_plus,
            z_prime.to(z_tilde_self.dtype),
            space_mode=space_mode,
        ) # the half step from z_prime, predict the next step
        b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (
            1.0 - sigma_plus
        ).clamp_min(1e-6)[..., None, None] # velocity for the second half

        vhat = (z1_hat_self.float() - z_tilde_self.float()) / (
            1.0 - sigma_self
        ).clamp_min(1e-6)[..., None, None] # velocity of the full step (coarse one)
        v_target = ((b_prime + b_doubleprime) / 2.0).detach() # average velocity of the two half steps as bootstrap target

        boot_per = (1.0 - sigma_self).pow(2) * (vhat - v_target).pow(2).mean(dim=(2, 3)) # scale back to the x-space, cause v = (x1 - xt) / (1 - t)
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    loss = ((loss_emp * B_emp) + (loss_self * B_self)) / B
    metrics = {
        "flow_mse": float(flow_per.mean().detach()),
        "bootstrap_mse": float(boot_mse.detach()),
        "loss_emp": float(loss_emp.detach()),
        "loss_self": float(loss_self.detach()),
        "sigma_mean": float(sigma_full.mean().detach()),
    }
    return loss, metrics


def flow_matching_loss(
    model: nn.Module,
    z1: torch.Tensor,
    actions: torch.Tensor,
    *,
    k_max: int | None = None,
    space_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Finest-step shortcut loss only (no bootstrap)."""
    if k_max is None:
        k_max = int(getattr(model, "k_max"))
    return shortcut_forcing_loss(
        model,
        z1,
        actions,
        k_max=k_max,
        B_self=0,
        global_step=0,
        bootstrap_start=0,
        space_mode=space_mode,
    )


class DynamicsModel(nn.Module):
    """Action-conditioned shortcut flow model on packed tokenizer latents."""

    def __init__(
        self,
        cfg: Mapping[str, Any] | DictConfig,
        *,
        n_latents: int,
        latent_dim: int,
    ):
        super().__init__()
        raw = config_to_dict(cfg)
        self.d_model = int(raw["embed_dim"])
        self.n_heads = int(raw["num_heads"])
        self.depth = int(raw["depth"])
        self.mlp_ratio = float(raw.get("mlp_ratio", 4.0))
        self.dropout = float(raw.get("dropout", 0.0))
        self.time_every = int(raw.get("time_every", 4))
        self.scale_pos_embeds = bool(raw.get("scale_pos_embeds", True))
        space_modes_raw = raw.get("space_modes")
        if space_modes_raw is not None:
            self.space_modes = tuple(str(m) for m in space_modes_raw)
            self.space_mode = str(raw.get("space_mode", self.space_modes[0]))
        else:
            self.space_mode = str(raw.get("space_mode", "wm_dynamics"))
            self.space_modes = (self.space_mode,)
        if self.space_mode not in self.space_modes:
            raise ValueError(
                f"dynamics.space_mode {self.space_mode!r} must be listed in space_modes {self.space_modes}"
            )
        self.packing_factor = int(raw.get("packing_factor", 1))
        self.n_register = int(raw.get("n_register", 0))
        self.n_agent = int(raw.get("n_agent", 1))
        self.action_dim = int(raw.get("action_dim", 6))

        self.k_max = int(raw.get("k_max", 64))
        assert _is_pow2(self.k_max), f"k_max must be a power of two, got {self.k_max}"
        self.emax = _emax_from_kmax(self.k_max)
        self.num_step_bins = self.emax + 1

        assert n_latents % self.packing_factor == 0
        self.n_spatial = n_latents // self.packing_factor
        self.d_spatial = latent_dim * self.packing_factor

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))

        self.action_base = nn.Parameter(torch.empty(self.d_model))
        action_hidden = int(self.d_model * 2.0)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, action_hidden),
            nn.SiLU(),
            nn.Linear(action_hidden, self.d_model),
        )

        self.step_embed = nn.Embedding(self.num_step_bins, self.d_model) # how long I should move forward, towards the next step
        self.signal_embed = nn.Embedding(self.k_max + 1, self.d_model) # where am I? current step

        segments = [
            (Modality.ACTION, 1),
            (Modality.NOISE, 1),
            (Modality.STEP, 1),
            (Modality.SPATIAL, self.n_spatial),
        ]
        if self.n_register > 0:
            segments.append((Modality.REGISTER, self.n_register))
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            n_latents=0,
            modality_ids=layout.modality_ids(),
            space_mode=self.space_modes,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=False,
        )

        self.flow_head = nn.Linear(self.d_model, self.d_spatial)
        self._init_weights()

    def _init_weights(self) -> None:
        if self.n_register > 0:
            nn.init.normal_(self.register_tokens, std=0.02)

        nn.init.normal_(self.action_base, std=0.02)
        action_last = self.action_encoder[2]
        assert isinstance(action_last, nn.Linear)
        nn.init.normal_(action_last.weight, std=1e-3)
        nn.init.zeros_(action_last.bias)

        nn.init.normal_(self.step_embed.weight, std=0.02)
        nn.init.normal_(self.signal_embed.weight, std=0.02)

        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def clean_conditioning(
        self, batch_time: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step/signal indices for clean latents (BC inference)."""
        B, T = batch_time
        step_idx = torch.full((B, T), self.emax, device=device, dtype=torch.long)
        signal_idx = torch.full((B, T), self.k_max, device=device, dtype=torch.long)
        return step_idx, signal_idx

    def forward(
        self,
        actions: torch.Tensor,
        step_idx: torch.Tensor,
        signal_idx: torch.Tensor,
        packed_z: torch.Tensor,
        agent_tokens: Optional[torch.Tensor] = None,
        *,
        space_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict clean packed latents. Returns (x1_hat, h_t)."""
        B, T = packed_z.shape[:2]
        spatial_tokens = self.spatial_proj(packed_z)
        action_tokens = self.action_encoder(actions).unsqueeze(2) + self.action_base.view(
            1, 1, 1, -1
        )
        signal_tokens = self.signal_embed(signal_idx.to(torch.long)).unsqueeze(2)
        step_tokens = self.step_embed(step_idx.to(torch.long)).unsqueeze(2)

        tokens = [action_tokens, signal_tokens, step_tokens, spatial_tokens]
        if self.n_register > 0:
            reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)
            tokens.append(reg)
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    (B, T, self.n_agent, self.d_model),
                    device=spatial_tokens.device,
                    dtype=spatial_tokens.dtype,
                )
            tokens.append(agent_tokens)

        x = torch.cat(tokens, dim=2)
        x = add_sinusoidal_positions(x, self.scale_pos_embeds)
        x = self.transformer(x, space_mode=space_mode)
        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_head(spatial_out)
        h_t = x[:, :, self.agent_slice, :] if self.n_agent > 0 else None
        return x1_hat, h_t


@torch.no_grad()
def sample_one_timestep_packed(
    model: nn.Module,
    past_packed: torch.Tensor,
    actions: torch.Tensor,
    flow_steps: int,
) -> torch.Tensor:
    """Sample one packed latent frame with K=flow_steps shortcut integration steps."""
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]
    k_max = int(getattr(model, "k_max"))
    emax = int(getattr(model, "emax"))
    sched = make_tau_schedule(k_max=k_max, flow_steps=flow_steps)

    K = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])

    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    step_idxs = torch.full((B, t + 1), emax, device=device, dtype=torch.long)
    step_idxs[:, -1] = e
    signal_idxs = torch.full((B, t + 1), k_max, device=device, dtype=torch.long)

    for i in range(K):
        signal_idxs[:, -1] = int(tau_idx[i])
        z_tilde = torch.cat([past_packed, z], dim=1)
        z1_hat, _ = model(actions[:, : t + 1], step_idxs, signal_idxs, z_tilde)
        x1_hat = z1_hat[:, -1:]
        tau_i = float(tau[i])
        denom = max(1e-4, 1.0 - tau_i)
        velocity = (x1_hat.float() - z.float()) / denom
        z = (z.float() + velocity * dt).to(dtype)

    return z[:, 0]


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    model: nn.Module,
    z_gt_packed: torch.Tensor,
    actions: torch.Tensor,
    ctx_length: int,
    horizon: int,
    flow_steps: int,
) -> torch.Tensor:
    """Action-conditioned autoregressive rollout: GT context latents, dataset actions, predicted horizon latents."""
    B, T = z_gt_packed.shape[:2]
    length = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, length - 1)
    horizon = min(horizon, length - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]
    for t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)
        z_next = sample_one_timestep_packed(model, past, actions, flow_steps)
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def sample_sliding_window_rollout_packed_sequence(
    model: nn.Module,
    z0_packed: torch.Tensor,
    actions: torch.Tensor,
    attn_window: int,
    rollout_length: int,
    flow_steps: int,
) -> torch.Tensor:
    """
    Rollout from a single GT frame (obs[0]) using dataset actions.

    For global step g (predicting frame g), past latents are z[0:g]. While g <= attn_window,
    the model attends to all past frames; afterward it attends only to the previous attn_window
    frames, with actions aligned to the same global indices (ref interactive ctx_window).
    """
    if rollout_length <= 0:
        raise ValueError(f"rollout_length must be > 0, got {rollout_length}")
    if attn_window <= 0:
        raise ValueError(f"attn_window must be > 0, got {attn_window}")
    if actions.shape[1] < rollout_length + 1:
        raise ValueError(
            f"actions must have length >= rollout_length + 1 (aligned), "
            f"got {actions.shape[1]} for rollout_length={rollout_length}"
        )

    outs = [z0_packed]
    for _ in range(rollout_length):
        g = len(outs)
        start = 0 if g <= attn_window else g - attn_window
        past = torch.stack(outs[start:g], dim=1)
        actions_local = actions[:, start : g + 1]
        z_next = sample_one_timestep_packed(model, past, actions_local, flow_steps)
        outs.append(z_next)

    return torch.stack(outs, dim=1)


@torch.no_grad()
def decode_packed_to_images(
    tokenizer: nn.Module,
    z_packed: torch.Tensor,
    patch_size: int,
    packing_factor: int,
    image_size: int,
    channels: int,
) -> torch.Tensor:
    """(B,T,n_spatial,d_spatial) -> (B,T,H,W,C) in [0,1]."""
    z_btld = unpack_spatial_to_bottleneck(z_packed, packing_factor)
    patches = tokenizer.decoder(z_btld)
    frames = temporal_unpatchify(patches, image_size, image_size, channels, patch_size)
    return frames.permute(0, 1, 3, 4, 2).clamp(0, 1)
