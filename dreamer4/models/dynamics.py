from __future__ import annotations

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


def flow_matching_loss(
    model: nn.Module,
    z1: torch.Tensor,
    actions: torch.Tensor,
    *,
    space_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """
    Simple flow matching on packed latents: corrupt with noise level sigma, predict clean z1.
    z_tilde = (1-sigma)*z0 + sigma*z1, target z1_hat ~= z1, weight (0.9*sigma + 0.1).
    """
    B, T = z1.shape[:2]
    device = z1.device
    sigma = torch.rand((B, T), device=device, dtype=torch.float32)
    z0 = torch.randn_like(z1)
    z_tilde = (1.0 - sigma)[..., None, None] * z0 + sigma[..., None, None] * z1
    z1_hat, _ = model(actions, sigma, z_tilde, space_mode=space_mode)
    flow_per = (z1_hat.float() - z1.float()).pow(2).mean(dim=(2, 3))
    weight = 0.9 * sigma + 0.1
    loss = (flow_per * weight).mean()
    metrics = {
        "flow_mse": float(flow_per.mean().detach()),
        "sigma_mean": float(sigma.mean().detach()),
    }
    return loss, metrics


class DynamicsModel(nn.Module):
    """Action-conditioned flow model on packed tokenizer latents (no shortcut forcing)."""

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
        # space_modes: masks registered on the transformer; space_mode: default when forward() omits it
        # (flow rollout / sample_one_timestep_packed never pass space_mode — they use space_mode).
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

        self.noise_mlp = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        segments = [
            (Modality.ACTION, 1),
            (Modality.NOISE, 1),
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

        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        actions: torch.Tensor,
        sigma: torch.Tensor,
        packed_z: torch.Tensor,
        agent_tokens: Optional[torch.Tensor] = None,
        *,
        space_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict clean packed latents. sigma: (B,T) in [0,1]. Returns (x1_hat, h_t)."""
        B, T = packed_z.shape[:2]
        spatial_tokens = self.spatial_proj(packed_z)
        action_tokens = self.action_encoder(actions).unsqueeze(2) + self.action_base.view(
            1, 1, 1, -1
        )
        noise_tokens = self.noise_mlp(sigma[..., None]).unsqueeze(2)

        tokens = [action_tokens, noise_tokens, spatial_tokens]
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
    """Sample one packed latent frame conditioned on clean past latents and actions."""
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    z = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    dt = 1.0 / flow_steps

    for i in range(flow_steps):
        tau_i = i / flow_steps
        z_tilde = torch.cat([past_packed, z], dim=1)
        sigma = torch.ones(B, t + 1, device=device, dtype=torch.float32)
        sigma[:, -1] = tau_i
        z1_hat, _ = model(actions[:, : t + 1], sigma, z_tilde)
        x1_hat = z1_hat[:, -1:]
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
