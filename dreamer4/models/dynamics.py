from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.models.action_encoder import ActionEncoder
from dreamer4.models.tokenizer import encode_images, temporal_unpatchify
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
    z1_hat, _ = model(actions, sigma, z_tilde)
    flow_per = (z1_hat.float() - z1.float()).pow(2).mean(dim=(2, 3))
    weight = 0.9 * sigma + 0.1
    loss = (flow_per * weight).mean()
    metrics = {
        "flow_mse": float(flow_per.mean().detach()),
        "sigma_mean": float(sigma.mean().detach()),
    }
    return loss, metrics


def _cfg_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


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
        raw = _cfg_dict(cfg)
        self.d_model = int(raw["embed_dim"])
        self.n_heads = int(raw["num_heads"])
        self.depth = int(raw["depth"])
        self.mlp_ratio = float(raw.get("mlp_ratio", 4.0))
        self.dropout = float(raw.get("dropout", 0.0))
        self.time_every = int(raw.get("time_every", 4))
        self.scale_pos_embeds = bool(raw.get("scale_pos_embeds", True))
        self.space_mode = str(raw.get("space_mode", "wm_dynamics"))
        self.packing_factor = int(raw.get("packing_factor", 1))
        self.n_register = int(raw.get("n_register", 0))
        self.n_agent = int(raw.get("n_agent", 1))
        self.action_dim = int(raw.get("action_dim", 6))

        assert n_latents % self.packing_factor == 0
        self.n_spatial = n_latents // self.packing_factor
        self.d_spatial = latent_dim * self.packing_factor

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.action_encoder = ActionEncoder(self.d_model, self.action_dim)
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
            space_mode=self.space_mode,
            dropout=self.dropout,
            mlp_ratio=self.mlp_ratio,
            time_every=self.time_every,
            latents_only_time=False,
        )

        self.flow_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(
        self,
        actions: torch.Tensor,
        sigma: torch.Tensor,
        packed_z: torch.Tensor,
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict clean packed latents. sigma: (B,T) in [0,1]. Returns (x1_hat, h_t)."""
        B, T = packed_z.shape[:2]
        spatial_tokens = self.spatial_proj(packed_z)
        action_tokens = self.action_encoder(actions)
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
        x = self.transformer(x)
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


def rollout_panel_uint8(
    gt_bthwc: torch.Tensor,
    pred_bthwc: torch.Tensor,
    ctx_length: int,
    max_items: int = 4,
    gap_px: int = 16,
) -> np.ndarray:
    """Tile GT and Pred over time; rows=GT/Pred, vertical gap between context and horizon."""
    T_match = min(gt_bthwc.shape[1], pred_bthwc.shape[1])
    gt = gt_bthwc[:, :T_match].permute(0, 1, 4, 2, 3)
    pred = pred_bthwc[:, :T_match].permute(0, 1, 4, 2, 3)
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)
    ctx = int(max(0, min(ctx_length, T)))

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)
        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            total_w = T_ * W_
            if split < total_w:
                left = y[..., :split]
                right = y[..., split:]
                gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
                y = torch.cat([left, gap, right], dim=3)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)
    panel = torch.cat([gt_t, pr_t], dim=2)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)
    return (big.clamp(0, 1) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def run_dynamics_rollout_eval(
    dynamics: nn.Module,
    tokenizer: nn.Module,
    image_bthwc: torch.Tensor,
    actions: torch.Tensor,
    *,
    patch_size: int,
    packing_factor: int,
    n_spatial: int,
    image_size: int,
    channels: int,
    ctx_length: int,
    horizon: int,
    flow_steps: int,
    max_items: int = 4,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Action-conditioned rollout eval: replay dataset actions, autoregressive latent prediction (no GT latent
    feedback after context), decode, compare to GT frames and repeat-last-frame baseline.

    Returns metrics, viz panel (uint8 HWC), gt frames (B,T,H,W,C), pred frames.
    """
    dynamics.eval()
    B, T = image_bthwc.shape[:2]
    length = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, length - 1)
    horizon = min(horizon, length - ctx_length)
    if horizon <= 0:
        raise ValueError(f"rollout horizon must be > 0 (T={T}, ctx={ctx_length})")

    frames = image_bthwc[:, :length]
    actions_eval = actions[:, :length]

    z_btld = encode_images(tokenizer, frames, patch_size)
    z_gt_packed = pack_bottleneck_to_spatial(z_btld, n_spatial, packing_factor)

    z_pred_packed = sample_autoregressive_packed_sequence(
        dynamics,
        z_gt_packed,
        actions_eval,
        ctx_length,
        horizon,
        flow_steps,
    )

    pred_frames = decode_packed_to_images(
        tokenizer,
        z_pred_packed,
        patch_size,
        packing_factor,
        image_size,
        channels,
    )

    floor = frames.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames[:, ctx_length - 1:ctx_length].expand(
            -1, horizon, -1, -1, -1
        )

    gt_h = frames[:, ctx_length:ctx_length + horizon]
    pred_h = pred_frames[:, ctx_length:ctx_length + horizon]
    floor_h = floor[:, ctx_length:ctx_length + horizon]

    mse_pred = (pred_h.float() - gt_h.float()).pow(2).mean()
    mse_floor = (floor_h.float() - gt_h.float()).pow(2).mean()
    psnr_pred = 10.0 * torch.log10(1.0 / mse_pred.clamp_min(1e-12))
    psnr_floor = 10.0 * torch.log10(1.0 / mse_floor.clamp_min(1e-12))

    metrics = {
        "rollout_mse": float(mse_pred.detach()),
        "rollout_mse_floor": float(mse_floor.detach()),
        "rollout_mse_ratio": float((mse_pred / mse_floor.clamp_min(1e-12)).detach()),
        "rollout_psnr": float(psnr_pred.detach()),
        "rollout_psnr_floor": float(psnr_floor.detach()),
        "rollout_psnr_gain": float((psnr_pred - psnr_floor).detach()),
    }

    panel = rollout_panel_uint8(frames, pred_frames, ctx_length, max_items=max_items)
    return metrics, panel, frames, pred_frames
