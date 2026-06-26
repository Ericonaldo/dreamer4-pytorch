from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.models.action_encoder import ActionEncoder
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
    z1_hat = model(actions, sigma, z_tilde)
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
        self.space_mode = str(raw.get("space_mode", "wm_agent"))
        self.packing_factor = int(raw.get("packing_factor", 1))
        self.n_register = int(raw.get("n_register", 0))
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

        layout = TokenLayout(n_latents=0, segments=tuple(segments))
        self.spatial_slice = layout.slices()[Modality.SPATIAL]

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
    ) -> torch.Tensor:
        """Predict clean packed latents. sigma: (B,T) in [0,1]."""
        B, T = packed_z.shape[:2]
        spatial_tokens = self.spatial_proj(packed_z)
        action_tokens = self.action_encoder(actions)
        noise_tokens = self.noise_mlp(sigma[..., None]).unsqueeze(2)

        tokens = [action_tokens, noise_tokens, spatial_tokens]
        if self.n_register > 0:
            reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)
            tokens.append(reg)

        x = torch.cat(tokens, dim=2)
        x = add_sinusoidal_positions(x, self.scale_pos_embeds)
        x = self.transformer(x)
        spatial_out = x[:, :, self.spatial_slice, :]
        return self.flow_head(spatial_out)
