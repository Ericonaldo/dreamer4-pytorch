"""Causal patch tokenizer: Encoder + Decoder + MAE reconstruction loss."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from dreamer4.config import config_to_dict
from dreamer4.models.transformer_blocks import (
    BlockCausalTransformer,
    MAEReplacer,
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
)


# ---------------------------------------------------------------------------
# Patch I/O
# ---------------------------------------------------------------------------


def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,T,C,H,W) in [0,1] -> (B,T,Np,Dp)."""
    B, T, C, H, W = videos_btchw.shape
    x = videos_btchw.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2).contiguous()
    Np, Dp = cols.shape[1], cols.shape[2]
    return cols.reshape(B, T, Np, Dp)


def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """(B,T,Np,Dp) -> (B,T,C,H,W)."""
    B, T, Np, Dp = patches_btnd.shape
    x = patches_btnd.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)
    return out.reshape(B, T, C, H, W)


def images_to_patches(image_bthwc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B,T,H,W,C) -> (B,T,C,H,W) -> patch tokens."""
    x = image_bthwc.permute(0, 1, 4, 2, 3).contiguous()
    return temporal_patchify(x, patch_size)


# ---------------------------------------------------------------------------
# Losses & metrics
# ---------------------------------------------------------------------------


def mae_recon_loss(
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
) -> torch.Tensor:
    """MSE on MAE-masked patches only."""
    mask = mae_mask_btNp1.to(dtype=torch.float32)
    diff = (pred_btnd.float() - target_btnd.float())
    sq = diff.pow(2) * mask
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def full_recon_loss(pred_btnd: torch.Tensor, target_btnd: torch.Tensor) -> torch.Tensor:
    """MSE on all patches (stable val metric, no mask randomness)."""
    return (pred_btnd.float() - target_btnd.float()).pow(2).mean()


def latent_temporal_std(z_btld: torch.Tensor) -> torch.Tensor:
    """Mean std of bottleneck latents across time; low values indicate temporal collapse."""
    if z_btld.shape[1] < 2:
        return torch.zeros((), device=z_btld.device, dtype=torch.float32)
    return z_btld.float().std(dim=1).mean()


# ---------------------------------------------------------------------------
# Training / inference API
# ---------------------------------------------------------------------------


def tokenizer_forward_loss(
    model: Tokenizer,
    image_bthwc: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, dict[str, float]]:
    patches = images_to_patches(image_bthwc, patch_size)
    z, (mae_mask, _) = model.encoder(patches)
    pred = model.decoder(z)
    loss = mae_recon_loss(pred, patches, mae_mask)
    metrics = {
        "loss_mae": float(loss.detach()),
        "masked_frac": float(mae_mask.float().mean().detach()),
        "z_temporal_std": float(latent_temporal_std(z).detach()),
    }
    return loss, metrics


def tokenizer_forward_with_aux(
    model: Tokenizer,
    image_bthwc: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + MAE loss; also returns pred, target patches, and MAE mask for viz."""
    patches = images_to_patches(image_bthwc, patch_size)
    z, (mae_mask, _) = model.encoder(patches)
    pred = model.decoder(z)
    loss = mae_recon_loss(pred, patches, mae_mask)
    metrics = {
        "loss_mae": float(loss.detach()),
        "loss_full": float(full_recon_loss(pred, patches).detach()),
        "masked_frac": float(mae_mask.float().mean().detach()),
        "z_temporal_std": float(latent_temporal_std(z).detach()),
    }
    return loss, metrics, pred, patches, mae_mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Patch encoder: MAE-masked patches + latent tokens -> tanh bottleneck."""

    def __init__(
        self,
        *,
        d_patch: int,
        n_latents: int,
        n_patches: int,
        d_bottleneck: int,
        scale_pos_embeds: bool,
        transformer_kwargs: Mapping[str, Any],
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
    ):
        super().__init__()
        d_model = transformer_kwargs["d_model"]
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.scale_pos_embeds = scale_pos_embeds

        self.patch_proj = nn.Linear(d_patch, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.transformer = BlockCausalTransformer(
            n_latents=n_latents,
            modality_ids=layout.modality_ids(),
            space_mode="encoder",
            **transformer_kwargs,
        )
        self.mae = MAEReplacer(d_model=d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, patch_tokens_btnd: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, Np, _ = patch_tokens_btnd.shape
        assert Np == self.n_patches

        proj = self.patch_proj(patch_tokens_btnd)
        proj_masked, mae_mask, keep_prob = self.mae(proj)

        lat = self.latents.view(1, 1, self.n_latents, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, proj_masked], dim=2)
        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)

        enc = self.transformer(tokens)
        z = torch.tanh(self.bottleneck_proj(enc[:, :, :self.n_latents, :])) # Why not use perceiver sampler?
        return z, (mae_mask, keep_prob)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_patch: int,
        n_latents: int,
        n_patches: int,
        d_bottleneck: int,
        scale_pos_embeds: bool,
        transformer_kwargs: Mapping[str, Any],
    ):
        super().__init__()
        d_model = transformer_kwargs["d_model"]
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.scale_pos_embeds = scale_pos_embeds

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        self.patch_head = nn.Linear(d_model, d_patch)

        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        self.transformer = BlockCausalTransformer(
            n_latents=n_latents,
            modality_ids=layout.modality_ids(),
            space_mode="decoder",
            **transformer_kwargs,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.patch_queries, std=0.02)

    def forward(self, z_btLd: torch.Tensor) -> torch.Tensor:
        B, T, L, _ = z_btLd.shape
        assert L == self.n_latents

        lat = torch.tanh(self.up_proj(z_btLd))
        qry = self.patch_queries.view(1, 1, self.n_patches, -1).expand(B, T, -1, -1)
        tokens = torch.cat([lat, qry], dim=2)
        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)

        x = self.transformer(tokens)
        return torch.sigmoid(self.patch_head(x[:, :, L:, :]))


class Tokenizer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size: int | None = None

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob

    def encode(self, patches_btnd: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(patches_btnd)
        return z

    def encode_images(
        self,
        image_bthwc: torch.Tensor,
        patch_size: int | None = None,
    ) -> torch.Tensor:
        """(B,T,H,W,C) -> bottleneck latents (B,T,n_latents,latent_dim)."""
        ps = patch_size if patch_size is not None else self.patch_size
        if ps is None:
            raise ValueError("patch_size required when tokenizer.patch_size is unset")
        return self.encode(images_to_patches(image_bthwc, ps))


# ---------------------------------------------------------------------------
# Build from config
# ---------------------------------------------------------------------------


def build_tokenizer(cfg: Mapping[str, Any] | DictConfig) -> Tokenizer:
    """Build Tokenizer from YAML `model` or `model.tokenizer` (OmegaConf dict)."""
    raw = config_to_dict(cfg)
    H = int(raw["image_size"])
    W = int(raw["image_size"])
    C = int(raw["channels"])
    patch = int(raw["patch_size"])

    d_model = int(raw["embed_dim"])
    n_heads = int(raw["num_heads"])
    depth = int(raw["depth"])
    n_latents = int(raw["n_latents"])
    d_bottleneck = int(raw["latent_dim"])

    assert H % patch == 0 and W % patch == 0
    assert d_model % n_heads == 0

    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    dropout = float(raw.get("dropout", 0.0))
    mlp_ratio = float(raw.get("mlp_ratio", 2.0))
    time_every = int(raw.get("time_every", 2))
    latents_only_time = bool(raw.get("latents_only_time", True))
    scale_pos_embeds = bool(raw.get("scale_pos_embeds", True))

    transformer_kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        time_every=time_every,
        latents_only_time=latents_only_time,
    )

    common_kwargs = dict(
        d_patch=d_patch,
        n_latents=n_latents,
        n_patches=n_patches,
        d_bottleneck=d_bottleneck,
        scale_pos_embeds=scale_pos_embeds,
        transformer_kwargs=transformer_kwargs,
    )
    enc = Encoder(**common_kwargs, mae_p_min=float(raw.get("mae_p_min", 0.0)), mae_p_max=float(raw.get("mae_p_max", 0.5)))
    dec = Decoder(**common_kwargs)
    model = Tokenizer(enc, dec)
    model.patch_size = patch
    model._cfg = raw
    return model
