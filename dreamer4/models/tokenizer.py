"""Causal patch tokenizer: Encoder + Decoder + MAE reconstruction loss."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from dreamer4.models.transformer_blocks import (
    BlockCausalTransformer,
    MAEReplacer,
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
)


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


class Encoder(nn.Module):
    """Patch encoder: MAE-masked patches + latent tokens -> tanh bottleneck."""

    def __init__(
        self,
        *,
        patch_dim: int,
        d_model: int,
        n_latents: int,
        n_patches: int,
        n_heads: int,
        depth: int,
        d_bottleneck: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        mae_p_min: float = 0.0,
        mae_p_max: float = 0.9,
        scale_pos_embeds: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.scale_pos_embeds = scale_pos_embeds

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=n_latents,
            modality_ids=modality_ids,
            space_mode="encoder",
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=latents_only_time,
        )
        self.mae = MAEReplacer(d_model=d_model, p_min=mae_p_min, p_max=mae_p_max)

        self.latents = nn.Parameter(torch.empty(n_latents, d_model))
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
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        n_patches: int,
        d_patch: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        latents_only_time: bool = True,
        scale_pos_embeds: bool = True,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.n_patches = n_patches
        self.scale_pos_embeds = scale_pos_embeds

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.patch_queries = nn.Parameter(torch.empty(n_patches, d_model))
        nn.init.normal_(self.patch_queries, std=0.02)
        self.patch_head = nn.Linear(d_model, d_patch)

        layout = TokenLayout(n_latents=n_latents, segments=((Modality.IMAGE, n_patches),))
        modality_ids = layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            n_latents=n_latents,
            modality_ids=modality_ids,
            space_mode="decoder",
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            time_every=time_every,
            latents_only_time=latents_only_time,
        )

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

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, (mae_mask, keep_prob) = self.encoder(patches_btnd)
        pred = self.decoder(z)
        return pred, mae_mask, keep_prob

    def encode(self, patches_btnd: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(patches_btnd)
        return z


def _as_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def _as_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def _as_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    return bool(cfg.get(key, default))


def _cfg_dict(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def build_tokenizer(cfg: Mapping[str, Any] | DictConfig) -> Tokenizer:
    """Build Tokenizer from YAML `model` or `model.tokenizer` (OmegaConf dict)."""
    raw = _cfg_dict(cfg)
    H = _as_int(raw, "image_size", 64)
    W = _as_int(raw, "image_size", 64)
    C = _as_int(raw, "channels", 3)
    patch = _as_int(raw, "patch_size", 16)

    d_model = _as_int(raw, "embed_dim", 128)
    n_heads = _as_int(raw, "num_heads", 4)
    depth = _as_int(raw, "depth", 2)
    n_latents = _as_int(raw, "n_latents", 8)
    d_bottleneck = _as_int(raw, "latent_dim", 32)

    assert H % patch == 0 and W % patch == 0
    assert d_model % n_heads == 0

    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C

    enc = Encoder(
        patch_dim=d_patch,
        d_model=d_model,
        n_latents=n_latents,
        n_patches=n_patches,
        n_heads=n_heads,
        depth=depth,
        d_bottleneck=d_bottleneck,
        dropout=_as_float(raw, "dropout", 0.0),
        mlp_ratio=_as_float(raw, "mlp_ratio", 2.0),
        time_every=_as_int(raw, "time_every", 2),
        latents_only_time=_as_bool(raw, "latents_only_time", True),
        mae_p_min=_as_float(raw, "mae_p_min", 0.0),
        mae_p_max=_as_float(raw, "mae_p_max", 0.5),
        scale_pos_embeds=_as_bool(raw, "scale_pos_embeds", True),
    )
    dec = Decoder(
        d_bottleneck=d_bottleneck,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=_as_float(raw, "dropout", 0.0),
        mlp_ratio=_as_float(raw, "mlp_ratio", 2.0),
        time_every=_as_int(raw, "time_every", 2),
        latents_only_time=_as_bool(raw, "latents_only_time", True),
        scale_pos_embeds=_as_bool(raw, "scale_pos_embeds", True),
    )
    model = Tokenizer(enc, dec)
    model._cfg = raw
    return model


def lpips_on_mae_recon(
    lpips_fn: nn.Module,
    pred_btnd: torch.Tensor,
    target_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    *,
    H: int,
    W: int,
    C: int,
    patch: int,
    subsample_frac: float = 1.0,
) -> torch.Tensor:
    """LPIPS on MAE-masked reconstruction (recon uses pred only on masked patches)."""
    recon_masked_btnd = torch.where(mae_mask_btNp1, pred_btnd, target_btnd)
    recon = temporal_unpatchify(recon_masked_btnd.float(), H, W, C, patch)
    tgt = temporal_unpatchify(target_btnd.float(), H, W, C, patch)

    if subsample_frac < 1.0:
        step = max(1, int(1.0 / subsample_frac))
        recon = recon[:, ::step]
        tgt = tgt[:, ::step]

    recon = (recon.clamp(0, 1) * 2.0 - 1.0).float()
    tgt = (tgt.clamp(0, 1) * 2.0 - 1.0).float()

    B, T = recon.shape[:2]
    recon = recon.reshape(B * T, C, H, W)
    tgt = tgt.reshape(B * T, C, H, W)

    device_type = "cuda" if recon.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        lp = lpips_fn(recon, tgt)
    return lp.mean()


def images_to_patches(image_bthwc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B,T,H,W,C) -> (B,T,C,H,W) -> patch tokens."""
    x = image_bthwc.permute(0, 1, 4, 2, 3).contiguous()
    return temporal_patchify(x, patch_size)


def encode_images(tokenizer: Tokenizer, image_bthwc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Encode images to bottleneck latents (B,T,n_latents,latent_dim)."""
    patches = images_to_patches(image_bthwc, patch_size)
    return tokenizer.encode(patches)


def tokenizer_forward_loss(
    model: Tokenizer,
    image_bthwc: torch.Tensor,
    patch_size: int,
    *,
    lpips_fn: nn.Module | None = None,
    lpips_weight: float = 0.0,
    lpips_frac: float = 1.0,
) -> Tuple[torch.Tensor, dict[str, float]]:
    patches = images_to_patches(image_bthwc, patch_size)
    z, (mae_mask, _) = model.encoder(patches)
    pred = model.decoder(z)
    mse = mae_recon_loss(pred, patches, mae_mask)
    metrics = {
        "loss_mae": float(mse.detach()),
        "masked_frac": float(mae_mask.float().mean().detach()),
        "z_temporal_std": float(latent_temporal_std(z).detach()),
    }

    if lpips_fn is not None and lpips_weight > 0.0:
        _, _, H, W, C = image_bthwc.shape
        lp = lpips_on_mae_recon(
            lpips_fn,
            pred,
            patches,
            mae_mask,
            H=H,
            W=W,
            C=C,
            patch=patch_size,
            subsample_frac=lpips_frac,
        )
        loss = mse + lpips_weight * lp
        metrics["loss_lpips"] = float(lp.detach())
    else:
        loss = mse

    return loss, metrics


def tokenizer_forward_with_aux(
    model: Tokenizer,
    image_bthwc: torch.Tensor,
    patch_size: int,
    *,
    lpips_fn: nn.Module | None = None,
    lpips_weight: float = 0.0,
    lpips_frac: float = 1.0,
) -> Tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward + MAE loss; also returns pred, target patches, and MAE mask for viz."""
    patches = images_to_patches(image_bthwc, patch_size)
    z, (mae_mask, _) = model.encoder(patches)
    pred = model.decoder(z)
    mse = mae_recon_loss(pred, patches, mae_mask)
    metrics = {
        "loss_mae": float(mse.detach()),
        "loss_full": float(full_recon_loss(pred, patches).detach()),
        "masked_frac": float(mae_mask.float().mean().detach()),
        "z_temporal_std": float(latent_temporal_std(z).detach()),
    }

    if lpips_fn is not None and lpips_weight > 0.0:
        _, _, H, W, C = image_bthwc.shape
        lp = lpips_on_mae_recon(
            lpips_fn,
            pred,
            patches,
            mae_mask,
            H=H,
            W=W,
            C=C,
            patch=patch_size,
            subsample_frac=lpips_frac,
        )
        loss = mse + lpips_weight * lp
        metrics["loss_lpips"] = float(lp.detach())
    else:
        loss = mse

    return loss, metrics, pred, patches, mae_mask


_ROW_LABELS = ("target", "masked", "recon_masked", "recon_full")


def _annotate_panel_rows(panel_hwc: np.ndarray, row_h: int, n_samples: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(panel_hwc)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for s in range(n_samples):
        for r, label in enumerate(_ROW_LABELS):
            y = s * 4 * row_h + r * row_h + 2
            draw.text((4, y), label, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0), font=font)
    return np.asarray(img)


def recon_panel_uint8(
    image_bthwc: torch.Tensor,
    pred_btnd: torch.Tensor,
    mae_mask_btNp1: torch.Tensor,
    patch_size: int,
    max_items: int = 4,
    max_T: int = 6,
) -> np.ndarray:
    """Panel image: rows = target | masked | recon_masked | recon_full per sample."""
    B, T, H, W, C = image_bthwc.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    x_btc_hw = image_bthwc[:Bv, :Tv].permute(0, 1, 4, 2, 3).contiguous()
    target_btnd = temporal_patchify(x_btc_hw, patch_size)
    mask = mae_mask_btNp1[:Bv, :Tv]
    pred = pred_btnd[:Bv, :Tv]

    masked_input_btnd = torch.where(mask, torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mask, pred, target_btnd)
    recon_full_btnd = pred

    target_img = temporal_unpatchify(target_btnd, H, W, C, patch_size)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch_size)
    rmask_img = temporal_unpatchify(recon_masked_btnd, H, W, C, patch_size)
    rfull_img = temporal_unpatchify(recon_full_btnd, H, W, C, patch_size)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    panel = torch.cat(
        [tile_time(target_img), tile_time(masked_img), tile_time(rmask_img), tile_time(rfull_img)],
        dim=2,
    )
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)
    big = (big.clamp(0, 1) * 255.0).to(torch.uint8)
    out = big.permute(1, 2, 0).cpu().numpy()
    return _annotate_panel_rows(out, H, Bv)
