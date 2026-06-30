"""Rollout and reconstruction panel images (uint8 HWC)."""

from __future__ import annotations

import numpy as np
import torch

from dreamer4.models.tokenizer import temporal_patchify, temporal_unpatchify
from eval.viz.annotate import annotate_multictx_panel, annotate_panel_rows, annotate_rollout_panel_rows

_RECON_ROW_LABELS = ("target", "masked", "recon_masked", "recon_full")


def tile_time_with_gap(
    x: torch.Tensor,
    ctx: int,
    gap_px: int,
    *,
    insert_gap: bool = True,
    gap_value: float = 0.0,
) -> torch.Tensor:
    """(B,T,C,H,W) -> (B,C,H,T*W) with optional gap after ctx frames."""
    B, T, C, H, W = x.shape
    y = x.permute(0, 2, 3, 1, 4).contiguous().view(B, C, H, T * W)
    if insert_gap and gap_px > 0 and 0 < ctx < T:
        split = ctx * W
        if split < T * W:
            left = y[..., :split]
            right = y[..., split:]
            gap = torch.full((B, C, H, gap_px), gap_value, device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=3)
    return y


def rollout_panels_multictx_uint8(
    gt_bthwc: torch.Tensor,
    pred_by_ctx_bkthwc: torch.Tensor,
    ctx_lengths: list[int],
    max_items: int = 4,
    gap_px: int = 16,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Combined panel (all trajectories stacked) + one annotated image per trajectory."""
    B, K, T = pred_by_ctx_bkthwc.shape[:3]
    gt = gt_bthwc[:, :T]
    H, W = gt.shape[3], gt.shape[4]
    Bv = min(B, max_items)
    max_ctx = int(min(ctx_lengths[-1], T))
    gt_row = tile_time_with_gap(
        gt[:Bv].permute(0, 1, 4, 2, 3),
        max_ctx,
        gap_px,
        insert_gap=gap_px > 0 and 0 < max_ctx < T,
        gap_value=1.0,
    )
    rows = [gt_row]
    for ki, ctx in enumerate(ctx_lengths):
        ctx = int(min(ctx, T))
        composite = gt[:Bv].clone()
        if ctx < T:
            composite[:, ctx:] = pred_by_ctx_bkthwc[:Bv, ki, ctx:]
        row = tile_time_with_gap(composite.permute(0, 1, 4, 2, 3), ctx, gap_px)
        rows.append(row)
    panel = torch.cat(rows, dim=2)
    per_sample: list[np.ndarray] = []
    for i in range(Bv):
        out = (panel[i].clamp(0, 1) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        per_sample.append(
            annotate_multictx_panel(out, H, W, ctx_lengths, 1, gap_px, T, include_gt_row=True)
        )
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)
    combined = (big.clamp(0, 1) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    combined = annotate_multictx_panel(
        combined, H, W, ctx_lengths, Bv, gap_px, T, include_gt_row=True
    )
    return combined, per_sample


def rollout_panel_multictx_uint8(
    gt_bthwc: torch.Tensor,
    pred_by_ctx_bkthwc: torch.Tensor,
    ctx_lengths: list[int],
    max_items: int = 4,
    gap_px: int = 16,
) -> np.ndarray:
    """Top GT row + rows ctx=1..K: GT context frames, rollout decode for the rest."""
    combined, _ = rollout_panels_multictx_uint8(
        gt_bthwc, pred_by_ctx_bkthwc, ctx_lengths, max_items=max_items, gap_px=gap_px
    )
    return combined


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
        return tile_time_with_gap(x, ctx, gap_px)

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)
    panel = torch.cat([gt_t, pr_t], dim=2)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)
    out = (big.clamp(0, 1) * 255.0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    return annotate_rollout_panel_rows(out, H, Bv)


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
    return annotate_panel_rows(out, H, Bv, _RECON_ROW_LABELS)
