"""Dynamics rollout evaluation with visualization panels."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dreamer4.eval_utils import dynamics_rollout_eval, dynamics_rollout_video


def stack_gt_pred_video_uint8(
    gt_bthwc: torch.Tensor,
    pred_bthwc: torch.Tensor,
) -> np.ndarray:
    """(T,H,W,C) GT on top, pred on bottom -> (T, 2H, W, C) uint8."""
    t = min(gt_bthwc.shape[1], pred_bthwc.shape[1])
    gt = (gt_bthwc[0, :t].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    pred = (pred_bthwc[0, :t].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    return np.concatenate([gt, pred], axis=1)


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
    return_per_traj: bool = False,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor, torch.Tensor] | tuple[
    dict[str, float], np.ndarray, torch.Tensor, torch.Tensor, list[np.ndarray]
]:
    from eval.viz.panels import rollout_panels_multictx_uint8

    result = dynamics_rollout_eval(
        dynamics,
        tokenizer,
        image_bthwc,
        actions,
        patch_size=patch_size,
        packing_factor=packing_factor,
        n_spatial=n_spatial,
        image_size=image_size,
        channels=channels,
        ctx_length=ctx_length,
        horizon=horizon,
        flow_steps=flow_steps,
    )
    panel, per_traj = rollout_panels_multictx_uint8(
        result.frames,
        result.pred_by_ctx_bkthwc,
        result.ctx_lengths,
        max_items=max_items,
    )
    if return_per_traj:
        return result.metrics, panel, result.frames, result.pred_frames, per_traj
    return result.metrics, panel, result.frames, result.pred_frames


@torch.no_grad()
def run_dynamics_rollout_video(
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
    attn_window: int,
    rollout_length: int,
    flow_steps: int,
    max_items: int = 4,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray], torch.Tensor, torch.Tensor]:
    result = dynamics_rollout_video(
        dynamics,
        tokenizer,
        image_bthwc,
        actions,
        patch_size=patch_size,
        packing_factor=packing_factor,
        n_spatial=n_spatial,
        image_size=image_size,
        channels=channels,
        attn_window=attn_window,
        rollout_length=rollout_length,
        flow_steps=flow_steps,
        max_items=max_items,
    )

    pred_videos: list[np.ndarray] = []
    compare_videos: list[np.ndarray] = []
    for i in range(result.pred_frames.shape[0]):
        pred_videos.append(
            (result.pred_frames[i].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        )
        compare_videos.append(
            stack_gt_pred_video_uint8(result.gt_frames[i : i + 1], result.pred_frames[i : i + 1])
        )
    return result.metrics, pred_videos, compare_videos, result.gt_frames, result.pred_frames
