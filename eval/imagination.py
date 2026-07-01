"""Render imagination rollout videos from an RL policy checkpoint.

Picks dataset episodes by cumulative return, encodes a context window (suffix-style,
matching RL training), rolls out policy-sampled actions in latent space, decodes
predicted latents, and writes mp4s.

Example::

    uv run python -m eval.imagination \\
      configs/walker_walk/policy_imagination_pmpo.yaml \\
      --rl-ckpt logs/walker_walk/rl_pmpo/checkpoints/step-step=7000.ckpt \\
      --out-dir logs/walker_walk/rl_pmpo/imagine_videos_step7000 \\
      --min-episode-return 900 --max-items 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig

from dreamer4.config import load_config
from dreamer4.data import align_dynamics_batch
from dreamer4.models.dynamics import (
    decode_packed_to_images,
    pack_bottleneck_to_spatial,
    sample_one_timestep_packed,
)
from dreamer4.models.policy import POLICY_ENV_ACTION_SLOT
from dreamer4.agent import load_policy_modules

from eval.data_stats import episode_cumulative_returns, select_episodes_by_return
from eval.dynamics_rollout import _load_rollout_batch_from_picked
from eval.viz.annotate import annotate_frames_uint8
from eval.viz.panels import rollout_panels_multictx_uint8
from eval.viz.rollout import stack_gt_pred_video_uint8


def _collect_imagined_latents(
    model,
    dynamics,
    packed_z_ctx: torch.Tensor,
    actions_ctx: torch.Tensor,
    policy,
    horizon: int,
    flow_steps: int,
    *,
    bc_space_mode: str,
    ctx_len: int,
) -> torch.Tensor:
    """Run imagination rollout; return predicted latents (B, H, n_spatial, d_spatial)."""
    z_sliding = packed_z_ctx[:, -ctx_len:].float()
    a_sliding = actions_ctx[:, -ctx_len:].float()

    with torch.no_grad():
        h_seq = model.agent_hidden(z_sliding, a_sliding, space_mode=bc_space_mode)
    h = h_seq[:, -1]

    imagined_latents: list[torch.Tensor] = []
    for _ in range(horizon):
        h_in = h.unsqueeze(1)
        with torch.no_grad():
            action_mtp, _, _, _ = policy.sample(h_in)
        action = action_mtp[:, 0, POLICY_ENV_ACTION_SLOT]
        actions_step = torch.cat([a_sliding, action.unsqueeze(1)], dim=1)
        with torch.no_grad():
            z_next = sample_one_timestep_packed(dynamics, z_sliding, actions_step, flow_steps)
        imagined_latents.append(z_next)
        z_sliding = torch.cat([z_sliding, z_next.unsqueeze(1)], dim=1)
        if z_sliding.shape[1] > ctx_len:
            z_sliding = z_sliding[:, -ctx_len:]
        a_sliding = torch.cat([a_sliding, action.unsqueeze(1)], dim=1)
        if a_sliding.shape[1] > ctx_len:
            a_sliding = a_sliding[:, -ctx_len:]
        h = model.agent_hidden(z_sliding, a_sliding, space_mode=bc_space_mode)[:, -1]

    return torch.stack(imagined_latents, dim=1)


def _load_rl_policy_model(cfg: DictConfig, rl_ckpt: Path, device: torch.device):
    model, tokenizer = load_policy_modules(cfg, device)
    ckpt = torch.load(rl_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix("model."): v
        for k, v in state.items()
        if k.startswith("model.") and "attn_mask" not in k
    }
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, tokenizer


def _imagine_composite_frames(
    model,
    tokenizer,
    *,
    gt_segment: torch.Tensor,
    actions_segment: torch.Tensor,
    ctx_k: int,
    flow_steps: int,
    patch_size: int,
    packing_factor: int,
    n_spatial: int,
    image_size: int,
    channels: int,
    bc_space_mode: str,
) -> torch.Tensor:
    """GT context + policy-imagined decode for remaining timesteps."""
    t_total = gt_segment.shape[1]
    rollout_len = t_total - ctx_k
    if rollout_len <= 0:
        return gt_segment

    ctx_images = gt_segment[:, :ctx_k]
    ctx_actions = actions_segment[:, :ctx_k]
    z_ctx = tokenizer.encode_images(ctx_images, patch_size)
    packed_z_ctx = pack_bottleneck_to_spatial(z_ctx, n_spatial, packing_factor)
    z_imagined = _collect_imagined_latents(
        model,
        model.dynamics,
        packed_z_ctx,
        ctx_actions,
        model.heads.policy,
        rollout_len,
        flow_steps,
        bc_space_mode=bc_space_mode,
        ctx_len=ctx_k,
    )
    im_frames = decode_packed_to_images(
        tokenizer,
        z_imagined,
        patch_size,
        packing_factor,
        image_size,
        channels,
    )
    composite = gt_segment.clone()
    composite[:, ctx_k:] = im_frames
    return composite


@torch.no_grad()
def render_imagination_rollout_panels(
    cfg: DictConfig,
    rl_ckpt: Path,
    batch,
    picked_meta: list[dict],
    *,
    context_len: int,
    horizon: int,
    flow_steps: int,
    device: torch.device,
    max_items: int,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    """Build panels: gt row + single ctx=K imagine row (K = context_len)."""
    model, tokenizer = _load_rl_policy_model(cfg, rl_ckpt, device)

    patch_size = int(cfg.model.tokenizer.patch_size)
    packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
    n_spatial = tokenizer.encoder.n_latents // packing_factor
    image_size = int(cfg.model.tokenizer.image_size)
    channels = int(cfg.model.tokenizer.channels)
    bc_space_mode = model.bc_space_mode

    image, action, _ = align_dynamics_batch(batch.image.to(device), batch.action.to(device))
    assert image is not None

    t_display = context_len + horizon
    need_t = horizon + t_display
    if image.shape[1] < need_t:
        raise ValueError(f"need at least {need_t} frames, got {image.shape[1]}")

    B = min(image.shape[0], max_items)
    gt_batch = image[:B, horizon : horizon + t_display]
    act_batch = action[:B, horizon : horizon + t_display]

    ctx_lengths = [context_len]
    pred_rows: list[torch.Tensor] = []
    for i in range(B):
        pred_rows.append(
            _imagine_composite_frames(
                model,
                tokenizer,
                gt_segment=gt_batch[i : i + 1],
                actions_segment=act_batch[i : i + 1],
                ctx_k=context_len,
                flow_steps=flow_steps,
                patch_size=patch_size,
                packing_factor=packing_factor,
                n_spatial=n_spatial,
                image_size=image_size,
                channels=channels,
                bc_space_mode=bc_space_mode,
            )
        )
    pred_by_ctx_bkthwc = torch.cat(pred_rows, dim=0).unsqueeze(1)

    gt_h = gt_batch[:, context_len : context_len + horizon]
    im_h = pred_by_ctx_bkthwc[:, 0, context_len : context_len + horizon]
    mse = (im_h.float() - gt_h.float()).pow(2).mean()
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

    panel, per_traj = rollout_panels_multictx_uint8(
        gt_batch, pred_by_ctx_bkthwc, ctx_lengths, max_items=B
    )
    out_meta = []
    for i in range(B):
        meta = dict(picked_meta[i])
        meta["context_len"] = context_len
        meta["horizon"] = horizon
        meta["rollout_mse"] = float(mse.detach()) if B == 1 else None
        meta["rollout_psnr"] = float(psnr.detach()) if B == 1 else None
        out_meta.append(meta)
    return panel, per_traj, out_meta


@torch.no_grad()
def render_imagination_rollout_videos(
    cfg: DictConfig,
    rl_ckpt: Path,
    batch,
    picked_meta: list[dict],
    *,
    context_len: int,
    horizon: int,
    flow_steps: int,
    device: torch.device,
    max_items: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Returns (pred_videos, compare_videos, per_episode_meta)."""
    model, tokenizer = _load_rl_policy_model(cfg, rl_ckpt, device)

    patch_size = int(cfg.model.tokenizer.patch_size)
    packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
    n_spatial = tokenizer.encoder.n_latents // packing_factor
    image_size = int(cfg.model.tokenizer.image_size)
    channels = int(cfg.model.tokenizer.channels)
    bc_space_mode = model.bc_space_mode

    image, action, _ = align_dynamics_batch(batch.image.to(device), batch.action.to(device))
    assert image is not None

    need_t = context_len + horizon
    if image.shape[1] < need_t + horizon:
        raise ValueError(
            f"need at least {need_t + horizon} frames for GT comparison, got {image.shape[1]}"
        )

    pred_videos: list[np.ndarray] = []
    compare_videos: list[np.ndarray] = []
    out_meta: list[dict[str, Any]] = []

    B = min(image.shape[0], max_items)
    for i in range(B):
        # Suffix context ending before imagined GT segment (matches RL window layout).
        ctx_images = image[i : i + 1, horizon : horizon + context_len]
        ctx_actions = action[i : i + 1, horizon : horizon + context_len]
        gt_ctx = ctx_images
        gt_future = image[i : i + 1, horizon + context_len : horizon + context_len + horizon]

        z_ctx = tokenizer.encode_images(ctx_images, patch_size)
        packed_z_ctx = pack_bottleneck_to_spatial(z_ctx, n_spatial, packing_factor)

        z_imagined = _collect_imagined_latents(
            model,
            model.dynamics,
            packed_z_ctx,
            ctx_actions,
            model.heads.policy,
            horizon,
            flow_steps,
            bc_space_mode=bc_space_mode,
            ctx_len=context_len,
        )
        im_frames = decode_packed_to_images(
            tokenizer,
            z_imagined,
            patch_size,
            packing_factor,
            image_size,
            channels,
        )

        gt_ctx_u8 = (gt_ctx[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        im_u8 = (im_frames[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        pred_video = np.concatenate([gt_ctx_u8, im_u8], axis=0)

        gt_full = torch.cat([gt_ctx, gt_future], dim=1)
        im_full = torch.cat([gt_ctx, im_frames], dim=1)
        compare_video = stack_gt_pred_video_uint8(gt_full, im_full)

        pred_videos.append(pred_video)
        compare_videos.append(compare_video)
        meta = dict(picked_meta[i])
        meta["context_len"] = context_len
        meta["horizon"] = horizon
        out_meta.append(meta)

    return pred_videos, compare_videos, out_meta


def _pick_episodes(
    cfg: DictConfig,
    *,
    min_return: float | None,
    max_return: float | None,
    max_items: int,
    seq_len: int,
) -> tuple[Any, list[dict]]:
    _, returns = episode_cumulative_returns(cfg.data.path)
    low = float(min_return if min_return is not None else 0.0)
    high = float(max_return) if max_return is not None else None
    picked = select_episodes_by_return(
        returns,
        low,
        episode_indices=None,
        top_k=max_items,
        max_return=high,
    )
    if not picked:
        raise ValueError(f"no episodes in return [{low}, {high})")
    print(f"Selected {len(picked)} episodes in [{low}, {high}): {picked}")
    return _load_rollout_batch_from_picked(cfg, picked, seq_len=seq_len)


def _pick_high_low_episodes(
    cfg: DictConfig,
    *,
    high_min: float,
    low_max: float,
    seq_len: int,
) -> tuple[Any, list[dict]]:
    _, returns = episode_cumulative_returns(cfg.data.path)
    high = select_episodes_by_return(returns, high_min, episode_indices=None, top_k=1)
    low = select_episodes_by_return(
        returns, 0.0, episode_indices=None, top_k=1, max_return=low_max
    )
    if not high:
        raise ValueError(f"no episodes with return >= {high_min}")
    if not low:
        raise ValueError(f"no episodes with return < {low_max}")
    picked = high + low
    print(f"Selected high/low episodes: {picked}")
    return _load_rollout_batch_from_picked(cfg, picked, seq_len=seq_len)


def _write_videos(
    out_dir: Path,
    pred_videos: list[np.ndarray],
    compare_videos: list[np.ndarray],
    meta: list[dict],
    *,
    fps: int,
    tag_prefix: str,
) -> None:
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for i, (pred, compare) in enumerate(zip(pred_videos, compare_videos)):
        ep = meta[i]
        ret = int(ep["return"])
        tag = f"{tag_prefix}_ep{ep['episode_idx']}_ret{ret}"
        horizon = int(ep.get("horizon", 8))
        ctx_len = len(pred) - horizon
        labels = [f"ctx {t}" for t in range(ctx_len)] + [f"im {t}" for t in range(horizon)]
        pred_ann = annotate_frames_uint8(pred, labels)
        compare_ann = annotate_frames_uint8(compare, labels)
        pred_path = video_dir / f"imagine_{tag}.mp4"
        compare_path = video_dir / f"imagine_{tag}_gt_pred.mp4"
        iio.imwrite(pred_path, pred_ann, fps=fps, codec="h264")
        iio.imwrite(compare_path, compare_ann, fps=fps, codec="h264")
        print(f"Saved {pred_path} and {compare_path}")


def _write_panels(
    out_dir: Path,
    panel: np.ndarray,
    per_traj: list[np.ndarray],
    meta: list[dict],
    *,
    rl_ckpt: Path,
    context_len: int,
    horizon: int,
    high_min: float | None,
    low_max: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "rollout_panel_all.png", panel)
    for i, traj_panel in enumerate(per_traj):
        iio.imwrite(out_dir / f"rollout_traj_{i:02d}.png", traj_panel)
    payload: dict[str, Any] = {
        "rl_ckpt": str(rl_ckpt),
        "context_len": context_len,
        "horizon": horizon,
        "episodes": meta,
    }
    if high_min is not None:
        payload["min_episode_return"] = high_min
    if low_max is not None:
        payload["max_episode_return"] = low_max
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {out_dir / 'rollout_panel_all.png'} ({len(per_traj)} trajectories)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imagination rollout videos from RL checkpoint")
    parser.add_argument("config", type=Path)
    parser.add_argument("--rl-ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-episode-return", type=float, default=None)
    parser.add_argument("--max-episode-return", type=float, default=None)
    parser.add_argument("--max-items", type=int, default=1)
    parser.add_argument("--tag-prefix", type=str, default="rollout")
    parser.add_argument("--write-videos", action="store_true", help="Write mp4 rollout videos")
    parser.add_argument("--write-panel", action="store_true", help="Write rollout_panel_all.png")
    parser.add_argument(
        "--high-low-panel",
        action="store_true",
        help="Pick one high (>=900) and one low (<200) episode for a combined panel",
    )
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    imag = cfg.get("imagination", {})
    context_len = int(imag.get("context_len_min", 8))
    horizon = int(imag.horizon)
    flow_steps = int(imag.flow_steps)
    seq_len = context_len + 2 * horizon

    seq_len = context_len + 2 * horizon

    if args.high_low_panel:
        batch, picked_meta = _pick_high_low_episodes(
            cfg, high_min=900.0, low_max=200.0, seq_len=seq_len
        )
        max_items = 2
    else:
        batch, picked_meta = _pick_episodes(
            cfg,
            min_return=args.min_episode_return,
            max_return=args.max_episode_return,
            max_items=args.max_items,
            seq_len=seq_len,
        )
        max_items = args.max_items

    if not args.write_videos and not args.write_panel:
        args.write_videos = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.write_panel:
        panel, per_traj, panel_meta = render_imagination_rollout_panels(
            cfg,
            args.rl_ckpt,
            batch,
            picked_meta,
            context_len=context_len,
            horizon=horizon,
            flow_steps=flow_steps,
            device=device,
            max_items=max_items,
        )
        _write_panels(
            args.out_dir,
            panel,
            per_traj,
            panel_meta,
            rl_ckpt=args.rl_ckpt,
            context_len=context_len,
            horizon=horizon,
            high_min=900.0 if args.high_low_panel else args.min_episode_return,
            low_max=200.0 if args.high_low_panel else args.max_episode_return,
        )

    if args.write_videos:
        pred_videos, compare_videos, meta = render_imagination_rollout_videos(
            cfg,
            args.rl_ckpt,
            batch,
            picked_meta,
            context_len=context_len,
            horizon=horizon,
            flow_steps=flow_steps,
            device=device,
            max_items=max_items,
        )
        _write_videos(
            args.out_dir,
            pred_videos,
            compare_videos,
            meta,
            fps=args.video_fps,
            tag_prefix=args.tag_prefix,
        )
        payload = {
            "rl_ckpt": str(args.rl_ckpt),
            "context_len": context_len,
            "horizon": horizon,
            "episodes": meta,
        }
        if args.min_episode_return is not None:
            payload["min_episode_return"] = args.min_episode_return
        if args.max_episode_return is not None:
            payload["max_episode_return"] = args.max_episode_return
        (args.out_dir / "video_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.out_dir / 'video_metrics.json'}")


if __name__ == "__main__":
    main()
