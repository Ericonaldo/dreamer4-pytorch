"""Standalone dynamics rollout evaluation for a trained DynamicsModel checkpoint.

Loads a frozen tokenizer (from ``tokenizer_ckpt`` in the dynamics config) and a
dynamics checkpoint, replays **dataset actions** (open-loop), and compares
decoded predictions to ground-truth frames.

Panel eval (default)
--------------------
For each trajectory, runs autoregressive latent sampling with multiple context
lengths ``ctx=1..rollout_ctx``. GT context frames are kept; later frames are
model predictions. Writes:

- ``rollout_panel_all.png`` — all trajectories stacked (GT row + ctx rows)
- ``rollout_traj_{i:02d}.png`` — one annotated panel per trajectory
- ``metrics.json`` — ``rollout_mse``, ``rollout_psnr``, and repeat-last-frame floor

Uses ``train.rollout_ctx``, ``train.rollout_horizon``, and
``train.rollout_flow_steps`` from the config (same as validation during training).

Long rollout videos (``--rollout-video``)
-----------------------------------------
Generates mp4s beyond the training horizon using a single GT frame ``obs[0]`` as
context. For global step ``g`` when predicting frame ``g``:

- while ``g <= L``: attend to **all** past latents (growing window)
- while ``g > L``: attend to the **previous L** latents only (sliding window)

Actions are aligned to the same global time indices as the latent window (ref
``interactive.py`` ``ctx_window``). ``L`` defaults to ``train.rollout_ctx``;
override with ``--attn-window``. Rollout length (steps after ``obs[0]``) defaults
to 64 (``--rollout-length``). Writes:

- ``videos/rollout_video_{i:02d}.mp4`` — decoded rollout
- ``videos/rollout_video_{i:02d}_gt_pred.mp4`` — GT on top, prediction on bottom
- ``video_metrics.json`` — MSE / PSNR on predicted frames (excluding ``obs[0]``)

Episode selection
-----------------
- ``--split train|val|all`` — episode hold-out split (``data.val_fraction``)
- ``--min-episode-return`` / ``--max-episode-return`` — pick episodes by cumulative return
- ``--by-reward-bands`` — run panel eval for each Walker Walk return band; writes
  ``summary.json`` and one subdirectory per band

Example::

    uv run python -m dreamer4.eval_dynamics_rollout \\
      configs/walker_walk/dynamics.yaml \\
      --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
      --out-dir logs/walker_walk/dynamics/rollout_eval \\
      --split val --max-items 4

    uv run python -m dreamer4.eval_dynamics_rollout \\
      configs/walker_walk/dynamics.yaml \\
      --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
      --out-dir logs/walker_walk/dynamics/rollout_videos \\
      --rollout-video --rollout-length 64 --attn-window 8 --max-items 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dreamer4.config import load_config
from dreamer4.data import (
    GranularEpisodeDataset,
    align_dynamics_batch,
    collate_episodes,
    episode_cumulative_returns,
    reward_band_counts,
    select_episodes_by_return,
    split_episode_indices,
    transition_window_offset,
)
from dreamer4.models import DynamicsModel, build_tokenizer
from dreamer4.models.dynamics import run_dynamics_rollout_eval, run_dynamics_rollout_video

# Walker-walk cumulative return bands (episode sum; max ~1000 at 1000 steps).
DEFAULT_REWARD_BANDS: list[tuple[str, float, float]] = [
    ("fallen_0_50", 0.0, 50.0),
    ("fallen_50_200", 50.0, 200.0),
    ("weak_200_500", 200.0, 500.0),
    ("partial_500_900", 500.0, 900.0),
    ("standing_900_970", 900.0, 970.0),
    ("expert_970_plus", 970.0, 1001.0),
]


def _load_state(module: torch.nn.Module, ckpt_path: str, *, prefix: str = "model.") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


def _episode_filter_for_split(cfg: DictConfig, split: str) -> set[int] | None:
    if split == "all":
        return None
    window_mode = str(cfg.data.get("window_mode", "transition"))
    probe = GranularEpisodeDataset(
        cfg.data.path, cfg.data.seq_len, cfg.data.obs_mode, window_mode=window_mode
    )
    val_fraction = float(cfg.data.get("val_fraction", 0.05))
    train_episodes, val_episodes = split_episode_indices(
        probe.num_episodes, val_fraction, int(cfg.data.get("val_seed", 0))
    )
    if split == "train":
        return set(train_episodes)
    if split == "val":
        return set(val_episodes)
    raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")


def _load_rollout_batch_from_picked(
    cfg: DictConfig,
    picked: list[tuple[int, float]],
    *,
    seq_len: int | None = None,
) -> tuple[Any, list[dict]]:
    window_mode = str(cfg.data.get("window_mode", "transition"))
    if window_mode != "transition":
        raise ValueError("episode picking requires data.window_mode=transition")
    if not picked:
        raise ValueError("no episodes selected")

    effective_seq_len = int(seq_len if seq_len is not None else cfg.data.seq_len)
    ds = GranularEpisodeDataset(
        cfg.data.path,
        effective_seq_len,
        cfg.data.obs_mode,
        episode_indices=[ep_idx for ep_idx, _ in picked],
        window_mode=window_mode,
    )
    items = []
    picked_meta = []
    for ep_idx, ret in picked:
        offset = transition_window_offset(
            ds.episode_length(ep_idx), effective_seq_len, position="middle"
        )
        items.append(ds.get_transition_window(ep_idx, offset))
        picked_meta.append({"episode_idx": ep_idx, "return": ret, "offset": offset})

    batch = collate_episodes(items)
    if batch.image is None:
        raise ValueError("Rollout eval requires images")
    return batch, picked_meta


def _load_rollout_batch(
    cfg: DictConfig,
    split: str,
    max_items: int,
    min_return: float | None,
    max_return: float | None,
    returns: np.ndarray | None = None,
    episode_filter: set[int] | None = None,
    *,
    seq_len: int | None = None,
) -> tuple[Any, list[dict] | None]:
    if min_return is not None or max_return is not None:
        if returns is None:
            _, returns = episode_cumulative_returns(cfg.data.path)
        if episode_filter is None:
            episode_filter = _episode_filter_for_split(cfg, split)
        low = float(min_return if min_return is not None else 0.0)
        high = float(max_return) if max_return is not None else None
        picked = select_episodes_by_return(
            returns,
            low,
            episode_indices=episode_filter,
            top_k=max_items,
            max_return=high,
        )
        if not picked:
            raise ValueError(
                f"no episodes in return [{low}, {high}) (split={split}, "
                f"max_return={float(returns.max()):.1f})"
            )
        print(f"Selected {len(picked)} episodes in [{low}, {high}): {picked}")
        return _load_rollout_batch_from_picked(cfg, picked, seq_len=seq_len)

    window_mode = str(cfg.data.get("window_mode", "transition"))
    if split not in ("train", "val"):
        raise ValueError(f"default batch loader requires split train|val, got {split!r}")
    episode_indices = list(_episode_filter_for_split(cfg, split) or [])
    effective_seq_len = int(seq_len if seq_len is not None else cfg.data.seq_len)
    ds = GranularEpisodeDataset(
        cfg.data.path,
        effective_seq_len,
        cfg.data.obs_mode,
        episode_indices=episode_indices,
        window_mode=window_mode,
    )
    batch = next(
        iter(
            DataLoader(
                ds,
                batch_size=max_items,
                shuffle=False,
                collate_fn=collate_episodes,
            )
        )
    )
    if batch.image is None:
        raise ValueError("Rollout eval requires images")
    return batch, None


def _dynamics_model_cfg(cfg: DictConfig) -> tuple[dict[str, Any], int, str]:
    """DynamicsModel kwargs, packing_factor, checkpoint state_dict prefix."""
    if cfg.model.get("dynamics") is not None and "embed_dim" in cfg.model.dynamics:
        dyn = OmegaConf.to_container(cfg.model.dynamics, resolve=True)
        pf = int(cfg.model.dynamics.get("packing_factor", 1))
        stage = str(cfg.get("stage", ""))
        prefix = "model.dynamics." if stage in ("bc", "bc_dynamics", "rl") else "model."
        return dyn, pf, prefix
    raw = OmegaConf.to_container(cfg.model, resolve=True)
    pf = int(cfg.model.get("packing_factor", 1))
    return raw, pf, "model."


def _build_models(cfg: DictConfig, dynamics_ckpt: Path, device: torch.device):
    tokenizer = build_tokenizer(cfg.model.tokenizer)
    if cfg.get("tokenizer_ckpt"):
        _load_state(tokenizer, cfg.tokenizer_ckpt, prefix="model.")
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    n_latents = tokenizer.encoder.n_latents
    latent_dim = tokenizer.encoder.bottleneck_proj.out_features
    dyn_cfg, packing_factor, ckpt_prefix = _dynamics_model_cfg(cfg)
    n_spatial = n_latents // packing_factor
    patch_size = int(cfg.model.tokenizer.patch_size)
    image_size = int(cfg.model.tokenizer.image_size)
    channels = int(cfg.model.tokenizer.channels)

    dynamics = DynamicsModel(dyn_cfg, n_latents=n_latents, latent_dim=latent_dim)
    _load_state(dynamics, str(dynamics_ckpt), prefix=ckpt_prefix)
    dynamics.eval()
    dynamics.to(device)
    tokenizer.to(device)

    rollout_kw = {
        "patch_size": patch_size,
        "packing_factor": packing_factor,
        "n_spatial": n_spatial,
        "image_size": image_size,
        "channels": channels,
        "ctx_length": int(cfg.train.get("rollout_ctx", 8)),
        "horizon": int(cfg.train.get("rollout_horizon", 8)),
        "flow_steps": int(cfg.train.get("rollout_flow_steps", 8)),
    }
    return dynamics, tokenizer, rollout_kw


def _write_rollout_videos(
    out_dir: Path,
    pred_videos: list[np.ndarray],
    compare_videos: list[np.ndarray],
    fps: int,
    *,
    annotate_steps: bool = False,
    names: list[str] | None = None,
) -> None:
    from dreamer4.video_utils import annotate_frames_uint8

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (pred, compare) in enumerate(zip(pred_videos, compare_videos)):
        tag = names[i] if names and i < len(names) else f"{i:02d}"
        pred_path = out_dir / f"rollout_video_{tag}.mp4"
        compare_path = out_dir / f"rollout_video_{tag}_gt_pred.mp4"
        if annotate_steps:
            pred = annotate_frames_uint8(pred)
            compare = annotate_frames_uint8(compare)
        iio.imwrite(pred_path, pred, fps=fps, codec="h264")
        iio.imwrite(compare_path, compare, fps=fps, codec="h264")
        print(f"Saved {pred_path} and {compare_path}")


def _run_rollout_video_and_save(
    cfg: DictConfig,
    dynamics: DynamicsModel,
    tokenizer: torch.nn.Module,
    batch: Any,
    out_dir: Path,
    max_items: int,
    *,
    attn_window: int,
    rollout_length: int,
    flow_steps: int,
    fps: int,
    extra_metrics: dict[str, Any],
    annotate_steps: bool = False,
) -> dict[str, Any]:
    device = next(dynamics.parameters()).device
    image, action, _ = align_dynamics_batch(batch.image.to(device), batch.action.to(device))
    assert image is not None

    patch_size = int(cfg.model.tokenizer.patch_size)
    _, packing_factor, _ = _dynamics_model_cfg(cfg)
    n_spatial = tokenizer.encoder.n_latents // packing_factor
    image_size = int(cfg.model.tokenizer.image_size)
    channels = int(cfg.model.tokenizer.channels)

    metrics, pred_videos, compare_videos, _, _ = run_dynamics_rollout_video(
        dynamics,
        tokenizer,
        image,
        action,
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
    metrics.update(extra_metrics)

    video_dir = out_dir / "videos"
    episode_meta = extra_metrics.get("episodes")
    video_names = None
    if episode_meta:
        video_names = [
            f"ep{meta['episode_idx']}_ret{meta['return']:.0f}" for meta in episode_meta[:max_items]
        ]
    _write_rollout_videos(
        video_dir,
        pred_videos,
        compare_videos,
        fps,
        annotate_steps=annotate_steps,
        names=video_names,
    )
    (out_dir / "video_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def _run_rollout_and_save(
    cfg: DictConfig,
    dynamics: DynamicsModel,
    tokenizer: torch.nn.Module,
    batch: Any,
    out_dir: Path,
    max_items: int,
    rollout_kw: dict[str, Any],
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    device = next(dynamics.parameters()).device
    image, action, _ = align_dynamics_batch(batch.image.to(device), batch.action.to(device))
    metrics, panel, _, _, per_traj = run_dynamics_rollout_eval(
        dynamics,
        tokenizer,
        image,
        action,
        max_items=max_items,
        return_per_traj=True,
        **rollout_kw,
    )
    metrics.update(extra_metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "rollout_panel_all.png", panel)
    for i, traj_panel in enumerate(per_traj):
        iio.imwrite(out_dir / f"rollout_traj_{i:02d}.png", traj_panel)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out_dir / 'rollout_panel_all.png'} ({len(per_traj)} trajectories)")
    return metrics


def _run_reward_bands(
    cfg: DictConfig,
    dynamics_ckpt: Path,
    out_dir: Path,
    split: str,
    max_items: int,
    bands: list[tuple[str, float, float]],
    device: torch.device,
) -> None:
    _, returns = episode_cumulative_returns(cfg.data.path)
    episode_filter = _episode_filter_for_split(cfg, split)
    population = reward_band_counts(returns, bands)

    dynamics, tokenizer, rollout_kw = _build_models(cfg, dynamics_ckpt, device)
    summary: dict[str, Any] = {
        "split": split,
        "dynamics_ckpt": str(dynamics_ckpt),
        "max_items_per_band": max_items,
        "dataset_return_mean": float(returns.mean()),
        "dataset_return_median": float(np.median(returns)),
        "dataset_return_max": float(returns.max()),
        "population": population,
        "bands": [],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for band_info in population:
        name = band_info["name"]
        low, high = band_info["low"], band_info["high"]
        entry: dict[str, Any] = {"name": name, "low": low, "high": high, **band_info}
        band_dir = out_dir / name

        if band_info["count"] == 0:
            entry["status"] = "skipped_empty"
            summary["bands"].append(entry)
            print(f"[{name}] skip: 0 episodes in [{low}, {high})")
            continue

        picked = select_episodes_by_return(
            returns,
            low,
            episode_indices=episode_filter,
            top_k=max_items,
            max_return=high,
        )
        if not picked:
            entry["status"] = "skipped_no_pick"
            summary["bands"].append(entry)
            continue

        try:
            batch, episode_meta = _load_rollout_batch_from_picked(cfg, picked)
            metrics = _run_rollout_and_save(
                cfg,
                dynamics,
                tokenizer,
                batch,
                band_dir,
                max_items,
                rollout_kw,
                {
                    "split": split,
                    "return_band": name,
                    "return_low": low,
                    "return_high": high,
                    "episodes": episode_meta,
                    "picked_return_mean": float(np.mean([r for _, r in picked])),
                },
            )
            entry["status"] = "ok"
            entry["rollout"] = metrics
            entry["episodes"] = episode_meta
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"[{name}] error: {exc}")

        summary["bands"].append(entry)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote band summary to {out_dir / 'summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dynamics open-loop rollout from a checkpoint (panels, metrics, optional mp4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Val split, static panels + metrics (uses train.rollout_ctx / rollout_horizon)
  python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \\
    --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
    --out-dir logs/walker_walk/dynamics/rollout_eval --split val

  # Long rollout videos: context=obs[0], sliding attention after L frames
  python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \\
    --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
    --out-dir logs/walker_walk/dynamics/rollout_videos \\
    --rollout-video --rollout-length 64 --attn-window 8 --video-fps 15

  # Rollout on high-return episodes only
  python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \\
    --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
    --out-dir logs/walker_walk/dynamics/rollout_expert \\
    --min-episode-return 970 --max-items 2

  # Per reward band (fallen / weak / expert, ...)
  python -m dreamer4.eval_dynamics_rollout configs/walker_walk/dynamics.yaml \\
    --dynamics-ckpt logs/walker_walk/dynamics/checkpoints/last.ckpt \\
    --out-dir logs/walker_walk/dynamics/rollout_bands --by-reward-bands
""",
    )
    parser.add_argument("config", type=Path, help="Dynamics YAML config")
    parser.add_argument("--dynamics-ckpt", type=Path, required=True, help="Lightning dynamics checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for images + metrics.json")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val", help="Episode split")
    parser.add_argument(
        "--min-episode-return",
        type=float,
        default=None,
        help="Pick episodes with cumulative reward >= this (inclusive)",
    )
    parser.add_argument(
        "--max-episode-return",
        type=float,
        default=None,
        help="Pick episodes with cumulative reward < this (exclusive); use with --min-episode-return",
    )
    parser.add_argument(
        "--by-reward-bands",
        action="store_true",
        help="Run rollout for each default reward band; writes summary.json + per-band subdirs",
    )
    parser.add_argument("--max-items", type=int, default=4, help="Number of trajectories per band / batch")
    parser.add_argument(
        "--rollout-video",
        action="store_true",
        help="Export long rollout mp4s (context=obs[0], sliding attention after L frames)",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=64,
        help="Autoregressive rollout steps after obs[0] (default 64)",
    )
    parser.add_argument(
        "--attn-window",
        type=int,
        default=None,
        help="Attention window L; first L steps attend to all past, then last L only (default: train.rollout_ctx)",
    )
    parser.add_argument("--video-fps", type=int, default=15, help="FPS for rollout mp4 export")
    parser.add_argument(
        "--annotate-steps",
        action="store_true",
        help="Overlay step 0, 1, ... on each rollout mp4 frame (top-right)",
    )
    parser.add_argument(
        "--skip-panel",
        action="store_true",
        help="Skip static rollout panels (e.g. when only exporting --rollout-video)",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    attn_window = int(
        args.attn_window if args.attn_window is not None else cfg.train.get("rollout_ctx", 8)
    )
    flow_steps = int(cfg.train.get("rollout_flow_steps", 8))

    if args.by_reward_bands:
        _run_reward_bands(
            cfg,
            args.dynamics_ckpt,
            args.out_dir,
            args.split,
            args.max_items,
            DEFAULT_REWARD_BANDS,
            device,
        )
        return

    dynamics, tokenizer, rollout_kw = _build_models(cfg, args.dynamics_ckpt, device)
    video_seq_len = args.rollout_length if args.rollout_video else None
    batch, episode_meta = _load_rollout_batch(
        cfg,
        args.split,
        args.max_items,
        args.min_episode_return,
        args.max_episode_return,
        seq_len=video_seq_len,
    )
    extra: dict[str, Any] = {"split": args.split}
    if args.min_episode_return is not None:
        extra["min_episode_return"] = args.min_episode_return
    if args.max_episode_return is not None:
        extra["max_episode_return"] = args.max_episode_return
    if episode_meta:
        extra["episodes"] = episode_meta

    if args.rollout_video:
        _run_rollout_video_and_save(
            cfg,
            dynamics,
            tokenizer,
            batch,
            args.out_dir,
            args.max_items,
            attn_window=attn_window,
            rollout_length=args.rollout_length,
            flow_steps=flow_steps,
            fps=args.video_fps,
            extra_metrics=extra,
            annotate_steps=args.annotate_steps,
        )
        if args.skip_panel:
            return

    _run_rollout_and_save(
        cfg,
        dynamics,
        tokenizer,
        batch,
        args.out_dir,
        args.max_items,
        rollout_kw,
        extra,
    )


if __name__ == "__main__":
    main()
