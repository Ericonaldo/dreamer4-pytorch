"""Evaluation utilities: offline dynamics rollout and online DMC policy rollouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Process, get_context
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.agent import BCPolicy, load_bc_modules, resolve_eval_action_horizon
from dreamer4.env import make_dmc_env
from dreamer4.models.dynamics import (
    decode_packed_to_images,
    pack_bottleneck_to_spatial,
    sample_autoregressive_packed_sequence,
    sample_sliding_window_rollout_packed_sequence,
)


# ---------------------------------------------------------------------------
# Dynamics rollout — open-loop evaluation with dataset actions (decode pixels)
# ---------------------------------------------------------------------------


@dataclass
class DynamicsRolloutResult:
    metrics: dict[str, float]
    frames: torch.Tensor
    pred_frames: torch.Tensor
    pred_by_ctx_bkthwc: torch.Tensor
    ctx_lengths: list[int]


@dataclass
class DynamicsRolloutVideoResult:
    metrics: dict[str, float]
    gt_frames: torch.Tensor
    pred_frames: torch.Tensor


def _rollout_metrics(
    gt_h: torch.Tensor,
    pred_h: torch.Tensor,
    floor_h: torch.Tensor,
    *,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
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
    if extra:
        metrics.update(extra)
    return metrics


@torch.no_grad()
def dynamics_rollout_eval(
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
) -> DynamicsRolloutResult:
    """Open-loop latent rollout with dataset actions; returns metrics and decoded frames."""
    dynamics.eval()
    B, T = image_bthwc.shape[:2]
    length = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, length - 1)
    horizon = min(horizon, length - ctx_length)
    if horizon <= 0:
        raise ValueError(f"rollout horizon must be > 0 (T={T}, ctx={ctx_length})")

    frames = image_bthwc[:, :length]
    actions_eval = actions[:, :length]

    z_btld = tokenizer.encode_images(frames, patch_size)
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

    ctx_lengths = list(range(1, ctx_length + 1))
    pred_by_ctx = []
    for k in ctx_lengths:
        z_k = sample_autoregressive_packed_sequence(
            dynamics,
            z_gt_packed,
            actions_eval,
            k,
            length - k,
            flow_steps,
        )
        pred_by_ctx.append(
            decode_packed_to_images(
                tokenizer,
                z_k,
                patch_size,
                packing_factor,
                image_size,
                channels,
            )
        )
    pred_by_ctx_bkthwc = torch.stack(pred_by_ctx, dim=1)

    floor = frames.clone()
    if horizon > 0:
        floor[:, ctx_length:ctx_length + horizon] = frames[:, ctx_length - 1:ctx_length].expand(
            -1, horizon, -1, -1, -1
        )

    metrics = _rollout_metrics(
        frames[:, ctx_length:ctx_length + horizon],
        pred_frames[:, ctx_length:ctx_length + horizon],
        floor[:, ctx_length:ctx_length + horizon],
    )
    return DynamicsRolloutResult(
        metrics=metrics,
        frames=frames,
        pred_frames=pred_frames,
        pred_by_ctx_bkthwc=pred_by_ctx_bkthwc,
        ctx_lengths=ctx_lengths,
    )


@torch.no_grad()
def dynamics_rollout_video(
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
) -> DynamicsRolloutVideoResult:
    """Long rollout from obs[0] with growing then sliding attention; decode to frames."""
    dynamics.eval()
    total = rollout_length + 1
    if image_bthwc.shape[1] < total:
        raise ValueError(
            f"need at least {total} observation frames, got {image_bthwc.shape[1]}"
        )

    frames = image_bthwc[:, :total]
    actions_eval = actions[:, :total]
    B = min(frames.shape[0], max_items)

    z_btld = tokenizer.encode_images(frames[:B], patch_size)
    z_gt_packed = pack_bottleneck_to_spatial(z_btld, n_spatial, packing_factor)
    z0 = z_gt_packed[:, 0]

    z_pred_packed = sample_sliding_window_rollout_packed_sequence(
        dynamics,
        z0,
        actions_eval[:B],
        attn_window,
        rollout_length,
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

    gt_b = frames[:B]
    floor = gt_b.clone()
    floor[:, 1:] = gt_b[:, :1].expand(-1, rollout_length, -1, -1, -1)
    metrics = _rollout_metrics(
        gt_b[:, 1:],
        pred_frames[:B, 1:],
        floor[:, 1:],
        extra={"rollout_length": float(rollout_length), "attn_window": float(attn_window)},
    )
    return DynamicsRolloutVideoResult(
        metrics=metrics,
        gt_frames=gt_b,
        pred_frames=pred_frames[:B],
    )


# ---------------------------------------------------------------------------
# Online env eval — real DMC rollouts with BC policy (metrics, optional frames)
# ---------------------------------------------------------------------------


def make_eval_env(cfg: DictConfig):
    eval_cfg = cfg.get("eval", {})
    return make_dmc_env(
        str(eval_cfg.get("task", "walker_walk")),
        repeat=int(eval_cfg.get("action_repeat", 1)),
        image_size=int(eval_cfg.get("image_size", 64)),
        proprio=bool(eval_cfg.get("proprio", False)),
        image=bool(eval_cfg.get("image", True)),
        camera=int(eval_cfg.get("camera_id", -1)),
        max_episode_steps=int(eval_cfg.get("max_episode_steps", 1000)),
    )


@dataclass
class EpisodeStats:
    return_: float
    length: int


class RandomPolicy:
    def __init__(self, action_dim: int = 6):
        self.action_dim = int(action_dim)

    def reset(self, ids: list[int] | None = None) -> None:
        return

    def act(self, image_uint8: np.ndarray, ids: list[int] | None = None) -> np.ndarray:
        del image_uint8, ids
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)


def run_episodes(env, policy, num_episodes: int) -> list[EpisodeStats]:
    stats: list[EpisodeStats] = []
    for _ in range(num_episodes):
        policy.reset()
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0
        while True:
            action = policy.act(obs["image"])
            obs = env.step(action)
            ep_return += float(obs["reward"])
            ep_len += 1
            if obs["is_last"]:
                break
        stats.append(EpisodeStats(return_=ep_return, length=ep_len))
    return stats


def summarize_episodes(stats: list[EpisodeStats]) -> dict[str, float]:
    returns = np.array([s.return_ for s in stats], dtype=np.float64)
    lengths = np.array([s.length for s in stats], dtype=np.float64)
    return {
        "episodes": len(stats),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "length_min": float(lengths.min()),
        "length_max": float(lengths.max()),
    }


def _split_episodes(total: int, n: int) -> list[int]:
    n = max(1, n)
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _rollout_on_device(
    cfg: DictConfig,
    device: torch.device,
    num_episodes: int,
    *,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
) -> list[EpisodeStats]:
    eval_cfg = cfg.get("eval", {})
    num_envs = min(int(eval_cfg.get("num_envs", 4)), num_episodes)
    envs = [make_eval_env(cfg) for _ in range(num_envs)]
    model, tokenizer = load_bc_modules(
        cfg, device, model_state=model_state, tokenizer_state=tokenizer_state
    )
    policy = BCPolicy(cfg, device, model=model, tokenizer=tokenizer, num_envs=num_envs)

    stats: list[EpisodeStats] = []
    obs = [env.reset() for env in envs]
    rets = np.zeros(num_envs, dtype=np.float64)
    lens = np.zeros(num_envs, dtype=np.int64)
    done = [False] * num_envs
    policy.reset()

    while len(stats) < num_episodes:
        active = [i for i in range(num_envs) if not done[i]]
        if not active:
            for i in range(num_envs):
                if len(stats) >= num_episodes:
                    done[i] = True
                    continue
                obs[i] = envs[i].reset()
                rets[i] = 0.0
                lens[i] = 0
                done[i] = False
            policy.reset([i for i in range(num_envs) if not done[i]])
            active = [i for i in range(num_envs) if not done[i]]
            if not active:
                break

        images = np.stack([obs[i]["image"] for i in active])
        actions = policy.act(images, ids=active)
        for j, i in enumerate(active):
            obs[i] = envs[i].step(actions[j])
            rets[i] += float(obs[i]["reward"])
            lens[i] += 1
            if obs[i]["is_last"]:
                stats.append(EpisodeStats(return_=rets[i], length=int(lens[i])))
                done[i] = True
    for env in envs:
        try:
            env.close()
        except Exception:
            pass
    return stats


def _worker(
    gpu_id: int | None,
    num_episodes: int,
    cfg_dict: dict[str, Any],
    model_state: dict[str, torch.Tensor] | None,
    tokenizer_state: dict[str, torch.Tensor] | None,
    out_queue,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = OmegaConf.create(cfg_dict)
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    stats = _rollout_on_device(
        cfg,
        device,
        num_episodes,
        model_state=model_state,
        tokenizer_state=tokenizer_state,
    )
    out_queue.put(stats)


def parse_eval_gpu_ids(raw: Any) -> list[int | None]:
    if raw is None:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [None]
    if isinstance(raw, str) and raw.lower() == "all":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [None]
    if isinstance(raw, int):
        return [raw] if torch.cuda.is_available() else [None]
    ids = [int(x) for x in raw]
    if not ids:
        return [0] if torch.cuda.is_available() else [None]
    return ids


def resolve_async_gpu_ids(cfg: DictConfig) -> list[int | None]:
    eval_cfg = cfg.get("eval", {})
    raw = eval_cfg.get("async_gpu_ids", eval_cfg.get("gpu_ids", "all"))
    return parse_eval_gpu_ids(raw)


def run_bc_env_eval(
    cfg: DictConfig,
    *,
    num_episodes: int | None = None,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
    gpu_ids: list[int] | None = None,
) -> dict[str, float]:
    episodes = int(num_episodes or cfg.get("eval", {}).get("episodes", 10))
    action_horizon = resolve_eval_action_horizon(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if model_state is not None:
        model_state = {k: v.detach().cpu() for k, v in model_state.items()}
    if tokenizer_state is not None:
        tokenizer_state = {k: v.detach().cpu() for k, v in tokenizer_state.items()}

    if gpu_ids is None:
        gpu_ids = parse_eval_gpu_ids("all")

    counts = _split_episodes(episodes, len(gpu_ids))
    if len(gpu_ids) == 1 or all(c == 0 for c in counts[1:]):
        gid = gpu_ids[0]
        stats = _rollout_on_device(
            cfg,
            torch.device(f"cuda:{gid}") if gid is not None and torch.cuda.is_available() else torch.device("cpu"),
            counts[0],
            model_state=model_state,
            tokenizer_state=tokenizer_state,
        )
        metrics = summarize_episodes(stats)
        metrics["action_horizon"] = float(action_horizon)
        return metrics

    ctx = get_context("spawn")
    out_queue = ctx.Queue()
    procs: list[Process] = []
    for gpu_id, count in zip(gpu_ids, counts):
        if count <= 0:
            continue
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, count, cfg_dict, model_state, tokenizer_state, out_queue),
        )
        p.start()
        procs.append(p)

    all_stats: list[EpisodeStats] = []
    for _ in procs:
        all_stats.extend(out_queue.get())
    for p in procs:
        p.join()
    metrics = summarize_episodes(all_stats)
    metrics["action_horizon"] = float(action_horizon)
    return metrics


def run_bc_policy_episode(
    cfg: DictConfig,
    device: torch.device,
    *,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Run one BC env episode; returns (T,H,W,C) uint8 frames, return, length."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = make_eval_env(cfg)

    if model_state is not None or tokenizer_state is not None:
        model, tokenizer = load_bc_modules(
            cfg, device, model_state=model_state, tokenizer_state=tokenizer_state
        )
        policy = BCPolicy(cfg, device, model=model, tokenizer=tokenizer)
    else:
        policy = BCPolicy(cfg, device)

    policy.reset()
    obs = env.reset()
    frames = [obs["image"].copy()]
    ep_return = 0.0
    ep_len = 0
    while not obs["is_last"]:
        action = policy.act(obs["image"])
        obs = env.step(action)
        frames.append(obs["image"].copy())
        ep_return += float(obs["reward"])
        ep_len += 1

    try:
        env.close()
    except Exception:
        pass

    return np.stack(frames, axis=0), ep_return, ep_len
