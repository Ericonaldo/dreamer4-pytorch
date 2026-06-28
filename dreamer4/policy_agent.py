"""Online DMC policies, BC env eval, and training-time async eval."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.models import BCModel, build_tokenizer
from dreamer4.models.policy import POLICY_ENV_ACTION_SLOT
from dreamer4.models.dynamics import pack_bottleneck_to_spatial
from dreamer4.models.tokenizer import encode_images


def _load_state(module: nn.Module, ckpt_path: str, *, prefix: str = "model.") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


def load_bc_modules(
    cfg: DictConfig,
    device: torch.device,
    *,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
) -> tuple[BCModel, nn.Module]:
    tokenizer = build_tokenizer(cfg.model.tokenizer)
    if tokenizer_state is not None:
        tokenizer.load_state_dict(tokenizer_state, strict=True)
    elif cfg.get("tokenizer_ckpt"):
        _load_state(tokenizer, cfg.tokenizer_ckpt, prefix="model.")

    n_latents = tokenizer.encoder.n_latents
    latent_dim = tokenizer.encoder.bottleneck_proj.out_features
    model = BCModel(
        cfg.model.dynamics,
        n_latents=n_latents,
        latent_dim=latent_dim,
        heads_cfg=cfg.model,
    )
    if model_state is not None:
        model.load_state_dict(model_state, strict=False)
    elif cfg.get("bc_ckpt"):
        ckpt = torch.load(cfg.bc_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        filtered = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.") and "attn_mask" not in k
        }
        model.load_state_dict(filtered, strict=False)
    elif cfg.get("dynamics_ckpt"):
        _load_state(model.dynamics, cfg.dynamics_ckpt, prefix="model.")

    tokenizer.eval()
    model.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    tokenizer.to(device)
    model.to(device)
    return model, tokenizer


def _pad_stack(seqs: list[torch.Tensor]) -> torch.Tensor:
    t_max = max(s.shape[0] for s in seqs)
    padded = []
    for s in seqs:
        if s.shape[0] < t_max:
            pad = s.new_zeros(t_max - s.shape[0], *s.shape[1:])
            s = torch.cat([pad, s], dim=0)
        padded.append(s)
    return torch.stack(padded, dim=0)


def _make_eval_env(cfg: DictConfig):
    from dreamer4.env import make_dmc_env

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


def resolve_eval_action_horizon(cfg: DictConfig) -> int:
    """Open-loop env steps per model forward; 1 = replan from current obs each step."""
    model_h = int(cfg.model.get("action_horizon", 8))
    eval_h = int(cfg.get("eval", {}).get("action_horizon", 1))
    # MTP slot l predicts a_{t+l}; after observing s_t use slots 1..L for env steps.
    return max(1, min(eval_h, model_h - 1))


class BCPolicy:
    """Online BC agent with optional batched env slots (encode → BCModel → MTP action)."""

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        *,
        model: BCModel | None = None,
        tokenizer: nn.Module | None = None,
        num_envs: int = 1,
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.max_history = int(cfg.eval.get("max_history", 16))
        self.action_dim = int(cfg.model.dynamics.action_dim)
        self.open_loop_steps = resolve_eval_action_horizon(cfg)

        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_bc_modules(cfg, device)
        else:
            self.model = model
            self.tokenizer = tokenizer

        n_latents = self.tokenizer.encoder.n_latents
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor

        self._z: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._a: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._pending: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._last_executed: list[np.ndarray | None] = [None] * self.num_envs

    def reset(self, ids: list[int] | None = None) -> None:
        if ids is None:
            ids = list(range(self.num_envs))
        for i in ids:
            self._z[i].clear()
            self._a[i].clear()
            self._pending[i].clear()
            self._last_executed[i] = None

    def _push_action(self, slot: int, action: np.ndarray) -> None:
        self._a[slot].append(torch.from_numpy(action.astype(np.float32)).to(self.device))
        if len(self._a[slot]) > self.max_history:
            self._a[slot].pop(0)

    def _commit_last_action(self, ids: list[int]) -> None:
        """Record the last env action before appending a new observation."""
        for i in ids:
            if self._z[i] and self._last_executed[i] is not None:
                self._push_action(i, self._last_executed[i])
                self._last_executed[i] = None

    def _aligned_actions(self, slot: int, t: int) -> torch.Tensor:
        """(t, A) with a_0=0; a_k is the action that produced z_k."""
        out = torch.zeros(t, self.action_dim, device=self.device)
        n_transitions = min(len(self._a[slot]), max(0, t - 1))
        if n_transitions > 0:
            a_hist = torch.stack(self._a[slot][-n_transitions:], dim=0)
            out[1 : 1 + n_transitions] = a_hist
        return out

    def _append_observations(self, images: np.ndarray, ids: list[int]) -> None:
        imgs = (
            torch.from_numpy(np.ascontiguousarray(images))
            .to(self.device)
            .float()
            .div_(255.0)
            .unsqueeze(1)
        )
        z = encode_images(self.tokenizer, imgs, self.patch_size)
        packed = pack_bottleneck_to_spatial(z, self.n_spatial, self.packing_factor)[:, 0]
        for j, i in enumerate(ids):
            self._z[i].append(packed[j])
            if len(self._z[i]) > self.max_history:
                self._z[i].pop(0)
                if self._a[i]:
                    self._a[i].pop(0)

    def _observe_env(self, images: np.ndarray, ids: list[int]) -> None:
        """Commit prior actions and encode current env frames into (z, a) history."""
        self._commit_last_action(ids)
        self._append_observations(images, ids)

    def _forward_mtp_actions(self, ids: list[int]) -> np.ndarray:
        z_seqs, a_seqs = [], []
        for i in ids:
            z_seq = torch.stack(self._z[i], dim=0)
            z_seqs.append(z_seq)
            a_seqs.append(self._aligned_actions(i, z_seq.shape[0]))

        z_batch = _pad_stack(z_seqs)
        a_batch = _pad_stack(a_seqs)
        outputs = self.model(z_batch, a_batch)
        # Slots 1..L predict a_{t+1}..a_{t+L} for open-loop env steps after observing s_t.
        end = POLICY_ENV_ACTION_SLOT + self.open_loop_steps
        return outputs.action[:, -1, POLICY_ENV_ACTION_SLOT:end].float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act(self, images: np.ndarray, ids: list[int] | None = None) -> np.ndarray:
        single = images.ndim == 3
        if single:
            images = images[np.newaxis, ...]
            ids = [0]

        actions_out: dict[int, np.ndarray] = {}
        need_forward: list[int] = []
        for j, i in enumerate(ids):
            if self._pending[i]:
                self._observe_env(images[j : j + 1], [i])
                action = self._pending[i].pop(0)
                self._last_executed[i] = action
                actions_out[i] = action
            else:
                need_forward.append(i)

        if need_forward:
            fwd_idx = [j for j, i in enumerate(ids) if i in need_forward]
            self._observe_env(images[fwd_idx], need_forward)
            mtp = self._forward_mtp_actions(need_forward)
            for j, i in enumerate(need_forward):
                step_actions = mtp[j]
                first = step_actions[0]
                self._last_executed[i] = first
                actions_out[i] = first
                if step_actions.shape[0] > 1:
                    self._pending[i].extend(step_actions[1:])

        ordered = np.stack([actions_out[i] for i in ids], axis=0)
        return ordered[0] if single else ordered


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
    envs = [_make_eval_env(cfg) for _ in range(num_envs)]
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
    """Parse eval.gpu_ids / async_gpu_ids: all | int | list[int]."""
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


def _resolve_async_gpu_ids(cfg: DictConfig) -> list[int | None]:
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
    """Multi-GPU eval: split episodes across GPUs, batched envs per GPU."""
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


def run_bc_policy_video(
    cfg: DictConfig,
    out_path: Path,
    *,
    fps: int = 20,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
    gpu_id: int | None = 0,
    annotate_steps: bool = False,
) -> dict[str, Any]:
    """Record one online env episode as mp4 (BC policy)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = _make_eval_env(cfg)

    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

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

    frames_arr = np.stack(frames, axis=0)
    if annotate_steps:
        from dreamer4.video_utils import annotate_frames_uint8

        labels = [f"step {t}" for t in range(len(frames))]
        labels[-1] = f"step {len(frames) - 1}  return {ep_return:.0f}"
        frames_arr = annotate_frames_uint8(frames_arr, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames_arr, fps=fps, codec="h264")
    return {
        "return": ep_return,
        "length": ep_len,
        "frames": len(frames),
        "video": str(out_path),
        "fps": fps,
    }


def _async_entry(
    step: int,
    run_dir: str,
    cfg_dict: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    tokenizer_state: dict[str, torch.Tensor],
    gpu_ids: list[int | None],
    out_queue,
) -> None:
    try:
        cfg = OmegaConf.create(cfg_dict)
        metrics = run_bc_env_eval(
            cfg,
            model_state=model_state,
            tokenizer_state=tokenizer_state,
            gpu_ids=gpu_ids,
        )
        eval_dir = Path(run_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        payload = {"step": step, **metrics}
        (eval_dir / f"env_step_{step:08d}.json").write_text(json.dumps(payload, indent=2) + "\n")
        out_queue.put((step, metrics, None))
    except Exception as exc:
        out_queue.put((step, {}, str(exc)))


class AsyncBCEval:
    """Non-blocking BC env eval for training (rank 0 only)."""

    def __init__(self) -> None:
        self._proc: Process | None = None
        self._queue = None
        self._ctx = None
        self._launch_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    @property
    def pending(self) -> bool:
        if self.running:
            return True
        t = self._launch_thread
        return t is not None and t.is_alive()

    def start(
        self,
        step: int,
        cfg: DictConfig,
        model: nn.Module,
        tokenizer: nn.Module,
        run_dir: Path,
    ) -> None:
        if self.pending:
            return
        self._ctx = get_context("spawn")
        self._queue = self._ctx.Queue()
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        gpu_ids = _resolve_async_gpu_ids(cfg)

        def _launch() -> None:
            try:
                model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                tokenizer_state = {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()}
                proc = self._ctx.Process(
                    target=_async_entry,
                    args=(
                        step,
                        str(run_dir),
                        cfg_dict,
                        model_state,
                        tokenizer_state,
                        gpu_ids,
                        self._queue,
                    ),
                    daemon=False,
                )
                proc.start()
                self._proc = proc
            finally:
                self._launch_thread = None

        self._launch_thread = threading.Thread(target=_launch, daemon=True)
        self._launch_thread.start()

    def poll(self, module) -> None:
        if self._queue is None:
            return
        try:
            step, metrics, err = self._queue.get_nowait()
        except queue.Empty:
            return
        proc = self._proc
        self._proc = None
        self._queue = None
        self._ctx = None
        if proc is not None and proc.is_alive():
            threading.Thread(target=proc.join, daemon=True).start()
        if err:
            from lightning.pytorch.utilities import rank_zero_warn

            rank_zero_warn(f"BC env eval failed at step {step}: {err}")
            return
        log_dict = {f"val/env_{key}": float(value) for key, value in metrics.items()}
        trainer = getattr(module, "trainer", None)
        loggers = getattr(trainer, "loggers", None) if trainer is not None else None
        if isinstance(loggers, (list, tuple)) and loggers:
            for logger in loggers:
                logger.log_metrics(log_dict, step=step)
        else:
            for key, value in metrics.items():
                prog = key == "return_mean"
                module.log(f"val/env_{key}", value, prog_bar=prog, sync_dist=False)

    def drain(self, module, *, timeout: float = 600.0) -> None:
        deadline = time.time() + timeout
        while self.pending and time.time() < deadline:
            self.poll(module)
            time.sleep(0.05)
