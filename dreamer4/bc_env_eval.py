"""BC policy rollout in DMC: batched envs per GPU, one EGL context, multi-GPU episode split."""

from __future__ import annotations

import json
import os
import queue
from multiprocessing import Process, get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from dreamer4.models import BCModel, build_tokenizer
from dreamer4.models.dynamics import pack_bottleneck_to_spatial
from dreamer4.models.tokenizer import encode_images
from dreamer4.policy_agent import EpisodeStats, summarize_episodes


def _split_episodes(total: int, n: int) -> list[int]:
    n = max(1, n)
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def _load_bc_modules(
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
        _load_ckpt(tokenizer, cfg.tokenizer_ckpt, prefix="model.")

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
        _load_ckpt(model.dynamics, cfg.dynamics_ckpt, prefix="model.")

    tokenizer.eval()
    model.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    tokenizer.to(device)
    model.to(device)
    return model, tokenizer


def _load_ckpt(module: nn.Module, ckpt_path: str, *, prefix: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


def _pad_stack(seqs: list[torch.Tensor]) -> torch.Tensor:
    t_max = max(s.shape[0] for s in seqs)
    padded = []
    for s in seqs:
        if s.shape[0] < t_max:
            pad = s.new_zeros(t_max - s.shape[0], *s.shape[1:])
            s = torch.cat([pad, s], dim=0)
        padded.append(s)
    return torch.stack(padded, dim=0)


class BatchedBCPolicy:
    """N parallel env slots with batched GPU forward (one process, one EGL context)."""

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        model: BCModel,
        tokenizer: nn.Module,
        num_envs: int,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.num_envs = int(num_envs)
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.max_history = int(cfg.eval.get("max_history", 32))
        self.action_dim = int(cfg.model.dynamics.action_dim)
        n_latents = tokenizer.encoder.n_latents
        self.n_spatial = n_latents // int(cfg.model.dynamics.get("packing_factor", 1))
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self._z: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._a: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]

    def reset(self, ids: list[int]) -> None:
        for i in ids:
            self._z[i].clear()
            self._a[i].clear()

    @torch.no_grad()
    def act(self, ids: list[int], images: np.ndarray) -> np.ndarray:
        imgs = torch.from_numpy(images).to(self.device).float().div_(255.0).unsqueeze(1)
        z = encode_images(self.tokenizer, imgs, self.patch_size)
        packed = pack_bottleneck_to_spatial(z, self.n_spatial, self.packing_factor)[:, 0]

        for j, i in enumerate(ids):
            self._z[i].append(packed[j])
            if len(self._z[i]) > self.max_history:
                self._z[i].pop(0)
                if self._a[i]:
                    self._a[i].pop(0)

        z_seqs, a_seqs = [], []
        for i in ids:
            z_seq = torch.stack(self._z[i], dim=0)
            t = z_seq.shape[0]
            if self._a[i]:
                a_hist = torch.stack(self._a[i], dim=0)
                if a_hist.shape[0] < t:
                    pad = torch.zeros(t - a_hist.shape[0], self.action_dim, device=self.device)
                    a_hist = torch.cat([a_hist, pad], dim=0)
            else:
                a_hist = z_seq.new_zeros(t, self.action_dim)
            z_seqs.append(z_seq)
            a_seqs.append(a_hist)

        z_batch = _pad_stack(z_seqs)
        a_batch = _pad_stack(a_seqs)
        outputs = self.model(z_batch, a_batch)
        actions = outputs.action[:, -1, 0].float().cpu().numpy()

        for j, i in enumerate(ids):
            self._a[i].append(torch.from_numpy(actions[j]).to(self.device))
            if len(self._a[i]) > self.max_history:
                self._a[i].pop(0)
        return actions


def _rollout_on_device(
    cfg: DictConfig,
    device: torch.device,
    num_episodes: int,
    *,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
) -> list[EpisodeStats]:
    from dreamer4.env import make_dmc_env

    eval_cfg = cfg.get("eval", {})
    num_envs = min(int(eval_cfg.get("num_envs", 4)), num_episodes)
    envs = [
        make_dmc_env(
            str(eval_cfg.get("task", "walker_walk")),
            repeat=int(eval_cfg.get("action_repeat", 2)),
            image_size=int(eval_cfg.get("image_size", 64)),
            proprio=bool(eval_cfg.get("proprio", False)),
            image=bool(eval_cfg.get("image", True)),
            camera=int(eval_cfg.get("camera_id", -1)),
            max_episode_steps=int(eval_cfg.get("max_episode_steps", 1000)),
        )
        for _ in range(num_envs)
    ]
    model, tokenizer = _load_bc_modules(
        cfg, device, model_state=model_state, tokenizer_state=tokenizer_state
    )
    policy = BatchedBCPolicy(cfg, device, model, tokenizer, num_envs)

    stats: list[EpisodeStats] = []
    obs = [env.reset() for env in envs]
    rets = np.zeros(num_envs, dtype=np.float64)
    lens = np.zeros(num_envs, dtype=np.int64)
    done = [False] * num_envs
    policy.reset(list(range(num_envs)))

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
        actions = policy.act(active, images)
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
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if model_state is not None:
        model_state = {k: v.detach().cpu() for k, v in model_state.items()}
    if tokenizer_state is not None:
        tokenizer_state = {k: v.detach().cpu() for k, v in tokenizer_state.items()}

    if gpu_ids is None:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 1
        gpu_ids = list(range(n)) if n > 0 else [None]

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
        return summarize_episodes(stats)

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
    return summarize_episodes(all_stats)


def _async_entry(
    step: int,
    run_dir: str,
    cfg_dict: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    tokenizer_state: dict[str, torch.Tensor],
    out_queue,
) -> None:
    try:
        cfg = OmegaConf.create(cfg_dict)
        metrics = run_bc_env_eval(
            cfg,
            model_state=model_state,
            tokenizer_state=tokenizer_state,
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

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(
        self,
        step: int,
        cfg: DictConfig,
        model: nn.Module,
        tokenizer: nn.Module,
        run_dir: Path,
    ) -> None:
        if self.running:
            return
        ctx = get_context("spawn")
        self._queue = ctx.Queue()
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        tokenizer_state = {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()}
        self._proc = ctx.Process(
            target=_async_entry,
            args=(step, str(run_dir), cfg_dict, model_state, tokenizer_state, self._queue),
        )
        self._proc.start()

    def poll(self, module) -> None:
        if self._queue is None:
            return
        try:
            step, metrics, err = self._queue.get_nowait()
        except queue.Empty:
            return
        if self._proc is not None:
            self._proc.join(timeout=1)
        self._proc = None
        self._queue = None
        if err:
            from lightning.pytorch.utilities import rank_zero_warn

            rank_zero_warn(f"BC env eval failed at step {step}: {err}")
            return
        for key, value in metrics.items():
            prog = key == "return_mean"
            module.log(f"val/env_{key}", value, prog_bar=prog, sync_dist=False, step=step)
