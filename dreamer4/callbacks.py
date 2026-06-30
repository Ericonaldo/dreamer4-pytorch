"""Training callbacks for validation."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_warn
from multiprocessing import Process, get_context
from omegaconf import DictConfig, OmegaConf

from dreamer4.eval_utils import DynamicsRolloutResult, resolve_async_gpu_ids, run_bc_env_eval


def _checkpoint_step(path: Path) -> int:
    """Extract global step from Lightning checkpoint filename for sorting."""
    stem = path.stem
    if stem.isdigit():
        return int(stem)
    if "-step=" in stem:
        tail = stem.rsplit("=", 1)[-1]
    elif stem.startswith("step-"):
        tail = stem[5:]
    else:
        return 0
    if "-v" in tail:
        tail = tail.split("-v", 1)[0]
    return int(tail) if tail.isdigit() else 0


class SaveCheckpointAfterPolicyWarmup(Callback):
    """RL: save ``warmup_end.ckpt`` when value-only warmup finishes (for easy resume)."""

    def __init__(self, checkpoint_dir: Path, warmup_steps: int):
        self.checkpoint_dir = checkpoint_dir
        self.warmup_steps = int(warmup_steps)
        self._saved = False

    def on_fit_start(self, trainer, pl_module) -> None:
        if int(trainer.global_step) >= self.warmup_steps:
            self._saved = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.warmup_steps <= 0 or self._saved:
            return
        step = int(trainer.global_step)
        if step != self.warmup_steps:
            return
        self._saved = True
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            rank_zero_warn(f"Saving post-warmup checkpoint at step {step}...")
            path = self.checkpoint_dir / "warmup_end.ckpt"
            trainer.save_checkpoint(str(path))
            rank_zero_warn(
                f"Saved post-warmup checkpoint at step {step}: {path} "
                f"(resume with train.resume_ckpt={path})"
            )
        trainer.strategy.barrier()
        if trainer.is_global_zero:
            rank_zero_warn(f"Post-warmup checkpoint barrier done at step {step}; continuing training")


class KeepLastCheckpoints(Callback):
    """Keep only the N most recent step checkpoints (ModelCheckpoint needs save_top_k=-1)."""

    def __init__(self, checkpoint_dir: Path, keep_last: int, every_n_steps: int):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if self.keep_last > 0 and step > 0 and step % self.every_n_steps == 0:
            if trainer.is_global_zero:
                self._prune()

    def _prune(self) -> None:
        ckpts = [p for p in self.checkpoint_dir.glob("*.ckpt") if p.name != "last.ckpt"]
        ckpts.sort(key=_checkpoint_step)
        for path in ckpts[:-self.keep_last]:
            path.unlink(missing_ok=True)


def _async_entry(
    step: int,
    run_dir: str,
    cfg_dict: dict[str, Any],
    model_state: dict[str, torch.Tensor],
    tokenizer_state: dict[str, torch.Tensor],
    gpu_ids: list[int | None],
    out_queue,
) -> None:
    """Child-process entry for async BC env eval (spawn target for ``AsyncBCEval.start``).

    Runs ``run_bc_env_eval`` in an isolated process so MuJoCo/DMC do not block training.
    Writes ``run_dir/eval/env_step_{step:08d}.json`` and sends ``(step, metrics, err)``
    to *out_queue* on completion or failure.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
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
    """Non-blocking BC env eval for training (rank 0 only).

    Spawns a separate process to roll out the policy in DMC while training continues.
    Typical usage from a Lightning module:

    - ``on_validation_epoch_end`` → ``start(...)``
    - ``on_train_batch_end`` → ``poll(...)``
    - ``on_train_end`` → ``drain(...)``
    """

    def __init__(self) -> None:
        """Initialize empty eval state (no process or queue until ``start``)."""
        self._proc: Process | None = None
        self._queue = None
        self._ctx = None
        self._launch_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """True while the eval child process is alive."""
        return self._proc is not None and self._proc.is_alive()

    @property
    def pending(self) -> bool:
        """True while an eval is running or still being launched (state-dict copy + spawn)."""
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
        """Launch async env eval at *step*; skip if a previous eval is still active."""
        if self.pending or self._queue is not None:
            from lightning.pytorch.utilities import rank_zero_warn

            rank_zero_warn(f"Skipping async env eval at step {step}: previous eval still active")
            return
        self._ctx = get_context("spawn")
        self._queue = self._ctx.Queue()
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        gpu_ids = resolve_async_gpu_ids(cfg)

        def _launch() -> None:
            # Copy weights on a daemon thread so the training step is not blocked.
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
        """Non-blocking check for eval results; log ``val/env_*`` metrics when ready."""
        if self._queue is None:
            return
        try:
            step, metrics, err = self._queue.get_nowait()
        except queue.Empty:
            return
        # Result received; release queue/context and reap the child process in the background.
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
        """Block until pending eval finishes or *timeout* is reached.

        Called from ``on_train_end`` so the last scheduled eval can log.
        """
        deadline = time.time() + timeout
        while self.pending and time.time() < deadline:
            self.poll(module)
            time.sleep(0.05)


def log_dynamics_rollout_viz(
    module,
    cfg: DictConfig,
    result: DynamicsRolloutResult,
    *,
    max_items: int,
    rollout_ctx: int,
    rollout_horizon: int,
    log_key: str = "dynamics/rollout_viz",
) -> None:
    """Write rollout panel PNG and optionally log to wandb (lazy-imports eval viz)."""
    from eval.viz.panels import rollout_panels_multictx_uint8

    panel, _ = rollout_panels_multictx_uint8(
        result.frames,
        result.pred_by_ctx_bkthwc,
        result.ctx_lengths,
        max_items=max_items,
    )
    step = int(module.trainer.global_step)
    run_dir = Path(cfg.log.dir) / cfg.log.run_name
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / f"rollout_step_{step:08d}.png"
    iio.imwrite(viz_path, panel)

    caption = (
        f"rows=gt+ctx=1..{rollout_ctx} | horizon={rollout_horizon} | "
        f"psnr_gain={result.metrics['rollout_psnr_gain']:.2f}"
    )
    for logger in module.trainer.loggers:
        if isinstance(logger, WandbLogger):
            import wandb

            logger.experiment.log({log_key: wandb.Image(panel, caption=caption)}, step=step)


def log_recon_panel(
    module,
    cfg: DictConfig,
    image: torch.Tensor,
    pred: torch.Tensor,
    mae_mask: torch.Tensor,
    patch_size: int,
) -> None:
    """Write tokenizer recon panel PNG and optionally log to wandb (lazy-imports eval viz)."""
    from eval.viz.panels import recon_panel_uint8

    panel = recon_panel_uint8(
        image,
        pred,
        mae_mask,
        patch_size,
        max_items=int(cfg.log.get("viz_max_items", 4)),
        max_T=int(cfg.log.get("viz_max_T", 6)),
    )
    step = int(module.trainer.global_step)
    run_dir = Path(cfg.log.dir) / cfg.log.run_name
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    viz_path = viz_dir / f"step_{step:08d}.png"
    iio.imwrite(viz_path, panel)

    caption = "rows=target/masked/recon_masked/recon_full"
    for logger in module.trainer.loggers:
        if isinstance(logger, WandbLogger):
            import wandb

            logger.experiment.log({"tokenizer/viz": wandb.Image(panel, caption=caption)}, step=step)
