from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dreamer4.config import save_config
from dreamer4.data import GranularEpisodeDataset, collate_episodes, split_episode_indices
from dreamer4.modules import STAGES


class ValidateEveryNSteps(Callback):
    """Run validation every N optimizer steps (works with DDP and small epoch sizes)."""

    def __init__(self, every_n_steps: int, limit_batches: int, val_dataloader: DataLoader):
        self.every_n_steps = every_n_steps
        self.limit_batches = limit_batches
        self.val_dataloader = val_dataloader

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            pl_module.eval()
            n_batches = max(1, min(len(self.val_dataloader), self.limit_batches))
            pl_module._val_n_batches = n_batches
            with torch.no_grad():
                for i, val_batch in enumerate(self.val_dataloader):
                    if i >= self.limit_batches:
                        break
                    val_batch = trainer.strategy.batch_to_device(val_batch)
                    pl_module.validation_step(val_batch, i)
            pl_module.on_validation_epoch_end()
            pl_module._val_n_batches = None
            pl_module.train()


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
    # Lightning may suffix duplicates: step-step=6500-v1
    if "-v" in tail:
        tail = tail.split("-v", 1)[0]
    return int(tail) if tail.isdigit() else 0


def _window_mode(cfg: DictConfig) -> str:
    mode = str(cfg.data.get("window_mode", "auto"))
    if mode != "auto":
        return mode
    return "transition" if cfg.stage in ("dynamics", "bc", "bc_dynamics") else "frame"


def _episode_dataset(cfg: DictConfig, episode_indices: list[int] | None) -> GranularEpisodeDataset:
    return GranularEpisodeDataset(
        path=cfg.data.path,
        seq_len=cfg.data.seq_len,
        obs_mode=cfg.data.obs_mode,
        episode_indices=episode_indices,
        window_mode=_window_mode(cfg),
        verbose=bool(cfg.data.get("verbose", True)),
    )


def _granular_worker_init_fn(_worker_id: int) -> None:
    """Granular mmap readers are not fork-safe; reopen in each DataLoader worker."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if isinstance(dataset, GranularEpisodeDataset):
        dataset.reset_reader()


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
    val_fraction = float(cfg.data.get("val_fraction", 0.0))
    probe = _episode_dataset(cfg, episode_indices=None)
    n_episodes = probe.num_episodes

    val_episodes: list[int] | None = None
    train_episodes: list[int] | None = None
    if val_fraction > 0:
        train_episodes, val_episodes = split_episode_indices(
            n_episodes,
            val_fraction,
            int(cfg.data.get("val_seed", 0)),
        )

    train_ds = _episode_dataset(cfg, train_episodes)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_episodes,
        drop_last=True,
        worker_init_fn=_granular_worker_init_fn if cfg.data.num_workers > 0 else None,
    )

    val_loader = None
    if val_episodes is not None:
        val_batch = int(cfg.train.get("val_batch_size", cfg.train.batch_size))
        val_loader = DataLoader(
            _episode_dataset(cfg, val_episodes),
            batch_size=val_batch,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_episodes,
            drop_last=False,
            worker_init_fn=_granular_worker_init_fn if cfg.data.num_workers > 0 else None,
        )
    return train_loader, val_loader


def build_trainer(cfg: DictConfig, run_dir: Path, has_val: bool, val_loader: DataLoader | None = None) -> L.Trainer:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_every = int(cfg.train.checkpoint_every)
    keep_last = int(cfg.train.get("checkpoint_keep_last", 5))
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="step-{step}",
            save_top_k=-1,
            every_n_train_steps=checkpoint_every,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if keep_last > 0:
        callbacks.append(KeepLastCheckpoints(checkpoint_dir, keep_last, checkpoint_every))

    loggers = [CSVLogger(save_dir=run_dir, name="csv")]
    if cfg.log.get("wandb", False):
        loggers.append(
            WandbLogger(
                project=cfg.log.project,
                name=cfg.log.run_name,
                save_dir=run_dir,
            )
        )

    strategy = "auto"
    devices = cfg.train.devices
    if isinstance(devices, int) and devices > 1:
        strategy = "ddp_find_unused_parameters_true" if cfg.stage in ("bc", "bc_dynamics") else "ddp"

    val_every = int(cfg.train.get("val_every", 0) or 0)
    step_val = has_val and val_every > 0
    limit_val_batches = 0 if step_val else (cfg.train.get("val_max_batches", 32) if has_val else 0)
    if step_val and val_loader is not None:
        callbacks.append(
            ValidateEveryNSteps(val_every, int(cfg.train.get("val_max_batches", 32)), val_loader)
        )

    return L.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator=cfg.train.accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.get("grad_clip"),
        log_every_n_steps=cfg.log.every_n_steps,
        check_val_every_n_epoch=0 if step_val else 1,
        limit_val_batches=limit_val_batches,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )


def _resolve_resume_ckpt(cfg: DictConfig) -> str | None:
    """Return checkpoint path for trainer.fit."""
    resume_ckpt = cfg.train.get("resume_ckpt")
    if not resume_ckpt:
        return None
    path = Path(resume_ckpt)
    if not path.is_file():
        raise FileNotFoundError(f"resume_ckpt not found: {path}")
    return str(path)


def train(cfg: DictConfig) -> None:
    stage = cfg.stage
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}, expected one of {list(STAGES)}")

    run_dir = Path(cfg.log.dir) / cfg.log.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    module_cls = STAGES[stage]
    module = module_cls(cfg)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = build_trainer(cfg, run_dir, has_val=val_loader is not None, val_loader=val_loader)
    ckpt_path = _resolve_resume_ckpt(cfg)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
