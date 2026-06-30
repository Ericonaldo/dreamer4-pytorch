from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dreamer4.callbacks import (
    KeepLastCheckpoints,
    SaveCheckpointAfterPolicyWarmup,
)
from dreamer4.config import load_config, save_config
from dreamer4.data import GranularEpisodeDataset, collate_episodes, split_episode_indices
from dreamer4.modules import STAGES


def _window_mode(cfg: DictConfig) -> str:
    mode = str(cfg.data.get("window_mode", "auto"))
    if mode != "auto":
        return mode
    return "transition" if cfg.stage in ("bc_dynamics", "rl") else "frame"


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


def build_trainer(cfg: DictConfig, run_dir: Path, has_val: bool) -> L.Trainer:
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
        ddp_unused = cfg.stage in ("bc_dynamics", "rl")
        strategy = "ddp_find_unused_parameters_true" if ddp_unused else "ddp"

    val_every = int(cfg.train.get("val_every", 0) or 0)
    step_val = has_val and val_every > 0
    val_max_batches = int(cfg.train.get("val_max_batches", 32))

    if cfg.stage == "rl":
        warmup_steps = int(cfg.get("imagination", {}).get("policy_warmup_steps", 0))
        if warmup_steps > 0:
            callbacks.append(SaveCheckpointAfterPolicyWarmup(checkpoint_dir, warmup_steps))

    return L.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator=cfg.train.accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.get("grad_clip"),
        log_every_n_steps=cfg.log.every_n_steps,
        val_check_interval=val_every if step_val else 1.0,
        check_val_every_n_epoch=None if step_val else 1,
        limit_val_batches=val_max_batches if has_val else 0,
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
    trainer = build_trainer(cfg, run_dir, has_val=val_loader is not None)
    ckpt_path = _resolve_resume_ckpt(cfg)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dreamer4")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("overrides", nargs="*", help="Config overrides, e.g. train.batch_size=32")
    args = parser.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
