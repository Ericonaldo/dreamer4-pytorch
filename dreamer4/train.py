from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dreamer4.config import save_config
from dreamer4.data import GranularEpisodeDataset, collate_episodes, split_episode_indices


class BaseModule(L.LightningModule):
    stage: str = "base"

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            betas=tuple(opt_cfg.betas),
            weight_decay=opt_cfg.weight_decay,
        )
        if not opt_cfg.get("use_scheduler", False):
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.train.max_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        self.log(f"{self.stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss


class TokenizerModule(BaseModule):
    stage = "tokenizer"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.tokenizer import build_tokenizer

        self.model = build_tokenizer(cfg.model)
        self.patch_size = int(cfg.model.patch_size)
        self._val_viz: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _tokenizer_loss(self, image_bthwc: torch.Tensor):
        from dreamer4.tokenizer import tokenizer_forward_loss

        return tokenizer_forward_loss(self.model, image_bthwc, self.patch_size)

    def _tokenizer_eval(self, image_bthwc: torch.Tensor):
        from dreamer4.tokenizer import tokenizer_forward_with_aux

        return tokenizer_forward_with_aux(self.model, image_bthwc, self.patch_size)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("Tokenizer training requires images; set data.obs_mode=image or both")

        if stage == "val":
            with torch.no_grad():
                loss, metrics, pred, _, mae_mask = self._tokenizer_eval(batch.image)
            if self.trainer.is_global_zero:
                self._val_viz = (batch.image.detach(), pred.detach(), mae_mask.detach())
            self.log("val/loss", loss, sync_dist=True)
            for key, value in metrics.items():
                self.log(f"val/{key}", value, sync_dist=True)
        else:
            loss, metrics = self._tokenizer_loss(batch.image)
            for key, value in metrics.items():
                prog = key == "loss_mae"
                self.log(f"{self.stage}/{key}", value, prog_bar=prog, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if self._val_viz is None or not self.trainer.is_global_zero:
            return

        from dreamer4.tokenizer import recon_panel_uint8

        image, pred, mae_mask = self._val_viz
        self._val_viz = None

        panel = recon_panel_uint8(
            image,
            pred,
            mae_mask,
            self.patch_size,
            max_items=int(self.cfg.log.get("viz_max_items", 4)),
            max_T=int(self.cfg.log.get("viz_max_T", 6)),
        )
        step = int(self.trainer.global_step)
        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        viz_dir = run_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"step_{step:08d}.png"
        iio.imwrite(viz_path, panel)

        caption = "rows=target/masked/recon_masked/recon_full"
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log(
                    {"tokenizer/viz": wandb.Image(panel, caption=caption)},
                    step=step,
                )


class DynamicsModule(BaseModule):
    stage = "dynamics"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import DynamicsModel
        from dreamer4.tokenizer import build_tokenizer

        tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            tokenizer_state = {
                k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")
            }
            tokenizer.load_state_dict(tokenizer_state, strict=True)
        for p in tokenizer.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.model = DynamicsModel(cfg.model, tokenizer)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("Dynamics training not yet implemented")


class BCModule(BaseModule):
    stage = "bc"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import AgentHeads

        self.model = AgentHeads(cfg.model)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("BC training not yet implemented")


class PolicyModule(BaseModule):
    stage = "policy"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.imagination_cfg = cfg.imagination

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("Policy training not yet implemented")


STAGES = {
    "tokenizer": TokenizerModule,
    "dynamics": DynamicsModule,
    "bc": BCModule,
    "policy": PolicyModule,
}


def _episode_dataset(cfg: DictConfig, indices: list[int] | None) -> GranularEpisodeDataset:
    return GranularEpisodeDataset(
        path=cfg.data.path,
        seq_len=cfg.data.seq_len,
        obs_mode=cfg.data.obs_mode,
        indices=indices,
    )


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader | None]:
    val_fraction = float(cfg.data.get("val_fraction", 0.0))
    probe = _episode_dataset(cfg, indices=None)
    n_episodes = len(probe)

    val_indices: list[int] | None = None
    train_indices: list[int] | None = None
    if val_fraction > 0:
        train_indices, val_indices = split_episode_indices(
            n_episodes,
            val_fraction,
            int(cfg.data.get("val_seed", 0)),
        )

    train_ds = _episode_dataset(cfg, train_indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_episodes,
        drop_last=True,
    )

    val_loader = None
    if val_indices is not None:
        val_batch = int(cfg.train.get("val_batch_size", cfg.train.batch_size))
        val_loader = DataLoader(
            _episode_dataset(cfg, val_indices),
            batch_size=val_batch,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_episodes,
            drop_last=False,
        )
    return train_loader, val_loader


def build_trainer(cfg: DictConfig, run_dir: Path, has_val: bool) -> L.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{step}",
            save_top_k=-1,
            every_n_train_steps=cfg.train.checkpoint_every,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

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
        strategy = "ddp"

    val_every = cfg.train.get("val_every")
    limit_val_batches = cfg.train.get("val_max_batches", 32) if has_val else 0

    return L.Trainer(
        max_steps=cfg.train.max_steps,
        accelerator=cfg.train.accelerator,
        devices=devices,
        strategy=strategy,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.get("grad_clip"),
        log_every_n_steps=cfg.log.every_n_steps,
        val_check_interval=val_every if has_val and val_every else None,
        limit_val_batches=limit_val_batches,
        default_root_dir=str(run_dir),
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
    )


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
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
