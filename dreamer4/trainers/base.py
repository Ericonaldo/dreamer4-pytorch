from __future__ import annotations

import math

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf


class BaseModule(L.LightningModule):
    stage: str = "base"

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self._resume_scheduler_synced = False

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

    def _sync_resume_scheduler(self) -> None:
        """Realign cosine after resume: ckpt state_dict restores old T_max from prior run."""
        if not bool(self.cfg.train.get("resume_reset_scheduler", False)):
            return
        step = int(self.trainer.global_step)
        if step <= 0 or not self.trainer.lr_scheduler_configs:
            return

        max_steps = int(self.cfg.train.max_steps)
        for config in self.trainer.lr_scheduler_configs:
            sched = config.scheduler
            if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
                sched.T_max = max_steps
                sched.last_epoch = step
                lrs = [
                    sched.eta_min
                    + (base_lr - sched.eta_min)
                    * (1 + math.cos(math.pi * step / max_steps))
                    / 2
                    for base_lr in sched.base_lrs
                ]
            else:
                sched.last_epoch = step
                lrs = sched.get_lr()
            for pg, lr in zip(sched.optimizer.param_groups, lrs):
                pg["lr"] = lr

        if self.trainer.is_global_zero:
            opt = self.trainer.optimizers[0]
            lr = opt.param_groups[0]["lr"]
            rank_zero_info(
                f"resume_reset_scheduler: global_step={step} T_max={max_steps} lr={lr:.6g}"
            )
        self._resume_scheduler_synced = True

    def on_train_start(self) -> None:
        if not self._resume_scheduler_synced:
            self._sync_resume_scheduler()

    def _num_val_batches(self) -> int:
        limit = int(self.cfg.train.get("val_max_batches", 32))
        loader = self.trainer.val_dataloaders
        if loader is None:
            return max(1, limit)
        if isinstance(loader, (list, tuple)):
            loader = loader[0]
        try:
            n = len(loader)
        except TypeError:
            n = limit
        return max(1, min(n, limit))

    def _pick_val_viz_batch_idx(self) -> None:
        n = self._num_val_batches()
        g = torch.Generator()
        g.manual_seed(int(self.trainer.global_step))
        self._val_viz_idx = int(torch.randint(0, n, (1,), generator=g).item())

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        self.log(f"{self.stage}/loss", loss, prog_bar=True, sync_dist=True)
        return loss
