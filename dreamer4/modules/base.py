from __future__ import annotations

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf


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
