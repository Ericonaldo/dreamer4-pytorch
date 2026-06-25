from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


class TokenizerModule(BaseModule):
    stage = "tokenizer"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import build_tokenizer

        self.model = build_tokenizer(cfg.model)
        self.patch_size = int(cfg.model.patch_size)
        self._val_viz: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        self.use_lpips = bool(cfg.train.get("use_lpips", False))
        self.lpips_weight = float(cfg.train.get("lpips_weight", 0.2))
        self.lpips_frac = float(cfg.train.get("lpips_frac", 0.5))
        self.lpips_net = str(cfg.train.get("lpips_net", "alex"))
        self.lpips_fn: nn.Module | None = None
        if self.use_lpips:
            import lpips

            self.lpips_fn = lpips.LPIPS(net=self.lpips_net)
            self.lpips_fn.eval()
            for p in self.lpips_fn.parameters():
                p.requires_grad_(False)

    def _lpips_kwargs(self) -> dict:
        if not self.use_lpips or self.lpips_fn is None:
            return {}
        return {
            "lpips_fn": self.lpips_fn,
            "lpips_weight": self.lpips_weight,
            "lpips_frac": self.lpips_frac,
        }

    def _tokenizer_loss(self, image_bthwc: torch.Tensor):
        from dreamer4.models import tokenizer_forward_loss

        return tokenizer_forward_loss(
            self.model, image_bthwc, self.patch_size, **self._lpips_kwargs()
        )

    def _tokenizer_eval(self, image_bthwc: torch.Tensor):
        from dreamer4.models import tokenizer_forward_with_aux

        return tokenizer_forward_with_aux(
            self.model, image_bthwc, self.patch_size, **self._lpips_kwargs()
        )

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

        from dreamer4.models import recon_panel_uint8

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
