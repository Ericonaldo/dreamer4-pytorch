from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dreamer4.callbacks import log_recon_panel
from dreamer4.modules.base import BaseModule


class TokenizerModule(BaseModule):
    stage = "tokenizer"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import build_tokenizer

        self.model = build_tokenizer(cfg.model)
        self.patch_size = int(cfg.model.patch_size)
        self._val_viz: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._val_viz_idx: int | None = None

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

    def _shared_step(self, batch, stage: str, *, capture_viz: bool = False) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("Tokenizer training requires images; set data.obs_mode=image or both")

        if stage == "val":
            with torch.no_grad():
                loss, metrics, pred, _, mae_mask = self._tokenizer_eval(batch.image)
            if capture_viz and self.trainer.is_global_zero:
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
        if self.trainer.is_global_zero and batch.image is not None and batch_idx == 0:
            self._pick_val_viz_batch_idx()
            self._val_viz = None
        capture = (
            self.trainer.is_global_zero
            and batch.image is not None
            and batch_idx == self._val_viz_idx
        )
        return self._shared_step(batch, "val", capture_viz=capture)

    def on_validation_epoch_end(self) -> None:
        if self._val_viz is None or not self.trainer.is_global_zero:
            return

        image, pred, mae_mask = self._val_viz
        self._val_viz = None
        self._val_viz_idx = None

        log_recon_panel(self, self.cfg, image, pred, mae_mask, self.patch_size)
