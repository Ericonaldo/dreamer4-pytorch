from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


class DynamicsModule(BaseModule):
    stage = "dynamics"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import DynamicsModel, build_tokenizer

        self.tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            tokenizer_state = {
                k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")
            }
            self.tokenizer.load_state_dict(tokenizer_state, strict=True)
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        n_latents = self.tokenizer.encoder.n_latents
        latent_dim = self.tokenizer.encoder.bottleneck_proj.out_features
        self.packing_factor = int(cfg.model.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.image_size = int(cfg.model.tokenizer.image_size)
        self.channels = int(cfg.model.tokenizer.channels)
        self.model = DynamicsModel(cfg.model, n_latents=n_latents, latent_dim=latent_dim)

        self.rollout_ctx = int(cfg.train.get("rollout_ctx", 8))
        self.rollout_horizon = int(cfg.train.get("rollout_horizon", 8))
        self.rollout_flow_steps = int(cfg.train.get("rollout_flow_steps", 8))
        self._val_rollout_batch: tuple[torch.Tensor, torch.Tensor] | None = None

    def _encode_packed(self, image_bthwc: torch.Tensor) -> torch.Tensor:
        from dreamer4.models.dynamics import pack_bottleneck_to_spatial
        from dreamer4.models.tokenizer import encode_images

        z = encode_images(self.tokenizer, image_bthwc, self.patch_size)
        return pack_bottleneck_to_spatial(z, self.n_spatial, self.packing_factor)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("Dynamics training requires images; set data.obs_mode=image or both")

        from dreamer4.models.dynamics import flow_matching_loss

        prefix = "val" if stage == "val" else self.stage
        with torch.no_grad():
            z1 = self._encode_packed(batch.image)
        loss, metrics = flow_matching_loss(self.model, z1, batch.action)
        for key, value in metrics.items():
            prog = stage == "train" and key == "flow_mse"
            self.log(f"{prefix}/{key}", value, prog_bar=prog, sync_dist=True)
        if stage == "val":
            self.log("val/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0 and self.trainer.is_global_zero and batch.image is not None:
            self._val_rollout_batch = (batch.image.detach(), batch.action.detach())
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        if self._val_rollout_batch is None or not self.trainer.is_global_zero:
            return

        from dreamer4.models.dynamics import run_dynamics_rollout_eval

        image, action = self._val_rollout_batch
        self._val_rollout_batch = None

        max_items = int(self.cfg.log.get("viz_max_items", 4))
        metrics, panel, _, _ = run_dynamics_rollout_eval(
            self.model,
            self.tokenizer,
            image,
            action,
            patch_size=self.patch_size,
            packing_factor=self.packing_factor,
            n_spatial=self.n_spatial,
            image_size=self.image_size,
            channels=self.channels,
            ctx_length=self.rollout_ctx,
            horizon=self.rollout_horizon,
            flow_steps=self.rollout_flow_steps,
            max_items=max_items,
        )

        for key, value in metrics.items():
            self.log(f"val/{key}", value, sync_dist=False)

        step = int(self.trainer.global_step)
        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        viz_dir = run_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_path = viz_dir / f"rollout_step_{step:08d}.png"
        iio.imwrite(viz_path, panel)

        caption = (
            f"rows=gt+ctx=1..{self.rollout_ctx} | horizon={self.rollout_horizon} | "
            f"psnr_gain={metrics['rollout_psnr_gain']:.2f}"
        )
        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                import wandb

                logger.experiment.log(
                    {"dynamics/rollout_viz": wandb.Image(panel, caption=caption)},
                    step=step,
                )
