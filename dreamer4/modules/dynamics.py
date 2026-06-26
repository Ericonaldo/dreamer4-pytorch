from __future__ import annotations

import torch
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
        self.model = DynamicsModel(cfg.model, n_latents=n_latents, latent_dim=latent_dim)

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
        return self._shared_step(batch, "val")
