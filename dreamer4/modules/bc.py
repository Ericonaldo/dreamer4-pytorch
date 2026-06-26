from __future__ import annotations

import torch
from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


def _load_state(module: torch.nn.Module, ckpt_path: str, *, prefix: str = "model.") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


class BCModule(BaseModule):
    stage = "bc"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import BCModel, build_tokenizer, bc_loss
        from dreamer4.models.dynamics import pack_bottleneck_to_spatial
        from dreamer4.models.tokenizer import encode_images

        self._pack = pack_bottleneck_to_spatial
        self._encode_images = encode_images
        self._bc_loss = bc_loss

        self.tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            _load_state(self.tokenizer, cfg.tokenizer_ckpt, prefix="model.")
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        n_latents = self.tokenizer.encoder.n_latents
        latent_dim = self.tokenizer.encoder.bottleneck_proj.out_features
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.action_horizon = int(cfg.model.get("action_horizon", 8))
        self.task_id = int(cfg.model.get("task_id", 0))

        self.model = BCModel(
            cfg.model.dynamics,
            n_latents=n_latents,
            latent_dim=latent_dim,
            heads_cfg=cfg.model,
        )
        if cfg.get("dynamics_ckpt"):
            _load_state(self.model.dynamics, cfg.dynamics_ckpt, prefix="model.")

        if cfg.train.get("freeze_dynamics", False):
            for p in self.model.dynamics.parameters():
                p.requires_grad_(False)

        self.action_weight = float(cfg.train.get("action_weight", 1.0))
        self.reward_weight = float(cfg.train.get("reward_weight", 1.0))
        self.value_weight = float(cfg.train.get("value_weight", 1.0))

    def _encode_packed(self, image_bthwc: torch.Tensor) -> torch.Tensor:
        z = self._encode_images(self.tokenizer, image_bthwc, self.patch_size)
        return self._pack(z, self.n_spatial, self.packing_factor)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("BC training requires images; set data.obs_mode=image or both")

        prefix = "val" if stage == "val" else self.stage
        with torch.no_grad():
            packed_z = self._encode_packed(batch.image)

        B = packed_z.shape[0]
        task = torch.full((B,), self.task_id, device=packed_z.device, dtype=torch.long)
        outputs = self.model(packed_z, batch.action, task)
        loss, metrics = self._bc_loss(
            outputs,
            batch.action,
            batch.reward,
            action_horizon=self.action_horizon,
            action_weight=self.action_weight,
            reward_weight=self.reward_weight,
            value_weight=self.value_weight,
        )
        for key, value in metrics.items():
            prog = stage == "train" and key == "action_mse"
            self.log(f"{prefix}/{key}", value, prog_bar=prog, sync_dist=True)
        if stage == "val":
            self.log("val/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
