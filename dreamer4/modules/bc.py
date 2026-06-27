from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig

from dreamer4.policy_agent import AsyncBCEval
from dreamer4.data import align_dynamics_batch
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
        self._env_eval = AsyncBCEval()

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

        self.model = BCModel(
            cfg.model.dynamics,
            n_latents=n_latents,
            latent_dim=latent_dim,
            heads_cfg=cfg.model,
        )
        if cfg.get("dynamics_ckpt"):
            _load_state(self.model.dynamics, cfg.dynamics_ckpt, prefix="model.")

        for p in self.model.dynamics.flow_head.parameters():
            p.requires_grad_(False)

        if cfg.train.get("freeze_dynamics", False):
            for p in self.model.dynamics.parameters():
                p.requires_grad_(False)

        self.action_weight = float(cfg.train.get("action_weight", 1.0))
        self.reward_weight = float(cfg.train.get("reward_weight", 1.0))

    def _encode_packed(self, image_bthwc: torch.Tensor) -> torch.Tensor:
        z = self._encode_images(self.tokenizer, image_bthwc, self.patch_size)
        return self._pack(z, self.n_spatial, self.packing_factor)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("BC training requires images; set data.obs_mode=image or both")

        prefix = "val" if stage == "val" else self.stage
        image, action, reward = align_dynamics_batch(batch.image, batch.action, batch.reward)
        with torch.no_grad():
            packed_z = self._encode_packed(image)

        outputs = self.model(packed_z, action)
        loss, metrics = self._bc_loss(
            outputs,
            action,
            reward,
            self.model.heads,
            action_horizon=self.action_horizon,
            action_weight=self.action_weight,
            reward_weight=self.reward_weight,
        )
        for key, value in metrics.items():
            prog = stage == "train" and key in ("action_nll", "action_mse", "action_out_mean")
            self.log(f"{prefix}/{key}", value, prog_bar=prog, sync_dist=True)
        if stage == "val":
            self.log("val/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_train_batch_end(self, *_) -> None:
        if self.trainer.is_global_zero:
            self._env_eval.poll(self)

    def on_train_end(self) -> None:
        if self.trainer.is_global_zero:
            self._env_eval.drain(self)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        eval_cfg = self.cfg.get("eval", {})
        if not eval_cfg.get("env_eval", True):
            return
        try:
            from dreamer4.env import make_dmc_env  # noqa: F401
        except ImportError as exc:
            rank_zero_warn(f"Skipping BC env eval (install dreamer4[dmc]): {exc}")
            return

        step = int(self.trainer.global_step)
        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        self._env_eval.start(step, self.cfg, self.model, self.tokenizer, run_dir)
