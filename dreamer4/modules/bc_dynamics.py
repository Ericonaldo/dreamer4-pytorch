from __future__ import annotations

from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig

from dreamer4.callbacks import AsyncBCEval
from dreamer4.data import align_dynamics_batch
from dreamer4.checkpoint import load_state
from dreamer4.modules.base import BaseModule


class BCDynamicsModule(BaseModule):
    """Joint BC + dynamics training on a shared backbone with per-objective space masks."""

    stage = "bc_dynamics"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import PolicyModel, build_tokenizer, bc_loss
        from dreamer4.models.dynamics import flow_matching_loss, pack_bottleneck_to_spatial

        self._pack = pack_bottleneck_to_spatial
        self._bc_loss = bc_loss
        self._flow_matching_loss = flow_matching_loss
        self._env_eval = AsyncBCEval()

        self.tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            load_state(self.tokenizer, cfg.tokenizer_ckpt, prefix="model.")
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        n_latents = self.tokenizer.encoder.n_latents
        latent_dim = self.tokenizer.encoder.bottleneck_proj.out_features
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.image_size = int(cfg.model.tokenizer.image_size)
        self.channels = int(cfg.model.tokenizer.channels)
        self.action_horizon = int(cfg.model.get("action_horizon", 8))

        self.model = PolicyModel(
            cfg.model.dynamics,
            n_latents=n_latents,
            latent_dim=latent_dim,
            heads_cfg=cfg.model,
        )

        self.dynamics_space_mode = self.model.dynamics_space_mode
        self.bc_space_mode = self.model.bc_space_mode

        self.flow_weight = float(cfg.train.get("flow_weight", 1.0))
        self.action_weight = float(cfg.train.get("action_weight", 1.0))
        self.reward_weight = float(cfg.train.get("reward_weight", 1.0))

        self.rollout_ctx = int(cfg.train.get("rollout_ctx", 8))
        self.rollout_horizon = int(cfg.train.get("rollout_horizon", 8))
        self.rollout_flow_steps = int(cfg.train.get("rollout_flow_steps", 8))
        self._val_rollout_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._val_rollout_viz_idx: int | None = None
        self._val_rollout_n_batches: int | None = None

    def _num_val_batches(self) -> int:
        limit = int(self.cfg.train.get("val_max_batches", 32))
        if self._val_rollout_n_batches is not None:
            return self._val_rollout_n_batches
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

    def _encode_packed(self, image_bthwc: torch.Tensor) -> torch.Tensor:
        z = self.tokenizer.encode_images(image_bthwc)
        return self._pack(z, self.n_spatial, self.packing_factor)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("BC+dynamics training requires images; set data.obs_mode=image or both")

        image, action, reward = align_dynamics_batch(batch.image, batch.action, batch.reward)
        with torch.no_grad():
            packed_z = self._encode_packed(image)

        flow_loss, flow_metrics = self._flow_matching_loss(
            self.model.dynamics,
            packed_z,
            action,
            space_mode=self.dynamics_space_mode,
        )
        bc_outputs = self.model(packed_z, action, space_mode=self.bc_space_mode)
        bc_loss_val, bc_metrics = self._bc_loss(
            bc_outputs,
            action,
            reward,
            self.model.heads,
            action_horizon=self.action_horizon,
            action_weight=self.action_weight,
            reward_weight=self.reward_weight,
        )

        loss = self.flow_weight * flow_loss + bc_loss_val
        dyn_prefix = "dynamics" if stage == "train" else "val"
        bc_prefix = "bc" if stage == "train" else "val"
        for key, value in flow_metrics.items():
            prog = stage == "train" and key == "flow_mse"
            self.log(f"{dyn_prefix}/{key}", value, prog_bar=prog, sync_dist=True)
        for key, value in bc_metrics.items():
            prog = stage == "train" and key in ("action_nll", "action_mse", "action_out_mean")
            self.log(f"{bc_prefix}/{key}", value, prog_bar=prog, sync_dist=True)
        if stage == "val":
            self.log(f"{self.stage}/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.is_global_zero and batch.image is not None:
            if batch_idx == 0:
                n_batches = self._num_val_batches()
                g = torch.Generator()
                g.manual_seed(int(self.trainer.global_step))
                self._val_rollout_viz_idx = int(torch.randint(0, n_batches, (1,), generator=g).item())
                self._val_rollout_batch = None
            if batch_idx == self._val_rollout_viz_idx:
                image, action, _ = align_dynamics_batch(batch.image, batch.action)
                self._val_rollout_batch = (image.detach(), action.detach())
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

        if self._val_rollout_batch is not None:
            from dreamer4.callbacks import log_dynamics_rollout_viz
            from dreamer4.eval_utils import dynamics_rollout_eval

            image, action = self._val_rollout_batch
            self._val_rollout_batch = None
            self._val_rollout_viz_idx = None
            self._val_rollout_n_batches = None

            max_items = int(self.cfg.log.get("viz_max_items", 4))
            result = dynamics_rollout_eval(
                self.model.dynamics,
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
            )

            for key, value in result.metrics.items():
                self.log(f"val/{key}", value, sync_dist=False)

            log_dynamics_rollout_viz(
                self,
                self.cfg,
                result,
                max_items=max_items,
                rollout_ctx=self.rollout_ctx,
                rollout_horizon=self.rollout_horizon,
            )

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
