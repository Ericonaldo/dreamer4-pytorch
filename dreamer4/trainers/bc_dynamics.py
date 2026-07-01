from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from dreamer4.callbacks import AsyncPolicyEval, log_dynamics_rollout_viz
from dreamer4.data import align_dynamics_batch
from dreamer4.eval_utils import dynamics_rollout_eval
from dreamer4.models import bc_loss, build_policy
from dreamer4.models.dynamics import pack_bottleneck_to_spatial, shortcut_forcing_loss
from dreamer4.trainers.base import BaseModule


class BCDynamicsModule(BaseModule):
    """Joint BC + dynamics training on a shared backbone with per-objective space masks."""

    stage = "bc_dynamics"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._env_eval = AsyncPolicyEval()

        self.tokenizer, self.model, self.n_spatial, self.packing_factor = build_policy(
            cfg, tokenizer_ckpt=cfg.get("tokenizer_ckpt")
        )
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.image_size = int(cfg.model.tokenizer.image_size)
        self.channels = int(cfg.model.tokenizer.channels)
        self.action_horizon = int(cfg.model.get("action_horizon", 8))

        self.dynamics_space_mode = self.model.dynamics_space_mode
        self.bc_space_mode = self.model.bc_space_mode

        self.flow_weight = float(cfg.train.get("flow_weight", 1.0))
        self.action_weight = float(cfg.train.get("action_weight", 1.0))
        self.reward_weight = float(cfg.train.get("reward_weight", 1.0))
        self.k_max = int(cfg.model.dynamics.get("k_max", 64))
        self.shortcut_self_fraction = float(cfg.train.get("shortcut_self_fraction", 0.25))
        self.shortcut_bootstrap_start = int(cfg.train.get("shortcut_bootstrap_start", 5000))

        self.rollout_ctx = int(cfg.train.get("rollout_ctx", 8))
        self.rollout_horizon = int(cfg.train.get("rollout_horizon", 8))
        self.rollout_flow_steps = int(cfg.train.get("rollout_flow_steps", 8))
        self._val_rollout_batch: tuple[torch.Tensor, torch.Tensor] | None = None
        self._val_viz_idx: int | None = None

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("BC+dynamics training requires images; set data.obs_mode=image or both")

        image, action, reward = align_dynamics_batch(batch.image, batch.action, batch.reward)
        with torch.no_grad():
            packed_z = pack_bottleneck_to_spatial(
                self.tokenizer.encode_images(image), self.n_spatial, self.packing_factor
            )

        B = packed_z.shape[0]
        B_self = int(round(self.shortcut_self_fraction * B))
        B_self = max(0, min(B - 1, B_self))
        flow_loss, flow_metrics = shortcut_forcing_loss(
            self.model.dynamics,
            packed_z,
            action,
            k_max=self.k_max,
            B_self=B_self,
            global_step=int(self.global_step),
            bootstrap_start=self.shortcut_bootstrap_start,
            space_mode=self.dynamics_space_mode,
        )
        bc_outputs = self.model(packed_z, action, space_mode=self.bc_space_mode)
        bc_loss_val, bc_metrics = bc_loss(
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
            prog = stage == "train" and key in ("flow_mse", "bootstrap_mse")
            self.log(f"{dyn_prefix}/{key}", value, prog_bar=prog, sync_dist=stage == "train")
        for key, value in bc_metrics.items():
            prog = stage == "train" and key in ("action_nll", "action_mse", "action_out_mean")
            self.log(f"{bc_prefix}/{key}", value, prog_bar=prog, sync_dist=stage == "train")
        if stage == "val":
            self.log(f"{self.stage}/loss", loss, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.is_global_zero and batch.image is not None and batch_idx == 0:
            self._pick_val_viz_batch_idx()
            self._val_rollout_batch = None
        if (
            self.trainer.is_global_zero
            and batch.image is not None
            and batch_idx == self._val_viz_idx
        ):
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
            image, action = self._val_rollout_batch
            self._val_rollout_batch = None
            self._val_viz_idx = None

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
        if not eval_cfg.get("env_eval", False):
            return

        step = int(self.trainer.global_step)
        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        self._env_eval.start(step, self.cfg, self.model, self.tokenizer, run_dir)
