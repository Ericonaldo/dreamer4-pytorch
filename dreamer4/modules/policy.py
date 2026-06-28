from __future__ import annotations

import copy
from pathlib import Path

import torch
from lightning.pytorch.utilities import rank_zero_warn
from omegaconf import DictConfig

from dreamer4.data import align_dynamics_batch
from dreamer4.imagination import imagine_latent_rollout
from dreamer4.modules.base import BaseModule
from dreamer4.modules.bc import _load_state
from dreamer4.models.policy import imagination_rl_loss, SymExpTwoHotEncoder, SymExpTwoHotHead
from dreamer4.policy_agent import AsyncBCEval


class PolicyModule(BaseModule):
    """Imagination RL: frozen dynamics + BC reward; train policy and value on imagined rollouts."""

    stage = "policy"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import BCModel, build_tokenizer
        from dreamer4.models.dynamics import pack_bottleneck_to_spatial
        from dreamer4.models.tokenizer import encode_images

        self._pack = pack_bottleneck_to_spatial
        self._encode_images = encode_images

        imag = cfg.imagination
        self.ctx_len = int(imag.context_length)
        self.horizon = int(imag.horizon)
        self.flow_steps = int(imag.flow_steps)
        self.gamma = float(imag.gamma)
        self.lambda_ = float(imag.lambda_)
        self.alpha = float(imag.alpha)
        self.beta = float(imag.beta)

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

        self.model = BCModel(
            cfg.model.dynamics,
            n_latents=n_latents,
            latent_dim=latent_dim,
            heads_cfg=cfg.model,
        )
        _load_state(self.model, cfg.bc_ckpt, prefix="model.")

        for p in self.model.dynamics.parameters():
            p.requires_grad_(False)
        for p in self.model.heads.reward_head.parameters():
            p.requires_grad_(False)

        self.policy_prior = copy.deepcopy(self.model.heads.policy)
        for p in self.policy_prior.parameters():
            p.requires_grad_(False)

        reward_bins = int(cfg.model.get("reward_bins", 255))
        reward_symexp_span = float(cfg.model.get("reward_symexp_span", 20.0))
        value_hidden = int(cfg.model.get("value_hidden", cfg.model.get("reward_hidden", 256)))
        value_layers = int(cfg.model.get("value_layers", cfg.model.get("reward_layers", 1)))
        value_enc = SymExpTwoHotEncoder(num_bins=reward_bins, symexp_span=reward_symexp_span)
        self.value_head = SymExpTwoHotHead(
            self.model.d_model,
            value_hidden,
            (value_enc.num_bins,),
            value_enc,
            layers=value_layers,
        )

        self.bc_space_mode = self.model.bc_space_mode
        self._env_eval = AsyncBCEval()

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
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

    def _encode_packed(self, image_bthwc: torch.Tensor) -> torch.Tensor:
        z = self._encode_images(self.tokenizer, image_bthwc, self.patch_size)
        return self._pack(z, self.n_spatial, self.packing_factor)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if batch.image is None:
            raise ValueError("Policy training requires images; set data.obs_mode=image or both")

        prefix = "val" if stage == "val" else self.stage
        image, action, _ = align_dynamics_batch(batch.image, batch.action, batch.reward)
        need = self.ctx_len + self.horizon + 1
        if image.shape[1] < need:
            raise ValueError(
                f"seq_len too short for imagination: need {need - 1} transitions "
                f"(context {self.ctx_len} + horizon {self.horizon}), got {image.shape[1] - 1}"
            )

        with torch.no_grad():
            packed_z = self._encode_packed(image)

        z_ctx = packed_z[:, :self.ctx_len]
        a_ctx = action[:, :self.ctx_len]

        rollout = imagine_latent_rollout(
            self.model,
            self.model.dynamics,
            z_ctx,
            a_ctx,
            self.model.heads.policy,
            self.horizon,
            self.flow_steps,
            bc_space_mode=self.bc_space_mode,
            ctx_len=self.ctx_len,
        )

        loss, metrics = imagination_rl_loss(
            rollout.hidden,
            rollout.actions,
            rollout.log_prob,
            self.model.heads,
            self.policy_prior,
            self.value_head,
            gamma=self.gamma,
            lambda_=self.lambda_,
            alpha=self.alpha,
            beta=self.beta,
        )

        for key, value in metrics.items():
            prog = stage == "train" and key in ("val_loss", "pi_loss", "mean_td_return")
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
            rank_zero_warn(f"Skipping policy env eval (install dreamer4[dmc]): {exc}")
            return

        step = int(self.trainer.global_step)
        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        self._env_eval.start(step, self.cfg, self.model, self.tokenizer, run_dir)
