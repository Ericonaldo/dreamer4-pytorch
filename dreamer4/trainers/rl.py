from __future__ import annotations

import copy
from pathlib import Path

import torch
from omegaconf import DictConfig

from dreamer4.callbacks import AsyncPolicyEval
from dreamer4.data import align_dynamics_batch
from dreamer4.models import build_policy
from dreamer4.models.dynamics import pack_bottleneck_to_spatial
from dreamer4.models.policy import (
    SymlogHead,
    imagine_latent_rollout,
    imagination_rl_loss,
)
from dreamer4.trainers.base import BaseModule


class RLModule(BaseModule):
    """Imagination RL: frozen dynamics + BC reward; train policy and value on imagined rollouts."""

    stage = "rl"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        imag = cfg.imagination
        self.seq_len = int(cfg.data.seq_len)
        self.context_len_min = int(imag.get("context_len_min", 8))
        self.horizon = int(imag.horizon)
        self.flow_steps = int(imag.flow_steps)
        self.gamma = float(imag.gamma)
        self.lambda_ = float(imag.lambda_)
        self.alpha = float(imag.get("alpha", 0.5))
        self.beta = float(imag.beta)
        self.normalize_advantages = bool(imag.get("normalize_advantages", False))
        self.policy_warmup_steps = int(imag.get("policy_warmup_steps", 200))
        self._policy_lr = float(imag.get("policy_lr", cfg.train.optimizer.lr))

        self.tokenizer, self.model, self.n_spatial, self.packing_factor = build_policy(
            cfg,
            tokenizer_ckpt=cfg.get("tokenizer_ckpt"),
            ckpt=cfg.get("bc_ckpt"),
        )

        for p in self.model.dynamics.parameters():
            p.requires_grad_(False)
        for p in self.model.heads.reward_head.parameters():
            p.requires_grad_(False)
        self.model.agent_tokens.requires_grad_(False)

        self.policy_prior = copy.deepcopy(self.model.heads.policy)
        for p in self.policy_prior.parameters():
            p.requires_grad_(False)

        value_hidden = int(cfg.model.get("value_hidden", cfg.model.get("reward_hidden", 256)))
        value_layers = int(cfg.model.get("value_layers", cfg.model.get("reward_layers", 1)))
        self.value_head = SymlogHead(
            self.model.d_model,
            value_hidden,
            layers=value_layers,
        )

        self._env_eval = AsyncPolicyEval()

    def configure_optimizers(self):
        opt_cfg = self.cfg.train.optimizer
        policy_lr = 0.0 if self.policy_warmup_steps > 0 else self._policy_lr
        param_groups = [
            {"params": list(self.value_head.parameters()), "name": "value"},
            {"params": list(self.model.heads.policy.parameters()), "name": "policy", "lr": policy_lr},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
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

    def _sample_context_window(
        self,
        image: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Random suffix context ending at the window tail; length uniform in [min_ctx, max_ctx]."""
        T = image.shape[1]
        max_ctx = min(self.seq_len, T)
        if max_ctx < 1:
            raise ValueError(f"need at least 1 context frame, got T={T}, max_ctx={max_ctx}")
        min_ctx = min(self.context_len_min, max_ctx)
        if min_ctx >= max_ctx:
            ctx_len = max_ctx
        else:
            ctx_len = int(torch.randint(min_ctx, max_ctx + 1, (1,)).item())
        start = T - ctx_len
        return image[:, start:], action[:, start:], ctx_len

    def _restore_policy_lr_after_warmup(self) -> None:
        if self.policy_warmup_steps <= 0:
            return
        if self.trainer is None:
            return
        if int(self.trainer.global_step) < self.policy_warmup_steps:
            return
        optimizers = self.trainer.optimizers
        if not optimizers:
            return
        opt = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        for pg in opt.param_groups:
            if pg.get("name") == "policy" and pg["lr"] != self._policy_lr:
                pg["lr"] = self._policy_lr

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        self._restore_policy_lr_after_warmup()

    def on_train_start(self) -> None:
        super().on_train_start()
        self._restore_policy_lr_after_warmup()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self._restore_policy_lr_after_warmup()

    def _imagine_from_batch(self, batch):
        if batch.image is None:
            raise ValueError("RL training requires images; set data.obs_mode=image or both")

        image, action, _ = align_dynamics_batch(batch.image, batch.action, batch.reward)
        if image.shape[1] < 2:
            raise ValueError(
                f"seq_len too short for imagination: need at least 2 aligned obs frames "
                f"(data.seq_len >= 1 in transition mode), got {image.shape[1]}"
            )

        image_ctx, action_ctx, ctx_len = self._sample_context_window(image, action)

        with torch.no_grad():
            packed_z = pack_bottleneck_to_spatial(
                self.tokenizer.encode_images(image_ctx), self.n_spatial, self.packing_factor
            )

        rollout = imagine_latent_rollout(
            self.model,
            self.model.dynamics,
            packed_z,
            action_ctx,
            self.model.heads.policy,
            self.horizon,
            self.flow_steps,
            ctx_len=ctx_len,
        )
        return rollout

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        rollout = self._imagine_from_batch(batch)

        train_policy = (
            stage == "train"
            and int(self.trainer.global_step) >= self.policy_warmup_steps
        )

        prefix = "val" if stage == "val" else self.stage
        loss, metrics = imagination_rl_loss(
            rollout.hidden,
            rollout.actions,
            self.model.heads,
            self.policy_prior,
            self.value_head,
            gamma=self.gamma,
            lambda_=self.lambda_,
            beta=self.beta,
            alpha=self.alpha,
            normalize_advantages=self.normalize_advantages,
            train_policy=train_policy,
        )

        for key, value in metrics.items():
            prog = stage == "train" and key in ("val_loss", "pi_loss", "mean_td_return")
            self.log(f"{prefix}/{key}", value, prog_bar=prog, sync_dist=stage == "train")
        if stage == "val":
            self.log("val/loss", loss, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if (
            self.policy_warmup_steps > 0
            and int(self.trainer.global_step) == self.policy_warmup_steps
        ):
            return None
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
        step = int(self.trainer.global_step)
        if self.policy_warmup_steps > 0 and step == self.policy_warmup_steps:
            return

        eval_cfg = self.cfg.get("eval", {})
        if not eval_cfg.get("env_eval", False):
            return

        run_dir = Path(self.cfg.log.dir) / self.cfg.log.run_name
        self._env_eval.start(step, self.cfg, self.model, self.tokenizer, run_dir)
