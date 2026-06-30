"""Online BC policy inference (load checkpoint → act in env)."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig

from dreamer4.checkpoint import load_state
from dreamer4.models import PolicyModel, build_tokenizer
from dreamer4.models.dynamics import pack_bottleneck_to_spatial
from dreamer4.models.policy import POLICY_ENV_ACTION_SLOT
from dreamer4.models.tokenizer import encode_images


def load_bc_modules(
    cfg: DictConfig,
    device: torch.device,
    *,
    model_state: dict[str, torch.Tensor] | None = None,
    tokenizer_state: dict[str, torch.Tensor] | None = None,
) -> tuple[PolicyModel, nn.Module]:
    tokenizer = build_tokenizer(cfg.model.tokenizer)
    if tokenizer_state is not None:
        tokenizer.load_state_dict(tokenizer_state, strict=True)
    elif cfg.get("tokenizer_ckpt"):
        load_state(tokenizer, cfg.tokenizer_ckpt, prefix="model.")

    n_latents = tokenizer.encoder.n_latents
    latent_dim = tokenizer.encoder.bottleneck_proj.out_features
    model = PolicyModel(
        cfg.model.dynamics,
        n_latents=n_latents,
        latent_dim=latent_dim,
        heads_cfg=cfg.model,
    )
    if model_state is not None:
        model.load_state_dict(model_state, strict=False)
    elif cfg.get("bc_ckpt"):
        ckpt = torch.load(cfg.bc_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        filtered = {
            k.removeprefix("model."): v
            for k, v in state.items()
            if k.startswith("model.") and "attn_mask" not in k
        }
        model.load_state_dict(filtered, strict=False)
    elif cfg.get("dynamics_ckpt"):
        load_state(model.dynamics, cfg.dynamics_ckpt, prefix="model.")

    tokenizer.eval()
    model.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    tokenizer.to(device)
    model.to(device)
    return model, tokenizer


def _pad_stack(seqs: list[torch.Tensor]) -> torch.Tensor:
    t_max = max(s.shape[0] for s in seqs)
    padded = []
    for s in seqs:
        if s.shape[0] < t_max:
            pad = s.new_zeros(t_max - s.shape[0], *s.shape[1:])
            s = torch.cat([pad, s], dim=0)
        padded.append(s)
    return torch.stack(padded, dim=0)


def resolve_eval_action_horizon(cfg: DictConfig) -> int:
    """Open-loop env steps per model forward; 1 = replan from current obs each step."""
    model_h = int(cfg.model.get("action_horizon", 8))
    eval_h = int(cfg.get("eval", {}).get("action_horizon", 1))
    return max(1, min(eval_h, model_h - 1))


class BCPolicy:
    """Online BC agent with optional batched env slots (encode → PolicyModel → MTP action)."""

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device,
        *,
        model: PolicyModel | None = None,
        tokenizer: nn.Module | None = None,
        num_envs: int = 1,
    ):
        self.device = device
        self.num_envs = int(num_envs)
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.max_history = int(cfg.eval.get("max_history", 16))
        self.action_dim = int(cfg.model.dynamics.action_dim)
        self.open_loop_steps = resolve_eval_action_horizon(cfg)

        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_bc_modules(cfg, device)
        else:
            self.model = model
            self.tokenizer = tokenizer

        n_latents = self.tokenizer.encoder.n_latents
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor

        self._z: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._a: list[list[torch.Tensor]] = [[] for _ in range(self.num_envs)]
        self._pending: list[list[np.ndarray]] = [[] for _ in range(self.num_envs)]
        self._last_executed: list[np.ndarray | None] = [None] * self.num_envs

    def reset(self, ids: list[int] | None = None) -> None:
        if ids is None:
            ids = list(range(self.num_envs))
        for i in ids:
            self._z[i].clear()
            self._a[i].clear()
            self._pending[i].clear()
            self._last_executed[i] = None

    def _push_action(self, slot: int, action: np.ndarray) -> None:
        self._a[slot].append(torch.from_numpy(action.astype(np.float32)).to(self.device))
        if len(self._a[slot]) > self.max_history:
            self._a[slot].pop(0)

    def _commit_last_action(self, ids: list[int]) -> None:
        for i in ids:
            if self._z[i] and self._last_executed[i] is not None:
                self._push_action(i, self._last_executed[i])
                self._last_executed[i] = None

    def _aligned_actions(self, slot: int, t: int) -> torch.Tensor:
        out = torch.zeros(t, self.action_dim, device=self.device)
        n_transitions = min(len(self._a[slot]), max(0, t - 1))
        if n_transitions > 0:
            a_hist = torch.stack(self._a[slot][-n_transitions:], dim=0)
            out[1 : 1 + n_transitions] = a_hist
        return out

    def _append_observations(self, images: np.ndarray, ids: list[int]) -> None:
        imgs = (
            torch.from_numpy(np.ascontiguousarray(images))
            .to(self.device)
            .float()
            .div_(255.0)
            .unsqueeze(1)
        )
        z = encode_images(self.tokenizer, imgs, self.patch_size)
        packed = pack_bottleneck_to_spatial(z, self.n_spatial, self.packing_factor)[:, 0]
        for j, i in enumerate(ids):
            self._z[i].append(packed[j])
            if len(self._z[i]) > self.max_history:
                self._z[i].pop(0)
                if self._a[i]:
                    self._a[i].pop(0)

    def _observe_env(self, images: np.ndarray, ids: list[int]) -> None:
        self._commit_last_action(ids)
        self._append_observations(images, ids)

    def _forward_mtp_actions(self, ids: list[int]) -> np.ndarray:
        z_seqs, a_seqs = [], []
        for i in ids:
            z_seq = torch.stack(self._z[i], dim=0)
            z_seqs.append(z_seq)
            a_seqs.append(self._aligned_actions(i, z_seq.shape[0]))

        z_batch = _pad_stack(z_seqs)
        a_batch = _pad_stack(a_seqs)
        outputs = self.model(z_batch, a_batch)
        end = POLICY_ENV_ACTION_SLOT + self.open_loop_steps
        return outputs.action[:, -1, POLICY_ENV_ACTION_SLOT:end].float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act(self, images: np.ndarray, ids: list[int] | None = None) -> np.ndarray:
        single = images.ndim == 3
        if single:
            images = images[np.newaxis, ...]
            ids = [0]

        actions_out: dict[int, np.ndarray] = {}
        need_forward: list[int] = []
        for j, i in enumerate(ids):
            if self._pending[i]:
                self._observe_env(images[j : j + 1], [i])
                action = self._pending[i].pop(0)
                self._last_executed[i] = action
                actions_out[i] = action
            else:
                need_forward.append(i)

        if need_forward:
            fwd_idx = [j for j, i in enumerate(ids) if i in need_forward]
            self._observe_env(images[fwd_idx], need_forward)
            mtp = self._forward_mtp_actions(need_forward)
            for j, i in enumerate(need_forward):
                step_actions = mtp[j]
                first = step_actions[0]
                self._last_executed[i] = first
                actions_out[i] = first
                if step_actions.shape[0] > 1:
                    self._pending[i].extend(step_actions[1:])

        ordered = np.stack([actions_out[i] for i in ids], axis=0)
        return ordered[0] if single else ordered
