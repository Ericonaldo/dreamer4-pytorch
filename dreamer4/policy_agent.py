from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from dreamer4.models import BCModel, build_tokenizer
from dreamer4.models.dynamics import pack_bottleneck_to_spatial
from dreamer4.models.tokenizer import encode_images


def _load_state(module: nn.Module, ckpt_path: str, *, prefix: str = "model.") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


@dataclass
class EpisodeStats:
    return_: float
    length: int


class RandomPolicy:
    def __init__(self, action_dim: int = 6):
        self.action_dim = int(action_dim)

    def reset(self) -> None:
        return

    def act(self, image_uint8: np.ndarray) -> np.ndarray:
        del image_uint8
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)


class BCPolicy:
    """Online BC agent: encode frames, teacher-forced actions, predict a_t from head."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.device = device
        self.patch_size = int(cfg.model.tokenizer.patch_size)
        self.max_history = int(cfg.eval.get("max_history", 32))
        self.action_dim = int(cfg.model.dynamics.action_dim)

        self.tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            _load_state(self.tokenizer, cfg.tokenizer_ckpt, prefix="model.")
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

        n_latents = self.tokenizer.encoder.n_latents
        latent_dim = self.tokenizer.encoder.bottleneck_proj.out_features
        self.packing_factor = int(cfg.model.dynamics.get("packing_factor", 1))
        self.n_spatial = n_latents // self.packing_factor

        self.model = BCModel(
            cfg.model.dynamics,
            n_latents=n_latents,
            latent_dim=latent_dim,
            heads_cfg=cfg.model,
        )
        if cfg.get("bc_ckpt"):
            ckpt = torch.load(cfg.bc_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            model_state = {
                k.removeprefix("model."): v
                for k, v in state.items()
                if k.startswith("model.") and "attn_mask" not in k
            }
            self.model.load_state_dict(model_state, strict=False)
        elif cfg.get("dynamics_ckpt"):
            _load_state(self.model.dynamics, cfg.dynamics_ckpt, prefix="model.")
        self.model.eval()
        self.model.to(device)
        self.tokenizer.to(device)

        self._z_packed: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []

    def reset(self) -> None:
        self._z_packed.clear()
        self._actions.clear()

    @torch.no_grad()
    def act(self, image_uint8: np.ndarray) -> np.ndarray:
        image = torch.from_numpy(image_uint8).to(self.device).float() / 255.0
        image = image.unsqueeze(0).unsqueeze(0)
        z = encode_images(self.tokenizer, image, self.patch_size)
        packed = pack_bottleneck_to_spatial(z, self.n_spatial, self.packing_factor)[0, 0]
        self._z_packed.append(packed)

        if len(self._z_packed) > self.max_history:
            self._z_packed.pop(0)
            if self._actions:
                self._actions.pop(0)

        T = len(self._z_packed)
        z_seq = torch.stack(self._z_packed, dim=0).unsqueeze(0)
        if self._actions:
            a_hist = torch.stack(self._actions, dim=0)
            placeholder = torch.zeros(self.action_dim, device=self.device)
            if a_hist.shape[0] < T:
                a_hist = torch.cat([a_hist, placeholder.unsqueeze(0)], dim=0)
            actions = a_hist.unsqueeze(0)
        else:
            actions = torch.zeros(1, 1, self.action_dim, device=self.device)

        outputs = self.model(z_seq, actions)
        action = outputs.action[0, -1, 0].float().cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._actions.append(torch.from_numpy(action).to(self.device))
        if len(self._actions) > self.max_history:
            self._actions.pop(0)
        return action


def run_episodes(
    env,
    policy,
    num_episodes: int,
) -> list[EpisodeStats]:
    stats: list[EpisodeStats] = []
    for _ in range(num_episodes):
        policy.reset()
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0
        while True:
            action = policy.act(obs["image"])
            obs = env.step(action)
            ep_return += float(obs["reward"])
            ep_len += 1
            if obs["is_last"]:
                break
        stats.append(EpisodeStats(return_=ep_return, length=ep_len))
    return stats


def summarize_episodes(stats: list[EpisodeStats]) -> dict[str, float]:
    returns = np.array([s.return_ for s in stats], dtype=np.float64)
    lengths = np.array([s.length for s in stats], dtype=np.float64)
    return {
        "episodes": len(stats),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "length_min": float(lengths.min()),
        "length_max": float(lengths.max()),
    }
