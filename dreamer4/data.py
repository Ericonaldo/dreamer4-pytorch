from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Batch:
    image: torch.Tensor | None
    proprio: torch.Tensor | None
    action: torch.Tensor
    reward: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    is_terminal: torch.Tensor
    length: torch.Tensor


def collate_episodes(items: list[dict[str, Any]]) -> Batch:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([item[key] for item in items])

    image = stack("image") if items[0]["image"] is not None else None
    proprio = stack("proprio") if items[0]["proprio"] is not None else None
    return Batch(
        image=image,
        proprio=proprio,
        action=stack("action"),
        reward=stack("reward"),
        is_first=stack("is_first"),
        is_last=stack("is_last"),
        is_terminal=stack("is_terminal"),
        length=stack("length"),
    )


def _proprio_vector(data: dict[str, Any]) -> np.ndarray:
    """orientations (14) + height (1) + velocity (9) = 24; or precomputed vector."""
    if "vector" in data:
        return np.asarray(data["vector"])
    orient = np.asarray(data["orientations"])
    height = np.asarray(data["height"])
    velocity = np.asarray(data["velocity"])
    if height.ndim == 1:
        height = height[..., None]
    return np.concatenate([orient, height, velocity], axis=-1)


def split_episode_indices(
    n_episodes: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Episode-level train/val split (indices into ShardedDatasetReader)."""
    if n_episodes < 2:
        raise ValueError(f"Need at least 2 episodes for train/val split, got {n_episodes}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    n_val = min(n_val, n_episodes - 1)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx


class GranularEpisodeDataset(Dataset):
    """Granular ShardedDatasetReader: elem = {'data': ..., 'length': T}."""

    def __init__(
        self,
        path: str,
        seq_len: int,
        obs_mode: str = "both",
        indices: list[int] | None = None,
        transform=None,
    ):
        import granular

        self.reader = granular.ShardedDatasetReader(path, granular.decoders)
        self.seq_len = seq_len
        self.obs_mode = obs_mode
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.reader)

    def __getitem__(self, index: int) -> dict[str, Any]:
        reader_idx = self.indices[index] if self.indices is not None else index
        elem = self.reader[reader_idx]
        data, length = elem["data"], min(int(elem["length"]), self.seq_len)

        image = proprio = None
        if self.obs_mode in ("image", "both"):
            image = torch.from_numpy(np.asarray(data["image"][:length], np.uint8).copy()).float() / 255.0
        if self.obs_mode in ("proprio", "both"):
            proprio = torch.from_numpy(_proprio_vector(data)[:length]).float()

        item = {
            "image": image,
            "proprio": proprio,
            "action": torch.as_tensor(np.asarray(data["action"][:length]).copy(), dtype=torch.float32),
            "reward": torch.as_tensor(np.asarray(data["reward"][:length]).copy(), dtype=torch.float32),
            "is_first": torch.as_tensor(np.asarray(data["is_first"][:length]).copy(), dtype=torch.bool),
            "is_last": torch.as_tensor(np.asarray(data["is_last"][:length]).copy(), dtype=torch.bool),
            "is_terminal": torch.as_tensor(np.asarray(data["is_terminal"][:length]).copy(), dtype=torch.bool),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
