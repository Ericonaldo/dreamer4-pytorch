from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CHUNK_SIZE = 100


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


def align_wm_obs_action(
    image: torch.Tensor | None,
    action: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Transition windows: (B,T+1,...) obs with (B,T,...) act -> encode first T frames."""
    if image is None:
        return None, action
    if image.shape[1] == action.shape[1] + 1:
        return image[:, :-1], action
    return image, action


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
    """Episode-level train/val split (indices into discovered episodes)."""
    if n_episodes < 2:
        raise ValueError(f"Need at least 2 episodes for train/val split, got {n_episodes}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_val = max(1, int(round(n_episodes * val_fraction)))
    n_val = min(n_val, n_episodes - 1)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    return train_idx, val_idx


def _discover_episodes(reader, n_chunks: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """Return inclusive global step ranges (start, end) for each complete episode."""
    events: list[tuple[int, str]] = []
    for i in range(n_chunks):
        elem = reader[i]
        length = min(int(elem["length"]), chunk_size)
        g0 = i * chunk_size
        is_first = np.asarray(elem["data"]["is_first"][:length])
        is_last = np.asarray(elem["data"]["is_last"][:length])
        for t in np.where(is_first)[0]:
            events.append((g0 + int(t), "S"))
        for t in np.where(is_last)[0]:
            events.append((g0 + int(t), "E"))

    events.sort()
    episodes: list[tuple[int, int]] = []
    cur_start: int | None = None
    for step, kind in events:
        if kind == "S":
            cur_start = step
        elif kind == "E" and cur_start is not None:
            episodes.append((cur_start, step))
            cur_start = None
    return episodes


def _build_valid_starts(
    episodes: list[tuple[int, int]],
    seq_len: int,
    *,
    window_mode: str,
    episode_indices: set[int] | None,
) -> list[tuple[int, int]]:
    """List of (episode_idx, global_start) for every in-episode window."""
    valid: list[tuple[int, int]] = []
    transition = window_mode == "transition"
    for ep_idx, (gs, ge) in enumerate(episodes):
        if episode_indices is not None and ep_idx not in episode_indices:
            continue
        ep_len = ge - gs + 1
        if transition:
            if ep_len <= seq_len:
                continue
            n_starts = ep_len - seq_len
        else:
            if ep_len < seq_len:
                continue
            n_starts = ep_len - seq_len + 1
        for offset in range(n_starts):
            valid.append((ep_idx, gs + offset))
    return valid


class _ChunkCache:
    def __init__(self, reader):
        self.reader = reader
        self._idx: int | None = None
        self._data: dict[str, Any] | None = None

    def get(self, chunk_idx: int) -> dict[str, Any]:
        if self._idx != chunk_idx:
            self._data = self.reader[chunk_idx]["data"]
            self._idx = chunk_idx
        assert self._data is not None
        return self._data


def _slice_global(
    cache: _ChunkCache,
    global_start: int,
    length: int,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Read `length` contiguous steps from the global timeline."""
    parts: dict[str, list[np.ndarray]] = {}
    pos = global_start
    remaining = length
    while remaining > 0:
        chunk_idx = pos // chunk_size
        local = pos % chunk_size
        data = cache.get(chunk_idx)
        take = min(remaining, chunk_size - local)
        sl = slice(local, local + take)
        for key in ("image", "action", "reward", "is_first", "is_last", "is_terminal",
                    "orientations", "height", "velocity"):
            if key not in data:
                continue
            arr = np.asarray(data[key][sl])
            parts.setdefault(key, []).append(arr.copy())
        pos += take
        remaining -= take
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


class GranularEpisodeDataset(Dataset):
    """
    Valid-start dataset over stitched Granular episodes (ref: ShardedFrameDataset / WMDataset).

    Precomputes every in-episode window that fits ``seq_len`` without crossing ``is_first`` /
    ``is_last`` boundaries. Short episodes are skipped; tail windows that would cross an episode
    end are never enumerated.

    ``window_mode``:
      - ``frame`` (tokenizer): ``seq_len`` frames for image/proprio/action/reward.
      - ``transition`` (dynamics/bc): ``seq_len+1`` obs, ``seq_len`` action/reward with
        ``obs[t] -- action[t] --> obs[t+1]`` and ``reward[t+1]`` (embodied Granular layout).
    """

    def __init__(
        self,
        path: str,
        seq_len: int,
        obs_mode: str = "both",
        episode_indices: list[int] | None = None,
        window_mode: str = "frame",
        chunk_size: int = CHUNK_SIZE,
        verbose: bool = True,
        transform=None,
        *,
        indices: list[int] | None = None,
    ):
        import granular

        if indices is not None:
            if episode_indices is not None:
                raise ValueError("Pass episode_indices or legacy indices, not both")
            episode_indices = indices

        if window_mode not in ("frame", "transition"):
            raise ValueError(f"window_mode must be 'frame' or 'transition', got {window_mode!r}")

        self.reader = granular.ShardedDatasetReader(path, granular.decoders)
        self.seq_len = int(seq_len)
        self.obs_mode = obs_mode
        self.window_mode = window_mode
        self.chunk_size = int(chunk_size)
        self.transform = transform
        self._cache = _ChunkCache(self.reader)

        n_chunks = len(self.reader)
        self.episodes = _discover_episodes(self.reader, n_chunks, self.chunk_size)
        ep_filter = set(episode_indices) if episode_indices is not None else None
        self.valid = _build_valid_starts(
            self.episodes,
            self.seq_len,
            window_mode=window_mode,
            episode_indices=ep_filter,
        )

        if verbose:
            ep_len = np.array([ge - gs + 1 for gs, ge in self.episodes], dtype=np.int64)
            print(
                f"[GranularEpisodeDataset] chunks={n_chunks:,} episodes={len(self.episodes):,} "
                f"valid_starts={len(self.valid):,} seq_len={self.seq_len} "
                f"window_mode={window_mode} episode_len median={int(np.median(ep_len)) if len(ep_len) else 0}"
            )
            if len(self.valid) == 0:
                print("[GranularEpisodeDataset] WARNING: no valid sequence starts")

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def __len__(self) -> int:
        return len(self.valid)

    def _window_bounds(self, global_start: int) -> tuple[int, int, int, int, int, int]:
        """Return obs_start, obs_len, act_start, act_len, rew_start, rew_len in global coords."""
        if self.window_mode == "transition":
            obs_len = self.seq_len + 1
            act_len = self.seq_len
            return global_start, obs_len, global_start, act_len, global_start + 1, act_len
        n = self.seq_len
        return global_start, n, global_start, n, global_start, n

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, global_start = self.valid[index]
        obs_start, obs_len, act_start, act_len, rew_start, rew_len = self._window_bounds(global_start)

        obs_data = _slice_global(self._cache, obs_start, obs_len, self.chunk_size)
        if self.window_mode == "transition":
            act_data = _slice_global(self._cache, act_start, act_len, self.chunk_size)
            rew_data = _slice_global(self._cache, rew_start, rew_len, self.chunk_size)
        else:
            act_data = obs_data
            rew_data = obs_data

        image = proprio = None
        if self.obs_mode in ("image", "both"):
            image = torch.from_numpy(np.asarray(obs_data["image"], np.uint8)).float() / 255.0
        if self.obs_mode in ("proprio", "both"):
            proprio = torch.from_numpy(_proprio_vector(obs_data)).float()

        item = {
            "image": image,
            "proprio": proprio,
            "action": torch.as_tensor(np.asarray(act_data["action"]), dtype=torch.float32).clamp(-1, 1),
            "reward": torch.as_tensor(np.asarray(rew_data["reward"]), dtype=torch.float32),
            "is_first": torch.as_tensor(np.asarray(act_data["is_first"]), dtype=torch.bool),
            "is_last": torch.as_tensor(np.asarray(act_data["is_last"]), dtype=torch.bool),
            "is_terminal": torch.as_tensor(np.asarray(act_data["is_terminal"]), dtype=torch.bool),
            "length": torch.tensor(act_len, dtype=torch.long),
        }
        if self.transform is not None:
            item = self.transform(item)
        return item
