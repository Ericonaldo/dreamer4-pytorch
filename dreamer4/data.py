"""Granular episode dataset and batch utilities for Dreamer4 training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CHUNK_SIZE = 100  # Granular shard size; global step i lives in chunk i // CHUNK_SIZE.


@dataclass
class Batch:
    """Batched episode windows from ``collate_episodes`` (B, T, ...)."""

    image: torch.Tensor | None
    proprio: torch.Tensor | None
    action: torch.Tensor
    reward: torch.Tensor
    is_first: torch.Tensor
    is_last: torch.Tensor
    is_terminal: torch.Tensor
    length: torch.Tensor


def collate_episodes(items: list[dict[str, Any]]) -> Batch:
    """Stack per-sample dicts from ``GranularEpisodeDataset`` into a training batch."""
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


def align_dynamics_batch(
    image: torch.Tensor | None,
    action: torch.Tensor,
    reward: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """
    Dreamer transition layout (ref nicklashansen / lucidrains / dreamer4-jax).

    Dataset transition windows provide:
      obs [s0..sT] (T+1), actions [a1..aT] (T), rewards [r1..rT] (T),
    with s_{t-1} -- a_t --> s_t and reward r_t.

    Dynamics time t uses frame s_t, action a_t (0 at t=0), reward r_t (0 at t=0).
    """
    if image is None:
        return None, action, reward
    if image.shape[1] != action.shape[1] + 1:
        return image, action, reward

    t1 = image.shape[1]
    action_aligned = action.new_zeros(action.shape[0], t1, *action.shape[2:])
    action_aligned[:, 1:] = action

    reward_aligned = None
    if reward is not None:
        reward_aligned = reward.new_zeros(reward.shape[0], t1, *reward.shape[2:])
        reward_aligned[:, 1:] = reward

    return image, action_aligned, reward_aligned


def _proprio_vector(data: dict[str, Any]) -> np.ndarray:
    """Build 24-d proprio from raw fields or a precomputed ``vector`` column."""
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
    """Shuffle episode indices and split into train / val lists for ``episode_indices``."""
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
    """Scan ``is_first`` / ``is_last`` flags and return inclusive global (start, end) per episode."""
    events: list[tuple[int, str]] = []
    for i in range(n_chunks):
        elem = reader[i, ("length", "data")]
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
    """Enumerate every valid in-episode window as ``(episode_idx, global_start)``."""
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
    """One-chunk LRU over a Granular reader; avoids re-reading the same shard per window."""

    def __init__(self, reader):
        self.reader = reader
        self._idx: int | None = None
        self._data: dict[str, Any] | None = None

    def get(self, chunk_idx: int) -> dict[str, Any]:
        """Return decoded ``data`` dict for ``chunk_idx``, reusing the cached chunk when possible."""
        if self._idx != chunk_idx:
            self._data = self.reader[chunk_idx, ("data",)]["data"]
            self._idx = chunk_idx
        assert self._data is not None
        return self._data


def _slice_global(
    cache: _ChunkCache,
    global_start: int,
    length: int,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Read ``length`` contiguous global steps, stitching across chunk boundaries."""
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
      - ``transition`` (dynamics/bc): ``seq_len+1`` obs ``[s0..sT]``, ``seq_len`` actions
        ``[a1..aT]`` and rewards ``[r1..rT]`` (Dreamer: ``s_{t-1} -- a_t --> s_t``).
        Use ``align_dynamics_batch`` to pad action/reward at ``t=0``.
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
        """Open the dataset once to discover episodes and precompute valid window starts."""
        if indices is not None:
            if episode_indices is not None:
                raise ValueError("Pass episode_indices or legacy indices, not both")
            episode_indices = indices

        if window_mode not in ("frame", "transition"):
            raise ValueError(f"window_mode must be 'frame' or 'transition', got {window_mode!r}")

        self.path = path
        self.seq_len = int(seq_len)
        self.obs_mode = obs_mode
        self.window_mode = window_mode
        self.chunk_size = int(chunk_size)
        self.transform = transform
        self._reader = None
        self._cache: _ChunkCache | None = None
        self._reader_pid: int | None = None

        import granular

        reader = granular.ShardedDatasetReader(path, granular.decoders)
        try:
            n_chunks = len(reader)
            self.episodes = _discover_episodes(reader, n_chunks, self.chunk_size)
        finally:
            reader.close()

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
        """Number of complete episodes discovered in the Granular dataset."""
        return len(self.episodes)

    def _ensure_reader(self) -> _ChunkCache:
        """Open a Granular reader in the current process (fork-safe for DataLoader workers)."""
        import granular

        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            if self._reader is not None:
                self._reader.close()
            self._reader = granular.ShardedDatasetReader(self.path, granular.decoders)
            self._cache = _ChunkCache(self._reader)
            self._reader_pid = pid
        assert self._cache is not None
        return self._cache

    def reset_reader(self) -> None:
        """Drop cached reader handles (e.g. from DataLoader worker_init_fn)."""
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._cache = None
        self._reader_pid = None

    def __len__(self) -> int:
        """Number of valid in-episode windows (dataset size for DataLoader)."""
        return len(self.valid)

    def episode_length(self, ep_idx: int) -> int:
        """Return step count for episode ``ep_idx`` (inclusive of terminal step)."""
        gs, ge = self.episodes[ep_idx]
        return ge - gs + 1

    def get_transition_window(self, ep_idx: int, offset: int) -> dict[str, Any]:
        """Load one transition window by episode and in-episode offset (eval helper)."""
        if self.window_mode != "transition":
            raise ValueError("get_transition_window requires window_mode=transition")
        gs, _ = self.episodes[ep_idx]
        key = (ep_idx, gs + offset)
        for i, valid_key in enumerate(self.valid):
            if valid_key == key:
                return self.__getitem__(i)
        raise KeyError(f"no transition window for episode {ep_idx} offset {offset}")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one window dict: tensors for image/proprio/action/reward and episode flags."""
        cache = self._ensure_reader()
        _, global_start = self.valid[index]

        if self.window_mode == "transition":
            obs_data = _slice_global(cache, global_start, self.seq_len + 1, self.chunk_size)
            act_data = _slice_global(cache, global_start, self.seq_len, self.chunk_size)
            rew_data = _slice_global(cache, global_start + 1, self.seq_len, self.chunk_size)
            act_len = self.seq_len
        else:
            obs_data = _slice_global(cache, global_start, self.seq_len, self.chunk_size)
            act_data = rew_data = obs_data
            act_len = self.seq_len

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
