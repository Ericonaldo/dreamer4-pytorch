from __future__ import annotations

import os
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


def episode_cumulative_returns(path: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Discover episodes and return per-episode sum of `reward` over inclusive [start, end]."""
    reader = _open_granular_reader(path)
    n_chunks = len(reader)
    episodes = _discover_episodes(reader, n_chunks, CHUNK_SIZE)
    cache = _ChunkCache(reader)
    returns = np.zeros(len(episodes), dtype=np.float64)
    for ep_idx, (gs, ge) in enumerate(episodes):
        rew = _slice_global(cache, gs, ge - gs + 1, CHUNK_SIZE)["reward"]
        returns[ep_idx] = float(np.sum(rew))
    reader.close()
    return episodes, returns


def select_episodes_by_return(
    returns: np.ndarray,
    min_return: float,
    *,
    episode_indices: set[int] | None = None,
    top_k: int | None = None,
    max_return: float | None = None,
) -> list[tuple[int, float]]:
    """Return [(ep_idx, return)]; with max_return uses [min_return, max_return)."""
    candidates: list[tuple[int, float]] = []
    for ep_idx, ret in enumerate(returns):
        if episode_indices is not None and ep_idx not in episode_indices:
            continue
        r = float(ret)
        if r < min_return:
            continue
        if max_return is not None and r >= max_return:
            continue
        candidates.append((ep_idx, r))
    if max_return is None:
        candidates.sort(key=lambda x: x[1], reverse=True)
    else:
        mid = (min_return + max_return) / 2.0
        candidates.sort(key=lambda x: abs(x[1] - mid))
    if top_k is not None:
        candidates = candidates[:top_k]
    return candidates


def reward_band_counts(returns: np.ndarray, bands: list[tuple[str, float, float]]) -> list[dict]:
    """Population counts per band [low, high)."""
    out = []
    for name, low, high in bands:
        mask = (returns >= low) & (returns < high)
        band_rets = returns[mask]
        out.append(
            {
                "name": name,
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "return_mean": float(band_rets.mean()) if len(band_rets) else None,
                "return_min": float(band_rets.min()) if len(band_rets) else None,
                "return_max": float(band_rets.max()) if len(band_rets) else None,
            }
        )
    return out


def transition_window_offset(ep_len: int, seq_len: int, *, position: str = "middle") -> int:
    """In-episode offset for a transition window (0 .. ep_len - seq_len - 1)."""
    if ep_len <= seq_len:
        raise ValueError(f"episode length {ep_len} must exceed seq_len {seq_len}")
    n_starts = ep_len - seq_len
    if position == "start":
        return 0
    if position == "middle":
        return n_starts // 2
    raise ValueError(f"unknown position {position!r}")


def _open_granular_reader(path: str):
    import granular

    return granular.ShardedDatasetReader(path, granular.decoders)


def _discover_episodes(reader, n_chunks: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """Return inclusive global step ranges (start, end) for each complete episode."""
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

        reader = _open_granular_reader(path)
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
        return len(self.episodes)

    def _ensure_reader(self) -> _ChunkCache:
        """Open a Granular reader in the current process (fork-safe for DataLoader workers)."""
        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            if self._reader is not None:
                self._reader.close()
            self._reader = _open_granular_reader(self.path)
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
        return len(self.valid)

    def episode_length(self, ep_idx: int) -> int:
        gs, ge = self.episodes[ep_idx]
        return ge - gs + 1

    def get_transition_window(self, ep_idx: int, offset: int) -> dict[str, Any]:
        """Load transition window at in-episode offset (same layout as __getitem__)."""
        if self.window_mode != "transition":
            raise ValueError("get_transition_window requires window_mode=transition")
        gs, _ = self.episodes[ep_idx]
        key = (ep_idx, gs + offset)
        for i, valid_key in enumerate(self.valid):
            if valid_key == key:
                return self.__getitem__(i)
        raise KeyError(f"no transition window for episode {ep_idx} offset {offset}")

    def _window_bounds(self, global_start: int) -> tuple[int, int, int, int, int, int]:
        """Return obs_start, obs_len, act_start, act_len, rew_start, rew_len in global coords."""
        if self.window_mode == "transition":
            obs_len = self.seq_len + 1
            act_len = self.seq_len
            return global_start, obs_len, global_start, act_len, global_start + 1, act_len
        n = self.seq_len
        return global_start, n, global_start, n, global_start, n

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache = self._ensure_reader()
        _, global_start = self.valid[index]
        obs_start, obs_len, act_start, act_len, rew_start, rew_len = self._window_bounds(global_start)

        obs_data = _slice_global(cache, obs_start, obs_len, self.chunk_size)
        if self.window_mode == "transition":
            act_data = _slice_global(cache, act_start, act_len, self.chunk_size)
            rew_data = _slice_global(cache, rew_start, rew_len, self.chunk_size)
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
