"""Episode return analysis and selection utilities for eval / inspection."""

from __future__ import annotations

import numpy as np

from dreamer4.data import (
    CHUNK_SIZE,
    _ChunkCache,
    _discover_episodes,
    _slice_global,
)


def episode_cumulative_returns(path: str) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Scan a Granular path and return episode ranges plus total reward per episode."""
    import granular

    reader = granular.ShardedDatasetReader(path, granular.decoders)
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
    """Filter episodes by return; sort by score (desc) or band midpoint (when ``max_return`` set)."""
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
    """Summarize how many episodes fall in each return band ``[low, high)``."""
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
