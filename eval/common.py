"""Shared helpers for offline eval scripts."""

from __future__ import annotations

from omegaconf import DictConfig

from dreamer4.data import GranularEpisodeDataset, split_episode_indices

DEFAULT_REWARD_BANDS: list[tuple[str, float, float]] = [
    ("fallen_0_50", 0.0, 50.0),
    ("fallen_50_200", 50.0, 200.0),
    ("weak_200_500", 200.0, 500.0),
    ("partial_500_900", 500.0, 900.0),
    ("standing_900_970", 900.0, 970.0),
    ("expert_970_plus", 970.0, 1001.0),
]


def episode_filter_for_split(cfg: DictConfig, split: str) -> set[int] | None:
    if split == "all":
        return None
    window_mode = str(cfg.data.get("window_mode", "transition"))
    probe = GranularEpisodeDataset(
        cfg.data.path, cfg.data.seq_len, cfg.data.obs_mode, window_mode=window_mode
    )
    val_fraction = float(cfg.data.get("val_fraction", 0.05))
    train_episodes, val_episodes = split_episode_indices(
        probe.num_episodes, val_fraction, int(cfg.data.get("val_seed", 0))
    )
    if split == "train":
        return set(train_episodes)
    if split == "val":
        return set(val_episodes)
    raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")
