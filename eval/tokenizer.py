"""Standalone tokenizer reconstruction eval with return-stratified episode sampling.

Writes per-split panels (target | masked | recon_masked | recon_full) and metrics.json.

Example::

    python -m eval.tokenizer configs/walker_walk/tokenizer_5m.yaml \\
      --tokenizer-ckpt logs/walker_walk/tokenizer_5m/checkpoints/step-step=20000.ckpt \\
      --out-dir analysis/tokenizer_5m_eval \\
      --splits train val \\
      --by-reward-bands
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from omegaconf import DictConfig

from dreamer4.checkpoint import load_state
from dreamer4.config import load_config
from dreamer4.data import (
    GranularEpisodeDataset,
    collate_episodes,
    episode_cumulative_returns,
    reward_band_counts,
    select_episodes_by_return,
    split_episode_indices,
)
from dreamer4.models import build_tokenizer
from dreamer4.models.tokenizer import tokenizer_forward_with_aux
from eval.common import DEFAULT_REWARD_BANDS
from eval.viz.panels import recon_panel_uint8


def _episode_filter_for_split(cfg: DictConfig, split: str) -> set[int] | None:
    if split == "all":
        return None
    probe = GranularEpisodeDataset(
        cfg.data.path,
        cfg.data.seq_len,
        cfg.data.obs_mode,
        window_mode="frame",
        verbose=False,
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


PRIORITY_BANDS: list[tuple[str, float, float]] = [
    ("low_under_200", 0.0, 200.0),
    ("high_over_900", 900.0, 1001.0),
]


def _frame_window_offset(ep_len: int, seq_len: int, *, position: str = "middle") -> int:
    if ep_len < seq_len:
        raise ValueError(f"episode length {ep_len} must be >= seq_len {seq_len}")
    n_starts = ep_len - seq_len + 1
    if position == "start":
        return 0
    if position == "middle":
        return n_starts // 2
    raise ValueError(f"unknown position {position!r}")


def _load_frame_batch_from_picked(
    cfg: DictConfig,
    picked: list[tuple[int, float]],
    *,
    seq_len: int | None = None,
) -> tuple[Any, list[dict]]:
    if not picked:
        raise ValueError("no episodes selected")

    effective_seq_len = int(seq_len if seq_len is not None else cfg.data.seq_len)
    ds = GranularEpisodeDataset(
        cfg.data.path,
        effective_seq_len,
        cfg.data.obs_mode,
        episode_indices=[ep_idx for ep_idx, _ in picked],
        window_mode="frame",
        verbose=False,
    )
    items: list[dict] = []
    picked_meta: list[dict] = []
    for ep_idx, ret in picked:
        offset = _frame_window_offset(ds.episode_length(ep_idx), effective_seq_len, position="middle")
        gs, _ = ds.episodes[ep_idx]
        key = (ep_idx, gs + offset)
        idx = next(i for i, valid_key in enumerate(ds.valid) if valid_key == key)
        items.append(ds[idx])
        picked_meta.append({"episode_idx": ep_idx, "return": ret, "offset": offset})

    batch = collate_episodes(items)
    if batch.image is None:
        raise ValueError("Tokenizer eval requires images")
    return batch, picked_meta


def _eval_batch(
    model: torch.nn.Module,
    batch,
    *,
    patch_size: int,
    max_items: int,
    max_T: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    image = batch.image.to(device)
    with torch.no_grad():
        _, metrics, pred, _, mae_mask = tokenizer_forward_with_aux(model, image, patch_size)
    panel = recon_panel_uint8(
        image.cpu(),
        pred.cpu(),
        mae_mask.cpu(),
        patch_size,
        max_items=max_items,
        max_T=max_T,
    )
    return metrics, panel


def _run_band(
    cfg: DictConfig,
    model: torch.nn.Module,
    *,
    split: str,
    band_name: str,
    low: float,
    high: float,
    returns: np.ndarray,
    episode_filter: set[int] | None,
    out_dir: Path,
    max_items: int,
    max_T: int,
    patch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": band_name, "low": low, "high": high}
    picked = select_episodes_by_return(
        returns,
        low,
        episode_indices=episode_filter,
        top_k=max_items,
        max_return=high,
    )
    if not picked:
        entry["status"] = "skipped_no_pick"
        return entry

    batch, episode_meta = _load_frame_batch_from_picked(cfg, picked)
    metrics, panel = _eval_batch(
        model,
        batch,
        patch_size=patch_size,
        max_items=max_items,
        max_T=max_T,
        device=device,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "recon_panel.png", panel)
    meta = {
        "split": split,
        "return_band": band_name,
        "return_low": low,
        "return_high": high,
        "episodes": episode_meta,
        "picked_return_mean": float(np.mean([r for _, r in picked])),
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    entry["status"] = "ok"
    entry["metrics"] = metrics
    entry["episodes"] = episode_meta
    entry["picked_return_mean"] = meta["picked_return_mean"]
    return entry


def _find_latest_ckpt(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    steps = sorted(ckpt_dir.glob("step-step=*.ckpt"), key=lambda p: p.stat().st_mtime)
    if steps:
        return steps[-1]
    last = ckpt_dir / "last.ckpt"
    return last if last.is_file() else None


def run_eval(
    cfg: DictConfig,
    *,
    tokenizer_ckpt: Path,
    out_dir: Path,
    splits: list[str],
    max_items: int,
    max_T: int,
    bands: list[tuple[str, float, float]],
    device: torch.device,
) -> dict[str, Any]:
    model = build_tokenizer(cfg.model)
    load_state(model, str(tokenizer_ckpt))
    model.to(device)
    model.eval()

    patch_size = int(cfg.model.patch_size)
    _, returns = episode_cumulative_returns(cfg.data.path)
    population = reward_band_counts(returns, bands)

    summary: dict[str, Any] = {
        "tokenizer_ckpt": str(tokenizer_ckpt),
        "max_items_per_band": max_items,
        "max_T": max_T,
        "dataset_return_mean": float(returns.mean()),
        "dataset_return_median": float(np.median(returns)),
        "dataset_return_max": float(returns.max()),
        "population": population,
        "splits": {},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        episode_filter = _episode_filter_for_split(cfg, split)
        split_summary: dict[str, Any] = {"bands": []}
        split_dir = out_dir / split
        for band_info in population:
            name = band_info["name"]
            low, high = band_info["low"], band_info["high"]
            if band_info["count"] == 0:
                split_summary["bands"].append({**band_info, "status": "skipped_empty"})
                print(f"[{split}/{name}] skip: 0 episodes in [{low}, {high})")
                continue
            print(f"[{split}/{name}] eval {max_items} episodes in [{low}, {high})")
            entry = _run_band(
                cfg,
                model,
                split=split,
                band_name=name,
                low=low,
                high=high,
                returns=returns,
                episode_filter=episode_filter,
                out_dir=split_dir / name,
                max_items=max_items,
                max_T=max_T,
                patch_size=patch_size,
                device=device,
            )
            split_summary["bands"].append({**band_info, **entry})
        summary["splits"][split] = split_summary

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenizer reconstruction eval with return bands")
    p.add_argument("config", type=Path)
    p.add_argument("--tokenizer-ckpt", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "all"])
    p.add_argument("--max-items", type=int, default=4)
    p.add_argument("--max-T", type=int, default=None)
    p.add_argument("--by-reward-bands", action="store_true", help="Use full default Walker Walk bands")
    p.add_argument("--priority-bands-only", action="store_true", help="Only <200 and >900 bands")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    cfg = load_config(args.config)
    run_name = cfg.log.run_name
    run_dir = Path(cfg.log.dir) / run_name
    ckpt = args.tokenizer_ckpt or _find_latest_ckpt(run_dir)
    if ckpt is None or not ckpt.is_file():
        raise SystemExit(f"No checkpoint found under {run_dir / 'checkpoints'}")

    max_T = int(args.max_T if args.max_T is not None else cfg.log.get("viz_max_T", 6))
    if args.priority_bands_only:
        bands = PRIORITY_BANDS
    elif args.by_reward_bands:
        bands = DEFAULT_REWARD_BANDS
    else:
        bands = PRIORITY_BANDS

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {ckpt}")
    print(f"out_dir: {args.out_dir}")
    print(f"splits: {args.splits}, bands: {[b[0] for b in bands]}")

    summary = run_eval(
        cfg,
        tokenizer_ckpt=ckpt,
        out_dir=args.out_dir,
        splits=args.splits,
        max_items=args.max_items,
        max_T=max_T,
        bands=bands,
        device=device,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "splits"}, indent=2))


if __name__ == "__main__":
    main()
