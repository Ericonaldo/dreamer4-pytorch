"""Standalone dynamics rollout eval: metrics + combined / per-trajectory viz panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import torch
from torch.utils.data import DataLoader

from dreamer4.config import load_config
from dreamer4.data import GranularEpisodeDataset, align_wm_obs_action, collate_episodes, split_episode_indices
from dreamer4.models import DynamicsModel, build_tokenizer
from dreamer4.models.dynamics import run_dynamics_rollout_eval


def _load_state(module: torch.nn.Module, ckpt_path: str, *, prefix: str = "model.") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    module.load_state_dict(filtered, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval dynamics rollout and export viz panels")
    parser.add_argument("config", type=Path, help="Dynamics YAML config")
    parser.add_argument("--dynamics-ckpt", type=Path, required=True, help="Lightning dynamics checkpoint")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for images + metrics.json")
    parser.add_argument("--max-items", type=int, default=4, help="Number of trajectories to visualize")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(cfg.model.tokenizer)
    if cfg.get("tokenizer_ckpt"):
        _load_state(tokenizer, cfg.tokenizer_ckpt, prefix="model.")
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    n_latents = tokenizer.encoder.n_latents
    latent_dim = tokenizer.encoder.bottleneck_proj.out_features
    packing_factor = int(cfg.model.get("packing_factor", 1))
    n_spatial = n_latents // packing_factor
    patch_size = int(cfg.model.tokenizer.patch_size)
    image_size = int(cfg.model.tokenizer.image_size)
    channels = int(cfg.model.tokenizer.channels)

    dynamics = DynamicsModel(cfg.model, n_latents=n_latents, latent_dim=latent_dim)
    _load_state(dynamics, str(args.dynamics_ckpt), prefix="model.")
    dynamics.eval()
    dynamics.to(device)
    tokenizer.to(device)

    val_fraction = float(cfg.data.get("val_fraction", 0.05))
    window_mode = str(cfg.data.get("window_mode", "transition"))
    probe = GranularEpisodeDataset(
        cfg.data.path, cfg.data.seq_len, cfg.data.obs_mode, window_mode=window_mode
    )
    _, val_episodes = split_episode_indices(
        probe.num_episodes, val_fraction, int(cfg.data.get("val_seed", 0))
    )
    val_ds = GranularEpisodeDataset(
        cfg.data.path,
        cfg.data.seq_len,
        cfg.data.obs_mode,
        episode_indices=val_episodes,
        window_mode=window_mode,
    )
    batch = next(
        iter(
            DataLoader(
                val_ds,
                batch_size=args.max_items,
                shuffle=False,
                collate_fn=collate_episodes,
            )
        )
    )
    if batch.image is None:
        raise ValueError("Rollout eval requires images")

    rollout_ctx = int(cfg.train.get("rollout_ctx", 8))
    rollout_horizon = int(cfg.train.get("rollout_horizon", 8))
    rollout_flow_steps = int(cfg.train.get("rollout_flow_steps", 8))

    image, action = align_wm_obs_action(batch.image.to(device), batch.action.to(device))
    metrics, panel, _, _, per_traj = run_dynamics_rollout_eval(
        dynamics,
        tokenizer,
        image,
        action,
        patch_size=patch_size,
        packing_factor=packing_factor,
        n_spatial=n_spatial,
        image_size=image_size,
        channels=channels,
        ctx_length=rollout_ctx,
        horizon=rollout_horizon,
        flow_steps=rollout_flow_steps,
        max_items=args.max_items,
        return_per_traj=True,
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_dir / "rollout_panel_all.png", panel)
    for i, traj_panel in enumerate(per_traj):
        iio.imwrite(out_dir / f"rollout_traj_{i:02d}.png", traj_panel)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out_dir / 'rollout_panel_all.png'} and {len(per_traj)} trajectory panels")


if __name__ == "__main__":
    main()
