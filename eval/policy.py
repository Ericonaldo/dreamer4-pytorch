"""DMC Walker Walk policy evaluation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import torch
from omegaconf import OmegaConf

from dreamer4.config import load_config
from dreamer4.eval_utils import (
    parse_eval_gpu_ids,
    run_policy_env_eval,
    run_policy_episode,
)
from eval.viz.annotate import annotate_frames_uint8


def run_policy_video(
    cfg,
    out_path: Path,
    *,
    fps: int = 20,
    gpu_id: int | None = 0,
    annotate_steps: bool = False,
) -> dict:
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    frames_arr, ep_return, ep_len = run_policy_episode(cfg, device)
    if annotate_steps:
        labels = [f"step {t}" for t in range(len(frames_arr))]
        labels[-1] = f"step {len(frames_arr) - 1}  return {ep_return:.0f}"
        frames_arr = annotate_frames_uint8(frames_arr, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, frames_arr, fps=fps, codec="h264")
    return {
        "return": ep_return,
        "length": ep_len,
        "frames": len(frames_arr),
        "video": str(out_path),
        "fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy in DMC Walker Walk")
    parser.add_argument("config", type=Path, help="BC or policy YAML config")
    parser.add_argument("--bc-ckpt", type=Path, default=None, help="BC Lightning checkpoint")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs for BC eval")
    parser.add_argument("--out", type=Path, default=None, help="Write metrics JSON here")
    parser.add_argument("--video-out", type=Path, default=None, help="Record one episode mp4 here")
    parser.add_argument("--video-fps", type=int, default=20, help="FPS for --video-out")
    parser.add_argument(
        "--annotate-video",
        action="store_true",
        help="Overlay step labels (and return on last frame) on --video-out mp4",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="Open-loop MTP steps per replan (1=closed-loop from current obs; default eval.action_horizon)",
    )
    parser.add_argument("--task", type=str, default=None, help="DMC task name, e.g. walker_walk")
    parser.add_argument("overrides", nargs="*", help="Config overrides")
    args = parser.parse_args()

    eval_defaults_path = args.config.parent / "policy_eval.yaml"
    if eval_defaults_path.is_file():
        cfg = OmegaConf.merge(OmegaConf.load(eval_defaults_path), load_config(args.config, args.overrides))
    else:
        cfg = load_config(args.config, args.overrides)
    if args.bc_ckpt is not None:
        cfg.bc_ckpt = str(args.bc_ckpt)
    if args.task is not None:
        cfg.eval.task = args.task
    if args.action_horizon is not None:
        cfg.eval.action_horizon = int(args.action_horizon)

    eval_cfg = cfg.get("eval", {})
    episodes = int(args.episodes or eval_cfg.get("episodes", 10))
    task = str(eval_cfg.get("task", "walker_walk"))

    metrics: dict = {"policy": "bc", "task": task}

    if args.video_out is not None:
        if not cfg.get("bc_ckpt"):
            raise ValueError("BC video requires bc_ckpt in config or --bc-ckpt")
        gpu_id = 0 if torch.cuda.is_available() else None
        if args.gpus is not None:
            gpu_id = 0
        metrics.update(
            run_policy_video(
                cfg,
                args.video_out,
                fps=int(args.video_fps),
                gpu_id=gpu_id,
                annotate_steps=args.annotate_video,
            )
        )

    if not cfg.get("bc_ckpt"):
        raise ValueError("BC eval requires bc_ckpt in config or --bc-ckpt")
    gpu_ids = parse_eval_gpu_ids(eval_cfg.get("gpu_ids", "all"))
    if args.gpus is not None:
        gpu_ids = list(range(args.gpus))
    metrics.update(run_policy_env_eval(cfg, num_episodes=episodes, gpu_ids=gpu_ids))

    print(json.dumps(metrics, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
