"""DMC Walker Walk policy evaluation (online env rollout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from dreamer4.config import load_config
from dreamer4.env import make_dmc_env
from dreamer4.policy_agent import (
    RandomPolicy,
    parse_eval_gpu_ids,
    run_bc_env_eval,
    run_bc_policy_video,
    run_episodes,
    summarize_episodes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate policy in DMC Walker Walk")
    parser.add_argument("config", type=Path, help="BC or policy YAML config")
    parser.add_argument("--policy", choices=["random", "bc"], default="random")
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

    eval_cfg = cfg.get("eval", {})
    episodes = int(args.episodes or eval_cfg.get("episodes", 10))
    task = str(eval_cfg.get("task", "walker_walk"))

    metrics: dict = {"policy": args.policy, "task": task}

    if args.video_out is not None:
        if args.policy != "bc":
            raise ValueError("--video-out requires --policy bc")
        if not cfg.get("bc_ckpt"):
            raise ValueError("BC video requires bc_ckpt in config or --bc-ckpt")
        import torch

        gpu_id = 0 if torch.cuda.is_available() else None
        if args.gpus is not None:
            gpu_id = 0
        video_metrics = run_bc_policy_video(
            cfg,
            args.video_out,
            fps=int(args.video_fps),
            gpu_id=gpu_id,
            annotate_steps=args.annotate_video,
        )
        metrics.update(video_metrics)

    if args.policy == "random":
        env = make_dmc_env(
            task,
            repeat=int(eval_cfg.get("action_repeat", 1)),
            image_size=int(eval_cfg.get("image_size", 64)),
            proprio=bool(eval_cfg.get("proprio", False)),
            image=bool(eval_cfg.get("image", True)),
            camera=int(eval_cfg.get("camera_id", -1)),
            max_episode_steps=int(eval_cfg.get("max_episode_steps", 1000)),
        )
        policy = RandomPolicy(action_dim=env.action_dim)
        metrics.update(summarize_episodes(run_episodes(env, policy, episodes)))
    else:
        if not cfg.get("bc_ckpt"):
            raise ValueError("BC eval requires bc_ckpt in config or --bc-ckpt")
        import torch

        eval_cfg = cfg.get("eval", {})
        gpu_ids = parse_eval_gpu_ids(eval_cfg.get("gpu_ids", "all"))
        if args.gpus is not None:
            gpu_ids = list(range(args.gpus))
        metrics.update(run_bc_env_eval(cfg, num_episodes=episodes, gpu_ids=gpu_ids))

    print(json.dumps(metrics, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
