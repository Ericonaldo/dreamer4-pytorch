"""Checkpoint loading utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


def load_state(
    module: nn.Module,
    ckpt_path: str,
    *,
    prefix: str = "model.",
) -> tuple[list[str], list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {
        k.removeprefix(prefix): v
        for k, v in state.items()
        if k.startswith(prefix) and "attn_mask" not in k
    }
    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return sorted(missing), sorted(unexpected)
