from __future__ import annotations

import torch
from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


class DynamicsModule(BaseModule):
    stage = "dynamics"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import DynamicsModel
        from dreamer4.models import build_tokenizer

        tokenizer = build_tokenizer(cfg.model.tokenizer)
        if cfg.get("tokenizer_ckpt"):
            ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            tokenizer_state = {
                k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")
            }
            tokenizer.load_state_dict(tokenizer_state, strict=True)
        for p in tokenizer.parameters():
            p.requires_grad = False
        self.tokenizer = tokenizer
        self.model = DynamicsModel(cfg.model, tokenizer)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("Dynamics training not yet implemented")
