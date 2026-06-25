from __future__ import annotations

from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


class PolicyModule(BaseModule):
    stage = "policy"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.imagination_cfg = cfg.imagination

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("Policy training not yet implemented")
