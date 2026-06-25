from __future__ import annotations

from omegaconf import DictConfig

from dreamer4.modules.base import BaseModule


class BCModule(BaseModule):
    stage = "bc"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        from dreamer4.models import AgentHeads

        self.model = AgentHeads(cfg.model)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        raise NotImplementedError("BC training not yet implemented")
