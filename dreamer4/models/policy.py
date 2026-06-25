from __future__ import annotations

import torch.nn as nn


class AgentHeads(nn.Module):
    """BC policy, reward, and value heads. Implementation pending."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError("AgentHeads not yet implemented")
