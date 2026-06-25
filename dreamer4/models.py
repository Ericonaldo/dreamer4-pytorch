from __future__ import annotations

import torch
import torch.nn as nn

from dreamer4.tokenizer import Tokenizer, build_tokenizer


class DynamicsModel(nn.Module):
    """Interactive dynamics model with shortcut forcing. Implementation pending."""

    def __init__(self, cfg, tokenizer: Tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("DynamicsModel not yet implemented")


class AgentHeads(nn.Module):
    """BC policy, reward, and value heads. Implementation pending."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, *args, **kwargs):
        raise NotImplementedError("AgentHeads not yet implemented")
