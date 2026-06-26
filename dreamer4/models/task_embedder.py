from __future__ import annotations

import torch
import torch.nn as nn


class TaskEmbedder(nn.Module):
    """Task-conditioned agent token init (ref TaskEmbedder). Walker uses n_tasks=1."""

    def __init__(
        self,
        d_model: int,
        n_agent: int = 1,
        *,
        use_ids: bool = True,
        n_tasks: int = 1,
        d_task: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_agent = int(n_agent)
        self.use_ids = bool(use_ids)

        if self.use_ids:
            self.task_table = nn.Embedding(int(n_tasks), self.d_model)
        else:
            self.task_proj = nn.Linear(int(d_task), self.d_model)

        self.agent_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.agent_base, std=0.02)

    def forward(self, task: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if self.use_ids:
            emb = self.task_table(task.to(torch.long))
        else:
            emb = self.task_proj(task)
        x = emb + self.agent_base.view(1, -1)
        return x[:, None, None, :].expand(B, T, self.n_agent, self.d_model)
