"""Block-causal transformer building blocks (from nicklashansen/dreamer4 model.py)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out


def temporal_patchify(videos_btchw: torch.Tensor, patch: int) -> torch.Tensor:
    """(B,T,C,H,W) in [0,1] -> (B,T,Np,Dp)."""
    B, T, C, H, W = videos_btchw.shape
    x = videos_btchw.reshape(B * T, C, H, W)
    cols = F.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2).contiguous()
    Np, Dp = cols.shape[1], cols.shape[2]
    return cols.reshape(B, T, Np, Dp)


def temporal_unpatchify(patches_btnd: torch.Tensor, H: int, W: int, C: int, patch: int) -> torch.Tensor:
    """(B,T,Np,Dp) -> (B,T,C,H,W)."""
    B, T, Np, Dp = patches_btnd.shape
    x = patches_btnd.reshape(B * T, Np, Dp).transpose(1, 2).contiguous()
    out = F.fold(x, output_size=(H, W), kernel_size=patch, stride=patch)
    return out.reshape(B, T, C, H, W)


def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)
    k = torch.floor(i / 2.0)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))


def add_sinusoidal_positions(tokens_btSd: torch.Tensor, scale: bool) -> torch.Tensor:
    B, T, S, D = tokens_btSd.shape
    pos_t = sinusoid_table(T, D, device=tokens_btSd.device)
    pos_s = sinusoid_table(S, D, device=tokens_btSd.device)
    pos = pos_t[None, :, None, :] + pos_s[None, None, :, :]
    if scale:
        pos = pos * (1.0 / math.sqrt(D))
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


class MAEReplacer(nn.Module):
    """Random patch masking for MAE training."""

    def __init__(self, d_model: int, p_min: float = 0.0, p_max: float = 0.9):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.mask_token = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, patches_btnd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, Np, D = patches_btnd.shape
        device = patches_btnd.device
        if self.p_min == 0.0 and self.p_max == 0.0:
            keep_prob = torch.ones((B, T, 1), device=device, dtype=patches_btnd.dtype)
            mae_mask = torch.zeros((B, T, Np, 1), device=device, dtype=torch.bool)
            return patches_btnd, mae_mask, keep_prob

        p_bt = torch.empty((B, T), device=device).uniform_(self.p_min, self.p_max)
        keep_prob = (1.0 - p_bt).unsqueeze(-1)
        keep = (torch.rand((B, T, Np), device=device) < keep_prob).unsqueeze(-1)
        mask_tok = self.mask_token.to(dtype=patches_btnd.dtype)
        replaced = torch.where(keep, patches_btnd, mask_tok.view(1, 1, 1, D))
        mae_mask = (~keep).to(torch.bool)
        return replaced, mae_mask, keep_prob


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = self.drop(u * F.silu(v))
        return self.drop(self.fc_out(h))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_nld: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)
        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        modality_ids: torch.Tensor,
        n_latents: int,
        mode: str,
        dropout: float,
    ):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)
        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)
        self.register_buffer("attn_mask", allow.unsqueeze(0).unsqueeze(0), persistent=False)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)
        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents
        same_mod = self.modality_ids[q_idx] == self.modality_ids[k_idx]
        if self.mode == "encoder":
            return torch.where(is_q_lat, torch.ones((S, S), dtype=torch.bool, device=device), same_mod)
        elif self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)
        else:
            raise ValueError(f"Unsupported space mode: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        if self.latents_only:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat_nld, is_causal=True)
            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x_btSd.clone()
            x[:, :, :L, :] = out
            return x
        x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        out = self.attn(x_nld, is_causal=True)
        return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
    ):
        super().__init__()
        self.do_time = (layer_index + 1) % time_every == 0
        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(
            d_model, n_heads, modality_ids, n_latents, space_mode, dropout
        )
        self.drop1 = nn.Dropout(dropout)
        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents)
            self.drop2 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            x = x + self.drop2(self.time(self.norm2(x)))
        return x + self.mlp(self.norm3(x))


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BlockCausalLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_latents=n_latents,
                    modality_ids=modality_ids,
                    space_mode=space_mode,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    layer_index=i,
                    time_every=time_every,
                    latents_only_time=latents_only_time,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
