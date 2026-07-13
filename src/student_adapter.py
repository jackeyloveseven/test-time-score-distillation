"""Small residual adapters for CFG/style residual distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


Tensor = torch.Tensor


@dataclass(frozen=True)
class ResidualAdapterConfig:
    latent_dim: int
    num_styles: int
    num_prompts: int
    hidden_dim: int = 256
    cond_dim: int = 64
    time_embed_dim: int = 64
    dropout: float = 0.0
    init_output_scale: float = 0.01
    low_rank_dim: int = 32
    use_gate: bool = True


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for scalar diffusion timesteps."""

    def __init__(self, embed_dim: int = 64, max_period: float = 10000.0) -> None:
        super().__init__()
        if embed_dim < 2:
            raise ValueError("embed_dim must be >= 2")
        self.embed_dim = int(embed_dim)
        self.max_period = float(max_period)

    def forward(self, timestep: Tensor) -> Tensor:
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.float().reshape(-1)
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timestep[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb


class ResidualAdapter(nn.Module):
    """Token-wise MLP that predicts a residual correction delta.

    The intended use is:

    ``pred_teacher = hard_residual + adapter(hard_residual, timestep, style, prompt)``.
    """

    def __init__(self, config: ResidualAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.time_embedding = SinusoidalTimestepEmbedding(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.style_embedding = nn.Embedding(config.num_styles, config.cond_dim)
        self.prompt_embedding = nn.Embedding(config.num_prompts, config.cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.cond_dim * 3, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.token_norm = nn.LayerNorm(config.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self._init_output(config.init_output_scale)

    def _init_output(self, scale: float) -> None:
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=float(scale))
            nn.init.zeros_(last.bias)

    def condition(self, timestep: Tensor, style_idx: Tensor, prompt_idx: Tensor) -> Tensor:
        time_cond = self.time_mlp(self.time_embedding(timestep / 1000.0))
        style_cond = self.style_embedding(style_idx.long().reshape(-1))
        prompt_cond = self.prompt_embedding(prompt_idx.long().reshape(-1))
        return self.cond_proj(torch.cat([time_cond, style_cond, prompt_cond], dim=-1))

    def forward(
        self,
        hard_residual: Tensor,
        timestep: Tensor,
        style_idx: Tensor,
        prompt_idx: Tensor,
    ) -> Tensor:
        if hard_residual.ndim != 3:
            raise ValueError("hard_residual must be [B, L, D]")
        cond = self.condition(timestep, style_idx, prompt_idx)
        cond = cond[:, None, :].expand(-1, hard_residual.shape[1], -1)
        x = torch.cat([self.token_norm(hard_residual.float()), cond], dim=-1)
        return self.mlp(x).to(dtype=hard_residual.dtype)


class LowRankResidualAdapter(nn.Module):
    """Conditioned low-rank token adapter for residual correction.

    It predicts token-wise low-rank coefficients and projects them through a
    learned basis in latent-channel space. An optional scalar gate keeps early
    outputs conservative.
    """

    def __init__(self, config: ResidualAdapterConfig) -> None:
        super().__init__()
        self.config = config
        rank = int(config.low_rank_dim)
        if rank < 1:
            raise ValueError("low_rank_dim must be >= 1")
        self.time_embedding = SinusoidalTimestepEmbedding(config.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.style_embedding = nn.Embedding(config.num_styles, config.cond_dim)
        self.prompt_embedding = nn.Embedding(config.num_prompts, config.cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.cond_dim * 3, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.token_norm = nn.LayerNorm(config.latent_dim)
        self.coeff_mlp = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, rank),
        )
        self.basis = nn.Parameter(torch.empty(rank, config.latent_dim))
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(config.latent_dim + config.cond_dim, config.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.hidden_dim // 2, 1),
            )
            if config.use_gate
            else None
        )
        self._init_parameters(config.init_output_scale)

    def _init_parameters(self, scale: float) -> None:
        nn.init.normal_(self.basis, mean=0.0, std=float(scale))
        last = self.coeff_mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=float(scale))
            nn.init.zeros_(last.bias)
        if self.gate_mlp is not None:
            last_gate = self.gate_mlp[-1]
            if isinstance(last_gate, nn.Linear):
                nn.init.zeros_(last_gate.weight)
                nn.init.constant_(last_gate.bias, -2.0)

    def condition(self, timestep: Tensor, style_idx: Tensor, prompt_idx: Tensor) -> Tensor:
        time_cond = self.time_mlp(self.time_embedding(timestep / 1000.0))
        style_cond = self.style_embedding(style_idx.long().reshape(-1))
        prompt_cond = self.prompt_embedding(prompt_idx.long().reshape(-1))
        return self.cond_proj(torch.cat([time_cond, style_cond, prompt_cond], dim=-1))

    def forward(
        self,
        hard_residual: Tensor,
        timestep: Tensor,
        style_idx: Tensor,
        prompt_idx: Tensor,
    ) -> Tensor:
        if hard_residual.ndim != 3:
            raise ValueError("hard_residual must be [B, L, D]")
        cond = self.condition(timestep, style_idx, prompt_idx)
        cond = cond[:, None, :].expand(-1, hard_residual.shape[1], -1)
        x = torch.cat([self.token_norm(hard_residual.float()), cond], dim=-1)
        coeff = self.coeff_mlp(x)
        delta = torch.matmul(coeff, self.basis.float())
        if self.gate_mlp is not None:
            gate = torch.sigmoid(self.gate_mlp(x))
            delta = delta * gate
        return delta.to(dtype=hard_residual.dtype)


__all__ = [
    "LowRankResidualAdapter",
    "ResidualAdapter",
    "ResidualAdapterConfig",
    "SinusoidalTimestepEmbedding",
]
