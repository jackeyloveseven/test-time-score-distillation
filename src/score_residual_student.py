"""Direct score/velocity residual student models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .student_adapter import SinusoidalTimestepEmbedding


Tensor = torch.Tensor


@dataclass(frozen=True)
class ScoreResidualStudentConfig:
    latent_dim: int
    max_tokens: int
    num_styles: int
    num_prompts: int
    hidden_dim: int = 512
    cond_dim: int = 128
    time_embed_dim: int = 128
    dropout: float = 0.0
    init_output_scale: float = 0.01
    model_type: str = "token_mlp"
    mixer_layers: int = 4
    mixer_kernel: int = 3
    grid_height: int = 0
    grid_width: int = 0


class ScoreResidualStudent(nn.Module):
    """Token-wise student that predicts style residual directly from z_t.

    Intended use:

    ``pred_residual = f_theta(z_t, timestep, style, prompt)``

    At inference this residual is added to the normal velocity:

    ``v = v_normal + alpha * pred_residual``
    """

    def __init__(self, config: ScoreResidualStudentConfig) -> None:
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
        self.position_embedding = nn.Embedding(config.max_tokens, config.cond_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(config.cond_dim * 3, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.latent_norm = nn.LayerNorm(config.latent_dim)
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
        z_t: Tensor,
        timestep: Tensor,
        style_idx: Tensor,
        prompt_idx: Tensor,
    ) -> Tensor:
        if z_t.ndim != 3:
            raise ValueError("z_t must be [B, L, D]")
        if z_t.shape[1] > self.config.max_tokens:
            raise ValueError(f"z_t has {z_t.shape[1]} tokens, max_tokens={self.config.max_tokens}")
        cond = self.condition(timestep, style_idx, prompt_idx)
        cond = cond[:, None, :].expand(-1, z_t.shape[1], -1)
        positions = torch.arange(z_t.shape[1], device=z_t.device)
        pos_cond = self.position_embedding(positions)[None, :, :]
        cond = cond + pos_cond
        x = torch.cat([self.latent_norm(z_t.float()), cond], dim=-1)
        return self.mlp(x).to(dtype=z_t.dtype)


class ScoreConditioner(nn.Module):
    """Shared timestep/style/prompt conditioning."""

    def __init__(self, config: ScoreResidualStudentConfig) -> None:
        super().__init__()
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

    def forward(self, timestep: Tensor, style_idx: Tensor, prompt_idx: Tensor) -> Tensor:
        time_cond = self.time_mlp(self.time_embedding(timestep / 1000.0))
        style_cond = self.style_embedding(style_idx.long().reshape(-1))
        prompt_cond = self.prompt_embedding(prompt_idx.long().reshape(-1))
        return self.cond_proj(torch.cat([time_cond, style_cond, prompt_cond], dim=-1))


class TokenMixerBlock(nn.Module):
    """Depthwise spatial mixing plus pointwise channel mixing."""

    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("mixer_kernel must be odd.")
        self.block = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
            ),
            nn.GroupNorm(num_groups=32 if hidden_dim % 32 == 0 else 1, num_channels=hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ScoreResidualTokenMixer(nn.Module):
    """Spatial token mixer for direct CFG correction.

    The token-wise MLP only sees each latent token independently. This model
    reshapes the latent sequence to its image grid and lets neighboring tokens
    interact through depthwise convolutions before predicting the correction.
    """

    def __init__(self, config: ScoreResidualStudentConfig) -> None:
        super().__init__()
        self.config = config
        self.conditioner = ScoreConditioner(config)
        self.position_embedding = nn.Embedding(config.max_tokens, config.cond_dim)
        self.latent_norm = nn.LayerNorm(config.latent_dim)
        self.input_proj = nn.Linear(config.latent_dim + config.cond_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                TokenMixerBlock(
                    hidden_dim=config.hidden_dim,
                    kernel_size=config.mixer_kernel,
                    dropout=config.dropout,
                )
                for _ in range(config.mixer_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.latent_dim)
        self._init_output(config.init_output_scale)

    def _init_output(self, scale: float) -> None:
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=float(scale))
        nn.init.zeros_(self.output_proj.bias)

    def _grid_shape(self, token_count: int) -> tuple[int, int]:
        height = int(self.config.grid_height)
        width = int(self.config.grid_width)
        if height > 0 and width > 0:
            if height * width != token_count:
                raise ValueError(f"Configured grid {height}x{width} does not match token count {token_count}")
            return height, width
        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(
                "Token mixer needs a rectangular grid. Set grid_height/grid_width "
                f"for token_count={token_count}."
            )
        return side, side

    def forward(
        self,
        z_t: Tensor,
        timestep: Tensor,
        style_idx: Tensor,
        prompt_idx: Tensor,
    ) -> Tensor:
        if z_t.ndim != 3:
            raise ValueError("z_t must be [B, L, D]")
        batch_size, token_count, _ = z_t.shape
        if token_count > self.config.max_tokens:
            raise ValueError(f"z_t has {token_count} tokens, max_tokens={self.config.max_tokens}")
        height, width = self._grid_shape(token_count)

        cond = self.conditioner(timestep, style_idx, prompt_idx)
        cond = cond[:, None, :].expand(-1, token_count, -1)
        positions = torch.arange(token_count, device=z_t.device)
        cond = cond + self.position_embedding(positions)[None, :, :]
        x = torch.cat([self.latent_norm(z_t.float()), cond], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2).reshape(batch_size, self.config.hidden_dim, height, width)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(batch_size, self.config.hidden_dim, token_count).transpose(1, 2)
        x = self.out_norm(x)
        return self.output_proj(x).to(dtype=z_t.dtype)


def build_score_residual_student(config: ScoreResidualStudentConfig) -> nn.Module:
    if config.model_type == "token_mlp":
        return ScoreResidualStudent(config)
    if config.model_type == "token_mixer":
        return ScoreResidualTokenMixer(config)
    raise ValueError(f"Unsupported score student model_type: {config.model_type}")


__all__ = [
    "ScoreResidualTokenMixer",
    "ScoreResidualStudent",
    "ScoreResidualStudentConfig",
    "build_score_residual_student",
]
