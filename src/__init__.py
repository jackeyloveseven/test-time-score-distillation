"""Source package for style residual distillation skeleton."""

from .attention_hooks import Klein9BCache, Klein9BController, register_klein9b_attn_hooks
from .adapter_runtime import ResidualAdapterRuntime
from .latent_ops import (
    adain_1d,
    feature_adain,
    sinkhorn_attn,
    sinkhorn_attn_freq,
    sinkhorn_latent_transport,
    sinkhorn_style_pattern,
)
from .residuals import (
    apply_velocity_guidance,
    compute_normal_velocity,
    compute_style_velocity_residual,
    compute_styled_velocity,
    compute_velocity_tuple,
)
from .score_residual_student import ScoreResidualStudent, ScoreResidualStudentConfig
from .score_student_runtime import ScoreResidualStudentRuntime
from .student_adapter import (
    LowRankResidualAdapter,
    ResidualAdapter,
    ResidualAdapterConfig,
    SinusoidalTimestepEmbedding,
)

__all__ = [
    "Klein9BCache",
    "Klein9BController",
    "register_klein9b_attn_hooks",
    "ResidualAdapterRuntime",
    "adain_1d",
    "feature_adain",
    "sinkhorn_attn",
    "sinkhorn_attn_freq",
    "sinkhorn_latent_transport",
    "sinkhorn_style_pattern",
    "apply_velocity_guidance",
    "compute_normal_velocity",
    "compute_style_velocity_residual",
    "compute_styled_velocity",
    "compute_velocity_tuple",
    "ScoreResidualStudent",
    "ScoreResidualStudentConfig",
    "ScoreResidualStudentRuntime",
    "ResidualAdapter",
    "ResidualAdapterConfig",
    "LowRankResidualAdapter",
    "SinusoidalTimestepEmbedding",
]
