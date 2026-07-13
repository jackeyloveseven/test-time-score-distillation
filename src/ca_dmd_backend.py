"""CA+DMD fake-score backend contracts and full-model critic helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch
import torch.nn.functional as F

from .native_edit_teacher import EditCondition, compute_native_velocity


Tensor = torch.Tensor
VelocityFn = Callable[..., Tensor]


def clean_flow_target_from_noise(fake_latents_clean: Tensor, noise: Tensor) -> Tensor:
    """Correct flow-matching target for clean ``x0`` under FLUX-style flow training."""

    return noise - fake_latents_clean


def flow_match_add_noise(fake_latents_clean: Tensor, noise: Tensor, sigma: Tensor | float) -> Tensor:
    sigma_t = sigma if torch.is_tensor(sigma) else torch.tensor(float(sigma), device=fake_latents_clean.device, dtype=fake_latents_clean.dtype)
    while sigma_t.ndim < fake_latents_clean.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return (1.0 - sigma_t) * fake_latents_clean + sigma_t * noise


@dataclass(frozen=True)
class FakeConditioning:
    prompt_embeds: Tensor
    text_ids: Tensor
    image_ids: Tensor
    edit_condition: EditCondition


@dataclass(frozen=True)
class FakeCriticTrainEntry:
    fake_latents_clean: Tensor
    conditioning: FakeConditioning
    timestep: Tensor
    sample_id: str | None = None
    reference_sha256: str | None = None
    prompt_id: str | None = None


@dataclass(frozen=True)
class FakeSchedulerLookup:
    timesteps: Tensor
    sigmas: Tensor


@dataclass(frozen=True)
class FakeSchedulePoint:
    timestep_index: int
    timestep: Tensor
    sigma: Tensor


def resolve_schedule_point(
    *,
    scheduler_lookup: FakeSchedulerLookup,
    timestep: Tensor | float,
    ref_tensor: Tensor,
) -> FakeSchedulePoint:
    if torch.is_tensor(timestep):
        timestep_tensor = timestep.detach().reshape(-1)[0].to(device="cpu", dtype=scheduler_lookup.timesteps.dtype)
    else:
        timestep_tensor = torch.tensor(float(timestep), device="cpu", dtype=scheduler_lookup.timesteps.dtype)
    timestep_index = int(torch.argmin((scheduler_lookup.timesteps - timestep_tensor).abs()).item())
    resolved_timestep = scheduler_lookup.timesteps[timestep_index].to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(1)
    resolved_sigma = scheduler_lookup.sigmas[timestep_index].to(device=ref_tensor.device, dtype=ref_tensor.dtype).reshape(1)
    return FakeSchedulePoint(
        timestep_index=timestep_index,
        timestep=resolved_timestep,
        sigma=resolved_sigma,
    )


def tensor_rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(grad.square().sum().item())
    return float(total ** 0.5)


def sample_parameter_slices(
    parameters: list[torch.nn.Parameter],
    *,
    per_tensor_values: int = 4096,
    max_total_values: int = 16384,
) -> list[tuple[int, Tensor]]:
    nonempty = [idx for idx, parameter in enumerate(parameters) if parameter.detach().numel() > 0]
    if not nonempty:
        return []
    anchor_indices = [nonempty[0], nonempty[len(nonempty) // 2], nonempty[-1]]
    unique_indices: list[int] = []
    for idx in anchor_indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
    remaining = int(max_total_values)
    samples: list[tuple[int, Tensor]] = []
    for idx in unique_indices:
        if remaining <= 0:
            break
        flat = parameters[idx].detach().float().reshape(-1)
        take = min(int(per_tensor_values), int(flat.numel()), remaining)
        if take <= 0:
            continue
        samples.append((idx, flat[:take].cpu().clone()))
        remaining -= take
    return samples


def sampled_parameter_fingerprint(samples: list[tuple[int, Tensor]]) -> float:
    return float(sum(sample.float().sum().item() for _, sample in samples))


def sampled_parameter_update_stats(
    before_samples: list[tuple[int, Tensor]],
    after_parameters: list[torch.nn.Parameter],
    *,
    update_eps: float = 0.0,
) -> tuple[float, int]:
    total_sq = 0.0
    total_count = 0
    nonzero = 0
    for parameter_index, before_values in before_samples:
        after_flat = after_parameters[parameter_index].detach().float().reshape(-1)[: before_values.numel()].cpu()
        delta = after_flat - before_values
        total_sq += float(delta.square().sum().item())
        total_count += int(delta.numel())
        nonzero += int((delta.abs() > float(update_eps)).sum().item())
    if total_count == 0:
        return 0.0, 0
    return float((total_sq / total_count) ** 0.5), nonzero


class FakeScoreBackend(Protocol):
    """Protocol for a real online fake-score backend used by ``ca_dmd``."""

    backend_name: str

    def train_step(
        self,
        *,
        fake_latents_clean: Tensor,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> dict[str, float]:
        ...

    def predict_velocity(
        self,
        *,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> Tensor:
        ...

    def summary_dict(self) -> dict[str, object]:
        ...

    def save_state(self, path: str | Path) -> None:
        ...


@dataclass(frozen=True)
class KleinFullFakeBackendConfig:
    fake_model_root: str = "/mnt/local/klein4b-base"
    optimizer: str = "sgd"
    lr: float = 1e-4
    grad_clip_norm: float | None = None
    updates_per_inner: int = 1
    train_timestep_mode: str = "uniform"
    train_seed: int = 1064
    fake_architecture: str = "standalone_full_model"
    init_checkpoint_root: str | None = None
    save_state_path: str | None = None
    resume_state_path: str | None = None
    experimental_scope: str = "plumbing_only"

    def __post_init__(self) -> None:
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError(f"Unsupported fake optimizer: {self.optimizer!r}")
        if self.updates_per_inner <= 0:
            raise ValueError(f"updates_per_inner must be >= 1, got {self.updates_per_inner}")
        if self.train_timestep_mode not in {"current", "uniform"}:
            raise ValueError(
                f"train_timestep_mode must be 'current' or 'uniform', got {self.train_timestep_mode!r}"
            )
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr!r}")
        if self.grad_clip_norm is not None:
            if not math.isfinite(float(self.grad_clip_norm)) or float(self.grad_clip_norm) <= 0.0:
                raise ValueError(f"grad_clip_norm must be finite and > 0 when set, got {self.grad_clip_norm!r}")


@dataclass(frozen=True)
class DMD2BackendConfig:
    diffsynth_root: str = "/mnt/local/yzh/prog/DiffSynth-Studio"
    env_name: str = "yzh-dmd"
    guidance_model_type: str = "lora"
    guidance_lora_strength: float = 1.0
    note: str = (
        "Bridge stub only. Alternating update / LoRA guidance integration must be "
        "implemented against DiffSynth DMD2 before style_mode='ca_dmd' is usable."
    )


class KleinFullFakeBackend:
    """Standalone full-model fake critic for bounded CA+DMD plumbing."""

    backend_name = "klein_full_fake"

    def __init__(
        self,
        *,
        fake_dit,
        conditioning: FakeConditioning | None,
        scheduler_lookup: FakeSchedulerLookup,
        config: KleinFullFakeBackendConfig | None = None,
        velocity_fn: VelocityFn | None = None,
    ) -> None:
        self.fake_dit = fake_dit
        self.conditioning = conditioning
        self.scheduler_lookup = FakeSchedulerLookup(
            timesteps=scheduler_lookup.timesteps.detach().to(device="cpu", dtype=torch.float32).clone(),
            sigmas=scheduler_lookup.sigmas.detach().to(device="cpu", dtype=torch.float32).clone(),
        )
        self.config = config or KleinFullFakeBackendConfig()
        self.velocity_fn = velocity_fn or compute_native_velocity
        self.parameters = [parameter for parameter in self.fake_dit.parameters() if parameter.requires_grad]
        if not self.parameters:
            raise ValueError("KleinFullFakeBackend requires at least one trainable fake parameter")
        if self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters, lr=float(self.config.lr), momentum=0.0)
        else:
            self.optimizer = torch.optim.AdamW(self.parameters, lr=float(self.config.lr))
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(int(self.config.train_seed))
        self.update_count = 0
        self.last_train_log: dict[str, float] = {}
        if self.config.resume_state_path:
            self._load_state(Path(self.config.resume_state_path))

    def _sample_noise_like(self, latents: Tensor) -> Tensor:
        noise = torch.randn(latents.shape, generator=self.rng, dtype=torch.float32)
        return noise.to(device=latents.device, dtype=latents.dtype)

    def _sample_train_schedule_point(self, *, fake_latents_clean: Tensor, timestep: Tensor) -> FakeSchedulePoint:
        if self.config.train_timestep_mode == "current":
            return resolve_schedule_point(
                scheduler_lookup=self.scheduler_lookup,
                timestep=timestep,
                ref_tensor=fake_latents_clean,
            )
        sampled_index = int(
            torch.randint(
                low=0,
                high=len(self.scheduler_lookup.timesteps),
                size=(1,),
                generator=self.rng,
                dtype=torch.int64,
            ).item()
        )
        sampled_timestep = self.scheduler_lookup.timesteps[sampled_index].to(
            device=fake_latents_clean.device,
            dtype=fake_latents_clean.dtype,
        ).reshape(1)
        sampled_sigma = self.scheduler_lookup.sigmas[sampled_index].to(
            device=fake_latents_clean.device,
            dtype=fake_latents_clean.dtype,
        ).reshape(1)
        return FakeSchedulePoint(
            timestep_index=sampled_index,
            timestep=sampled_timestep,
            sigma=sampled_sigma,
        )

    def _velocity(
        self,
        *,
        latents: Tensor,
        timestep: Tensor,
        conditioning: FakeConditioning,
        no_grad: bool,
    ) -> Tensor:
        return self.velocity_fn(
            dit=self.fake_dit,
            latents=latents,
            timestep=timestep.to(device=latents.device, dtype=latents.dtype),
            embedded_guidance=0.0,
            prompt_embeds=conditioning.prompt_embeds,
            text_ids=conditioning.text_ids,
            image_ids=conditioning.image_ids,
            edit_condition=conditioning.edit_condition,
            no_grad=no_grad,
        )

    def _resolve_conditioning(self, cond: dict[str, Any] | None) -> FakeConditioning:
        if cond is not None:
            conditioning = cond.get("conditioning")
            if conditioning is not None:
                if not isinstance(conditioning, FakeConditioning):
                    raise TypeError(
                        "cond['conditioning'] must be a FakeConditioning instance, "
                        f"got {type(conditioning)!r}"
                    )
                return conditioning
        if self.conditioning is None:
            raise RuntimeError(
                "No fake conditioning was provided. Pass a constructor default conditioning "
                "or cond={'conditioning': FakeConditioning(...)} / cond={'train_entries': [...]}."
            )
        return self.conditioning

    def _coerce_train_entries(
        self,
        *,
        fake_latents_clean: Tensor,
        timestep: Tensor,
        cond: dict[str, Any] | None,
    ) -> list[FakeCriticTrainEntry]:
        if cond is not None and cond.get("train_entries") is not None:
            raw_entries = cond["train_entries"]
            if not isinstance(raw_entries, list) or not raw_entries:
                raise ValueError("cond['train_entries'] must be a non-empty list")
            entries: list[FakeCriticTrainEntry] = []
            for raw_entry in raw_entries:
                if isinstance(raw_entry, FakeCriticTrainEntry):
                    entry = raw_entry
                elif isinstance(raw_entry, dict):
                    conditioning = raw_entry.get("conditioning")
                    if not isinstance(conditioning, FakeConditioning):
                        raise TypeError(
                            "Each cond['train_entries'][i]['conditioning'] must be FakeConditioning"
                        )
                    entry_timestep = raw_entry.get("timestep", timestep)
                    entry = FakeCriticTrainEntry(
                        fake_latents_clean=raw_entry["fake_latents_clean"].detach(),
                        conditioning=conditioning,
                        timestep=entry_timestep,
                        sample_id=raw_entry.get("sample_id"),
                        reference_sha256=raw_entry.get("reference_sha256"),
                        prompt_id=raw_entry.get("prompt_id"),
                    )
                else:
                    raise TypeError(
                        "Each cond['train_entries'][i] must be FakeCriticTrainEntry or dict, "
                        f"got {type(raw_entry)!r}"
                    )
                entries.append(
                    FakeCriticTrainEntry(
                        fake_latents_clean=entry.fake_latents_clean.detach(),
                        conditioning=entry.conditioning,
                        timestep=entry.timestep.detach(),
                        sample_id=entry.sample_id,
                        reference_sha256=entry.reference_sha256,
                        prompt_id=entry.prompt_id,
                    )
                )
            return entries
        return [
            FakeCriticTrainEntry(
                fake_latents_clean=fake_latents_clean.detach(),
                conditioning=self._resolve_conditioning(cond),
                timestep=timestep.detach(),
            )
        ]

    def _single_train_loss(
        self,
        *,
        entry: FakeCriticTrainEntry,
    ) -> tuple[Tensor, dict[str, Any]]:
        train_point = self._sample_train_schedule_point(
            fake_latents_clean=entry.fake_latents_clean,
            timestep=entry.timestep,
        )
        noise = self._sample_noise_like(entry.fake_latents_clean)
        z_train = flow_match_add_noise(entry.fake_latents_clean, noise, train_point.sigma)
        target_v = clean_flow_target_from_noise(entry.fake_latents_clean, noise)
        pred_v = self._velocity(
            latents=z_train,
            timestep=train_point.timestep,
            conditioning=entry.conditioning,
            no_grad=False,
        )
        loss = F.mse_loss(pred_v.float(), target_v.float(), reduction="mean")
        stats = {
            "fake_train_timestep": float(train_point.timestep.detach().reshape(-1)[0].item()),
            "fake_train_timestep_index": float(train_point.timestep_index),
            "fake_train_sigma": float(train_point.sigma.detach().reshape(-1)[0].item()),
            "fake_train_target_norm": float(target_v.detach().float().norm().item()),
            "fake_train_pred_norm_before": float(pred_v.detach().float().norm().item()),
            "fake_train_entry_loss": float(loss.detach().item()),
            "z_train": z_train.detach(),
            "target_v": target_v.detach(),
            "train_timestep_tensor": train_point.timestep.detach(),
        }
        return loss, stats

    def _accumulated_train_update(
        self,
        *,
        entries: list[FakeCriticTrainEntry],
        scoring_sigma: Tensor | float,
    ) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        parameter_before = sample_parameter_slices(self.parameters)
        fingerprint_before = sampled_parameter_fingerprint(parameter_before)
        entry_stats: list[dict[str, Any]] = []
        accumulated_loss: Tensor | None = None
        for entry in entries:
            entry_loss, entry_log = self._single_train_loss(entry=entry)
            scaled_loss = entry_loss / float(len(entries))
            scaled_loss.backward()
            entry_stats.append(entry_log)
            if accumulated_loss is None:
                accumulated_loss = entry_loss.detach()
            else:
                accumulated_loss = accumulated_loss + entry_loss.detach()
        if accumulated_loss is None:
            raise RuntimeError("accumulated critic update requires at least one train entry")
        loss_before_value = float((accumulated_loss / float(len(entries))).item())
        grad_norm_preclip = parameter_grad_norm(self.parameters)
        grad_norm_postclip = grad_norm_preclip
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=float(self.config.grad_clip_norm))
            grad_norm_postclip = parameter_grad_norm(self.parameters)
        self.optimizer.step()
        self.update_count += 1
        update_rms, update_nonzero = sampled_parameter_update_stats(parameter_before, self.parameters)
        fingerprint_after = sampled_parameter_fingerprint(sample_parameter_slices(self.parameters))
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            loss_after_total = 0.0
            pred_norm_after_total = 0.0
            target_norm_total = 0.0
            timesteps: list[float] = []
            timestep_indices: list[float] = []
            sigmas: list[float] = []
            per_entry_before: list[float] = []
            per_entry_after: list[float] = []
            sample_ids: list[str] = []
            reference_hashes: list[str] = []
            prompt_ids: list[str] = []
            for entry, entry_log in zip(entries, entry_stats):
                pred_v_after = self._velocity(
                    latents=entry_log["z_train"],
                    timestep=entry_log["train_timestep_tensor"],
                    conditioning=entry.conditioning,
                    no_grad=True,
                )
                loss_after = F.mse_loss(pred_v_after.float(), entry_log["target_v"].float(), reduction="mean")
                loss_after_total += float(loss_after.detach().item())
                pred_norm_after_total += float(pred_v_after.detach().float().norm().item())
                target_norm_total += float(entry_log["target_v"].detach().float().norm().item())
                timesteps.append(entry_log["fake_train_timestep"])
                timestep_indices.append(entry_log["fake_train_timestep_index"])
                sigmas.append(entry_log["fake_train_sigma"])
                per_entry_before.append(entry_log["fake_train_entry_loss"])
                per_entry_after.append(float(loss_after.detach().item()))
                if entry.sample_id is not None:
                    sample_ids.append(str(entry.sample_id))
                if entry.reference_sha256 is not None:
                    reference_hashes.append(str(entry.reference_sha256))
                if entry.prompt_id is not None:
                    prompt_ids.append(str(entry.prompt_id))
        scoring_sigma_value = (
            float(scoring_sigma.detach().reshape(-1)[0].item())
            if torch.is_tensor(scoring_sigma)
            else float(scoring_sigma)
        )
        entry_count = float(len(entries))
        log = {
            "fake_train_timestep": float(timesteps[-1]),
            "fake_train_timestep_index": float(timestep_indices[-1]),
            "fake_train_sigma": float(sigmas[-1]),
            "fake_scoring_sigma": scoring_sigma_value,
            "fake_train_loss_before": loss_before_value,
            "fake_train_loss_after": float(loss_after_total / entry_count),
            "fake_train_grad_norm_preclip": grad_norm_preclip,
            "fake_train_grad_norm_postclip": grad_norm_postclip,
            "fake_train_target_norm": float(target_norm_total / entry_count),
            "fake_train_pred_norm_before": float(
                sum(entry_log["fake_train_pred_norm_before"] for entry_log in entry_stats) / entry_count
            ),
            "fake_train_pred_norm_after": float(pred_norm_after_total / entry_count),
            "fake_train_lr": float(self.optimizer.param_groups[0]["lr"]),
            "fake_train_update_count": float(self.update_count),
            "fake_param_sample_update_rms": update_rms,
            "fake_param_sample_update_nonzero": float(update_nonzero),
            "fake_param_sample_fingerprint_before": fingerprint_before,
            "fake_param_sample_fingerprint_after": fingerprint_after,
            "fake_train_entry_count": entry_count,
        }
        if len(entries) > 1:
            log["fake_train_timestep_list"] = [float(value) for value in timesteps]
            log["fake_train_timestep_index_list"] = [float(value) for value in timestep_indices]
            log["fake_train_sigma_list"] = [float(value) for value in sigmas]
            log["fake_train_loss_before_list"] = [float(value) for value in per_entry_before]
            log["fake_train_loss_after_list"] = [float(value) for value in per_entry_after]
            if sample_ids:
                log["fake_train_sample_ids"] = list(sample_ids)
            if reference_hashes:
                log["fake_train_reference_sha256"] = list(reference_hashes)
            if prompt_ids:
                log["fake_train_prompt_ids"] = list(prompt_ids)
        return log

    def train_step(
        self,
        *,
        fake_latents_clean: Tensor,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> dict[str, float]:
        del z_sigma
        train_entries = self._coerce_train_entries(
            fake_latents_clean=fake_latents_clean.detach(),
            timestep=timestep.detach(),
            cond=cond,
        )
        logs = [
            self._accumulated_train_update(
                entries=train_entries,
                scoring_sigma=scoring_sigma,
            )
            for _ in range(int(self.config.updates_per_inner))
        ]
        last_log = dict(logs[-1])
        last_log["fake_train_updates_per_inner"] = float(self.config.updates_per_inner)
        last_log["fake_train_timestep_mode"] = 0.0 if self.config.train_timestep_mode == "current" else 1.0
        self.last_train_log = last_log
        return last_log

    def predict_velocity(
        self,
        *,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> Tensor:
        del scoring_sigma
        conditioning = self._resolve_conditioning(cond)
        with torch.no_grad():
            return self._velocity(
                latents=z_sigma.detach(),
                timestep=timestep.detach(),
                conditioning=conditioning,
                no_grad=True,
            ).detach()

    def summary_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "fake_model_root": self.config.fake_model_root,
            "fake_architecture": self.config.fake_architecture,
            "fake_init_checkpoint_root": self.config.init_checkpoint_root or self.config.fake_model_root,
            "optimizer": self.config.optimizer,
            "lr": float(self.config.lr),
            "grad_clip_norm": self.config.grad_clip_norm,
            "updates_per_inner": int(self.config.updates_per_inner),
            "train_timestep_mode": self.config.train_timestep_mode,
            "train_seed": int(self.config.train_seed),
            "update_count": int(self.update_count),
            "experimental_scope": self.config.experimental_scope,
            "resume_state_path": self.config.resume_state_path,
            "save_state_path": self.config.save_state_path,
            "last_train_log": dict(self.last_train_log),
        }

    def _load_state(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.fake_dit.load_state_dict(payload["fake_dit"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload.get("update_count", 0))
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.set_state(rng_state)

    def save_state(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fake_dit": self.fake_dit.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "rng_state": self.rng.get_state(),
                "config": self.summary_dict(),
            },
            out_path,
        )


class DMD2GuidanceBackendBridge:
    """Explicit placeholder for the real DMD2-compatible backend."""

    backend_name = "dmd2_guidance_bridge"

    def __init__(self, config: DMD2BackendConfig | None = None) -> None:
        self.config = config or DMD2BackendConfig()

    def train_step(
        self,
        *,
        fake_latents_clean: Tensor,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> dict[str, float]:
        del fake_latents_clean, z_sigma, timestep, scoring_sigma, cond
        raise NotImplementedError(self._error_message())

    def predict_velocity(
        self,
        *,
        z_sigma: Tensor,
        timestep: Tensor,
        scoring_sigma: Tensor | float,
        cond: dict[str, Any],
    ) -> Tensor:
        del z_sigma, timestep, scoring_sigma, cond
        raise NotImplementedError(self._error_message())

    def summary_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "config": self.config.__dict__,
        }

    def save_state(self, path: str | Path) -> None:
        raise NotImplementedError(self._error_message())

    def _error_message(self) -> str:
        root = Path(self.config.diffsynth_root)
        return (
            "A credible 'ca_dmd' backend is not wired yet. "
            f"Expected to adapt DiffSynth DMD2 under {root / 'diffsynth/diffusion/dmd2'} "
            "with a Base9B/Klein-compatible online guidance model and alternating updates. "
            f"Configured env={self.config.env_name!r}, guidance_model_type={self.config.guidance_model_type!r}. "
            "Use style_mode='ca_only' until that backend is implemented."
        )


def build_fake_score_backend(kind: str | None, **kwargs: Any) -> FakeScoreBackend | None:
    if kind in {None, "", "none"}:
        return None
    if kind == "dmd2_guidance":
        return DMD2GuidanceBackendBridge()
    if kind == "klein_full_fake":
        return KleinFullFakeBackend(**kwargs)
    raise ValueError(f"Unsupported fake-score backend kind: {kind}")


__all__ = [
    "DMD2BackendConfig",
    "DMD2GuidanceBackendBridge",
    "FakeConditioning",
    "FakeCriticTrainEntry",
    "FakeSchedulePoint",
    "FakeSchedulerLookup",
    "FakeScoreBackend",
    "KleinFullFakeBackend",
    "KleinFullFakeBackendConfig",
    "build_fake_score_backend",
    "clean_flow_target_from_noise",
    "flow_match_add_noise",
    "resolve_schedule_point",
]
