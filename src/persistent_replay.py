"""Persistent clean-latent replay helpers for the full fake critic."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from .ca_dmd_backend import FakeConditioning, FakeCriticTrainEntry
from .native_edit_teacher import EditCondition


Tensor = torch.Tensor


def _clone_cpu_detached(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu").clone()


def _randperm_indices(length: int, *, generator: torch.Generator) -> list[int]:
    if length <= 1:
        return list(range(length))
    return torch.randperm(length, generator=generator, dtype=torch.int64).tolist()


@dataclass(frozen=True)
class ReplayConditioning:
    prompt_embeds_cpu: Tensor
    text_ids_cpu: Tensor
    image_ids_cpu: Tensor
    edit_latents_cpu: Tensor
    edit_image_ids_cpu: Tensor
    source: str

    def __post_init__(self) -> None:
        for name in (
            "prompt_embeds_cpu",
            "text_ids_cpu",
            "image_ids_cpu",
            "edit_latents_cpu",
            "edit_image_ids_cpu",
        ):
            tensor = getattr(self, name)
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must stay on CPU inside replay storage")
            if tensor.requires_grad or tensor.grad_fn is not None:
                raise ValueError(f"{name} must be detached inside replay storage")

    @classmethod
    def from_fake_conditioning(cls, conditioning: FakeConditioning) -> "ReplayConditioning":
        return cls(
            prompt_embeds_cpu=_clone_cpu_detached(conditioning.prompt_embeds),
            text_ids_cpu=_clone_cpu_detached(conditioning.text_ids),
            image_ids_cpu=_clone_cpu_detached(conditioning.image_ids),
            edit_latents_cpu=_clone_cpu_detached(conditioning.edit_condition.edit_latents),
            edit_image_ids_cpu=_clone_cpu_detached(conditioning.edit_condition.edit_image_ids),
            source=str(conditioning.edit_condition.source),
        )

    def to_fake_conditioning(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> FakeConditioning:
        prompt_embeds = self.prompt_embeds_cpu.to(device=device, dtype=dtype or self.prompt_embeds_cpu.dtype)
        text_ids = self.text_ids_cpu.to(device=device)
        image_ids = self.image_ids_cpu.to(device=device)
        edit_latents = self.edit_latents_cpu.to(device=device, dtype=dtype or self.edit_latents_cpu.dtype)
        edit_image_ids = self.edit_image_ids_cpu.to(device=device)
        return FakeConditioning(
            prompt_embeds=prompt_embeds.detach(),
            text_ids=text_ids.detach(),
            image_ids=image_ids.detach(),
            edit_condition=EditCondition(
                edit_latents=edit_latents.detach(),
                edit_image_ids=edit_image_ids.detach(),
                source=self.source,
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "prompt_embeds_cpu": _clone_cpu_detached(self.prompt_embeds_cpu),
            "text_ids_cpu": _clone_cpu_detached(self.text_ids_cpu),
            "image_ids_cpu": _clone_cpu_detached(self.image_ids_cpu),
            "edit_latents_cpu": _clone_cpu_detached(self.edit_latents_cpu),
            "edit_image_ids_cpu": _clone_cpu_detached(self.edit_image_ids_cpu),
            "source": self.source,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ReplayConditioning":
        return cls(
            prompt_embeds_cpu=_clone_cpu_detached(state["prompt_embeds_cpu"]),
            text_ids_cpu=_clone_cpu_detached(state["text_ids_cpu"]),
            image_ids_cpu=_clone_cpu_detached(state["image_ids_cpu"]),
            edit_latents_cpu=_clone_cpu_detached(state["edit_latents_cpu"]),
            edit_image_ids_cpu=_clone_cpu_detached(state["edit_image_ids_cpu"]),
            source=str(state["source"]),
        )


@dataclass(frozen=True)
class PersistentReplayEntry:
    sample_id: str
    split: str
    source_dataset: str
    reference_path: str
    reference_sha256: str
    content_prompt: str
    prompt_id: str
    seed: int
    main_model_manifest_sha256: str
    teacher_model_manifest_sha256: str
    fake_init_model_manifest_sha256: str
    clean_latent_cpu: Tensor
    conditioning: ReplayConditioning
    prompt_text: str | None = None
    style_id: str | None = None
    clean_latent_sha256: str | None = None
    originating_outer_step: int | None = None
    reference_timestep: float | None = None

    def __post_init__(self) -> None:
        if self.clean_latent_cpu.device.type != "cpu":
            raise ValueError("clean_latent_cpu must stay on CPU inside replay storage")
        if self.clean_latent_cpu.requires_grad or self.clean_latent_cpu.grad_fn is not None:
            raise ValueError("clean_latent_cpu must be detached inside replay storage")
        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.split:
            raise ValueError("split must be non-empty")
        if not self.reference_sha256:
            raise ValueError("reference_sha256 must be non-empty")
        if not self.prompt_id:
            raise ValueError("prompt_id must be non-empty")

    @classmethod
    def from_tensors(
        cls,
        *,
        sample_id: str,
        split: str,
        source_dataset: str,
        reference_path: str,
        reference_sha256: str,
        content_prompt: str,
        prompt_id: str,
        seed: int,
        main_model_manifest_sha256: str,
        teacher_model_manifest_sha256: str,
        fake_init_model_manifest_sha256: str,
        clean_latent: Tensor,
        conditioning: FakeConditioning,
        prompt_text: str | None = None,
        style_id: str | None = None,
        clean_latent_sha256: str | None = None,
        originating_outer_step: int | None = None,
        reference_timestep: float | None = None,
    ) -> "PersistentReplayEntry":
        return cls(
            sample_id=str(sample_id),
            split=str(split),
            source_dataset=str(source_dataset),
            reference_path=str(reference_path),
            reference_sha256=str(reference_sha256),
            content_prompt=str(content_prompt),
            prompt_id=str(prompt_id),
            seed=int(seed),
            main_model_manifest_sha256=str(main_model_manifest_sha256),
            teacher_model_manifest_sha256=str(teacher_model_manifest_sha256),
            fake_init_model_manifest_sha256=str(fake_init_model_manifest_sha256),
            clean_latent_cpu=_clone_cpu_detached(clean_latent),
            conditioning=ReplayConditioning.from_fake_conditioning(conditioning),
            prompt_text=prompt_text,
            style_id=style_id,
            clean_latent_sha256=clean_latent_sha256,
            originating_outer_step=originating_outer_step,
            reference_timestep=reference_timestep,
        )

    def stratum_key(self) -> tuple[str, str]:
        return (self.reference_sha256, self.prompt_id)

    def to_train_entry(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        default_timestep: Tensor | float,
    ) -> FakeCriticTrainEntry:
        if self.reference_timestep is None:
            timestep = default_timestep
        else:
            timestep = torch.tensor([float(self.reference_timestep)])
        if torch.is_tensor(timestep):
            timestep_tensor = timestep.to(device=device, dtype=dtype or self.clean_latent_cpu.dtype).detach()
        else:
            timestep_tensor = torch.tensor([float(timestep)], device=device, dtype=dtype or self.clean_latent_cpu.dtype)
        return FakeCriticTrainEntry(
            fake_latents_clean=self.clean_latent_cpu.to(device=device, dtype=dtype or self.clean_latent_cpu.dtype).detach(),
            conditioning=self.conditioning.to_fake_conditioning(device=device, dtype=dtype),
            timestep=timestep_tensor,
            sample_id=self.sample_id,
            reference_sha256=self.reference_sha256,
            prompt_id=self.prompt_id,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "source_dataset": self.source_dataset,
            "reference_path": self.reference_path,
            "reference_sha256": self.reference_sha256,
            "content_prompt": self.content_prompt,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "main_model_manifest_sha256": self.main_model_manifest_sha256,
            "teacher_model_manifest_sha256": self.teacher_model_manifest_sha256,
            "fake_init_model_manifest_sha256": self.fake_init_model_manifest_sha256,
            "clean_latent_cpu": _clone_cpu_detached(self.clean_latent_cpu),
            "conditioning": self.conditioning.state_dict(),
            "prompt_text": self.prompt_text,
            "style_id": self.style_id,
            "clean_latent_sha256": self.clean_latent_sha256,
            "originating_outer_step": self.originating_outer_step,
            "reference_timestep": self.reference_timestep,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PersistentReplayEntry":
        return cls(
            sample_id=str(state["sample_id"]),
            split=str(state["split"]),
            source_dataset=str(state["source_dataset"]),
            reference_path=str(state["reference_path"]),
            reference_sha256=str(state["reference_sha256"]),
            content_prompt=str(state["content_prompt"]),
            prompt_id=str(state["prompt_id"]),
            seed=int(state["seed"]),
            main_model_manifest_sha256=str(state["main_model_manifest_sha256"]),
            teacher_model_manifest_sha256=str(state["teacher_model_manifest_sha256"]),
            fake_init_model_manifest_sha256=str(state["fake_init_model_manifest_sha256"]),
            clean_latent_cpu=_clone_cpu_detached(state["clean_latent_cpu"]),
            conditioning=ReplayConditioning.from_state_dict(state["conditioning"]),
            prompt_text=state.get("prompt_text"),
            style_id=state.get("style_id"),
            clean_latent_sha256=state.get("clean_latent_sha256"),
            originating_outer_step=state.get("originating_outer_step"),
            reference_timestep=state.get("reference_timestep"),
        )


class PersistentReplayBuffer:
    def __init__(self, *, capacity: int = 256, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)
        self._entries: list[PersistentReplayEntry] = []
        self._next_insert_idx = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def next_insert_idx(self) -> int:
        return self._next_insert_idx

    def add(self, entry: PersistentReplayEntry) -> PersistentReplayEntry | None:
        stored_entry = PersistentReplayEntry.from_state_dict(entry.state_dict())
        overwritten: PersistentReplayEntry | None = None
        if len(self._entries) < self.capacity:
            self._entries.append(stored_entry)
        else:
            overwritten = self._entries[self._next_insert_idx]
            self._entries[self._next_insert_idx] = stored_entry
        self._next_insert_idx = (self._next_insert_idx + 1) % self.capacity
        return overwritten

    def entries(self, *, split: str | None = None) -> list[PersistentReplayEntry]:
        items = list(self._entries)
        if split is None:
            return items
        return [entry for entry in items if entry.split == split]

    def composition_summary(self) -> dict[str, Any]:
        split_counts = Counter(entry.split for entry in self._entries)
        strata_counts = Counter(entry.stratum_key() for entry in self._entries)
        return {
            "capacity": self.capacity,
            "size": len(self._entries),
            "next_insert_idx": self._next_insert_idx,
            "split_counts": dict(sorted(split_counts.items())),
            "strata_counts": {
                f"{reference_sha256}:{prompt_id}": count
                for (reference_sha256, prompt_id), count in sorted(strata_counts.items())
            },
        }

    def sample_stratified(
        self,
        count: int,
        *,
        split: str = "train",
        exclude_sample_ids: set[str] | None = None,
    ) -> list[PersistentReplayEntry]:
        if count <= 0:
            return []
        excluded = exclude_sample_ids or set()
        groups: dict[tuple[str, str], list[PersistentReplayEntry]] = defaultdict(list)
        for entry in self._entries:
            if entry.split != split:
                continue
            if entry.sample_id in excluded:
                continue
            groups[entry.stratum_key()].append(entry)
        if not groups:
            return []
        keys = sorted(groups)
        key_order = [keys[idx] for idx in _randperm_indices(len(keys), generator=self._rng)]
        for key in key_order:
            values = groups[key]
            perm = _randperm_indices(len(values), generator=self._rng)
            groups[key] = [values[idx] for idx in perm]
        samples: list[PersistentReplayEntry] = []
        while len(samples) < count:
            made_progress = False
            for key in key_order:
                values = groups[key]
                if not values:
                    continue
                samples.append(values.pop(0))
                made_progress = True
                if len(samples) >= count:
                    break
            if not made_progress:
                break
        return samples

    def build_accumulation_entries(
        self,
        *,
        latest_entry: PersistentReplayEntry,
        grad_accum_steps: int,
        replay_mode: str,
        split: str = "train",
    ) -> list[PersistentReplayEntry]:
        if grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {grad_accum_steps!r}")
        if replay_mode not in {"replay", "no_replay"}:
            raise ValueError(f"Unsupported replay_mode {replay_mode!r}")
        if replay_mode == "no_replay":
            return [latest_entry for _ in range(int(grad_accum_steps))]
        latest_count = int(grad_accum_steps // 2)
        replay_count = int(grad_accum_steps - latest_count)
        replay_entries = self.sample_stratified(
            replay_count,
            split=split,
            exclude_sample_ids={latest_entry.sample_id},
        )
        while len(replay_entries) < replay_count:
            replay_entries.append(latest_entry)
        latest_remaining = latest_count
        replay_remaining = list(replay_entries)
        out: list[PersistentReplayEntry] = []
        while len(out) < grad_accum_steps:
            if latest_remaining > 0:
                out.append(latest_entry)
                latest_remaining -= 1
            if replay_remaining and len(out) < grad_accum_steps:
                out.append(replay_remaining.pop(0))
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "next_insert_idx": self._next_insert_idx,
            "rng_state": self._rng.get_state(),
            "entries": [entry.state_dict() for entry in self._entries],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PersistentReplayBuffer":
        buffer = cls(
            capacity=int(state["capacity"]),
            seed=int(state["seed"]),
        )
        buffer._next_insert_idx = int(state["next_insert_idx"])
        buffer._entries = [PersistentReplayEntry.from_state_dict(entry) for entry in state["entries"]]
        buffer._rng.set_state(state["rng_state"])
        return buffer


def validate_reference_hash_split(entries: list[PersistentReplayEntry]) -> dict[str, Any]:
    train_hashes = sorted({entry.reference_sha256 for entry in entries if entry.split == "train"})
    heldout_hashes = sorted({entry.reference_sha256 for entry in entries if entry.split == "heldout"})
    overlap = sorted(set(train_hashes) & set(heldout_hashes))
    if overlap:
        raise ValueError(
            "Manifest split violates reference-hash isolation; overlapping hashes: "
            + ", ".join(overlap)
        )
    return {
        "train_reference_hashes": train_hashes,
        "heldout_reference_hashes": heldout_hashes,
        "train_reference_count": len(train_hashes),
        "heldout_reference_count": len(heldout_hashes),
        "overlap_count": len(overlap),
    }


__all__ = [
    "PersistentReplayBuffer",
    "PersistentReplayEntry",
    "ReplayConditioning",
    "validate_reference_hash_split",
]
