"""Regression tests for the TPAMI native-teacher adaptation."""

from __future__ import annotations

import argparse
import ast
import math
import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_tpami_with_style_teacher.py"
PERSISTENT_RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_persistent_fake_critic.py"


def load_runner_function(function_name: str):
    source = RUNNER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            function_source = ast.get_source_segment(source, node)
            if function_source is None:
                raise RuntimeError(f"Unable to extract source for {function_name}")
            namespace = {"Path": Path, "argparse": argparse}
            exec(function_source, namespace)
            return namespace[function_name]
    raise RuntimeError(f"Function {function_name} not found in runner")


resolve_style_path = load_runner_function("resolve_style_path")

from src.attention_hooks import Klein9BCache, Klein9BController
from src.ca_dmd_backend import (
    FakeConditioning,
    FakeCriticTrainEntry,
    FakeSchedulerLookup,
    KleinFullFakeBackendConfig,
    build_fake_score_backend,
    clean_flow_target_from_noise,
    resolve_schedule_point,
)
from src.csv_rows import csv_fieldnames_for_rows
from src.flux2_score import (
    build_pseudo_target_loss,
    resolve_sigma_info,
    rms_normalize_field,
    velocity_delta_to_score_delta,
    velocity_to_score,
)
from src.gradient_audit import StyleGradientCalibrationState, apply_gradient_audit, compute_style_calibration_scale
from src.kv_match import capture_reference_kv
from src.kv_match_objective import apply_reference_kv_grag, kv_match_loss_from_attention_layers
from src.klein9b_loading import resolve_dual_model_vram_contract
from src.loss_isolation import build_loss_isolation_contract, final_adain_if_enabled
from src.native_edit_teacher import (
    EditCondition,
    assert_static_condition_tensor,
    evaluate_native_edit_triplet,
    freeze_condition_tensor,
)
from src.persistent_replay import (
    PersistentReplayBuffer,
    PersistentReplayEntry,
    validate_reference_hash_split,
)
from src.prompt_text import parse_content_prompt_text
from src.style_objectives import pseudo_guidance_from_velocities, score_match_loss_from_velocities, validate_style_mode


class DummyScheduler:
    def __init__(self, timesteps, sigmas):
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)


class TPAMITests(unittest.TestCase):
    def _fake_conditioning(self, prompt_value: float, edit_value: float, *, source: str = "test") -> FakeConditioning:
        return FakeConditioning(
            prompt_embeds=torch.full((1, 2, 3), float(prompt_value)),
            text_ids=torch.zeros(1, 2, 4),
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.full((1, 4, 3), float(edit_value)),
                edit_image_ids=torch.zeros(1, 4, 4),
                source=source,
            ),
        )

    def _replay_entry(
        self,
        *,
        sample_id: str,
        reference_sha256: str,
        prompt_id: str,
        split: str = "train",
        latent_value: float = 1.0,
    ) -> PersistentReplayEntry:
        return PersistentReplayEntry.from_tensors(
            sample_id=sample_id,
            split=split,
            source_dataset="p0-mini",
            reference_path=f"/tmp/{reference_sha256}.png",
            reference_sha256=reference_sha256,
            content_prompt=f"content-{prompt_id}",
            prompt_id=prompt_id,
            seed=64,
            main_model_manifest_sha256="main-hash",
            teacher_model_manifest_sha256="teacher-hash",
            fake_init_model_manifest_sha256="fake-hash",
            clean_latent=torch.full((1, 4, 3), float(latent_value), requires_grad=True),
            conditioning=self._fake_conditioning(latent_value, latent_value + 1.0, source=sample_id),
            prompt_text=f"prompt-{prompt_id}",
            style_id=reference_sha256,
            clean_latent_sha256=f"latent-{sample_id}",
            reference_timestep=500.0,
        )

    def test_sigma_resolution_uses_scheduler_lookup(self) -> None:
        scheduler = DummyScheduler([1000.0, 702.3, 395.47], [1.0, 0.7023, 0.39547])
        ref = torch.zeros(1, 4, 3)
        info = resolve_sigma_info(scheduler, torch.tensor([701.9]), ref_tensor=ref)
        self.assertEqual(info.timestep_index, 1)
        self.assertAlmostEqual(float(info.sigma.item()), 0.7023, places=4)

    def test_resolve_style_path_accepts_unpadded_numeric_style_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            style_root = Path(tmpdir)
            (style_root / "08.png").write_bytes(b"style")
            args = argparse.Namespace(style_image=None, style_root=str(style_root), style_id="8")
            resolved = resolve_style_path(args)
            self.assertEqual(resolved, style_root / "08.png")

    def test_score_delta_is_zero_at_sigma_one(self) -> None:
        delta_v = torch.randn(1, 8, 4)
        delta_s, coeff = velocity_delta_to_score_delta(delta_v, torch.tensor(1.0))
        self.assertAlmostEqual(float(coeff.item()), 0.0, places=6)
        self.assertTrue(torch.allclose(delta_s, torch.zeros_like(delta_s)))

    def test_plan_a_removes_shared_state_before_detach(self) -> None:
        z = torch.randn(1, 6, 4, requires_grad=True)
        pre_scale = torch.nn.Parameter(torch.tensor(0.25))
        sigma = torch.tensor(0.4)
        v_pre = pre_scale * torch.ones_like(z)
        v_style = torch.full_like(z, 0.1)

        wrong_delta = velocity_to_score(z, v_pre, sigma) - velocity_to_score(z.detach(), v_style, sigma)
        wrong_loss = wrong_delta.float().pow(2).mean()
        wrong_loss.backward(retain_graph=True)
        self.assertGreater(float(z.grad.abs().sum().item()), 0.0)
        z.grad.zero_()
        pre_scale.grad = None

        result = score_match_loss_from_velocities(
            v_pre,
            v_style,
            sigma,
            normalization="raw",
            loss_kind="mse",
        )
        result.loss.backward()
        self.assertTrue(z.grad is None or torch.allclose(z.grad, torch.zeros_like(z.grad)))
        self.assertIsNotNone(pre_scale.grad)
        self.assertGreater(float(pre_scale.grad.abs().item()), 0.0)

    def test_native_triplet_same_state_and_reference_only_change(self) -> None:
        calls = []

        def velocity_fn(**kwargs):
            calls.append(
                {
                    "latents_id": id(kwargs["latents"]),
                    "timestep_id": id(kwargs["timestep"]),
                    "prompt_id": id(kwargs["prompt_embeds"]),
                    "text_ids_id": id(kwargs["text_ids"]),
                    "image_ids_id": id(kwargs["image_ids"]),
                    "has_edit": kwargs["edit_condition"] is not None,
                    "no_grad": kwargs["no_grad"],
                }
            )
            latents = kwargs["latents"]
            if kwargs["edit_condition"] is None:
                return latents * 2
            return latents * 3

        latents = torch.randn(1, 4, 3, requires_grad=True)
        timestep = torch.tensor([500.0])
        prompt_embeds = torch.randn(1, 8, 16)
        text_ids = torch.zeros(1, 8, 4)
        image_ids = torch.zeros(1, 4, 4)
        edit_condition = EditCondition(
            edit_latents=torch.randn(1, 4, 3),
            edit_image_ids=torch.zeros(1, 4, 4),
            source="test",
        )
        triplet = evaluate_native_edit_triplet(
            dit=None,
            latents=latents,
            timestep=timestep,
            embedded_guidance=0.0,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
            edit_condition=edit_condition,
            require_pre_grad=True,
            velocity_fn=velocity_fn,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["latents_id"], calls[1]["latents_id"])
        self.assertEqual(calls[0]["timestep_id"], calls[1]["timestep_id"])
        self.assertEqual(calls[0]["prompt_id"], calls[1]["prompt_id"])
        self.assertEqual(calls[0]["text_ids_id"], calls[1]["text_ids_id"])
        self.assertEqual(calls[0]["image_ids_id"], calls[1]["image_ids_id"])
        self.assertFalse(calls[0]["has_edit"])
        self.assertTrue(calls[1]["has_edit"])
        self.assertTrue(triplet.without_shared_with_pre)

    def test_native_triplet_can_use_content_only_negative_prompt(self) -> None:
        calls = []

        def velocity_fn(**kwargs):
            calls.append(
                {
                    "prompt_id": id(kwargs["prompt_embeds"]),
                    "text_ids_id": id(kwargs["text_ids"]),
                    "has_edit": kwargs["edit_condition"] is not None,
                }
            )
            if kwargs["edit_condition"] is None:
                return kwargs["latents"] * 2.0
            return kwargs["latents"] * 3.0

        latents = torch.randn(1, 4, 3, requires_grad=True)
        positive_prompt_embeds = torch.randn(1, 8, 16)
        positive_text_ids = torch.zeros(1, 8, 4)
        negative_prompt_embeds = torch.randn(1, 8, 16)
        negative_text_ids = torch.ones(1, 8, 4)
        triplet = evaluate_native_edit_triplet(
            dit=None,
            latents=latents,
            timestep=torch.tensor([400.0]),
            embedded_guidance=0.0,
            prompt_embeds=positive_prompt_embeds,
            text_ids=positive_text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.randn(1, 4, 3),
                edit_image_ids=torch.zeros(1, 4, 4),
                source="test",
            ),
            require_pre_grad=True,
            share_without_with_pre=False,
            velocity_fn=velocity_fn,
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["prompt_id"], id(negative_prompt_embeds))
        self.assertEqual(calls[0]["text_ids_id"], id(negative_text_ids))
        self.assertEqual(calls[1]["prompt_id"], id(negative_prompt_embeds))
        self.assertEqual(calls[1]["text_ids_id"], id(negative_text_ids))
        self.assertEqual(calls[2]["prompt_id"], id(positive_prompt_embeds))
        self.assertEqual(calls[2]["text_ids_id"], id(positive_text_ids))
        self.assertFalse(calls[0]["has_edit"])
        self.assertFalse(calls[1]["has_edit"])
        self.assertTrue(calls[2]["has_edit"])
        self.assertFalse(triplet.without_shared_with_pre)

    def test_teacher_detach_and_pre_gradient(self) -> None:
        latents = torch.randn(1, 4, 3, requires_grad=True)

        def velocity_fn(**kwargs):
            if kwargs["edit_condition"] is None:
                return kwargs["latents"] * 2.0
            return kwargs["latents"] * 3.0

        triplet = evaluate_native_edit_triplet(
            dit=None,
            latents=latents,
            timestep=torch.tensor([400.0]),
            embedded_guidance=0.0,
            prompt_embeds=torch.randn(1, 8, 16),
            text_ids=torch.zeros(1, 8, 4),
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.randn(1, 4, 3),
                edit_image_ids=torch.zeros(1, 4, 4),
                source="test",
            ),
            require_pre_grad=True,
            velocity_fn=velocity_fn,
        )
        self.assertTrue(triplet.pre_velocity.requires_grad)
        self.assertFalse(triplet.with_velocity.requires_grad)
        self.assertFalse(triplet.without_velocity.requires_grad)

        result = score_match_loss_from_velocities(
            triplet.pre_velocity,
            triplet.with_velocity,
            torch.tensor(0.5),
            normalization="raw",
            loss_kind="mse",
        )
        result.loss.backward()
        self.assertIsNotNone(latents.grad)
        self.assertGreater(float(latents.grad.abs().sum().item()), 0.0)

    def test_condition_freeze_removes_prompt_graph(self) -> None:
        base = torch.randn(1, 8, 4, requires_grad=True)
        prompt_live = base * 2.0
        frozen = freeze_condition_tensor(prompt_live, device=prompt_live.device)
        self.assertFalse(frozen.requires_grad)
        self.assertIsNone(frozen.grad_fn)
        self.assertTrue(torch.allclose(frozen, prompt_live.detach()))

    def test_parse_content_prompt_text_builds_clean_content_prompt(self) -> None:
        args = argparse.Namespace(
            content_prompt=None,
            prompt=None,
            prompt_id="kp0",
            content_prompt_template="Create {content}.",
        )
        self.assertEqual(
            parse_content_prompt_text(
                content_prompt=args.content_prompt,
                prompt=args.prompt,
                prompt_id=args.prompt_id,
                content_prompt_template=args.content_prompt_template,
                prompt_lookup={"kp0": "A panda eating bamboo"},
            ),
            "Create A panda eating bamboo.",
        )

    def test_parse_content_prompt_text_requires_explicit_override_for_custom_prompt(self) -> None:
        args = argparse.Namespace(
            content_prompt=None,
            prompt="Create A panda eating bamboo. Use the visual style of the reference image.",
            prompt_id="kp0",
            content_prompt_template="Create {content}.",
        )
        with self.assertRaises(ValueError):
            parse_content_prompt_text(
                content_prompt=args.content_prompt,
                prompt=args.prompt,
                prompt_id=args.prompt_id,
                content_prompt_template=args.content_prompt_template,
                prompt_lookup={"kp0": "A panda eating bamboo"},
            )

    def test_static_condition_assert_rejects_live_graph_and_accepts_detached(self) -> None:
        live = torch.randn(1, 4, 3, requires_grad=True) * 3.0
        with self.assertRaises(RuntimeError):
            assert_static_condition_tensor("teacher_prompt_embeds", live)
        detached = freeze_condition_tensor(live)
        assert_static_condition_tensor("teacher_prompt_embeds", detached)

    def test_reference_kv_grag_only_compresses_image_tokens(self) -> None:
        ref_k = torch.tensor([[[[10.0], [20.0], [1.0], [5.0]]]])
        ref_v = torch.tensor([[[[30.0], [40.0], [2.0], [6.0]]]])
        out_k, out_v = apply_reference_kv_grag([ref_k], [ref_v], txt_len=2, shallow=0.5, deep=0.5)
        self.assertTrue(torch.allclose(out_k[0][:, :, :2, :], ref_k[:, :, :2, :]))
        self.assertTrue(torch.allclose(out_v[0][:, :, :2, :], ref_v[:, :, :2, :]))
        expected_k_img = torch.tensor([[[[2.0], [4.0]]]])
        expected_v_img = torch.tensor([[[[3.0], [5.0]]]])
        self.assertTrue(torch.allclose(out_k[0][:, :, 2:, :], expected_k_img))
        self.assertTrue(torch.allclose(out_v[0][:, :, 2:, :], expected_v_img))

    def test_kv_match_loss_detaches_q_target_branch_but_keeps_self_out_grad(self) -> None:
        self_out = torch.randn(1, 2, 5, 3, requires_grad=True)
        q_live = torch.randn(1, 2, 5, 3, requires_grad=True)
        ref_k = torch.randn(1, 2, 5, 3)
        ref_v = torch.randn(1, 2, 5, 3)

        result = kv_match_loss_from_attention_layers(
            self_out_list=[self_out, self_out * 0.5],
            q_list=[q_live, q_live * 0.5],
            ref_k_list=[ref_k, ref_k],
            ref_v_list=[ref_v, ref_v],
            sinkhorn_tau=0.12,
            sinkhorn_iters=3,
            ot_layer_frac=0.5,
            sinkhorn_mode="freq",
            freq_hi_boost=2.0,
        )
        result.loss.backward()
        self.assertIsNotNone(self_out.grad)
        self.assertGreater(float(self_out.grad.abs().sum().item()), 0.0)
        self.assertTrue(q_live.grad is None or torch.allclose(q_live.grad, torch.zeros_like(q_live.grad)))
        self.assertEqual(result.layer_count, 2)
        self.assertEqual(result.ot_layer_count, 1)

    def test_capture_reference_kv_preserves_passed_timestep_tensor(self) -> None:
        cache = Klein9BCache()
        controller = Klein9BController(self_layers=(0, 1))
        style_latent_t = torch.randn(1, 4, 3)
        timestep_style_model = torch.tensor([0.375], dtype=torch.bfloat16)
        null_prompt_embeds = torch.randn(1, 2, 8)
        null_text_ids = torch.zeros(1, 2, 4)
        style_image_ids = torch.zeros(1, 4, 4)
        seen: dict[str, object] = {}

        class DummyDit:
            def __call__(self, **kwargs):
                seen["hidden_states_id"] = id(kwargs["hidden_states"])
                seen["timestep_id"] = id(kwargs["timestep"])
                seen["timestep_value"] = kwargs["timestep"].detach().clone()
                seen["txt_ids_id"] = id(kwargs["txt_ids"])
                seen["img_ids_id"] = id(kwargs["img_ids"])
                controller.txt_len = int(kwargs["encoder_hidden_states"].shape[1])
                q = kwargs["hidden_states"].unsqueeze(1)
                cache.add(q, q + 1.0, q + 2.0, q + 3.0)
                return kwargs["hidden_states"]

        ref_k_list, ref_v_list, txt_len = capture_reference_kv(
            dit=DummyDit(),
            cache=cache,
            controller=controller,
            style_latent_t=style_latent_t,
            model_timestep=timestep_style_model,
            null_prompt_embeds=null_prompt_embeds,
            null_text_ids=null_text_ids,
            style_image_ids=style_image_ids,
            kv_grag_shallow=1.0,
            kv_grag_deep=1.0,
        )
        self.assertEqual(seen["hidden_states_id"], id(style_latent_t))
        self.assertEqual(seen["timestep_id"], id(timestep_style_model))
        self.assertTrue(torch.equal(seen["timestep_value"], timestep_style_model))
        self.assertEqual(seen["txt_ids_id"], id(null_text_ids))
        self.assertEqual(seen["img_ids_id"], id(style_image_ids))
        self.assertEqual(txt_len, null_prompt_embeds.shape[1])
        self.assertEqual(len(ref_k_list), 1)
        self.assertEqual(len(ref_v_list), 1)

    def test_gradient_audit_autograd_grad_does_not_pollute_latents_grad(self) -> None:
        latents = torch.randn(1, 3, 2, requires_grad=True)
        state = StyleGradientCalibrationState(target_rms=None, calibration_eps=1e-12, effective_style_weight=2.0)
        style_loss = latents.sum()
        dist_loss = (2.0 * latents).sum()
        content_loss = (3.0 * latents).sum()

        result = apply_gradient_audit(
            style_mode="score_match",
            latents=latents,
            style_loss=style_loss,
            dist_loss=dist_loss,
            content_loss=content_loss,
            style_weight=2.0,
            dist_weight=0.15,
            content_weight=0.2,
            calibration_state=state,
            should_measure=True,
            outer_step=7,
            inner_iter=0,
        )
        self.assertIsNone(latents.grad)
        self.assertTrue(state.measured_once)
        self.assertFalse(state.frozen)
        self.assertEqual(state.measured_outer_step, 7)
        self.assertEqual(state.measured_inner_iter, 0)
        self.assertEqual(state.measured_style_mode, "score_match")
        total_loss = result.weighted_style_loss + result.weighted_dist_loss + result.weighted_content_loss
        total_loss.backward()
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.allclose(latents.grad, torch.full_like(latents, 2.9)))

    def test_style_grad_target_formula_is_correct(self) -> None:
        scale = compute_style_calibration_scale(
            raw_style_grad_rms=1.0,
            style_weight=2.0,
            target_rms=0.5,
            calibration_eps=1e-12,
        )
        self.assertAlmostEqual(scale, 0.25, places=7)

    def test_style_grad_calibration_freezes_after_first_measurement(self) -> None:
        latents = torch.randn(1, 2, 2, requires_grad=True)
        state = StyleGradientCalibrationState(target_rms=0.5, calibration_eps=1e-12, effective_style_weight=2.0)

        first = apply_gradient_audit(
            style_mode="ca_only",
            latents=latents,
            style_loss=latents.sum(),
            dist_loss=(latents * 0.0).sum(),
            content_loss=(latents * 0.0).sum(),
            style_weight=2.0,
            dist_weight=0.15,
            content_weight=0.2,
            calibration_state=state,
            should_measure=True,
            outer_step=7,
            inner_iter=0,
        )
        self.assertTrue(state.measured_once)
        self.assertTrue(state.frozen)
        self.assertAlmostEqual(state.scale, 0.25, places=7)
        self.assertAlmostEqual(first.effective_style_weight, 0.5, places=7)

        latents_2 = torch.randn(1, 2, 2, requires_grad=True)
        second = apply_gradient_audit(
            style_mode="ca_only",
            latents=latents_2,
            style_loss=(4.0 * latents_2).sum(),
            dist_loss=(latents_2 * 0.0).sum(),
            content_loss=(latents_2 * 0.0).sum(),
            style_weight=2.0,
            dist_weight=0.15,
            content_weight=0.2,
            calibration_state=state,
            should_measure=False,
            outer_step=8,
            inner_iter=0,
        )
        self.assertAlmostEqual(second.effective_style_weight, 0.5, places=7)
        self.assertEqual(state.measured_outer_step, 7)
        self.assertEqual(state.measured_inner_iter, 0)

    def test_default_none_does_not_change_total_loss(self) -> None:
        latents = torch.randn(1, 2, 2, requires_grad=True)
        state = StyleGradientCalibrationState(target_rms=None, calibration_eps=1e-12, effective_style_weight=2.0)
        style_loss = (latents.square()).mean()
        dist_loss = (latents + 1.0).abs().mean()
        content_loss = (latents - 2.0).abs().mean()
        result = apply_gradient_audit(
            style_mode="kv_match",
            latents=latents,
            style_loss=style_loss,
            dist_loss=dist_loss,
            content_loss=content_loss,
            style_weight=2.0,
            dist_weight=0.15,
            content_weight=0.2,
            calibration_state=state,
            should_measure=False,
            outer_step=7,
            inner_iter=0,
        )
        expected = 2.0 * style_loss + 0.15 * dist_loss + 0.2 * content_loss
        actual = result.weighted_style_loss + result.weighted_dist_loss + result.weighted_content_loss
        self.assertTrue(torch.allclose(actual, expected))

    def test_gradient_audit_helper_supports_score_match_ca_only_and_kv_match(self) -> None:
        for mode in ("score_match", "ca_only", "kv_match"):
            latents = torch.randn(1, 2, 2, requires_grad=True)
            state = StyleGradientCalibrationState(target_rms=None, calibration_eps=1e-12, effective_style_weight=2.0)
            result = apply_gradient_audit(
                style_mode=mode,
                latents=latents,
                style_loss=latents.sum(),
                dist_loss=latents.sum(),
                content_loss=latents.sum(),
                style_weight=2.0,
                dist_weight=0.15,
                content_weight=0.2,
                calibration_state=state,
                should_measure=True,
                outer_step=7,
                inner_iter=0,
            )
            self.assertEqual(result.log_row["gradient_audit_style_mode"], mode)
            self.assertTrue(state.measured_once)

    def test_gradient_audit_can_measure_on_probe_without_touching_main_graph(self) -> None:
        latents = torch.randn(1, 2, 2, requires_grad=True)
        probe_latents = latents.detach().clone().requires_grad_(True)
        state = StyleGradientCalibrationState(target_rms=0.5, calibration_eps=1e-12, effective_style_weight=2.0)
        result = apply_gradient_audit(
            style_mode="score_match",
            latents=latents,
            style_loss=(latents * 2.0).sum(),
            dist_loss=(latents * 3.0).sum(),
            content_loss=(latents * 4.0).sum(),
            measurement_latents=probe_latents,
            measurement_style_loss=probe_latents.sum(),
            measurement_dist_loss=(probe_latents * 0.0).sum(),
            measurement_content_loss=(probe_latents * 0.0).sum(),
            style_weight=2.0,
            dist_weight=0.15,
            content_weight=0.2,
            calibration_state=state,
            should_measure=True,
            outer_step=7,
            inner_iter=0,
        )
        self.assertAlmostEqual(state.scale, 0.25, places=7)
        self.assertIsNone(latents.grad)
        self.assertIsNone(probe_latents.grad)
        total_loss = result.weighted_style_loss + result.weighted_dist_loss + result.weighted_content_loss
        total_loss.backward()
        self.assertTrue(torch.allclose(latents.grad, torch.full_like(latents, 2.25)))
        self.assertIsNone(probe_latents.grad)

    def test_rms_normalization_is_resolution_stable(self) -> None:
        g_small = torch.randn(1, 16, 8)
        g_large = torch.randn(1, 128, 8)
        norm_small = rms_normalize_field(g_small)
        norm_large = rms_normalize_field(g_large)
        rms_small = norm_small.float().square().mean().sqrt().item()
        rms_large = norm_large.float().square().mean().sqrt().item()
        self.assertAlmostEqual(rms_small, 1.0, places=4)
        self.assertAlmostEqual(rms_large, 1.0, places=4)

        x0_small = torch.zeros_like(g_small)
        x0_large = torch.zeros_like(g_large)
        _, target_small, _ = build_pseudo_target_loss(
            x0_small,
            g_small,
            step_size=0.2,
            normalize_grad_mode="rms",
        )
        _, target_large, _ = build_pseudo_target_loss(
            x0_large,
            g_large,
            step_size=0.2,
            normalize_grad_mode="rms",
        )
        self.assertAlmostEqual(float(target_small.float().square().mean().sqrt().item()), 0.2, places=3)
        self.assertAlmostEqual(float(target_large.float().square().mean().sqrt().item()), 0.2, places=3)

    def test_plan_b_is_straight_through_on_latent_only(self) -> None:
        z = torch.randn(1, 5, 3, requires_grad=True)
        alpha = torch.nn.Parameter(torch.tensor(2.0))
        beta = torch.nn.Parameter(torch.tensor(-1.5))
        sigma = torch.tensor(0.35)

        v_without = alpha * z
        v_with = (alpha * z + 1.0).detach()
        v_fake = (beta * z - 1.0).detach()
        x0_st = z - sigma * v_without.detach()

        result = pseudo_guidance_from_velocities(
            "ca_dmd",
            x0_pred=x0_st,
            v_with_detached=v_with,
            v_without_detached=v_without.detach(),
            v_fake_detached=v_fake,
            sigma=sigma,
            cfg_scale=2.0,
            lambda_ca=1.0,
            lambda_dm=0.5,
            pseudo_step_size=0.3,
            normalize_grad_mode="rms",
        )
        result.loss.backward()

        self.assertTrue(alpha.grad is None or torch.allclose(alpha.grad, torch.zeros_like(alpha.grad)))
        self.assertTrue(beta.grad is None or torch.allclose(beta.grad, torch.zeros_like(beta.grad)))

        expected = (x0_st.detach() - result.x0_target.detach()) / x0_st.numel()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.allclose(z.grad, expected, atol=1e-6, rtol=1e-5))

    def test_plan_b_update_moves_with_score_direction_for_ca_only_and_ca_dmd(self) -> None:
        lr = 1.0

        x_ca = torch.zeros(1, 4, 3, requires_grad=True)
        ca_only = pseudo_guidance_from_velocities(
            "ca_only",
            x0_pred=x_ca,
            v_with_detached=torch.ones_like(x_ca),
            v_without_detached=torch.zeros_like(x_ca),
            sigma=torch.tensor(0.5),
            cfg_scale=2.0,
            lambda_ca=1.0,
            pseudo_step_size=0.2,
            normalize_grad_mode="rms",
        )
        ca_only.loss.backward()
        x_ca_new = x_ca.detach() - lr * x_ca.grad.detach()
        self.assertGreater(float(((x_ca_new - x_ca.detach()) * ca_only.pseudo_grad_score.detach()).sum().item()), 0.0)

        x_dmd = torch.zeros(1, 4, 3, requires_grad=True)
        ca_dmd = pseudo_guidance_from_velocities(
            "ca_dmd",
            x0_pred=x_dmd,
            v_with_detached=2.0 * torch.ones_like(x_dmd),
            v_without_detached=torch.zeros_like(x_dmd),
            v_fake_detached=-torch.ones_like(x_dmd),
            sigma=torch.tensor(0.5),
            cfg_scale=2.0,
            lambda_ca=1.0,
            lambda_dm=0.5,
            pseudo_step_size=0.2,
            normalize_grad_mode="rms",
        )
        ca_dmd.loss.backward()
        x_dmd_new = x_dmd.detach() - lr * x_dmd.grad.detach()
        self.assertGreater(float(((x_dmd_new - x_dmd.detach()) * ca_dmd.pseudo_grad_score.detach()).sum().item()), 0.0)

    def test_dmd2_clean_flow_target_regression(self) -> None:
        fake_latents = torch.randn(1, 4, 3)
        noise = torch.randn_like(fake_latents)
        sigma = 0.4
        noisy_latents = (1.0 - sigma) * fake_latents + sigma * noise
        correct = clean_flow_target_from_noise(fake_latents, noise)
        wrong = noise - noisy_latents
        self.assertTrue(torch.allclose(correct, noise - fake_latents))
        self.assertFalse(torch.allclose(correct, wrong))

    def test_full_fake_backend_uses_scheduler_sigma_lookup_and_updates_sampled_params(self) -> None:
        class DummyFakeDiT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def velocity_fn(**kwargs):
            latents = kwargs["latents"]
            prompt_term = kwargs["prompt_embeds"].float().mean() * 0.01
            edit_term = kwargs["edit_condition"].edit_latents.float().mean() * 0.02
            return kwargs["dit"].scale * latents + kwargs["dit"].bias + prompt_term + edit_term

        fake_dit = DummyFakeDiT()
        conditioning = FakeConditioning(
            prompt_embeds=torch.randn(1, 2, 3).detach(),
            text_ids=torch.zeros(1, 2, 4).detach(),
            image_ids=torch.zeros(1, 4, 4).detach(),
            edit_condition=EditCondition(
                edit_latents=torch.randn(1, 4, 3).detach(),
                edit_image_ids=torch.zeros(1, 4, 4).detach(),
                source="test",
            ),
        )
        scheduler_lookup = FakeSchedulerLookup(
            timesteps=torch.tensor([900.0, 500.0, 100.0]),
            sigmas=torch.tensor([0.9, 0.5, 0.1]),
        )
        backend = build_fake_score_backend(
            "klein_full_fake",
            fake_dit=fake_dit,
            conditioning=conditioning,
            scheduler_lookup=scheduler_lookup,
            config=KleinFullFakeBackendConfig(
                fake_model_root="/tmp/fake",
                optimizer="sgd",
                lr=0.5,
                updates_per_inner=1,
                train_timestep_mode="current",
                train_seed=123,
            ),
            velocity_fn=velocity_fn,
        )
        self.assertIsNotNone(backend)
        point = resolve_schedule_point(
            scheduler_lookup=scheduler_lookup,
            timestep=torch.tensor([490.0]),
            ref_tensor=torch.zeros(1, 4, 3),
        )
        self.assertEqual(point.timestep_index, 1)
        self.assertAlmostEqual(float(point.sigma.item()), 0.5, places=6)

        fake_latents_clean = torch.ones(1, 4, 3, requires_grad=True)
        before_scale = float(fake_dit.scale.detach().item())
        log = backend.train_step(
            fake_latents_clean=fake_latents_clean,
            z_sigma=torch.zeros_like(fake_latents_clean),
            timestep=torch.tensor([490.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={},
        )
        self.assertIsNone(fake_latents_clean.grad)
        self.assertNotEqual(before_scale, float(fake_dit.scale.detach().item()))
        self.assertAlmostEqual(log["fake_train_sigma"], 0.5, places=6)
        self.assertAlmostEqual(log["fake_scoring_sigma"], 0.5, places=6)
        self.assertGreater(log["fake_param_sample_update_rms"], 0.0)
        self.assertGreater(log["fake_param_sample_update_nonzero"], 0.0)
        self.assertLessEqual(log["fake_train_loss_after"], log["fake_train_loss_before"])

        predicted = backend.predict_velocity(
            z_sigma=torch.zeros_like(fake_latents_clean),
            timestep=torch.tensor([500.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={},
        )
        self.assertFalse(predicted.requires_grad)
        self.assertEqual(predicted.shape, fake_latents_clean.shape)

    def test_full_fake_backend_dynamic_conditioning_changes_velocity_and_accumulates_once(self) -> None:
        class DummyFakeDiT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))

        train_latents_seen: list[torch.Tensor] = []

        def velocity_fn(**kwargs):
            latents = kwargs["latents"]
            prompt_term = kwargs["prompt_embeds"].float().mean()
            edit_term = kwargs["edit_condition"].edit_latents.float().mean()
            if not kwargs["no_grad"]:
                train_latents_seen.append(latents.detach().clone())
            return kwargs["dit"].scale * latents + prompt_term + edit_term

        cond_a = FakeConditioning(
            prompt_embeds=torch.zeros(1, 2, 3),
            text_ids=torch.zeros(1, 2, 4),
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.zeros(1, 4, 3),
                edit_image_ids=torch.zeros(1, 4, 4),
                source="a",
            ),
        )
        cond_b = FakeConditioning(
            prompt_embeds=torch.full((1, 2, 3), 2.0),
            text_ids=torch.zeros(1, 2, 4),
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.full((1, 4, 3), 3.0),
                edit_image_ids=torch.zeros(1, 4, 4),
                source="b",
            ),
        )
        backend = build_fake_score_backend(
            "klein_full_fake",
            fake_dit=DummyFakeDiT(),
            conditioning=None,
            scheduler_lookup=FakeSchedulerLookup(
                timesteps=torch.tensor([900.0, 500.0, 100.0]),
                sigmas=torch.tensor([0.9, 0.5, 0.1]),
            ),
            config=KleinFullFakeBackendConfig(
                fake_model_root="/tmp/fake",
                optimizer="sgd",
                lr=0.01,
                updates_per_inner=1,
                train_timestep_mode="current",
                train_seed=123,
            ),
            velocity_fn=velocity_fn,
        )
        self.assertIsNotNone(backend)

        z_sigma = torch.ones(1, 4, 3)
        predicted_a = backend.predict_velocity(
            z_sigma=z_sigma,
            timestep=torch.tensor([500.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={"conditioning": cond_a},
        )
        predicted_b = backend.predict_velocity(
            z_sigma=z_sigma,
            timestep=torch.tensor([500.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={"conditioning": cond_b},
        )
        self.assertFalse(torch.allclose(predicted_a, predicted_b))

        step_calls = 0
        original_step = backend.optimizer.step

        def counted_step(*args, **kwargs):
            nonlocal step_calls
            step_calls += 1
            return original_step(*args, **kwargs)

        backend.optimizer.step = counted_step  # type: ignore[method-assign]
        train_entry_a = FakeCriticTrainEntry(
            fake_latents_clean=torch.ones(1, 4, 3, requires_grad=True),
            conditioning=cond_a,
            timestep=torch.tensor([900.0]),
            sample_id="a",
            reference_sha256="ref-a",
            prompt_id="kp0",
        )
        train_entry_b = FakeCriticTrainEntry(
            fake_latents_clean=torch.ones(1, 4, 3, requires_grad=True),
            conditioning=cond_b,
            timestep=torch.tensor([100.0]),
            sample_id="b",
            reference_sha256="ref-b",
            prompt_id="kp1",
        )
        log = backend.train_step(
            fake_latents_clean=torch.ones(1, 4, 3, requires_grad=True),
            z_sigma=torch.zeros(1, 4, 3),
            timestep=torch.tensor([500.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={"train_entries": [train_entry_a, train_entry_b]},
        )
        self.assertEqual(step_calls, 1)
        self.assertEqual(log["fake_train_entry_count"], 2.0)
        self.assertEqual(log["fake_train_timestep_index_list"], [0.0, 2.0])
        self.assertEqual(len(log["fake_train_sigma_list"]), 2)
        self.assertAlmostEqual(log["fake_train_sigma_list"][0], 0.9, places=6)
        self.assertAlmostEqual(log["fake_train_sigma_list"][1], 0.1, places=6)
        self.assertEqual(log["fake_train_sample_ids"], ["a", "b"])
        self.assertIsNone(train_entry_a.fake_latents_clean.grad)
        self.assertIsNone(train_entry_b.fake_latents_clean.grad)

    def test_full_fake_backend_uses_independent_noise_per_accumulated_entry(self) -> None:
        class DummyFakeDiT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))

        train_latents_seen: list[torch.Tensor] = []

        def velocity_fn(**kwargs):
            if not kwargs["no_grad"]:
                train_latents_seen.append(kwargs["latents"].detach().clone())
            return kwargs["dit"].scale * kwargs["latents"]

        conditioning = FakeConditioning(
            prompt_embeds=torch.zeros(1, 2, 3),
            text_ids=torch.zeros(1, 2, 4),
            image_ids=torch.zeros(1, 4, 4),
            edit_condition=EditCondition(
                edit_latents=torch.zeros(1, 4, 3),
                edit_image_ids=torch.zeros(1, 4, 4),
                source="same",
            ),
        )
        backend = build_fake_score_backend(
            "klein_full_fake",
            fake_dit=DummyFakeDiT(),
            conditioning=None,
            scheduler_lookup=FakeSchedulerLookup(
                timesteps=torch.tensor([500.0]),
                sigmas=torch.tensor([0.5]),
            ),
            config=KleinFullFakeBackendConfig(
                fake_model_root="/tmp/fake",
                optimizer="sgd",
                lr=0.01,
                updates_per_inner=1,
                train_timestep_mode="current",
                train_seed=456,
            ),
            velocity_fn=velocity_fn,
        )
        self.assertIsNotNone(backend)
        entry_a = FakeCriticTrainEntry(
            fake_latents_clean=torch.ones(1, 4, 3),
            conditioning=conditioning,
            timestep=torch.tensor([500.0]),
        )
        entry_b = FakeCriticTrainEntry(
            fake_latents_clean=torch.ones(1, 4, 3),
            conditioning=conditioning,
            timestep=torch.tensor([500.0]),
        )
        backend.train_step(
            fake_latents_clean=torch.ones(1, 4, 3),
            z_sigma=torch.zeros(1, 4, 3),
            timestep=torch.tensor([500.0]),
            scoring_sigma=torch.tensor([0.5]),
            cond={"train_entries": [entry_a, entry_b]},
        )
        self.assertGreaterEqual(len(train_latents_seen), 2)
        self.assertFalse(torch.allclose(train_latents_seen[0], train_latents_seen[1]))

    def test_persistent_replay_entry_roundtrip_and_train_entry_conversion(self) -> None:
        entry = self._replay_entry(
            sample_id="sample-a",
            reference_sha256="ref-a",
            prompt_id="kp0",
            latent_value=2.0,
        )
        self.assertEqual(entry.clean_latent_cpu.device.type, "cpu")
        self.assertFalse(entry.clean_latent_cpu.requires_grad)
        restored = PersistentReplayEntry.from_state_dict(entry.state_dict())
        self.assertEqual(restored.sample_id, "sample-a")
        train_entry = restored.to_train_entry(
            device="cpu",
            dtype=torch.float32,
            default_timestep=torch.tensor([900.0]),
        )
        self.assertEqual(train_entry.sample_id, "sample-a")
        self.assertEqual(train_entry.reference_sha256, "ref-a")
        self.assertEqual(train_entry.prompt_id, "kp0")
        self.assertAlmostEqual(float(train_entry.timestep.item()), 500.0, places=6)
        self.assertFalse(train_entry.fake_latents_clean.requires_grad)

    def test_persistent_replay_buffer_capacity_and_state_roundtrip(self) -> None:
        buffer = PersistentReplayBuffer(capacity=2, seed=17)
        buffer.add(self._replay_entry(sample_id="a", reference_sha256="ref-a", prompt_id="kp0", latent_value=1.0))
        buffer.add(self._replay_entry(sample_id="b", reference_sha256="ref-b", prompt_id="kp1", latent_value=2.0))
        overwritten = buffer.add(
            self._replay_entry(sample_id="c", reference_sha256="ref-c", prompt_id="kp0", latent_value=3.0)
        )
        self.assertIsNotNone(overwritten)
        self.assertEqual(overwritten.sample_id, "a")
        self.assertEqual(len(buffer), 2)
        self.assertEqual(sorted(entry.sample_id for entry in buffer.entries()), ["b", "c"])
        summary = buffer.composition_summary()
        self.assertEqual(summary["size"], 2)
        restored = PersistentReplayBuffer.from_state_dict(buffer.state_dict())
        self.assertEqual(restored.capacity, 2)
        self.assertEqual(restored.next_insert_idx, buffer.next_insert_idx)
        self.assertEqual(sorted(entry.sample_id for entry in restored.entries()), ["b", "c"])

    def test_persistent_replay_buffer_stratified_sampling_is_deterministic(self) -> None:
        buffer_a = PersistentReplayBuffer(capacity=8, seed=123)
        buffer_b = PersistentReplayBuffer(capacity=8, seed=123)
        entries = [
            self._replay_entry(sample_id="a0", reference_sha256="ref-a", prompt_id="kp0", latent_value=1.0),
            self._replay_entry(sample_id="a1", reference_sha256="ref-a", prompt_id="kp0", latent_value=2.0),
            self._replay_entry(sample_id="b0", reference_sha256="ref-b", prompt_id="kp1", latent_value=3.0),
            self._replay_entry(sample_id="b1", reference_sha256="ref-b", prompt_id="kp1", latent_value=4.0),
        ]
        for entry in entries:
            buffer_a.add(entry)
            buffer_b.add(entry)
        sample_ids_a = [entry.sample_id for entry in buffer_a.sample_stratified(4, split="train")]
        sample_ids_b = [entry.sample_id for entry in buffer_b.sample_stratified(4, split="train")]
        self.assertEqual(sample_ids_a, sample_ids_b)
        self.assertEqual(len(sample_ids_a), 4)
        self.assertEqual({sample_ids_a[0][0], sample_ids_a[1][0]}, {"a", "b"})

    def test_persistent_replay_buffer_builds_latest_replay_ratio(self) -> None:
        buffer = PersistentReplayBuffer(capacity=8, seed=5)
        latest = self._replay_entry(sample_id="latest", reference_sha256="ref-z", prompt_id="kp0", latent_value=9.0)
        buffer.add(self._replay_entry(sample_id="a0", reference_sha256="ref-a", prompt_id="kp0", latent_value=1.0))
        buffer.add(self._replay_entry(sample_id="b0", reference_sha256="ref-b", prompt_id="kp1", latent_value=2.0))
        batch = buffer.build_accumulation_entries(
            latest_entry=latest,
            grad_accum_steps=4,
            replay_mode="replay",
            split="train",
        )
        self.assertEqual(len(batch), 4)
        self.assertEqual(sum(entry.sample_id == "latest" for entry in batch), 2)
        self.assertEqual(sum(entry.sample_id != "latest" for entry in batch), 2)
        no_replay_batch = buffer.build_accumulation_entries(
            latest_entry=latest,
            grad_accum_steps=4,
            replay_mode="no_replay",
            split="train",
        )
        self.assertEqual([entry.sample_id for entry in no_replay_batch], ["latest"] * 4)

    def test_validate_reference_hash_split_rejects_overlap(self) -> None:
        entries = [
            self._replay_entry(sample_id="train-a", reference_sha256="same-ref", prompt_id="kp0", split="train"),
            self._replay_entry(sample_id="heldout-a", reference_sha256="same-ref", prompt_id="kp1", split="heldout"),
        ]
        with self.assertRaises(ValueError):
            validate_reference_hash_split(entries)

    def test_persistent_replay_resume_cursor_matches_uninterrupted_training(self) -> None:
        class DummyFakeDiT(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.1))
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def velocity_fn(**kwargs):
            latents = kwargs["latents"]
            prompt_term = kwargs["prompt_embeds"].float().mean() * 0.01
            edit_term = kwargs["edit_condition"].edit_latents.float().mean() * 0.02
            return kwargs["dit"].scale * latents + kwargs["dit"].bias + prompt_term + edit_term

        scheduler_lookup = FakeSchedulerLookup(
            timesteps=torch.tensor([500.0]),
            sigmas=torch.tensor([0.5]),
        )
        entries = [
            self._replay_entry(sample_id="s0", reference_sha256="ref-a", prompt_id="kp0", latent_value=1.0),
            self._replay_entry(sample_id="s1", reference_sha256="ref-b", prompt_id="kp1", latent_value=2.0),
            self._replay_entry(sample_id="s2", reference_sha256="ref-c", prompt_id="kp0", latent_value=3.0),
        ]

        def make_backend(resume_state_path: str | None = None):
            backend = build_fake_score_backend(
                "klein_full_fake",
                fake_dit=DummyFakeDiT(),
                conditioning=None,
                scheduler_lookup=scheduler_lookup,
                config=KleinFullFakeBackendConfig(
                    fake_model_root="/tmp/fake",
                    optimizer="sgd",
                    lr=0.05,
                    updates_per_inner=1,
                    train_timestep_mode="current",
                    train_seed=123,
                    resume_state_path=resume_state_path,
                ),
                velocity_fn=velocity_fn,
            )
            self.assertIsNotNone(backend)
            return backend

        def run_training(backend, buffer, *, start_train_index: int, start_sample_update_index: int, stop_after_updates: int | None = None):
            total_updates = int(backend.summary_dict()["update_count"])
            default_timestep = torch.tensor([500.0], dtype=torch.float32)
            for train_idx in range(start_train_index, len(entries)):
                latest = entries[train_idx]
                sample_start = start_sample_update_index if train_idx == start_train_index else 0
                if sample_start == 0:
                    buffer.add(latest)
                for sample_update_idx in range(sample_start, 2):
                    batch = buffer.build_accumulation_entries(
                        latest_entry=latest,
                        grad_accum_steps=4,
                        replay_mode="replay",
                        split="train",
                    )
                    train_entries = [
                        entry.to_train_entry(device="cpu", dtype=torch.float32, default_timestep=default_timestep)
                        for entry in batch
                    ]
                    backend.train_step(
                        fake_latents_clean=train_entries[0].fake_latents_clean,
                        z_sigma=torch.zeros_like(train_entries[0].fake_latents_clean),
                        timestep=train_entries[0].timestep,
                        scoring_sigma=torch.tensor([0.5]),
                        cond={"train_entries": train_entries},
                    )
                    total_updates += 1
                    if stop_after_updates is not None and total_updates == stop_after_updates:
                        next_train_index = train_idx
                        next_sample_update_index = sample_update_idx + 1
                        if next_sample_update_index >= 2:
                            next_train_index = train_idx + 1
                            next_sample_update_index = 0
                        return {
                            "next_train_index": next_train_index,
                            "next_sample_update_index": next_sample_update_index,
                            "total_updates": total_updates,
                        }
            return {
                "next_train_index": len(entries),
                "next_sample_update_index": 0,
                "total_updates": total_updates,
            }

        full_backend = make_backend()
        full_buffer = PersistentReplayBuffer(capacity=8, seed=77)
        full_cursor = run_training(full_backend, full_buffer, start_train_index=0, start_sample_update_index=0)
        self.assertEqual(full_cursor["total_updates"], 6)

        with TemporaryDirectory() as tmpdir:
            part_backend = make_backend()
            part_buffer = PersistentReplayBuffer(capacity=8, seed=77)
            resume_cursor = run_training(
                part_backend,
                part_buffer,
                start_train_index=0,
                start_sample_update_index=0,
                stop_after_updates=3,
            )
            fake_state_path = Path(tmpdir) / "fake_backend.pt"
            trainer_state_path = Path(tmpdir) / "trainer_state.pt"
            part_backend.save_state(fake_state_path)
            torch.save(
                {
                    "replay_buffer": part_buffer.state_dict(),
                    "next_train_index": resume_cursor["next_train_index"],
                    "next_sample_update_index": resume_cursor["next_sample_update_index"],
                },
                trainer_state_path,
            )
            saved = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
            resumed_backend = make_backend(str(fake_state_path))
            resumed_buffer = PersistentReplayBuffer.from_state_dict(saved["replay_buffer"])
            resumed_cursor = run_training(
                resumed_backend,
                resumed_buffer,
                start_train_index=int(saved["next_train_index"]),
                start_sample_update_index=int(saved["next_sample_update_index"]),
            )

        self.assertEqual(resumed_cursor["total_updates"], full_cursor["total_updates"])
        for param_full, param_resumed in zip(full_backend.fake_dit.parameters(), resumed_backend.fake_dit.parameters()):
            self.assertTrue(torch.allclose(param_full.detach(), param_resumed.detach(), atol=1e-7, rtol=0.0))
        self.assertEqual(
            [entry.sample_id for entry in full_buffer.entries()],
            [entry.sample_id for entry in resumed_buffer.entries()],
        )

    def test_ca_dmd_mode_guard_and_placeholder_backend_errors(self) -> None:
        with self.assertRaises(ValueError):
            validate_style_mode("ca_dmd", "none")

        bridge = build_fake_score_backend("dmd2_guidance")
        self.assertIsNotNone(bridge)
        with self.assertRaises(NotImplementedError):
            bridge.train_step(
                fake_latents_clean=torch.zeros(1, 4, 3),
                z_sigma=torch.zeros(1, 4, 3),
                timestep=torch.tensor([500.0]),
                scoring_sigma=torch.tensor([0.5]),
                cond={},
            )
        with self.assertRaises(NotImplementedError):
            bridge.predict_velocity(
                z_sigma=torch.zeros(1, 4, 3),
                timestep=torch.tensor([500.0]),
                scoring_sigma=torch.tensor([0.5]),
                cond={},
            )

    def test_score_match_forces_teacher_full_cuda_even_with_legacy_vram_limit(self) -> None:
        contract = resolve_dual_model_vram_contract(
            "score_match",
            legacy_vram_limit_gb=16.0,
        )
        self.assertEqual(contract.effective_main_vram_limit_gb, 16.0)
        self.assertIsNone(contract.effective_teacher_vram_limit_gb)
        self.assertTrue(contract.teacher_requires_full_cuda)
        self.assertTrue(any("overriding teacher_vram_limit_gb=16.0 to None" in note for note in contract.notes))

    def test_kv_match_forces_main_full_cuda_and_skips_external_teacher(self) -> None:
        contract = resolve_dual_model_vram_contract(
            "kv_match",
            legacy_vram_limit_gb=16.0,
            teacher_vram_limit_gb=24.0,
        )
        self.assertIsNone(contract.effective_main_vram_limit_gb)
        self.assertFalse(contract.teacher_model_required)
        self.assertTrue(contract.main_requires_full_cuda)
        self.assertTrue(any("overriding main_vram_limit_gb=16.0 to None" in note for note in contract.notes))

    def test_csv_fieldnames_handle_heterogeneous_rows(self) -> None:
        rows = [
            {"outer_step": 0, "inner_iter": -1, "sigma": None, "skipped": "pre_start_time"},
            {"outer_step": 1, "inner_iter": 0, "sigma": 0.5, "style_loss": 1.2, "dist_loss": 0.3},
        ]
        fieldnames = csv_fieldnames_for_rows(rows)
        self.assertEqual(fieldnames[:3], ["outer_step", "inner_iter", "sigma"])
        self.assertEqual(fieldnames[3:], ["dist_loss", "skipped", "style_loss"])

    def test_final_adain_helper_preserves_default_behavior_and_disable_behavior(self) -> None:
        latents = torch.tensor([[[1.0], [3.0], [5.0]]])
        style_latent = torch.tensor([[[10.0], [12.0], [14.0]]])
        enabled = final_adain_if_enabled(latents, style_latent, enabled=True)
        disabled = final_adain_if_enabled(latents, style_latent, enabled=False)
        self.assertTrue(torch.allclose(disabled, latents))
        self.assertFalse(torch.allclose(enabled, latents))
        self.assertAlmostEqual(float(enabled.mean().item()), float(style_latent.mean().item()), places=6)

    def test_loss_isolation_contract_reports_true_loss_only_when_final_adain_is_disabled(self) -> None:
        contract = build_loss_isolation_contract(
            disable_slt=True,
            adain_alpha=0.0,
            disable_final_adain=True,
        )
        self.assertTrue(contract["true_loss_only"])
        self.assertFalse(contract["final_adain_enabled"])

        non_isolated = build_loss_isolation_contract(
            disable_slt=True,
            adain_alpha=0.0,
            disable_final_adain=False,
        )
        self.assertFalse(non_isolated["true_loss_only"])
        self.assertTrue(non_isolated["final_adain_enabled"])

    def test_scheduler_step_contract_is_single_call_in_runner_source(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("scheduler.step("), 1)

    def test_runner_reuses_same_kv_model_timestep_for_capture_and_inner_forward(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("kv_model_timestep = timestep_style.to(device=latents.device, dtype=latents.dtype) / 1000", text)
        self.assertIn("model_timestep=kv_model_timestep", text)
        self.assertIn("timestep=kv_model_timestep", text)

    def test_runner_clears_kv_cache_around_main_denoise_forward(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("kv_cache.clear()"), 4)
        self.assertGreaterEqual(text.count("kv_controller.step()"), 4)
        self.assertIn('"denoise_cache_cleared": args.style_mode == "kv_match"', text)

    def test_runner_preserves_rng_and_restores_kv_probe_state(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("torch.get_rng_state()", text)
        self.assertIn("torch.cuda.get_rng_state_all()", text)
        self.assertIn("restore_kv_cache_state", text)
        self.assertIn("restore_kv_controller_state", text)
        self.assertIn("measurement_latents", text)

    def test_runner_exposes_final_adain_flag_and_summary_fields(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('--disable-final-adain', text)
        self.assertIn('"final_adain_enabled": not args.disable_final_adain', text)
        self.assertIn('"loss_isolation_contract": build_loss_isolation_contract(', text)
        self.assertIn('"disable_final_adain": False', text)

    def test_runner_exports_clean_latent_before_final_adain_when_requested(self) -> None:
        text = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn('--export-clean-latent-path', text)
        self.assertIn('"clean_latent_export": clean_latent_export_summary', text)
        self.assertIn('final_adain_applied_after_export', text)
        self.assertLess(
            text.index("clean_latent_export_summary = None"),
            text.index("latents = final_adain_if_enabled("),
        )

    def test_persistent_runner_source_contains_manifest_load_once_cache_train_eval_summary(self) -> None:
        text = PERSISTENT_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("--manifest-jsonl", text)
        self.assertIn("build_conditioning_cache", text)
        self.assertIn("validate_reference_hash_split(entries)", text)
        self.assertIn('FlowMatchScheduler("FLUX.2")', text)
        self.assertIn("dynamic_shift_len=int(latent_contract[\"latent_h\"] * latent_contract[\"latent_w\"])", text)
        self.assertIn('"next_sample_update_index": int(next_sample_update_index)', text)
        self.assertIn("scoring_point=(scoring_timestep, scoring_sigma, scoring_index)", text)
        self.assertIn("fake_model_load_count", text)
        self.assertIn("before_heldout_eval", text)
        self.assertIn("after_heldout_eval", text)
        self.assertIn("Saved summary", text)


if __name__ == "__main__":
    unittest.main()
