"""Regression tests for carrier-anchored score residual composition."""

from __future__ import annotations

import unittest

import torch

from src.carrier_score import content_affinity_transport
from src.flux2_score import score_delta_to_velocity_delta, velocity_delta_to_score_delta
from src.score_residual_student import ScoreResidualStudentConfig, build_score_residual_student
from scripts.train_score_residual_student import compose_prediction


class ScoreResidualStudentTests(unittest.TestCase):
    def test_carrier_anchored_prediction_matches_reference_independent_transport(self) -> None:
        raw_goal = torch.tensor([[[1.0, 0.0], [0.8, 0.0], [-1.0, 0.0]]])
        z_t = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]])
        hard = torch.zeros_like(raw_goal)
        predicted, loss_field = compose_prediction(
            raw_goal,
            z_t,
            hard,
            "carrier_anchored",
            carrier_temperature=0.05,
            carrier_strength=1.0,
            carrier_highpass_gain=0.0,
        )
        expected, affinity = content_affinity_transport(
            raw_goal,
            z_t,
            temperature=0.05,
            strength=1.0,
            highpass_gain=0.0,
        )
        self.assertTrue(torch.allclose(predicted, expected))
        self.assertTrue(torch.allclose(loss_field, expected))
        self.assertFalse(affinity.requires_grad)
        self.assertFalse(torch.allclose(predicted, raw_goal))

    def test_hard_delta_remains_explicit_carrier_baseline(self) -> None:
        raw_goal = torch.randn(1, 4, 3)
        hard = torch.randn_like(raw_goal)
        predicted, loss_field = compose_prediction(raw_goal, raw_goal, hard, "hard_delta")
        self.assertTrue(torch.allclose(predicted, hard + raw_goal))
        self.assertTrue(torch.allclose(loss_field, raw_goal))

    def test_native_score_goal_converts_back_to_velocity_for_sampling(self) -> None:
        delta_velocity = torch.ones(1, 3, 2)
        delta_score, _ = velocity_delta_to_score_delta(delta_velocity, torch.tensor(0.5))
        recovered = score_delta_to_velocity_delta(delta_score, torch.tensor(0.5))
        self.assertTrue(torch.allclose(recovered, delta_velocity))
        endpoint_score, _ = velocity_delta_to_score_delta(delta_velocity, torch.tensor(1.0))
        endpoint_velocity = score_delta_to_velocity_delta(endpoint_score, torch.tensor(1.0))
        self.assertTrue(torch.allclose(endpoint_velocity, torch.zeros_like(delta_velocity)))

    def test_image_code_conditioning_changes_goal_without_style_id_lookup(self) -> None:
        config = ScoreResidualStudentConfig(
            latent_dim=4,
            max_tokens=4,
            num_styles=1,
            num_prompts=2,
            hidden_dim=16,
            cond_dim=8,
            time_embed_dim=8,
            conditioning_mode="image_code",
            image_code_dim=6,
        )
        model = build_score_residual_student(config).eval()
        z_t = torch.randn(2, 4, 4)
        time = torch.tensor([500.0, 500.0])
        style_idx = torch.zeros(2, dtype=torch.long)
        prompt_idx = torch.zeros(2, dtype=torch.long)
        code_a = torch.zeros(2, 6)
        code_b = torch.arange(6, dtype=torch.float32).repeat(2, 1)
        output_a = model(z_t, time, style_idx, prompt_idx, image_code=code_a)
        output_b = model(z_t, time, style_idx, prompt_idx, image_code=code_b)
        self.assertFalse(torch.allclose(output_a, output_b))


if __name__ == "__main__":
    unittest.main()
