"""Regression tests for carrier-anchored score residual composition."""

from __future__ import annotations

import unittest

import torch

from src.carrier_score import content_affinity_transport
from src.flux2_score import score_delta_to_velocity_delta, velocity_delta_to_score_delta
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


if __name__ == "__main__":
    unittest.main()
