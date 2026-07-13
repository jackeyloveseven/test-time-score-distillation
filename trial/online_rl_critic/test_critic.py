import unittest

import torch

from trial.online_rl_critic.critic import OnlineStageCritic


class OnlineStageCriticTest(unittest.TestCase):
    def test_state_has_expected_shape(self):
        state = OnlineStageCritic.state_vector(
            sigma=0.5,
            outer_fraction=0.2,
            inner_fraction=0.5,
            style_loss=1.0,
            dist_loss=0.1,
            content_loss=0.2,
            total_loss=1.3,
            grad_rms=0.01,
            update_rms=0.02,
            last_reward=0.0,
        )
        self.assertEqual(tuple(state.shape), (10,))

    def test_action_and_td_update(self):
        critic = OnlineStageCritic(device=torch.device("cpu"), epsilon=0.0, seed=7)
        state = torch.zeros(10)
        decision = critic.select_action(state)
        self.assertIn(decision.action_scale, (0.5, 1.0, 1.5))
        loss = critic.observe_transition(
            state=state,
            action_index=decision.action_index,
            reward=0.25,
            next_state=torch.ones(10),
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(critic.update_count, 1)

    def test_reward_is_clipped(self):
        self.assertEqual(OnlineStageCritic.relative_loss_reward(1.0, -10.0), 1.0)
        self.assertEqual(OnlineStageCritic.relative_loss_reward(1.0, 10.0), -1.0)


if __name__ == "__main__":
    unittest.main()
