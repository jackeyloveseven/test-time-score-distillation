import unittest

import torch

from trial.online_rl_critic.grpo_policy import OnlineGRPOPolicy


class OnlineGRPOPolicyTest(unittest.TestCase):
    def test_group_relative_update(self):
        policy = OnlineGRPOPolicy(device=torch.device("cpu"), group_size=3)
        state = torch.zeros(10)
        decisions = [policy.select_action(state) for _ in range(3)]
        logs = []
        for index, (decision, reward) in enumerate(zip(decisions, (-0.2, 0.4, 0.1))):
            logs.append(
                policy.add_transition(
                    state=state + float(index),
                    action_index=decision.action_index,
                    old_log_prob=decision.log_prob,
                    reward=reward,
                )
            )
        self.assertEqual(policy.update_count, 1)
        self.assertGreater(logs[-1]["grpo_group_std"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(logs[-1]["grpo_policy_loss"])))

    def test_same_state_reward_group_update(self):
        policy = OnlineGRPOPolicy(device=torch.device("cpu"))
        update = policy.update_from_same_state_rewards(
            state=torch.zeros(10),
            rewards=[-0.1, 0.2, 0.05],
        )
        self.assertEqual(policy.update_count, 1)
        self.assertEqual(len(update["grpo_advantages"]), 3)
        self.assertGreater(float(update["grpo_group_std"]), 0.0)

    def test_same_state_relative_advantage_prefers_best_action(self):
        torch.manual_seed(0)
        policy = OnlineGRPOPolicy(device=torch.device("cpu"), lr=0.05, entropy_weight=0.0)
        state = torch.zeros(10)
        policy.update_from_same_state_rewards(state=state, rewards=[0.0, 0.1, 0.2])
        self.assertEqual(policy.select_greedy_action(state).action_scale, 1.5)


if __name__ == "__main__":
    unittest.main()
