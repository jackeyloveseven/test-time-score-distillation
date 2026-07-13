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


if __name__ == "__main__":
    unittest.main()
