"""GRPO-style relative-advantage policy for the isolated critic trial."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GRPOAction:
    action_index: int
    action_scale: float
    log_prob: float
    entropy: float


class OnlineGRPOPolicy:
    """Tiny categorical policy trained from relative rewards in action groups."""

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float = 3e-4,
        group_size: int = 4,
        clip_eps: float = 0.20,
        entropy_weight: float = 0.01,
        temperature: float = 1.0,
        hidden_dim: int = 64,
        action_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
    ) -> None:
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        self.device = torch.device(device)
        self.action_scales = tuple(float(x) for x in action_scales)
        self.group_size = int(group_size)
        self.clip_eps = float(clip_eps)
        self.entropy_weight = float(entropy_weight)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(10, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), len(self.action_scales)),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=float(lr), weight_decay=0.0)
        self.group: list[dict[str, object]] = []
        self.update_count = 0

    def _logits(self, states: Tensor) -> Tensor:
        states_t = states.detach().to(device=self.device, dtype=torch.float32)
        return self.net(states_t) / max(self.temperature, 1e-6)

    def select_action(self, state: Tensor) -> GRPOAction:
        logits = self._logits(state.reshape(1, -1))
        distribution = torch.distributions.Categorical(logits=logits)
        sample = distribution.sample()
        index = int(sample.item())
        return GRPOAction(
            action_index=index,
            action_scale=self.action_scales[index],
            log_prob=float(distribution.log_prob(sample).item()),
            entropy=float(distribution.entropy().item()),
        )

    def select_greedy_action(self, state: Tensor) -> GRPOAction:
        """Select the highest-probability action after a same-state policy update."""

        logits = self._logits(state.reshape(1, -1))
        distribution = torch.distributions.Categorical(logits=logits)
        index = int(torch.argmax(logits.reshape(-1)).item())
        action = torch.tensor(index, device=self.device)
        return GRPOAction(
            action_index=index,
            action_scale=self.action_scales[index],
            log_prob=float(distribution.log_prob(action).item()),
            entropy=float(distribution.entropy().item()),
        )

    def action_for_index(self, state: Tensor, index: int) -> GRPOAction:
        """Build an action record for an externally selected candidate index."""

        logits = self._logits(state.reshape(1, -1))
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.tensor(int(index), device=self.device)
        return GRPOAction(
            action_index=int(index),
            action_scale=self.action_scales[int(index)],
            log_prob=float(distribution.log_prob(action).item()),
            entropy=float(distribution.entropy().item()),
        )

    def add_transition(
        self,
        *,
        state: Tensor,
        action_index: int,
        old_log_prob: float,
        reward: float,
    ) -> dict[str, float]:
        self.group.append(
            {
                "state": state.detach().float().cpu(),
                "action_index": int(action_index),
                "old_log_prob": float(old_log_prob),
                "reward": float(reward),
            }
        )
        if len(self.group) < self.group_size:
            return {
                "grpo_policy_loss": 0.0,
                "grpo_group_mean": 0.0,
                "grpo_group_std": 0.0,
                "grpo_advantage": 0.0,
                "grpo_update_count": float(self.update_count),
            }

        entries = self.group[: self.group_size]
        self.group = self.group[self.group_size :]
        states = torch.stack([entry["state"] for entry in entries]).to(self.device)
        actions = torch.tensor([int(entry["action_index"]) for entry in entries], device=self.device)
        old_log_probs = torch.tensor([float(entry["old_log_prob"]) for entry in entries], device=self.device)
        rewards = torch.tensor([float(entry["reward"]) for entry in entries], device=self.device)
        group_mean = rewards.mean()
        group_std = rewards.std(unbiased=False)
        advantages = (rewards - group_mean) / (group_std + 1e-6)

        distribution = torch.distributions.Categorical(logits=self._logits(states))
        new_log_probs = distribution.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs.detach())
        unclipped = ratios * advantages.detach()
        clipped = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.detach()
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = distribution.entropy().mean()
        total_loss = policy_loss - self.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.update_count += 1

        return {
            "grpo_policy_loss": float(policy_loss.detach().item()),
            "grpo_entropy": float(entropy.detach().item()),
            "grpo_group_mean": float(group_mean.detach().item()),
            "grpo_group_std": float(group_std.detach().item()),
            "grpo_advantage": float(advantages[-1].detach().item()),
            "grpo_update_count": float(self.update_count),
        }

    def update_from_same_state_rewards(
        self,
        *,
        state: Tensor,
        rewards: list[float] | Tensor,
    ) -> dict[str, object]:
        """Apply one relative-advantage update from candidate actions at one state."""

        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).reshape(-1)
        if rewards_t.numel() != len(self.action_scales):
            raise ValueError(
                f"same-state reward group must have {len(self.action_scales)} entries, got {rewards_t.numel()}"
            )
        states = state.detach().float().to(self.device).reshape(1, -1).expand(rewards_t.numel(), -1)
        actions = torch.arange(rewards_t.numel(), device=self.device)
        group_mean = rewards_t.mean()
        group_std = rewards_t.std(unbiased=False)
        advantages = (rewards_t - group_mean) / (group_std + 1e-6)

        old_distribution = torch.distributions.Categorical(logits=self._logits(states).detach())
        old_log_probs = old_distribution.log_prob(actions).detach()
        distribution = torch.distributions.Categorical(logits=self._logits(states))
        new_log_probs = distribution.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs)
        unclipped = ratios * advantages.detach()
        clipped = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.detach()
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = distribution.entropy().mean()
        total_loss = policy_loss - self.entropy_weight * entropy
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.update_count += 1
        return {
            "grpo_policy_loss": float(policy_loss.detach().item()),
            "grpo_entropy": float(entropy.detach().item()),
            "grpo_group_mean": float(group_mean.detach().item()),
            "grpo_group_std": float(group_std.detach().item()),
            "grpo_advantages": [float(value) for value in advantages.detach().cpu().tolist()],
            "grpo_rewards": [float(value) for value in rewards_t.detach().cpu().tolist()],
            "grpo_update_count": float(self.update_count),
        }
