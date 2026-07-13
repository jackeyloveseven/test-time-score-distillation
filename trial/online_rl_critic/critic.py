"""Small online Q critic for the real-time latent-gradient trial.

This module is deliberately an experiment, not a replacement for the native
teacher. It chooses a scalar multiplier for the already-computed latent update
and learns from the loss change observed at the next inner stage. The reward is
therefore a diagnostic proxy for whether an action improved the current
objective; it is not a claim of a trained image reward model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ActionDecision:
    action_index: int
    action_scale: float
    q_value: float
    epsilon: float


class OnlineStageCritic:
    """Tiny tabular-action Q critic trained online with one-step TD targets."""

    def __init__(
        self,
        *,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.90,
        epsilon: float = 0.30,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        hidden_dim: int = 64,
        action_scales: tuple[float, ...] = (0.5, 1.0, 1.5),
        seed: int = 20260713,
    ) -> None:
        if not action_scales:
            raise ValueError("action_scales must not be empty")
        self.device = torch.device(device)
        self.action_scales = tuple(float(x) for x in action_scales)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = random.Random(int(seed))
        self.net = nn.Sequential(
            nn.Linear(10 + 1, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=float(lr), weight_decay=0.0)
        self.update_count = 0

    @staticmethod
    def state_vector(
        *,
        sigma: float,
        outer_fraction: float,
        inner_fraction: float,
        style_loss: float,
        dist_loss: float,
        content_loss: float,
        total_loss: float,
        grad_rms: float,
        update_rms: float,
        last_reward: float,
    ) -> Tensor:
        def signed_log(value: float) -> float:
            value_f = float(value)
            return float(torch.sign(torch.tensor(value_f)).item()) * float(torch.log1p(torch.tensor(abs(value_f))).item())

        return torch.tensor(
            [
                float(sigma),
                float(outer_fraction),
                float(inner_fraction),
                signed_log(style_loss),
                signed_log(dist_loss),
                signed_log(content_loss),
                signed_log(total_loss),
                signed_log(grad_rms),
                signed_log(update_rms),
                float(last_reward),
            ],
            dtype=torch.float32,
        )

    def _q_values(self, state: Tensor) -> Tensor:
        state_t = state.detach().to(device=self.device, dtype=torch.float32).reshape(1, -1)
        action_t = torch.tensor(self.action_scales, device=self.device, dtype=torch.float32).reshape(-1, 1)
        state_rep = state_t.expand(action_t.shape[0], -1)
        return self.net(torch.cat([state_rep, action_t], dim=-1)).reshape(-1)

    def select_action(self, state: Tensor) -> ActionDecision:
        q_values = self._q_values(state).detach()
        if self.rng.random() < self.epsilon:
            index = self.rng.randrange(len(self.action_scales))
        else:
            index = int(torch.argmax(q_values).item())
        return ActionDecision(
            action_index=index,
            action_scale=self.action_scales[index],
            q_value=float(q_values[index].item()),
            epsilon=float(self.epsilon),
        )

    def observe_transition(
        self,
        *,
        state: Tensor,
        action_index: int,
        reward: float,
        next_state: Tensor,
        done: bool = False,
    ) -> float:
        state_t = state.detach().to(device=self.device, dtype=torch.float32).reshape(1, -1)
        next_t = next_state.detach().to(device=self.device, dtype=torch.float32)
        action_t = torch.tensor([[self.action_scales[int(action_index)]]], device=self.device, dtype=torch.float32)
        q_value = self.net(torch.cat([state_t, action_t], dim=-1)).reshape(())
        with torch.no_grad():
            next_q = torch.max(self._q_values(next_t))
            target = torch.tensor(float(reward), device=self.device) + (0.0 if done else self.gamma * next_q)
        loss = nn.functional.smooth_l1_loss(q_value, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.detach().item())

    @staticmethod
    def relative_loss_reward(previous_loss: float, current_loss: float) -> float:
        denom = abs(float(previous_loss)) + 1e-6
        reward = (float(previous_loss) - float(current_loss)) / denom
        return float(max(-1.0, min(1.0, reward)))

    def state_dict(self) -> dict[str, object]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "action_scales": self.action_scales,
        }
