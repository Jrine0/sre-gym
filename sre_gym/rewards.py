"""PBRS (Potential-Based Reward Shaping) engine for dense feedback.

Formula: ShapedReward = SparseReward + α(γΦ(s') - Φ(s))
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sre_gym.models import clamp_score


@dataclass
class PBRSConfig:
    """Configuration for potential-based reward shaping."""

    alpha: float = 0.1  # Shaping coefficient
    gamma: float = 0.99  # Discount factor for potential
    penalty_per_step: float = 0.05  # Stall penalty
    max_penalty: float = 1.0
    sparse_success_reward: float = 1.0  # Reward granted on successful task completion


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    sparse_reward: float = 0.0
    shaping_reward: float = 0.0
    step_penalty: float = 0.0
    cheat_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "sparse_reward": self.sparse_reward,
            "shaping_reward": self.shaping_reward,
            "step_penalty": self.step_penalty,
            "cheat_penalty": self.cheat_penalty,
            "total": self.total,
        }


class PBRSEngine:
    """Computes PBRS rewards based on system health potential function.

    Potential Φ(s) = healthy_pods / total_pods
    """

    def __init__(self, config: PBRSConfig | None = None):
        self.config = config or PBRSConfig()
        self._last_potential: float | None = None

    def compute_potential(self, healthy_pods: int, total_pods: int) -> float:
        """Compute potential function Φ(s) = health ratio."""
        if total_pods == 0:
            return 0.0
        return healthy_pods / total_pods

    def compute_reward(
        self,
        healthy_pods: int,
        total_pods: int,
        is_terminal: bool = False,
        is_success: bool = False,
        cheated: bool = False,
    ) -> RewardBreakdown:
        """Compute shaped reward using PBRS formula.

        Args:
            healthy_pods: Number of healthy pods in current state
            total_pods: Total expected pods
            is_terminal: Whether this is a terminal step
            is_success: Whether task was successfully completed
            cheated: Whether agent tried to game the reward

        Returns:
            RewardBreakdown with individual components and total
        """
        current_potential = self.compute_potential(healthy_pods, total_pods)

        shaping = 0.0
        if self._last_potential is not None:
            delta_phi = self.config.gamma * current_potential - self._last_potential
            shaping = self.config.alpha * delta_phi

        self._last_potential = current_potential

        sparse = self.config.sparse_success_reward if is_terminal and is_success else 0.0

        step_penalty = -self.config.penalty_per_step if not is_terminal else 0.0
        cheat_penalty = -0.2 if cheated else 0.0

        total = sparse + shaping + step_penalty + cheat_penalty
        total = clamp_score(total)  # Clamp to strict (0, 1)

        return RewardBreakdown(
            sparse_reward=sparse,
            shaping_reward=shaping,
            step_penalty=step_penalty,
            cheat_penalty=cheat_penalty,
            total=total,
        )

    def reset(self) -> None:
        """Reset engine state for new episode."""
        self._last_potential = None
