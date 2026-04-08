"""Tests for SRE Gym environment."""

import pytest
from sre_gym.env import SREGymEnv, EnvConfig
from sre_gym.models import K8sAction, K8sActionType
from sre_gym.rewards import PBRSEngine, PBRSConfig, RewardBreakdown
from sre_gym.grader import AssertionEngine


class TestPBRSEngine:
    """Tests for PBRS reward shaping."""

    def test_potential_calculation(self):
        """Φ(s) should return healthy/total."""
        engine = PBRSEngine()
        assert engine.compute_potential(0, 1) == 0.0
        assert engine.compute_potential(1, 1) == 1.0
        assert engine.compute_potential(3, 4) == 0.75

    def test_reward_bounded_0_1(self):
        """All rewards should be clamped to [0, 1]."""
        engine = PBRSEngine()
        for hp in range(5):
            for tp in range(1, 6):
                breakdown = engine.compute_reward(hp, tp)
                assert 0.0 <= breakdown.total <= 1.0, f"Reward {breakdown.total} out of bounds"

    def test_reset_clears_state(self):
        """reset() should clear last potential."""
        engine = PBRSEngine()
        engine.compute_reward(1, 2)
        assert engine._last_potential is not None
        engine.reset()
        assert engine._last_potential is None

    def test_success_reward_at_terminal(self):
        """Terminal success should grant sparse reward."""
        engine = PBRSEngine(PBRSConfig(sparse_success_reward=1.0))
        engine.compute_reward(1, 1, is_terminal=False)
        breakdown = engine.compute_reward(1, 1, is_terminal=True, is_success=True)
        assert breakdown.sparse_reward == 1.0


class TestK8sModels:
    """Tests for Pydantic models."""

    def test_k8s_action_validation(self):
        """K8sAction should validate required fields."""
        action = K8sAction(action_type=K8sActionType.APPLY_MANIFEST)
        assert action.action_type == K8sActionType.APPLY_MANIFEST
        assert action.manifest is None

    def test_k8s_observation_bounds(self):
        """K8sObservation health_score should be in [0, 1]."""
        from sre_gym.models import K8sObservation
        obs = K8sObservation(
            kubectl_output="test",
            health_score=0.5,
            step_number=1,
        )
        assert 0.0 <= obs.health_score <= 1.0

    def test_task_config_defaults(self):
        """TaskConfig should have sensible defaults."""
        from sre_gym.models import TaskConfig
        config = TaskConfig(difficulty="easy", name="test", description="desc")
        assert config.max_steps >= 1
        assert config.penalty_per_step >= 0


class TestEnvConfig:
    """Tests for environment configuration."""

    def test_all_difficulties_available(self):
        """All three difficulty levels should be valid."""
        for diff in ["easy", "medium", "hard"]:
            config = EnvConfig(task_difficulty=diff)
            assert config.task_difficulty == diff

    def test_invalid_difficulty_raises(self):
        """Invalid difficulty should raise ValueError."""
        with pytest.raises(ValueError):
            SREGymEnv(EnvConfig(task_difficulty="invalid"))


class TestAssertionEngine:
    """Tests for grader assertions."""

    def test_engine_initialization(self):
        """AssertionEngine should initialize without error."""
        grader = AssertionEngine(namespace="default")
        assert grader.namespace == "default"

    def test_health_score_bounds(self):
        """compute_health_score should return [0, 1]."""
        grader = AssertionEngine()
        score = grader.compute_health_score(healthy_pods=1, total_pods=2)
        assert 0.0 <= score <= 1.0

    def test_stability_index_calculation(self):
        """Stability index formula: (uptime - penalties) / duration."""
        grader = AssertionEngine()
        stability = grader.compute_stability_index(
            episode_duration=10.0,
            system_uptime=0.8,
            penalties=0.1,
        )
        expected = (0.8 - 0.1) / 10.0
        assert abs(stability - expected) < 0.001
