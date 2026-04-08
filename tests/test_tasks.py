"""Tests for task implementations."""

import pytest
from sre_gym.tasks.easy import EasyTask, EasyTaskConfig
from sre_gym.tasks.medium import MediumTask, MediumTaskConfig
from sre_gym.tasks.hard import HardTask, HardTaskConfig


class TestEasyTask:
    """Tests for easy task (ConfigMap missing)."""

    def test_config_defaults(self):
        """Easy task config should have correct defaults."""
        config = EasyTaskConfig()
        assert config.difficulty == "easy"
        assert config.max_steps == 15
        assert config.sparse_success_reward == 1.0

    def test_task_initialization(self):
        """Task should initialize without error."""
        task = EasyTask()
        assert task.config.difficulty == "easy"
        assert task.config.name == "CrashLoopBackOff - Missing ConfigMap"

    def test_hint_contains_configmap(self):
        """Hint should reference ConfigMap fix."""
        task = EasyTask()
        hint = task.get_hint()
        assert "ConfigMap" in hint or "configmap" in hint.lower()


class TestMediumTask:
    """Tests for medium task (OOMKilled)."""

    def test_config_defaults(self):
        """Medium task config should have correct defaults."""
        config = MediumTaskConfig()
        assert config.difficulty == "medium"
        assert config.max_steps == 20

    def test_task_initialization(self):
        """Task should initialize without error."""
        task = MediumTask()
        assert task.config.difficulty == "medium"

    def test_hint_mentions_memory(self):
        """Hint should reference memory limits."""
        task = MediumTask()
        hint = task.get_hint()
        assert "memory" in hint.lower() or "Mi" in hint


class TestHardTask:
    """Tests for hard task (cascading failure)."""

    def test_config_defaults(self):
        """Hard task config should have correct defaults."""
        config = HardTaskConfig()
        assert config.difficulty == "hard"
        assert config.max_steps == 30
        assert config.namespace == "default"

    def test_task_initialization(self):
        """Task should initialize without error."""
        task = HardTask()
        assert task.config.difficulty == "hard"
        assert task.grader is not None  # Just check grader is initialized

    def test_manifest_defines_three_services(self):
        """Hard task manifest should define 3 microservices."""
        from sre_gym.tasks.hard import get_cascading_manifest
        manifest = get_cascading_manifest()
        assert "db-primary" in manifest
        assert "api-service" in manifest
        assert "frontend" in manifest


class TestCurriculumProgression:
    """Tests for curriculum difficulty ladder."""

    def test_difficulty_escalation(self):
        """Harder tasks should have more steps."""
        easy = EasyTaskConfig()
        medium = MediumTaskConfig()
        hard = HardTaskConfig()
        assert easy.max_steps < medium.max_steps < hard.max_steps

    def test_rewards_in_range(self):
        """All tasks should have rewards in [0, 1]."""
        configs = [EasyTaskConfig(), MediumTaskConfig(), HardTaskConfig()]
        for config in configs:
            assert 0.0 <= config.sparse_success_reward <= 1.0
            assert config.penalty_per_step >= 0.0
