#!/usr/bin/env python3
"""OpenEnv client for SRE Gym."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from openenv.core import EnvClient
from sre_gym.models import K8sAction, K8sObservation


class SREGymClient(EnvClient[K8sAction, K8sObservation, dict]):
    """Client for SRE Gym environment."""

    async def parse_observation(self, data: dict) -> K8sObservation:
        """Parse observation from server response."""
        return K8sObservation(**data)

    async def parse_reward(self, reward: float) -> float:
        """Parse reward from server response."""
        return reward
