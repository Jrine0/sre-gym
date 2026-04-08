#!/usr/bin/env python3
"""OpenEnv FastAPI server for SRE Gym."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sre_gym.env import SREGymEnv, EnvConfig
from openenv.core.env_server import create_fastapi_app
from sre_gym.models import K8sAction, K8sObservation


# Create the environment instance
env = SREGymEnv(EnvConfig(task_difficulty="easy"))

# Create FastAPI app with OpenEnv
app = create_fastapi_app(env, K8sAction, K8sObservation)


def run():
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
