"""OpenEnv-compliant SRE Gym environment for K8s troubleshooting."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

from sre_gym.models import K8sAction, K8sObservation, K8sState, TaskConfig
from sre_gym.rewards import PBRSConfig, PBRSEngine, RewardBreakdown
from sre_gym.grader import AssertionEngine
from sre_gym.tasks.easy import EasyTask
from sre_gym.tasks.medium import MediumTask
from sre_gym.tasks.hard import HardTask


def _check_kubectl_available() -> bool:
    """Check if kubectl is available and cluster is reachable."""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Check at module load
KUBECTL_AVAILABLE = _check_kubectl_available()


class SimState:
    """Simulates K8s state when kubectl is not available."""

    def __init__(self):
        self.configmaps: set[str] = set()
        self.pods_fixed: set[str] = set()
        self.pods_deleted: set[str] = set()

    def reset(self):
        self.configmaps.clear()
        self.pods_fixed.clear()
        self.pods_deleted.clear()

    def apply_manifest(self, manifest: str) -> str:
        """Simulate applying a manifest."""
        if not manifest:
            return "Error: No manifest"

        lines = manifest.split("\n")
        in_configmap = False
        cm_name = None

        for line in lines:
            line = line.strip()
            if line == "---":
                in_configmap = False
                cm_name = None
            if line.startswith("kind:"):
                if "ConfigMap" in line:
                    in_configmap = True
            if line.startswith("name:") and in_configmap and cm_name is None:
                cm_name = line.split("name:")[1].strip()
                self.configmaps.add(cm_name)
                return f"configmap/{cm_name} created"

        # Check for pod names
        for line in lines:
            if line.startswith("name:"):
                pod_name = line.split("name:")[1].strip()
                if "api" in pod_name or "app" in pod_name or "frontend" in pod_name:
                    if "backend-api" in self.configmaps or "db-config" in self.configmaps:
                        self.pods_fixed.add(pod_name)

        return "manifest applied"

    def delete_resource(self, kind: str, name: str) -> str:
        """Simulate deleting a resource."""
        if kind == "pod":
            self.pods_deleted.add(name)
        return f"{kind}/{name} deleted"

    def get_pod_phase(self, pod_name: str) -> str:
        """Get simulated pod phase."""
        if pod_name == "backend-api":
            if "db-config" in self.configmaps:
                return "Running"
            return "CrashLoopBackOff"
        elif pod_name == "memory-app":
            return "OOMKilled"
        elif pod_name in ("db-primary", "api-service", "frontend"):
            return "CrashLoopBackOff"
        return "Unknown"


# Global simulation state
SIM_STATE = SimState()


@dataclass
class EnvConfig:
    """Configuration for the SRE Gym environment."""

    task_difficulty: str = "easy"  # easy | medium | hard
    namespace: str = "default"
    max_steps: int = 30
    penalty_per_step: float = 0.05
    sparse_success_reward: float = 1.0
    shaping_alpha: float = 0.1
    shaping_gamma: float = 0.99


class SREGymEnv:
    """OpenEnv-compliant environment for K8s SRE training.

    Implements the standard RL interface: step(), reset(), close()

    Usage:
        env = SREGymEnv(task="easy")
        obs = env.reset()
        for _ in range(20):
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
        env.close()
    """

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self._task_instance = self._create_task()
        self._grader = AssertionEngine(namespace=self.config.namespace)
        self._reward_engine = PBRSEngine(
            PBRSConfig(
                alpha=self.config.shaping_alpha,
                gamma=self.config.shaping_gamma,
                penalty_per_step=self.config.penalty_per_step,
            )
        )
        self._state: K8sState | None = None
        self._episode_start: float | None = None

    def _create_task(self) -> EasyTask | MediumTask | HardTask:
        """Instantiate the appropriate task based on difficulty."""
        match self.config.task_difficulty:
            case "easy":
                from sre_gym.tasks.easy import EasyTaskConfig
                return EasyTask(EasyTaskConfig())
            case "medium":
                from sre_gym.tasks.medium import MediumTaskConfig
                return MediumTask(MediumTaskConfig())
            case "hard":
                from sre_gym.tasks.hard import HardTaskConfig
                return HardTask(HardTaskConfig())
            case _:
                raise ValueError(
                    f"Unknown difficulty: {self.config.task_difficulty}. "
                    "Use: easy | medium | hard"
                )

    def reset(self, task_id: str | None = None) -> K8sObservation:
        """Reset environment to initial state, return first observation.

        Args:
            task_id: Optional task identifier (easy/medium/hard) for OpenEnv API.
                    If provided, overrides the configured task_difficulty.
        """
        self._reward_engine.reset()

        # Reset simulation state if not using kubectl
        if not KUBECTL_AVAILABLE:
            SIM_STATE.reset()

        # Support task_id from OpenEnv validator
        if task_id and task_id in ("easy", "medium", "hard"):
            self.config.task_difficulty = task_id
            self._task_instance = self._create_task()

        # Clean up any existing pods from previous episode
        self._cleanup()
        time.sleep(1)

        try:
            self._state = self._task_instance.setup()
        except Exception as e:
            # In simulation mode, create a default state
            self._state = K8sState(
                healthy_pods=0,
                total_pods=1,
                failed_pods=["backend-api"],
            )

        self._episode_start = time.time()
        self._state.episode_start_ts = self._episode_start

        # Initial evaluation to get observation
        try:
            _, obs = self._task_instance.evaluate(self._state)
        except Exception as e:
            # Fallback observation
            obs = K8sObservation(
                kubectl_output=f"Error in evaluate: {e}",
                pod_status="CrashLoopBackOff",
                health_score=0.0,
                step_number=0,
            )

        obs.step_number = 0
        self._state.step_number = 0
        return obs

    def step(self, action: K8sAction) -> tuple[K8sObservation, float, bool, dict]:
        """Execute one step of the environment.

        Args:
            action: K8sAction selected by the agent

        Returns:
            (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        step_number = self._state.step_number + 1
        self._state.step_number = step_number

        # Execute action
        kubectl_output = self._execute_action(action)
        time.sleep(1)  # Allow cluster to settle

        # Re-evaluate state after action
        done, obs = self._task_instance.evaluate(self._state)

        # Update internal state
        self._state.healthy_pods = 1 if obs.health_score >= 1.0 else 0
        self._state.failed_pods = [] if obs.health_score >= 1.0 else self._state.failed_pods

        # Compute PBRS reward
        reward_breakdown = self._reward_engine.compute_reward(
            healthy_pods=self._state.healthy_pods,
            total_pods=self._state.total_pods,
            is_terminal=done,
            is_success=obs.health_score >= 1.0,
        )

        # Apply step penalty
        self._state.penalties_accumulated += abs(reward_breakdown.step_penalty)

        obs.step_number = step_number
        obs.kubectl_output = kubectl_output

        info = {
            "reward_breakdown": reward_breakdown.to_dict(),
            "state": {
                "healthy_pods": self._state.healthy_pods,
                "total_pods": self._state.total_pods,
                "penalties": self._state.penalties_accumulated,
            },
        }

        return obs, reward_breakdown.total, done, info

    def _execute_action(self, action: K8sAction) -> str:
        """Execute a kubectl action and return output."""
        # Use simulation mode if kubectl not available
        if not KUBECTL_AVAILABLE:
            return self._execute_action_sim(action)

        cmd = ["kubectl"]
        match action.action_type.value:
            case "apply_manifest":
                if not action.manifest:
                    return "Error: No manifest provided for apply_manifest"
                result = subprocess.run(
                    cmd + ["apply", "-f", "-"],
                    input=action.manifest,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr

            case "delete_resource":
                if not action.resource_kind or not action.resource_name:
                    return "Error: resource_kind and resource_name required"
                result = subprocess.run(
                    cmd + [
                        "delete",
                        action.resource_kind,
                        action.resource_name,
                        "-n", action.namespace,
                        "--ignore-not-found",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr

            case "scale_deployment":
                if not action.resource_name:
                    return "Error: resource_name required for scale"
                replicas = action.options.get("replicas", 1)
                result = subprocess.run(
                    cmd + [
                        "scale", "deployment", action.resource_name,
                        "--replicas", str(replicas),
                        "-n", action.namespace,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr

            case "edit_resource":
                return "edit_resource: use kubectl edit manually or apply corrected manifest"

            case "exec_command":
                if not action.resource_name or not action.command:
                    return "Error: resource_name and command required for exec"
                result = subprocess.run(
                    cmd + [
                        "exec", "-it", action.resource_name,
                        "-n", action.namespace, "--",
                    ] + action.command,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr

            case "noop":
                return "No-op: agent chose to wait and observe"

            case _:
                return f"Unknown action type: {action.action_type}"

    def _execute_action_sim(self, action: K8sAction) -> str:
        """Execute action in simulation mode (when kubectl unavailable)."""
        match action.action_type.value:
            case "apply_manifest":
                if not action.manifest:
                    return "Error: No manifest"
                return SIM_STATE.apply_manifest(action.manifest)

            case "delete_resource":
                if not action.resource_kind or not action.resource_name:
                    return "Error: resource_kind and resource_name required"
                return SIM_STATE.delete_resource(action.resource_kind, action.resource_name)

            case "noop":
                return "No-op: agent chose to wait and observe"

            case _:
                return f"Action {action.action_type.value} simulated"

    def _cleanup(self) -> None:
        """Remove pods/resources from previous episode using label selectors."""
        if not KUBECTL_AVAILABLE:
            # Skip cleanup if kubectl not available
            return

        # Delete pods created by SRE Gym tasks using label selector
        subprocess.run(
            ["kubectl", "delete", "pods", "-n", "default", "-l", "sre-gym-task=true", "--ignore-not-found"],
            capture_output=True,
        )
        # Also clean up by known pod names for backward compatibility
        pods = ["backend-api", "memory-app", "db-primary", "api-service", "frontend"]
        for pod in pods:
            subprocess.run(
                ["kubectl", "delete", "pod", pod, "-n", "default", "--ignore-not-found"],
                capture_output=True,
            )
        # Clean up ConfigMap
        subprocess.run(
            ["kubectl", "delete", "configmap", "db-config", "-n", "default", "--ignore-not-found"],
            capture_output=True,
        )

    def close(self) -> None:
        """Clean up resources when done."""
        self._cleanup()

    def state(self) -> dict[str, Any]:
        """Return current state for OpenEnv API."""
        if self._state is None:
            return {}
        return {
            "step_number": self._state.step_number,
            "healthy_pods": self._state.healthy_pods,
            "total_pods": self._state.total_pods,
            "failed_pods": self._state.failed_pods,
            "penalties_accumulated": self._state.penalties_accumulated,
        }

    @property
    def observation_space(self) -> dict[str, Any]:
        """Return observation space definition."""
        return {
            "kubectl_output": str,
            "error_message": str | None,
            "pod_status": str | None,
            "health_score": float,
            "step_number": int,
        }

    @property
    def action_space(self) -> dict[str, Any]:
        """Return action space definition."""
        return {"description": "K8sAction with action_type, manifest, resource identifiers"}

    def render(self, mode: str = "human") -> str | None:
        """Render current state (for debugging/visualization)."""
        if self._state:
            return (
                f"Step {self._state.step_number} | "
                f"Healthy: {self._state.healthy_pods}/{self._state.total_pods} | "
                f"Penalties: {self._state.penalties_accumulated:.3f}"
            )
        return None
