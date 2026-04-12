"""OpenEnv-compliant SRE Gym environment for K8s troubleshooting."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any

from sre_gym.models import K8sAction, K8sObservation, K8sState, TaskConfig, clamp_score
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
    """Standalone simulation state — zero external dependencies.

    When kubectl is unavailable, this class handles ALL environment logic:
    action execution, state evaluation, and observation generation.
    No task objects, no grader imports, no subprocess calls.
    """

    def __init__(self):
        self.configmaps: set[str] = set()
        self.pods_deleted: set[str] = set()
        self.task_difficulty: str = "easy"
        self._total_pods: int = 1
        self._failed_pods: list[str] = []
        # Track what pods have been applied (keyed by pod name)
        self._pod_manifests: dict[str, dict] = {}

    def reset(self, difficulty: str) -> None:
        """Reset simulation for the given difficulty.

        Sets up the broken initial state that the agent must fix.
        """
        self.configmaps.clear()
        self.pods_deleted.clear()
        self._pod_manifests.clear()
        self.task_difficulty = difficulty

        if difficulty == "easy":
            self._total_pods = 1
            self._failed_pods = ["backend-api"]
        elif difficulty == "medium":
            self._total_pods = 1
            self._failed_pods = ["memory-app"]
            # memory-app starts with 64Mi limit (too low, causes OOM)
            self._pod_manifests["memory-app"] = {"memory_limit": "64Mi"}
        elif difficulty == "hard":
            self._total_pods = 3
            self._failed_pods = ["db-primary", "api-service", "frontend"]
        else:
            self._total_pods = 1
            self._failed_pods = ["backend-api"]

    def _parse_memory_limit(self, manifest: str) -> str | None:
        """Extract memory limit from a pod manifest, or None if not found."""
        lines = manifest.split("\n")
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("memory:"):
                # e.g. "memory: 256Mi" or "memory: \"256Mi\""
                parts = line.split(":", 1)[1].strip().strip('"').strip("'")
                return parts
        return None

    def apply_manifest(self, manifest: str) -> str:
        """Execute apply_manifest in simulation mode."""
        if not manifest:
            return "Error: No manifest provided"

        lines = manifest.split("\n")
        created = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("kind:"):
                kind = line.split(":", 1)[1].strip()
                idx = lines.index(line)

                if kind == "ConfigMap":
                    for j in range(idx + 1, min(idx + 10, len(lines))):
                        name_line = lines[j].strip()
                        if name_line.startswith("name:"):
                            cm_name = name_line.split(":", 1)[1].strip()
                            self.configmaps.add(cm_name)
                            created.append(f"configmap/{cm_name} created")
                            break

                elif kind == "Pod":
                    pod_name = None
                    memory_limit = None
                    for j in range(idx + 1, min(idx + 20, len(lines))):
                        name_line = lines[j].strip()
                        if name_line.startswith("name:"):
                            pod_name = name_line.split(":", 1)[1].strip()
                        elif "memory" in name_line and memory_limit is None:
                            memory_limit = self._parse_memory_limit(name_line)
                    if pod_name:
                        # Update manifest for this pod
                        self._pod_manifests[pod_name] = {"memory_limit": memory_limit}
                        # Remove from deleted list (pod will be re-created)
                        self.pods_deleted.discard(pod_name)
                        created.append(f"pod/{pod_name} created")

        if created:
            return "\n".join(created)
        return "manifest applied"

    def delete_resource(self, kind: str, name: str) -> str:
        """Execute delete_resource in simulation mode."""
        if kind == "pod":
            self.pods_deleted.add(name)
        return f"{kind}/{name} deleted"

    def scale_deployment(self, name: str, replicas: int) -> str:
        """Execute scale_deployment in simulation mode."""
        return f"deployment.apps/{name} scaled"

    def exec_command(self, name: str, command: list[str]) -> str:
        """Execute exec_command in simulation mode."""
        return f"exec completed on {name}: {' '.join(command)}"

    def get_pod_phase(self, pod_name: str) -> str:
        """Return simulated pod phase based on task and current state."""
        if pod_name in self.pods_deleted:
            return "Terminating"

        if self.task_difficulty == "easy":
            if pod_name == "backend-api":
                return "Running" if "db-config" in self.configmaps else "CrashLoopBackOff"
        elif self.task_difficulty == "medium":
            if pod_name == "memory-app":
                manifest = self._pod_manifests.get(pod_name, {})
                limit = manifest.get("memory_limit", "64Mi")
                if limit and self._memory_ok(limit):
                    return "Running"
                return "OOMKilled"
        elif self.task_difficulty == "hard":
            phases = {
                "db-primary": "CrashLoopBackOff",
                "api-service": "CrashLoopBackOff",
                "frontend": "CrashLoopBackOff",
            }
            return phases.get(pod_name, "Unknown")

        return "Running"

    def _memory_ok(self, limit: str) -> bool:
        """Check if memory limit is sufficient (>64Mi counts as OK)."""
        if not limit:
            return False
        import re
        m = re.match(r"(\d+)(Mi|Gi)", limit)
        if not m:
            return False
        value, unit = int(m.group(1)), m.group(2)
        if unit == "Gi":
            value *= 1024
        return value > 64

    def evaluate(self, step_number: int) -> tuple[bool, K8sObservation]:
        """Evaluate current simulation state and return (done, observation).

        This replaces all task.evaluate() calls when kubectl is unavailable.
        """
        difficulty = self.task_difficulty

        if difficulty == "easy":
            phase = self.get_pod_phase("backend-api")
            healthy = 1 if phase == "Running" else 0
            is_success = healthy == 1
            done = is_success or step_number >= 15
            kubectl_output = f"Pod backend-api is {phase}"
            error_msg = None
            if not is_success:
                if "db-config" not in self.configmaps:
                    error_msg = "ConfigMap 'db-config' not found. Create it with kubectl create configmap."
                else:
                    error_msg = f"Pod backend-api is {phase}. Check events for details."
            health_score = healthy / self._total_pods
            pod_status = phase

        elif difficulty == "medium":
            phase = self.get_pod_phase("memory-app")
            healthy = 1 if phase == "Running" else 0
            is_success = healthy == 1
            done = is_success or step_number >= 20
            kubectl_output = f"Pod memory-app is {phase}"
            error_msg = None
            if not is_success:
                if phase == "OOMKilled":
                    error_msg = "Pod OOMKilled. Increase memory limit: use 'Mi' not 'MB'. Example: limits.memory: 256Mi"
                else:
                    error_msg = f"Pod memory-app is {phase}. Check resource limits."
            health_score = healthy / self._total_pods
            pod_status = phase

        elif difficulty == "hard":
            phases = {name: self.get_pod_phase(name) for name in ["db-primary", "api-service", "frontend"]}
            healthy_count = sum(1 for p in phases.values() if p == "Running")
            is_success = healthy_count == 3
            done = is_success or step_number >= 30
            kubectl_output = str(phases)
            error_msg = None
            if not is_success:
                if phases.get("db-primary") != "Running":
                    error_msg = "Root cause: DB pod CrashLoopBackOff. Increase memory/cpu limits."
                elif phases.get("api-service") != "Running":
                    error_msg = "API service unhealthy. Check DB connectivity."
            health_score = healthy_count / self._total_pods
            pod_status = ", ".join(f"{k}={v}" for k, v in phases.items())
        else:
            is_success = False
            done = step_number >= 15
            kubectl_output = "Unknown task"
            error_msg = "Unknown task difficulty"
            health_score = 0.0
            pod_status = "Unknown"

        obs = K8sObservation(
            kubectl_output=kubectl_output,
            error_message=error_msg,
            pod_status=pod_status,
            health_score=health_score,
            step_number=step_number,
        )
        return done, obs


# Global simulation state — single instance
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
        # Always validate difficulty — this ensures ValueError is raised even
        # in simulation mode (when kubectl is unavailable and we skip task creation)
        if self.config.task_difficulty not in ("easy", "medium", "hard"):
            raise ValueError(
                f"Unknown difficulty: {self.config.task_difficulty}. "
                "Use: easy | medium | hard"
            )
        # In simulation mode, task/grader are not needed — reset() handles everything.
        # Only create them when kubectl is available.
        self._task_instance: EasyTask | MediumTask | HardTask | None = None
        self._grader: AssertionEngine | None = None
        self._state: K8sState | None = None
        self._episode_start: float | None = None

        # PBRS engine — lightweight, no external deps, always safe to init
        self._reward_engine = PBRSEngine(
            PBRSConfig(
                alpha=self.config.shaping_alpha,
                gamma=self.config.shaping_gamma,
                penalty_per_step=self.config.penalty_per_step,
            )
        )

        # Task/grader only needed for real cluster mode
        if KUBECTL_AVAILABLE:
            self._init_kubectl_mode()

    def _init_kubectl_mode(self) -> None:
        """Initialize task and grader (only when kubectl is available)."""
        self._task_instance = self._create_task()
        self._grader = AssertionEngine(namespace=self.config.namespace)

    def _create_task(self) -> EasyTask | MediumTask | HardTask:
        """Instantiate the appropriate task based on difficulty."""
        match self.config.task_difficulty:
            case "easy":
                from sre_gym.tasks.easy import EasyTask, EasyTaskConfig
                return EasyTask(EasyTaskConfig())
            case "medium":
                from sre_gym.tasks.medium import MediumTask, MediumTaskConfig
                return MediumTask(MediumTaskConfig())
            case "hard":
                from sre_gym.tasks.hard import HardTask, HardTaskConfig
                return HardTask(HardTaskConfig())
            case _:
                raise ValueError(
                    f"Unknown difficulty: {self.config.task_difficulty}. "
                    "Use: easy | medium | hard"
                )

    def reset(self, task_id: str | None = None) -> K8sObservation:
        """Reset environment to initial state, return first observation.

        When kubectl is unavailable, uses fully-standalone simulation mode with
        zero external dependencies.
        """
        # Override difficulty if task_id provided
        if task_id and task_id in ("easy", "medium", "hard"):
            self.config.task_difficulty = task_id

        difficulty = self.config.task_difficulty

        # --- SIMULATION MODE (kubectl unavailable) ---
        if not KUBECTL_AVAILABLE:
            self._episode_start = time.time()

            # Reset simulation state for this difficulty
            SIM_STATE.reset(difficulty)

            # Set up internal state matching simulation
            self._state = K8sState(
                healthy_pods=0,
                total_pods=SIM_STATE._total_pods,
                failed_pods=list(SIM_STATE._failed_pods),
                penalties_accumulated=0.0,
                episode_start_ts=self._episode_start,
                step_number=0,
            )

            # Evaluate initial broken state
            done, obs = SIM_STATE.evaluate(0)
            obs.health_score = clamp_score(obs.health_score)

            # Reset PBRS engine
            if self._reward_engine is not None:
                self._reward_engine.reset()

            obs.step_number = 0
            return obs

        # --- KUBECTL MODE (real cluster) ---
        if self._reward_engine is not None:
            self._reward_engine.reset()

        # Recreate task if difficulty changed
        if task_id and task_id in ("easy", "medium", "hard"):
            self._task_instance = self._create_task()

        # Clean up previous episode
        self._cleanup()
        time.sleep(1)

        self._state = self._task_instance.setup()
        self._episode_start = time.time()
        self._state.episode_start_ts = self._episode_start

        _, obs = self._task_instance.evaluate(self._state)
        obs.health_score = self._clamp_score(obs.health_score)
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

        # --- SIMULATION MODE ---
        if not KUBECTL_AVAILABLE:
            kubectl_output = self._execute_action_sim(action)
            time.sleep(0.1)

            # Evaluate using simulation state
            done, obs = SIM_STATE.evaluate(step_number)

            # Update internal state
            self._state.healthy_pods = 1 if obs.health_score >= 1.0 else 0
            self._state.failed_pods = [] if obs.health_score >= 1.0 else list(SIM_STATE._failed_pods)

            # Compute PBRS reward
            if self._reward_engine is not None:
                reward_breakdown = self._reward_engine.compute_reward(
                    healthy_pods=self._state.healthy_pods,
                    total_pods=self._state.total_pods,
                    is_terminal=done,
                    is_success=obs.health_score >= 1.0,
                )
            else:
                reward_breakdown = RewardBreakdown(
                    shaping_reward=0.0,
                    step_penalty=-0.05,
                    sparse_reward=1.0 if done and obs.health_score >= 1.0 else 0.0,
                    total=1.0 if done and obs.health_score >= 1.0 else -0.05,
                )

            self._state.penalties_accumulated += abs(reward_breakdown.step_penalty)
            obs.step_number = step_number
            obs.kubectl_output = kubectl_output

            # Clamp all scores to strict (0, 1) for validator compliance
            final_reward = clamp_score(reward_breakdown.total)
            obs.health_score = clamp_score(obs.health_score)

            info = {
                "reward_breakdown": reward_breakdown.to_dict(),
                "state": {
                    "healthy_pods": self._state.healthy_pods,
                    "total_pods": self._state.total_pods,
                    "penalties": self._state.penalties_accumulated,
                },
            }
            return obs, final_reward, done, info

        # --- KUBECTL MODE ---
        kubectl_output = self._execute_action(action)
        time.sleep(1)

        done, obs = self._task_instance.evaluate(self._state)

        # Update internal state
        self._state.healthy_pods = 1 if obs.health_score >= 1.0 else 0
        self._state.failed_pods = [] if obs.health_score >= 1.0 else self._state.failed_pods

        reward_breakdown = self._reward_engine.compute_reward(
            healthy_pods=self._state.healthy_pods,
            total_pods=self._state.total_pods,
            is_terminal=done,
            is_success=obs.health_score >= 1.0,
        )

        self._state.penalties_accumulated += abs(reward_breakdown.step_penalty)
        obs.step_number = step_number
        obs.kubectl_output = kubectl_output

        # Clamp all scores to strict (0, 1) for validator compliance
        final_reward = self._clamp_score(reward_breakdown.total)
        obs.health_score = self._clamp_score(obs.health_score)

        info = {
            "reward_breakdown": reward_breakdown.to_dict(),
            "state": {
                "healthy_pods": self._state.healthy_pods,
                "total_pods": self._state.total_pods,
                "penalties": self._state.penalties_accumulated,
            },
        }

        return obs, final_reward, done, info

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

            case "scale_deployment":
                replicas = action.options.get("replicas", 1) if action.options else 1
                return SIM_STATE.scale_deployment(action.resource_name or "", replicas)

            case "exec_command":
                return SIM_STATE.exec_command(action.resource_name or "", action.command or [])

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
