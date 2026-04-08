"""Medium task: OOMKilled pod triage by adjusting resource limits."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field

from sre_gym.models import K8sAction, K8sObservation, K8sState, TaskConfig
from sre_gym.grader import AssertionEngine


class MediumTaskConfig(TaskConfig):
    """Configuration for the medium task."""

    difficulty: Literal["medium"] = "medium"
    name: str = "OOMKilled - Memory Limit Triage"
    description: str = (
        "A pod is OOMKilled because its memory limit is set too low. "
        "Fix by adjusting the memory limit to an appropriate value."
    )
    max_steps: int = Field(default=20, ge=1)
    penalty_per_step: float = Field(default=0.05, ge=0.0)
    sparse_success_reward: float = Field(default=1.0, ge=0.0, le=1.0)
    namespace: str = "default"


def get_oom_manifest(memory_limit: str = "64Mi") -> str:
    """Return pod manifest with restrictive memory limit."""
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: memory-app
  namespace: default
  labels:
    sre-gym-task: "true"
spec:
  containers:
  - name: app
    image: polinux/stress:latest
    args:
    - vm
    - "1"
    - --vm-bytes
    - "150M"
    - --vm-hang
    - "1"
    resources:
      requests:
        memory: "64Mi"
      limits:
        memory: "{memory_limit}"
"""


class MediumTask:
    """Medium task: fix OOMKilled pod by adjusting memory limits."""

    def __init__(self, config: MediumTaskConfig | None = None):
        self.config = config or MediumTaskConfig()
        self.grader = AssertionEngine(namespace="default")

    def setup(self) -> K8sState:
        """Set up broken state: apply pod with too-low memory limit."""
        import subprocess

        manifest = get_oom_manifest("64Mi")
        subprocess.run(
            ["kubectl", "delete", "pod", "memory-app", "--ignore-not-found"],
            capture_output=True,
        )
        subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=manifest,
            capture_output=True,
            text=True,
        )
        time.sleep(3)  # Allow stress workload to consume memory

        state = K8sState(
            healthy_pods=0,
            total_pods=1,
            failed_pods=["memory-app"],
        )
        return state

    def evaluate(self, state: K8sState) -> tuple[bool, K8sObservation]:
        """Evaluate state and return (done, observation).

        Note: Reward is computed by PBRS engine in env.py, not here.
        """
        phase = self.grader.get_pod_phase("memory-app")
        healthy = 1 if phase == "Running" else 0

        is_success = healthy == 1
        done = is_success or state.step_number >= self.config.max_steps

        if is_success:
            obs = K8sObservation(
                kubectl_output=f"Pod memory-app is {phase}",
                pod_status=phase,
                health_score=1.0,
                step_number=state.step_number,
            )
        else:
            event = self.grader.get_event_message("memory-app") or ""
            is_oom = "OOMKilled" in event or phase == "Unknown"

            error_msg = None
            if is_oom:
                error_msg = "Pod OOMKilled. Increase memory limit: use 'Mi' not 'MB'. Example: limits.memory: 256Mi"
            elif phase == "CrashLoopBackOff":
                error_msg = "Pod crashing. Check resource limits and ensure values use 'Mi' suffix."

            obs = K8sObservation(
                kubectl_output=event or f"Pod status: {phase}",
                error_message=error_msg,
                pod_status=phase,
                health_score=healthy / state.total_pods,
                step_number=state.step_number,
            )

        return done, obs

    def get_hint(self) -> str:
        """Return a diagnostic hint for the agent (not prescriptive)."""
        return (
            "The pod 'memory-app' is experiencing issues. "
            "Use 'kubectl describe pod memory-app' to check events for OOMKilled or resource warnings. "
            "Use 'kubectl get pod memory-app -o yaml' to inspect current resource limits. "
            "Memory values in Kubernetes should use the 'Mi' suffix (mebibytes)."
        )
