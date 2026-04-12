"""Easy task: Pod CrashLoopBackOff due to missing ConfigMap."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field

from sre_gym.models import K8sAction, K8sObservation, K8sState, TaskConfig, clamp_score
from sre_gym.grader import AssertionEngine


class EasyTaskConfig(TaskConfig):
    """Configuration for the easy task."""

    difficulty: Literal["easy"] = "easy"
    name: str = "CrashLoopBackOff - Missing ConfigMap"
    description: str = (
        "A pod is in CrashLoopBackOff because it references a ConfigMap "
        "that doesn't exist. Fix by creating the missing ConfigMap."
    )
    max_steps: int = Field(default=15, ge=1)
    penalty_per_step: float = Field(default=0.05, ge=0.0)
    sparse_success_reward: float = Field(default=1.0, ge=0.0, le=1.0)
    namespace: str = "default"


def get_broken_manifest(config: EasyTaskConfig) -> str:
    """Return the broken pod manifest that references non-existent ConfigMap."""
    return """apiVersion: v1
kind: Pod
metadata:
  name: backend-api
  namespace: default
  labels:
    sre-gym-task: "true"
spec:
  containers:
  - name: api
    image: nginx:1.25
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: db-config
          key: connection_string
"""


def get_fix_manifest() -> str:
    """Return the correct ConfigMap + fixed pod manifest."""
    return """apiVersion: v1
kind: ConfigMap
metadata:
  name: db-config
  namespace: default
data:
  connection_string: "postgresql://localhost:5432/mydb"
---
apiVersion: v1
kind: Pod
metadata:
  name: backend-api
  namespace: default
  labels:
    sre-gym-task: "true"
spec:
  containers:
  - name: api
    image: nginx:1.25
    env:
    - name: DATABASE_URL
      valueFrom:
        configMapKeyRef:
          name: db-config
          key: connection_string
"""


class EasyTask:
    """Easy task: fix CrashLoopBackOff by creating missing ConfigMap."""

    def __init__(self, config: EasyTaskConfig | None = None):
        self.config = config or EasyTaskConfig()
        self.grader = AssertionEngine(namespace=self.config.namespace)

    def setup(self) -> K8sState:
        """Set up the broken state: apply broken manifest, delete ConfigMap."""
        import subprocess

        # Only run kubectl commands if available
        if subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=5,
        ).returncode == 0:
            # Apply broken pod (might already exist)
            broken = get_broken_manifest(self.config)
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=broken,
                capture_output=True,
                text=True,
            )
            # Delete the ConfigMap to ensure it's missing
            subprocess.run(
                ["kubectl", "delete", "configmap", "db-config", "-n", "default", "--ignore-not-found"],
                capture_output=True,
            )
            time.sleep(2)  # Allow pod to start crashing

        state = K8sState(
            healthy_pods=0,
            total_pods=1,
            failed_pods=["backend-api"],
        )
        return state

    def evaluate(self, state: K8sState) -> tuple[bool, K8sObservation]:
        """Evaluate current state and return (done, observation).

        Note: Reward is computed by PBRS engine in env.py, not here.
        """
        phase = self.grader.get_pod_phase("backend-api")
        healthy = 1 if phase == "Running" else 0

        is_success = healthy == 1 and phase == "Running"
        done = is_success or state.step_number >= self.config.max_steps

        if is_success:
            obs = K8sObservation(
                kubectl_output=f"Pod backend-api is {phase}",
                pod_status=phase,
                health_score=clamp_score(1.0),
                step_number=state.step_number,
            )
        else:
            event = self.grader.get_event_message("backend-api") or ""
            error_msg = None
            if "NotFound" in event or "ConfigMap" in event:
                error_msg = "ConfigMap 'db-config' not found. Create it with kubectl create configmap."
            elif phase == "CrashLoopBackOff":
                error_msg = "Pod in CrashLoopBackOff. Check if referenced ConfigMap exists."

            obs = K8sObservation(
                kubectl_output=event or f"Pod status: {phase}",
                error_message=error_msg,
                pod_status=phase,
                health_score=clamp_score(healthy / state.total_pods),
                step_number=state.step_number,
            )

        return done, obs

    def get_hint(self) -> str:
        """Return a diagnostic hint for the agent (not prescriptive)."""
        return (
            "The pod 'backend-api' is failing. Use kubectl describe to inspect events and see what's wrong. "
            "Look for error messages about missing resources. "
            "Check what resources the pod references using: kubectl get pod backend-api -o yaml | grep -A 10 configMapKeyRef"
        )
