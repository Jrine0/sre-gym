"""Hard task: Cascading latency failure across three microservices."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import Field

from sre_gym.models import K8sAction, K8sObservation, K8sState, TaskConfig
from sre_gym.grader import AssertionEngine


class HardTaskConfig(TaskConfig):
    """Configuration for the hard task."""

    difficulty: Literal["hard"] = "hard"
    name: str = "Cascading Failure - DB Timeout via LB Latency"
    description: str = (
        "Three microservices (frontend, api, db) have a cascading failure. "
        "The database times out, causing API latency spikes, which overloads the frontend. "
        "Diagnose and resolve the root cause."
    )
    max_steps: int = Field(default=30, ge=1)
    penalty_per_step: float = Field(default=0.05, ge=0.0)
    sparse_success_reward: float = Field(default=1.0, ge=0.0, le=1.0)
    namespace: str = "default"


def get_cascading_manifest() -> str:
    """Return manifests for the three-service setup with intentional issues."""
    return """---
apiVersion: v1
kind: Pod
metadata:
  name: db-primary
  namespace: default
  labels:
    app: database
    tier: backend
    sre-gym-task: "true"
spec:
  containers:
  - name: postgres
    image: postgres:15-alpine
    env:
    - name: POSTGRES_DB
      value: "appdb"
    - name: POSTGRES_USER
      value: "admin"
    - name: POSTGRES_PASSWORD
      value: "secret123"
    resources:
      limits:
        cpu: "100m"
        memory: "128Mi"
---
apiVersion: v1
kind: Pod
metadata:
  name: api-service
  namespace: default
  labels:
    app: api
    tier: middle
    sre-gym-task: "true"
spec:
  containers:
  - name: api
    image: nginx:1.25-alpine
    env:
    - name: DB_HOST
      value: "db-primary.default"
    - name: DB_TIMEOUT
      value: "100ms"
    resources:
      limits:
        cpu: "50m"
        memory: "64Mi"
---
apiVersion: v1
kind: Pod
metadata:
  name: frontend
  namespace: default
  labels:
    app: frontend
    tier: frontend
    sre-gym-task: "true"
spec:
  containers:
  - name: web
    image: nginx:1.25-alpine
    env:
    - name: API_ENDPOINT
      value: "http://api-service.default:80"
    resources:
      limits:
        cpu: "50m"
        memory: "64Mi"
"""


class HardTask:
    """Hard task: resolve cascading failure across three microservices."""

    def __init__(self, config: HardTaskConfig | None = None):
        self.config = config or HardTaskConfig()
        self.grader = AssertionEngine(namespace="default")
        self._start_time: float | None = None

    def setup(self) -> K8sState:
        """Set up cascading failure state."""
        import subprocess

        # Only run kubectl commands if available
        if subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=5,
        ).returncode == 0:
            # Delete existing pods by name (kubectl delete -f - with manifest input doesn't work for deletion)
            for pod in ["db-primary", "api-service", "frontend"]:
                subprocess.run(
                    ["kubectl", "delete", "pod", pod, "-n", "default", "--ignore-not-found"],
                    capture_output=True,
                )
            subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=get_cascading_manifest(),
                capture_output=True,
                text=True,
            )
            time.sleep(5)

        self._start_time = time.time()

        state = K8sState(
            healthy_pods=0,
            total_pods=3,
            failed_pods=["db-primary", "api-service", "frontend"],
        )
        return state

    def evaluate(self, state: K8sState) -> tuple[bool, K8sObservation]:
        """Evaluate state and return (done, observation).

        Note: Reward is computed by PBRS engine in env.py, not here.
        """
        phases = {
            name: self.grader.get_pod_phase(name)
            for name in ["db-primary", "api-service", "frontend"]
        }

        healthy_count = sum(1 for p in phases.values() if p == "Running")
        all_healthy = healthy_count == 3

        is_success = all_healthy
        done = is_success or state.step_number >= self.config.max_steps

        system_uptime = healthy_count / state.total_pods

        root_cause = None
        if not all_healthy:
            if phases.get("db-primary") != "Running":
                root_cause = "Root cause: DB pod resource-constrained. Increase memory/cpu limits."
            elif phases.get("api-service") != "Running":
                root_cause = "API service unhealthy. Check DB connectivity and timeout settings."

        obs = K8sObservation(
            kubectl_output=str(phases),
            error_message=root_cause,
            pod_status=", ".join(f"{k}={v}" for k, v in phases.items()),
            health_score=system_uptime,
            step_number=state.step_number,
        )

        return done, obs

    def get_hint(self) -> str:
        """Return a diagnostic hint for the agent (not prescriptive)."""
        return (
            "A cascading failure is affecting the microservices. "
            "Start by checking all pod statuses with 'kubectl get pods' and 'kubectl describe pod <name>' for events. "
            "Look for resource constraints (OOMKilled, CPU throttling) or connection errors. "
            "Check the database pod first as it's at the bottom of the call chain."
        )
