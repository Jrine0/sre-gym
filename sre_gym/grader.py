"""AssertionEngine for grading agent actions against real K8s cluster state."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


def _check_kubectl() -> bool:
    """Check if kubectl is available."""
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


KUBECTL_AVAILABLE = _check_kubectl()


@dataclass
class AssertionResult:
    """Result of an assertion check."""

    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: dict | None = None


class AssertionEngine:
    """Checks agent actions against real container status using kubectl."""

    def __init__(self, namespace: str = "default", context: str | None = None):
        self.namespace = namespace
        self.context = context

    def _kubectl(self, args: list[str]) -> tuple[int, str, str]:
        """Run kubectl command and return (returncode, stdout, stderr)."""
        cmd = ["kubectl"]
        if self.context:
            cmd.extend(["--context", self.context])
        cmd.extend(["-n", self.namespace] + args)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr

    def get_pod_phase(self, pod_name: str) -> str | None:
        """Get current phase of a pod."""
        if not KUBECTL_AVAILABLE:
            # Return simulated phase
            if pod_name == "backend-api":
                from sre_gym.env import SIM_STATE
                if "db-config" in SIM_STATE.configmaps:
                    return "Running"
                return "CrashLoopBackOff"
            return "CrashLoopBackOff"

        _, stdout, _ = self._kubectl([
            "get", "pod", pod_name, "-o", "jsonpath={.status.phase}"
        ])
        return stdout.strip() or None

    def get_pod_status_json(self, pod_name: str) -> dict:
        """Get full pod status as JSON."""
        if not KUBECTL_AVAILABLE:
            return {"status": {"phase": self.get_pod_phase(pod_name)}}

        _, stdout, stderr = self._kubectl([
            "get", "pod", pod_name, "-o", "json"
        ])
        if stderr:
            return {"error": stderr.strip()}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"error": "Failed to parse pod status"}

    def get_event_message(self, pod_name: str) -> str | None:
        """Get most recent event for pod."""
        if not KUBECTL_AVAILABLE:
            if pod_name == "backend-api":
                from sre_gym.env import SIM_STATE
                if "db-config" not in SIM_STATE.configmaps:
                    return "Error: couldn't find configmap: configmap/db-config not found"
            return "Waiting for pod events"

        _, stdout, _ = self._kubectl([
            "get", "events",
            "--sort-by=.lastTimestamp",
            "-o", f"jsonpath={{.items[?(@.involvedObject.name=='{pod_name}')].message}}"
        ])
        return stdout.strip() or None

    def count_running_pods(self, label_selector: str = "") -> int:
        """Count pods currently in Running state."""
        if not KUBECTL_AVAILABLE:
            return 0

        args = ["get", "pods", "-o", "jsonpath={.items[*].status.phase}"]
        if label_selector:
            args.extend(["-l", label_selector])

        _, stdout, _ = self._kubectl(args)
        phases = stdout.strip().split()
        return sum(1 for p in phases if p == "Running")

    def count_total_pods(self, label_selector: str = "") -> int:
        """Count total pods matching selector."""
        if not KUBECTL_AVAILABLE:
            return 1

        args = ["get", "pods"]
        if label_selector:
            args.extend(["-l", label_selector])
        args.append("--no-headers")

        _, stdout, _ = self._kubectl(args)
        return len(stdout.strip().split("\n")) if stdout.strip() else 0

    def assert_pod_running(self, pod_name: str) -> AssertionResult:
        """Assert a specific pod is in Running state (binary: 1.0 or 0.0)."""
        phase = self.get_pod_phase(pod_name)
        if phase == "Running":
            return AssertionResult(
                passed=True,
                score=1.0,
                message=f"Pod {pod_name} is Running"
            )
        elif phase is None:
            return AssertionResult(
                passed=False,
                score=0.0,
                message=f"Pod {pod_name} not found"
            )
        else:
            event = self.get_event_message(pod_name) or "No event available"
            return AssertionResult(
                passed=False,
                score=0.0,
                message=f"Pod {pod_name} is {phase}",
                details={"phase": phase, "event": event}
            )

    def compute_health_score(
        self,
        healthy_pods: int | None = None,
        total_pods: int | None = None,
        label_selector: str = "",
    ) -> float:
        """Compute health score = healthy_pods / total_pods."""
        if healthy_pods is None:
            healthy_pods = self.count_running_pods(label_selector)
        if total_pods is None:
            total_pods = self.count_total_pods(label_selector) or 1
        return healthy_pods / total_pods

    def compute_stability_index(
        self,
        episode_duration: float,
        system_uptime: float,
        penalties: float,
    ) -> float:
        """Compute Stability Index = (System Uptime - Penalties) / Episode Duration."""
        if episode_duration <= 0:
            return 0.0
        return max(0.0, (system_uptime - penalties) / episode_duration)
