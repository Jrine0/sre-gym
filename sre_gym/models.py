"""Pydantic models for K8s actions, observations, and state."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, JsonValue


class K8sActionType(str, Enum):
    APPLY_MANIFEST = "apply_manifest"
    DELETE_RESOURCE = "delete_resource"
    SCALEDeployment = "scale_deployment"
    EDIT_RESOURCE = "edit_resource"
    EXEC_COMMAND = "exec_command"
    NOOP = "noop"


class K8sAction(BaseModel):
    """Action an agent can take in the SRE gym environment."""

    action_type: K8sActionType = Field(description="Type of kubectl action")
    manifest: str | None = Field(default=None, description="YAML manifest content")
    resource_kind: str | None = Field(default=None, description="e.g., pod, deployment")
    resource_name: str | None = Field(default=None, description="Name of the resource")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    command: list[str] | None = Field(default=None, description="Exec command args")
    options: dict[str, JsonValue] = Field(default_factory=dict)

    model_config = {"str_strip_whitespace": True}


class K8sObservation(BaseModel):
    """Observation returned after each step."""

    kubectl_output: str = Field(description="Raw stdout from kubectl command")
    error_message: str | None = Field(
        default=None,
        description="Descriptive error if failed, e.g., 'Invalid memory format: use Mi not MB'"
    )
    pod_status: str | None = Field(default=None, description="Current pod phase")
    event_summary: str | None = Field(default=None, description="K8s event summary")
    health_score: float = Field(
        ge=0.0, le=1.0, description="System health score based on pod states"
    )
    step_number: int = Field(ge=0, description="Current step in episode")


class K8sState(BaseModel):
    """Internal state of the Kubernetes cluster for the current episode."""

    healthy_pods: int = Field(ge=0, description="Number of Running pods")
    total_pods: int = Field(ge=0, description="Total expected pods")
    failed_pods: list[str] = Field(default_factory=list)
    event_log: list[str] = Field(default_factory=list, description="Recent K8s events")
    penalties_accumulated: float = Field(default=0.0)
    episode_start_ts: float | None = None
    system_uptime: float = 0.0
    step_number: int = Field(default=0, ge=0, description="Current step in episode")


class TaskConfig(BaseModel):
    """Configuration for a single task."""

    difficulty: Literal["easy", "medium", "hard"]
    name: str
    description: str
    max_steps: int = Field(default=20, ge=1)
    penalty_per_step: float = Field(default=0.05, ge=0.0)
    sparse_success_reward: float = Field(default=1.0, ge=0.0, le=1.0)
