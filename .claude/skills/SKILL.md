# SRE Environment Builder

Use this skill to scaffold, implement, and validate a new SRE-focused OpenEnv environment for K8s troubleshooting.

## Activation Triggers

- "Build a K8s gym environment"
- "Create an SRE OpenEnv"
- "Implement a self-healing infrastructure task"

## Execution Workflow

### 1. Model Definition
Use Pydantic for K8sAction, K8sObservation, and K8sState.

```python
from pydantic import BaseModel, Field

class K8sAction(BaseModel):
    action_type: K8sActionType
    manifest: str | None = None
    # ...
```

### 2. Environment Logic
Inherit from `openenv.core.env_server.Environment`.

```python
from sre_gym.env import SREGymEnv
env = SREGymEnv()
```

### 3. Task Implementation

| Difficulty | Task Archetype | Grader Logic |
|------------|---------------|--------------|
| Easy | Pod CrashLoopBackOff due to missing ConfigMap | Binary: 1.0 if pod Running |
| Medium | OOMKilled triage under resource budget | Linear: 0.5 + 0.5×(1−overhead) |
| Hard | Cascading latency failure (3 microservices) | Stability Index |

### 4. Grader Implementation
Use `AssertionEngine` to check real container status:

```python
from sre_gym.grader import AssertionEngine
grader = AssertionEngine()
phase = grader.get_pod_phase("pod-name")
```

### 5. Validation
Run `python -m sre_gym.cli validate --verbose`

## Output Contract

- Environment must run on **2 vCPUs and 8GB RAM**
- All rewards must be scaled **0.0 to 1.0**
- `inference.py` must use the **OpenAI client** and structured `<tool_call>` logs
