# SRE Gym Debugger Skill

## Common Issues and Fixes

### 1. Environment Requires kubectl (but validator doesn't have it)

**Problem**: `env.reset()` fails with traceback when kubectl is not available in the container.

**Traceback**:
```
File "/tmp/workspace/sre_gym/env.py", line 186, in reset
  self._state = self._task_instance.setup()
```

**Root Cause**: The environment tries to run kubectl commands immediately on initialization.

**Solution**: Add simulation mode that falls back when kubectl is unavailable:

```python
# In env.py - check kubectl at module load
def _check_kubectl_available() -> bool:
    try:
        result = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

KUBECTL_AVAILABLE = _check_kubectl_available()

# Create simulation state tracker
class SimState:
    def __init__(self):
        self.configmaps: set[str] = set()
    # Track state for simulation mode

# Use simulation when kubectl unavailable
if not KUBECTL_AVAILABLE:
    return self._simulate_action(action)
```

---

### 2. Tasks Return Wrong Tuple Format

**Problem**: Tasks return `(reward, done, obs)` but env expects `(done, obs)`.

**Fix**: All tasks MUST return `(done, obs)` - reward is computed by PBRS in env.py.

```python
# WRONG
def evaluate(self, state: K8sState) -> tuple[float, bool, K8sObservation]:
    return reward, done, obs

# CORRECT
def evaluate(self, state: K8sState) -> tuple[bool, K8sObservation]:
    return done, obs
```

---

### 3. K8sAction command Field Must Be List

**Problem**: LLM returns command as string, but Pydantic expects list.

**Fix**: Normalize in inference:

```python
command = action_data.get("command")
if command and isinstance(command, str):
    command = command.split()
```

---

### 4. OpenEnv Validation Errors

**"[project.scripts] server entry point should reference main function"**

Fix: `pyproject.toml` must have:
```toml
[project.scripts]
server = "server.app:main"
```

**"server/app.py missing main() function"**

Fix: `server/app.py` must have:
```python
def main():
    # server code
    pass

if __name__ == "__main__":
    main()
```

---

### 5. Task Config Using @dataclass + Pydantic

**Problem**: `@dataclass` doesn't work with Pydantic `Field()`.

**Fix**: Use Pydantic directly:
```python
# WRONG
@dataclass
class EasyTaskConfig(TaskConfig):
    max_steps: int = 15

# CORRECT
class EasyTaskConfig(TaskConfig):
    max_steps: int = Field(default=15, ge=1)
```

---

### 6. K8sState Missing step_number

**Problem**: Code references `state.step_number` but it's not in the model.

**Fix**: Add to `K8sState`:
```python
class K8sState(BaseModel):
    step_number: int = Field(default=0, ge=0)
```

---

### 7. Namespace Inconsistency

**Problem**: Some code uses `config.namespace`, others hardcode "default".

**Fix**: Always use `namespace: str = "default"` in task configs and reference it consistently.

---

### 8. Easy Task Manifest Hardcodes Namespace

**Problem**: `get_broken_manifest()` uses f-string `namespace: {config.namespace}` but cleanup uses hardcoded "default".

**Fix**: Hardcode "default" in manifests:
```python
return f"""apiVersion: v1
kind: Pod
metadata:
  name: backend-api
  namespace: default  # Always default
spec:
```

---

## Testing Checklist

Before submitting, verify:

```bash
# 1. Validation passes
python -m sre_gym.cli validate

# 2. Tests pass
pytest tests/ -v

# 3. Inference runs WITHOUT kubectl
python inference.py --task easy --episodes 3

# 4. OpenEnv validates
openenv validate
```

---

## Common Grader Fixes

When kubectl unavailable, grader MUST return simulated values:

```python
def get_pod_phase(self, pod_name: str) -> str | None:
    if not KUBECTL_AVAILABLE:
        # Return based on simulation state
        from sre_gym.env import SIM_STATE
        if pod_name == "backend-api":
            if "db-config" in SIM_STATE.configmaps:
                return "Running"
            return "CrashLoopBackOff"
        return "Unknown"
```

---

## Environment Variables

For inference to work, set:
```bash
export OPENAI_API_KEY="gsk_xxx"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
```

Or create `.env` file:
```
OPENAI_API_KEY=gsk_xxx
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
```

---

## Key Files to Check

| File | Common Issues |
|------|--------------|
| `sre_gym/env.py` | Needs `KUBECTL_AVAILABLE`, `SimState`, simulation mode |
| `sre_gym/grader.py` | Needs kubectl check, simulated returns |
| `sre_gym/tasks/*.py` | Return `(done, obs)`, not `(reward, done, obs)` |
| `pyproject.toml` | Must have `[project.scripts] server = "server.app:main"` |
| `server/app.py` | Must have `def main()` with `if __name__ == "__main__"` |
| `inference.py` | Normalize command field, handle API key errors |

---

## Validation Commands

```bash
# Local validation
python -m sre_gym.cli validate

# OpenEnv validation
openenv validate

# Test without kubectl
kubectl_off()  # Just don't have kubectl installed
python inference.py --task easy --episodes 1

# Test with kubectl
kind create cluster --name sre-gym
python inference.py --task easy --episodes 1
```
