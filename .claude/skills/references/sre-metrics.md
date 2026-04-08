# SRE Metrics: Mapping K8s Events to Reward Components

## Pod Lifecycle States

| State | Description | Reward Signal |
|-------|-------------|---------------|
| Pending | Scheduler waiting | Incomplete |
| Running | Container running | +0.25 toward healthy |
| Succeeded | Job completed | Terminal success |
| Failed | Unrecoverable error | Terminal failure |
| Unknown | Node unreachable | Penalty |
| CrashLoopBackOff | Restarting repeatedly | Negative signal |
| OOMKilled | Memory limit exceeded | Negative signal |

## Event-to-Reward Mapping

### Easy Task (ConfigMap Missing)
```
Event: "configmap 'db-config' not found"
→ Error message: "Create ConfigMap with kubectl create configmap"
→ Reward: Structured hint, no terminal reward until fixed
```

### Medium Task (OOMKilled)
```
Event: "last state: terminated...OOMKilled"
→ Error message: "Invalid memory format: use 'Mi' not 'MB'"
→ Linear reward: 0.5 + 0.5×(1 - overhead)
```

### Hard Task (Cascading Failure)
```
Multiple events across pods:
- db-primary: OOMKilled
- api-service: Connection timeout
- frontend: High latency
→ Stability Index: (Uptime - Penalties) / Duration
```

## Health Score Calculation

```python
def health_score(healthy_pods: int, total_pods: int) -> float:
    """
    Φ(s) for PBRS reward shaping.
    Returns normalized health ratio [0, 1].
    """
    if total_pods == 0:
        return 0.0
    return healthy_pods / total_pods
```

## Grader Integration

```python
from sre_gym.grader import AssertionEngine

grader = AssertionEngine()
# Get real pod status
phase = grader.get_pod_phase("backend-api")
# Compute health
score = grader.compute_health_score()
# Stability index for hard tasks
stability = grader.compute_stability_index(episode_duration, uptime, penalties)
```
