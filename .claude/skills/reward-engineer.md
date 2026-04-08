# Reward Engineer (PBRS Pattern)

Use this skill to design and implement non-exploitable, dense reward functions for developer tooling agents.

## Core Formula

```
ShapedReward = SparseReward + α(γΦ(s') - Φ(s))
```

Where:
- `α` = shaping coefficient (typically 0.1)
- `γ` = discount factor (typically 0.99)
- `Φ(s)` = potential function based on system health

## Application Guidelines

### Φ(s) for SRE Tasks
```python
Φ(s) = healthy_pods / total_pods
```

### Φ(s) for Code Tasks
```python
Φ(s) = % of passing unit tests + reduction in cyclomatic complexity
```

## Penalties

| Penalty Type | Value | Purpose |
|-------------|-------|---------|
| Stall penalty | -0.05/step | Discourage infinite loops |
| Cheating penalty | -0.2 | Penalize removing failing tests |

## Cheating Defense

- Monitor code diffs for test deletions
- Verify fixes address root cause, not symptom
- Track reward trajectory for anomalous patterns

## References

- `references/pbrs-stability.md`: Detailed math for policy invariance
- `references/sre-metrics.md`: Mapping K8s events to reward components
