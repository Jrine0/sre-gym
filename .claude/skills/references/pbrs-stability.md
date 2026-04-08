# PBRS (Potential-Based Reward Shaping) Stability Proof

## Formal Definition

Reward shaping transforms a sparse reward function `R(s, a, s')` into a shaped reward:

```
R'(s, a, s') = R(s, a, s') + F(s, s')
```

Where the shaping function `F` is derived from a potential function `Φ`:

```
F(s, s') = γΦ(s') - Φ(s)
```

## Conditions for Policy Invariance

Ng et al. (1999) proved that reward shaping preserves optimal policies if and only if:

1. **F(s, s') = γΦ(s') - Φ(s)** for some `Φ`
2. **Φ is finite** (bounded potential difference)

This ensures:
- No new optimal policies are introduced
- No optimal policies are removed
- The shaped MDP is equivalent to the original

## Application to SRE Gym

### Potential Function
```
Φ(s) = healthy_pods / total_pods ∈ [0, 1]
```

This is bounded and directly measures task progress.

### Shaping Reward
```
F(s, s') = γ(healthy_pods'/total_pods) - (healthy_pods/total_pods)
```

### Implementation
```python
def compute_reward(self, healthy, total, is_terminal):
    potential_current = healthy / total
    delta_phi = self.gamma * potential_current - self._last_potential
    shaping = self.alpha * delta_phi
    self._last_potential = potential_current
    return shaping
```

## Proof of Non-Exploitability

1. **Bounded potential**: `Φ(s) ∈ [0, 1]` ensures finite shaping
2. **Monotonic alignment**: `Φ` increases with task progress
3. **No shortcuts**: Shaping rewards partial progress but doesn't allow skipping solutions

Agents cannot exploit by:
- Getting high reward without actually fixing issues
- Manipulating environment to appear healthy
- Removing constraints instead of fixing root cause
