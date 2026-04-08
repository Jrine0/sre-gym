# SRE Gym - Self-Healing Kubernetes SRE Gym

An RL training environment for AI agents to diagnose and fix Kubernetes production errors using kubectl tools. Trains agents on realistic K8s failure scenarios with dense PBRS rewards.

## Overview

SRE Gym simulates real-world Site Reliability Engineering tasks where an AI agent must:
1. **Diagnose** Kubernetes issues (CrashLoopBackOff, OOMKilled, cascading failures)
2. **Plan** corrective actions using kubectl
3. **Execute** fixes and verify resolution

This is a **real-world utility** problem - SREs spend hours debugging these exact issues in production.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Validate environment
python -m sre_gym.cli validate

# Run a task manually
python -m sre_gym.cli run --task easy --episodes 1
```

## Environment Description

### What It Simulates

| Task | Real Scenario | Agent Action |
|------|--------------|--------------|
| Easy | Pod referencing missing ConfigMap fails with CrashLoopBackOff | Create ConfigMap |
| Medium | Pod OOMKilled due to memory limit misconfiguration | Increase memory limit |
| Hard | Cascading failure across 3 microservices (DB → API → Frontend) | Diagnose root cause |

### Why This Is Real-World

- **ConfigMap issues** are extremely common in K8s deployments
- **OOMKilled** is the #1 cause of pod failures in production
- **Cascading failures** require root-cause analysis skills
- Every production SRE team encounters these daily

## Action Space

| Action Type | Description | Parameters |
|-------------|-------------|------------|
| `apply_manifest` | Apply YAML manifest | `manifest`, `namespace` |
| `delete_resource` | Delete a resource | `resource_kind`, `resource_name`, `namespace` |
| `scale_deployment` | Scale deployment | `resource_name`, `namespace`, `options.replicas` |
| `exec_command` | Execute in pod | `resource_name`, `command`, `namespace` |
| `noop` | Wait and observe | - |

### Example Action (apply_manifest)

```python
from sre_gym.models import K8sAction, K8sActionType

action = K8sAction(
    action_type=K8sActionType.APPLY_MANIFEST,
    manifest="""apiVersion: v1
kind: ConfigMap
metadata:
  name: db-config
data:
  connection_string: "postgresql://localhost:5432/mydb"
""",
    namespace="default"
)
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `kubectl_output` | string | Raw stdout from kubectl commands |
| `error_message` | string\|null | Error description if pod is failing |
| `pod_status` | string\|null | Kubernetes phase (Running, CrashLoopBackOff, etc.) |
| `health_score` | float | System health [0.0, 1.0] |
| `step_number` | int | Current step in episode |

## Task Difficulty

### Easy: CrashLoopBackOff - Missing ConfigMap
- **Objective**: Fix pod failing due to missing ConfigMap
- **Max Steps**: 15
- **Expected Success Rate**: >90% for good agents
- **Grader**: Checks if pod reaches Running state

### Medium: OOMKilled - Memory Limit Triage
- **Objective**: Fix OOMKilled pod by adjusting memory limits
- **Max Steps**: 20
- **Expected Success Rate**: >75% for good agents
- **Grader**: Checks pod Running with sufficient memory

### Hard: Cascading Failure - Multi-Service
- **Objective**: Diagnose and fix cascading failure across 3 microservices
- **Max Steps**: 30
- **Expected Success Rate**: >50% for frontier agents
- **Grader**: All 3 pods must be Running

## Reward Function

Uses **Potential-Based Reward Shaping (PBRS)**:

```
Total Reward = Sparse Reward + α(γΦ(s') - Φ(s)) - Step Penalty
```

- **Dense shaping**: Rewards based on health score improvement each step
- **Step penalty**: Small negative reward for no-op actions (-0.05)
- **Sparse terminal**: +1.0 reward when task is completed
- **All rewards clamped**: [0.0, 1.0]

This provides useful signal throughout the episode, not just at the end.

## Baseline Scores

Run inference to get baseline scores:

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"

# Run baseline
python inference.py --task easy --episodes 10
```

Expected output format:
```
[START] task=easy episode=1/10 timestamp=...
[STEP] step=1 action=noop reward=0.05 observation=CrashLoopBackOff
[STEP] step=2 action=apply_manifest reward=0.10 observation=...
...
[END] episode=1 total_reward=0.85 success=true steps=5 duration=12.34
```

## Setup Instructions

### Prerequisites

- Python 3.10+
- Kubernetes cluster (kind, k3s, or cloud)
- kubectl configured

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/meta-hack.git
cd meta-hack

# Install with dev dependencies
pip install -e ".[dev]"

# Create kind cluster
kind create cluster --name sre-gym

# Validate
python -m sre_gym.cli validate
```

### Docker

```bash
# Build
docker build -t sre-gym .

# Run
docker run --rm -it --privileged \
  -v ~/.kube:/root/.kube \
  sre-gym python -m sre_gym.cli run --task easy
```

### Hugging Face Spaces

```bash
# Install HF CLI
pip install huggingface_hub
huggingface-cli login

# Create space
huggingface-cli create-space sre-gym --type gradio

# Push code
git push
```

## Project Structure

```
meta-hack/
├── sre_gym/              # Main package
│   ├── env.py            # Environment (step/reset/state)
│   ├── models.py         # Pydantic models
│   ├── grader.py         # Assertion engine
│   ├── rewards.py        # PBRS reward shaping
│   ├── inference.py       # LLM agent
│   ├── cli.py            # CLI tools
│   ├── mcp_server.py     # MCP tool definitions
│   └── tasks/            # Task definitions
│       ├── easy.py       # CrashLoopBackOff
│       ├── medium.py     # OOMKilled
│       └── hard.py       # Cascading failure
├── inference.py           # Baseline inference script
├── app.py               # HF Spaces Gradio app
├── openenv.yaml         # OpenEnv spec
├── Dockerfile           # Container image
├── pyproject.toml       # Package config
└── README.md            # This file
```

## API Reference

### Python API

```python
from sre_gym.env import SREGymEnv, EnvConfig

# Create environment
env = SREGymEnv(EnvConfig(task_difficulty="easy", max_steps=15))

# Reset to initial state
obs = env.reset()
print(f"Initial: {obs.pod_status}")  # CrashLoopBackOff

# Run episode
done = False
total_reward = 0.0
while not done:
    action = agent.select_action(obs)  # Your agent
    obs, reward, done, info = env.step(action)
    total_reward += reward
    print(f"Step {obs.step_number}: {obs.pod_status}, reward={reward:.2f}")

env.close()
print(f"Total reward: {total_reward:.2f}")
```

### CLI

```bash
# Run manual mode
python -m sre_gym.cli run --task easy --episodes 1

# Validate environment
python -m sre_gym.cli validate

# Run with LLM agent
OPENAI_API_KEY=xxx python inference.py --task easy --episodes 10
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | API key for LLM (required) |
| `API_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `MODEL_NAME` | `gpt-4o` | Model identifier |

## License

Apache 2.0
