---
title: SRE Gym
emoji: 🐛
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

# SRE Gym - Self-Healing Kubernetes SRE Gym

An RL training environment for AI agents to diagnose and fix Kubernetes production errors using kubectl tools. Trains agents on realistic K8s failure scenarios with dense PBRS rewards.

## Overview

SRE Gym simulates real-world Site Reliability Engineering tasks where an AI agent must:
1. **Diagnose** Kubernetes issues (CrashLoopBackOff, OOMKilled, cascading failures)
2. **Plan** corrective actions using kubectl
3. **Execute** fixes and verify resolution

## Tasks

| Difficulty | Task | Issue | Fix |
|------------|------|-------|-----|
| Easy | CrashLoopBackOff | Missing ConfigMap | Create ConfigMap |
| Medium | OOMKilled | Memory limit too low | Increase memory |
| Hard | Cascading Failure | 3 microservices down | Diagnose root cause |

## Quick Start

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"

# Run baseline inference
python inference.py --task easy --episodes 10
```

## Output Format

```
[START] task=easy episode=1/10 timestamp=...
[STEP] step=1 action=noop reward=0.05 observation=CrashLoopBackOff
[STEP] step=2 action=apply_manifest reward=0.10 observation=Running
[END] episode=1 total_reward=0.85 success=true steps=5 duration=12.34
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | API key for LLM |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | API endpoint |
| `MODEL_NAME` | `llama-3.3-70b-versatile` | Model identifier |

## Free LLM Providers

- **Groq**: https://console.groq.com (30 req/min free)
- **HuggingFace**: https://huggingface.co/settings/inference-catalog

## Project

- GitHub: https://github.com/Jrine0/sre-gym
- RL training environment for K8s troubleshooting
- Apache 2.0 License
