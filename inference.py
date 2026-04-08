#!/usr/bin/env python3
"""Baseline inference script for SRE Gym.

This script runs an LLM agent against the SRE Gym environment and produces
structured logs in the required [START]/[STEP]/[END] format.

Usage:
    python inference.py --task easy --episodes 10

Environment Variables (from .env file or system):
    API_BASE_URL: The API endpoint for the LLM
        - OpenAI: https://api.openai.com/v1
        - Groq (FREE): https://api.groq.com/openai/v1
        - Hugging Face: https://api-inference.huggingface.co/v1
    MODEL_NAME: The model identifier
        - OpenAI: gpt-4o
        - Groq: llama-3.3-70b-versatile
        - HF: meta-llama/Llama-3.2-3B-Instruct
    OPENAI_API_KEY: API key for the provider
    HF_TOKEN: Hugging Face token (optional)

Output Format:
    [START] task=easy episode=1/10 timestamp=2026-01-01T00:00:00Z
    [STEP] step=1 action=apply_manifest reward=0.05 observation=...
    [END] episode=1 total_reward=0.85 success=true steps=5 duration=12.34
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Load .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from sre_gym.env import SREGymEnv, EnvConfig
from sre_gym.models import K8sAction, K8sActionType


# =============================================================================
# Configuration from Environment Variables
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")


# =============================================================================
# Logging
# =============================================================================

def log_start(task: str, episode: int, total_episodes: int) -> None:
    """Log episode start in required format."""
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"[START] task={task} episode={episode}/{total_episodes} timestamp={timestamp}", flush=True)


def log_step(step: int, action: dict[str, Any], observation: dict[str, Any], reward: float) -> None:
    """Log step in required format."""
    action_type = action.get("action_type", "unknown")
    obs_summary = observation.get("pod_status", "unknown")
    error = observation.get("error_message", "")
    if error:
        obs_summary = f"{obs_summary}: {error[:50]}"
    print(f"[STEP] step={step} action={action_type} reward={reward:.4f} observation={obs_summary}", flush=True)


def log_end(episode: int, total_reward: float, success: bool, steps: int, duration: float) -> None:
    """Log episode end in required format."""
    print(f"[END] episode={episode} total_reward={total_reward:.4f} success={str(success).lower()} steps={steps} duration={duration:.2f}", flush=True)


# =============================================================================
# Agent Implementation
# =============================================================================

class K8sAgent:
    """LLM agent that uses kubectl tools to fix K8s issues."""

    def __init__(self, model: str = MODEL_NAME, api_base: str = API_BASE_URL):
        # Get API key - check HF_TOKEN first for HF Inference, then OPENAI_API_KEY
        api_key = os.environ.get("OPENAI_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")

        # For Hugging Face Inference API, use HF_TOKEN as Bearer
        if "huggingface" in api_base.lower():
            self.client = OpenAI(
                api_key=hf_token,
                base_url=api_base,
            )
            self.use_hf_api = True
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            self.use_hf_api = False

        self.model = model
        self._system_prompt = (
            "You are an SRE agent debugging a Kubernetes cluster. "
            "Use kubectl commands to diagnose and fix failing pods. "
            "Be concise and precise with kubectl syntax.\n\n"
            "Available actions (use exactly these action_type values):\n"
            "- apply_manifest: Apply YAML manifest to create/update resources\n"
            "- delete_resource: Delete a pod, configmap, or other resource\n"
            "- scale_deployment: Scale a deployment to N replicas\n"
            "- exec_command: Execute command inside a pod\n"
            "- noop: Wait and observe (incurs step penalty)\n\n"
            "Respond with JSON only in this format:\n"
            '{"action_type": "apply_manifest", "manifest": "...", "namespace": "default"}\n'
            "or\n"
            '{"action_type": "noop"}'
        )
        self._conversation_history: list[dict] = []

    def select_action(self, observation: dict, task_hint: str = "") -> K8sAction:
        """Given observation, return the next K8sAction to take."""
        user_msg = f"""Current state:
- Pod status: {observation.get('pod_status', 'unknown')}
- Health score: {observation.get('health_score', 0.0):.2f}
- Error: {observation.get('error_message', 'none')}
- kubectl output: {observation.get('kubectl_output', '')[:500]}
- Step: {observation.get('step_number', 0)}

{task_hint}

Respond with JSON action."""

        messages = [
            {"role": "system", "content": self._system_prompt},
        ]
        if self._conversation_history:
            for entry in self._conversation_history[-3:]:
                messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})
                messages.append({"role": "assistant", "content": f"Action: {json.dumps(entry['action'])}"})
        messages.append({"role": "user", "content": user_msg})

        try:
            if self.use_hf_api:
                # Hugging Face Inference API format
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=500,
                )
            else:
                # Standard OpenAI-compatible API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=500,
                )
            content = response.choices[0].message.content or "{}"

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            action_data = json.loads(content.strip())
        except (json.JSONDecodeError, Exception) as e:
            print(f"# WARNING: LLM call failed: {e}, defaulting to noop", flush=True)
            action_data = {"action_type": "noop"}

        # Normalize command field - it can be a string or list
        command = action_data.get("command")
        if command is not None and isinstance(command, str):
            command = command.split() if command.strip() else None

        action = K8sAction(
            action_type=K8sActionType(action_data.get("action_type", "noop")),
            manifest=action_data.get("manifest"),
            resource_kind=action_data.get("resource_kind"),
            resource_name=action_data.get("resource_name"),
            namespace=action_data.get("namespace", "default"),
            command=command,
            options=action_data.get("options", {}),
        )

        self._conversation_history.append({
            "observation": obs,
            "action": action_data,
        })
        return action

    def reset(self) -> None:
        """Reset conversation history for new episode."""
        self._conversation_history = []


# =============================================================================
# Main Inference Loop
# =============================================================================

def run_episode(env: SREGymEnv, agent: K8sAgent, task: str, episode: int, total_episodes: int, max_steps: int | None = None) -> dict:
    """Run a single episode and return results."""
    log_start(task, episode, total_episodes)

    task_hint = ""
    if hasattr(env._task_instance, "get_hint"):
        task_hint = env._task_instance.get_hint()

    obs = env.reset()
    agent.reset()
    start_time = time.time()

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        if max_steps and step > max_steps:
            break

        obs_dict = obs.model_dump()
        action = agent.select_action(obs_dict, task_hint)
        action_dict = action.model_dump()

        obs, reward, done, info = env.step(action)

        total_reward += reward
        log_step(step, action_dict, obs.model_dump(), reward)

        if done:
            break

    success = obs.health_score >= 1.0
    duration = time.time() - start_time
    log_end(episode, total_reward, success, step, duration)

    return {
        "task": task,
        "episode": episode,
        "total_reward": total_reward,
        "success": success,
        "steps": step,
        "duration": duration,
    }


def main():
    """Run baseline inference against SRE Gym."""
    import argparse
    parser = argparse.ArgumentParser(description="SRE Gym Baseline Inference")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy",
                        help="Task difficulty")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    args = parser.parse_args()

    # Validate required environment variables
    api_key = os.environ.get("OPENAI_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_key and not hf_token:
        print("ERROR: Either OPENAI_API_KEY or HF_TOKEN environment variable must be set", flush=True)
        print("", flush=True)
        print("Free options:", flush=True)
        print("  1. Groq (FREE): https://console.groq.com - get API key", flush=True)
        print("     export OPENAI_API_KEY='gsk_xxxx'", flush=True)
        print("     export API_BASE_URL='https://api.groq.com/openai/v1'", flush=True)
        print("     export MODEL_NAME='llama-3.3-70b-versatile'", flush=True)
        print("", flush=True)
        print("  2. Hugging Face Inference (FREE tier): https://huggingface.co/settings/inference-catalog", flush=True)
        print("     export HF_TOKEN='hf_xxxx'", flush=True)
        print("     export API_BASE_URL='https://api-inference.huggingface.co/v1'", flush=True)
        print("     export MODEL_NAME='meta-llama/Llama-3.2-3B-Instruct'", flush=True)
        sys.exit(1)

    print(f"# SRE Gym Baseline Inference", flush=True)
    print(f"# Task: {args.task}", flush=True)
    print(f"# Episodes: {args.episodes}", flush=True)
    print(f"# Model: {args.model} @ {API_BASE_URL}", flush=True)
    print("", flush=True)

    env = SREGymEnv(EnvConfig(task_difficulty=args.task))
    agent = K8sAgent(model=args.model, api_base=API_BASE_URL)

    results = []
    for ep in range(1, args.episodes + 1):
        result = run_episode(env, agent, args.task, ep, args.episodes, args.max_steps)
        results.append(result)

    env.close()

    # Summary
    successes = sum(1 for r in results if r["success"])
    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    print("", flush=True)
    print(f"# Summary", flush=True)
    print(f"# Task: {args.task}", flush=True)
    print(f"# Episodes: {args.episodes}", flush=True)
    print(f"# Success Rate: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)", flush=True)
    print(f"# Average Reward: {avg_reward:.4f}", flush=True)
    print(f"# Average Steps: {avg_steps:.1f}", flush=True)

    # Output JSON for programmatic access
    print("", flush=True)
    print(f"# JSON: {json.dumps(results)}", flush=True)


if __name__ == "__main__":
    main()
