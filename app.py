#!/usr/bin/env python3
"""Hugging Face Spaces - OpenEnv + Gradio for SRE Gym.

This app exposes both:
1. OpenEnv FastAPI endpoints (/reset, /step, /state)
2. Gradio web interface
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

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
sys.path.insert(0, str(Path(__file__).parent))

from sre_gym.env import SREGymEnv, EnvConfig
from sre_gym.models import K8sAction, K8sActionType


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")


# =============================================================================
# FastAPI OpenEnv Endpoints
# =============================================================================

app = FastAPI(title="SRE Gym - OpenEnv")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env: SREGymEnv | None = None


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "name": "sre-gym"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset(task_id: str | None = None):
    """Reset environment and return initial observation."""
    global _env
    try:
        _env = SREGymEnv(EnvConfig(task_difficulty=task_id or "easy"))
        obs = _env.reset()
        return obs.model_dump()
    except Exception as e:
        return {"error": str(e)}


@app.post("/step")
async def step(action: dict):
    """Execute one step with the given action."""
    global _env
    if _env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    try:
        k8s_action = K8sAction(
            action_type=K8sActionType(action.get("action_type", "noop")),
            manifest=action.get("manifest"),
            resource_kind=action.get("resource_kind"),
            resource_name=action.get("resource_name"),
            namespace=action.get("namespace", "default"),
            command=action.get("command"),
            options=action.get("options", {}),
        )
        obs, reward, done, info = _env.step(k8s_action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/state")
async def state():
    """Get current environment state."""
    global _env
    if _env is None:
        return {"error": "Environment not initialized. Call /reset first."}
    return _env.state()


@app.get("/tasks")
async def tasks():
    """List available tasks."""
    return {
        "tasks": [
            {"id": "easy", "name": "CrashLoopBackOff - Missing ConfigMap"},
            {"id": "medium", "name": "OOMKilled - Memory Limit Triage"},
            {"id": "hard", "name": "Cascading Failure - Multi-Service"},
        ]
    }


# =============================================================================
# LLM Agent
# =============================================================================

class K8sAgent:
    """LLM agent for K8s troubleshooting."""

    def __init__(self, model: str = MODEL_NAME):
        api_key = os.environ.get("OPENAI_API_KEY", "")
        hf_token = os.environ.get("HF_TOKEN", "")

        if "huggingface" in API_BASE_URL.lower():
            self.client = OpenAI(api_key=hf_token, base_url=API_BASE_URL)
        else:
            self.client = OpenAI(api_key=api_key, base_url=API_BASE_URL)

        self.model = model
        self._system_prompt = (
            "You are an SRE agent debugging Kubernetes. Use kubectl to fix issues.\n"
            "Actions: apply_manifest, delete_resource, noop\n"
            "Respond with JSON only."
        )
        self._history = []

    def select_action(self, obs: dict) -> K8sAction:
        user_msg = f"Status: {obs.get('pod_status')}\nError: {obs.get('error_message')}\nStep: {obs.get('step_number')}"

        messages = [{"role": "system", "content": self._system_prompt}]
        if self._history:
            for entry in self._history[-2:]:
                messages.append({"role": "user", "content": str(entry["observation"])})
                messages.append({"role": "assistant", "content": json.dumps(entry["action"])})
        messages.append({"role": "user", "content": user_msg})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            content = response.choices[0].message.content or "{}"
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            action_data = json.loads(content.strip())
        except Exception:
            action_data = {"action_type": "noop"}

        action = K8sAction(
            action_type=K8sActionType(action_data.get("action_type", "noop")),
            manifest=action_data.get("manifest"),
            resource_kind=action_data.get("resource_kind"),
            resource_name=action_data.get("resource_name"),
            namespace=action_data.get("namespace", "default"),
        )
        self._history.append({"observation": obs, "action": action_data})
        return action

    def reset(self):
        self._history = []


# =============================================================================
# Gradio Interface
# =============================================================================

def run_episode(task: str, max_steps: int, model_name: str) -> tuple[str, dict]:
    """Run a single episode and return log and results."""
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("HF_TOKEN"):
        return "ERROR: OPENAI_API_KEY or HF_TOKEN not set", {"error": "API key required"}

    log_lines = []

    env = SREGymEnv(EnvConfig(task_difficulty=task, max_steps=max_steps))
    agent = K8sAgent(model=model_name)

    timestamp = datetime.now(timezone.utc).isoformat()
    log_lines.append(f"[START] task={task} episode=1/1 timestamp={timestamp}")

    obs = env.reset()
    agent.reset()
    start_time = time.time()

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        obs_dict = obs.model_dump()
        action = agent.select_action(obs_dict)
        obs, reward, done, info = env.step(action)

        total_reward += reward
        status = obs_dict.get("pod_status", "unknown")
        log_lines.append(f"[STEP] step={step} action={action.action_type.value} reward={reward:.4f} observation={status}")

        if done:
            break

    success = obs.health_score >= 1.0
    duration = time.time() - start_time
    log_lines.append(f"[END] episode=1 total_reward={total_reward:.4f} success={str(success).lower()} steps={step} duration={duration:.2f}")

    env.close()

    results = {
        "task": task,
        "total_reward": round(total_reward, 4),
        "success": success,
        "steps": step,
        "duration": round(duration, 2),
    }

    return "\n".join(log_lines), results


# =============================================================================
# Mount Gradio app
# =============================================================================

gradio_app = gr.Blocks(title="SRE Gym")

with gradio_app:
    gr.Markdown("# SRE Gym - Kubernetes Troubleshooting Agent")
    gr.Markdown("Train AI agents to diagnose and fix K8s production faults.")

    with gr.Row():
        with gr.Column():
            task = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task Difficulty"
            )
            max_steps = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Max Steps")
            model_name = gr.Textbox(value=MODEL_NAME, label="Model Name")
            run_btn = gr.Button("Run Episode", variant="primary")

        with gr.Column():
            log_output = gr.Textbox(label="Episode Log", lines=15)
            results_output = gr.JSON(label="Results")

    gr.Markdown("""
    ## Tasks
    - **Easy**: CrashLoopBackOff - Missing ConfigMap
    - **Medium**: OOMKilled - Memory Limit too low
    - **Hard**: Cascading Failure - 3 microservices down

    ## OpenEnv API
    - POST /reset?task_id=easy
    - POST /step
    - GET /state
    """)

    run_btn.click(
        fn=run_episode,
        inputs=[task, max_steps, model_name],
        outputs=[log_output, results_output]
    )

# Mount Gradio app
app = gr.mount_gradio_app(app, gradio_app, path="/")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
