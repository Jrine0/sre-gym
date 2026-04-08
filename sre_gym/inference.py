"""Inference script using OpenAI client with structured <tool_call> logs.

This script runs the SRE Gym with an LLM agent that uses kubectl tools
to diagnose and fix Kubernetes issues.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from openai import OpenAI

from sre_gym.env import SREGymEnv, EnvConfig
from sre_gym.models import K8sAction, K8sActionType

# =============================================================================
# Logging configuration
# =============================================================================

log = structlog.get_logger()


@dataclass
class ToolCallLog:
    """Structured log entry for a tool call in <tool_call> format."""

    timestamp: float
    step: int
    reasoning: str
    action: dict[str, Any]
    observation: dict[str, Any]
    reward: float


@dataclass
class EpisodeLog:
    """Complete log for an episode."""

    task: str
    difficulty: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    steps: int = 0
    duration: float = 0.0


# =============================================================================
# Agent implementation
# =============================================================================

class K8sAgent:
    """LLM agent that uses kubectl tools to fix K8s issues."""

    def __init__(self, model: str = "gpt-4o"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._system_prompt = (
            "You are an SRE agent debugging a Kubernetes cluster. "
            "Use kubectl commands to diagnose and fix failing pods. "
            "Be concise and precise with kubectl syntax.\n\n"
            "Available actions:\n"
            "- apply_manifest: Apply YAML manifest\n"
            "- delete_resource: Delete a resource\n"
            "- scale_deployment: Scale a deployment\n"
            "- exec_command: Execute command in pod\n"
            "- noop: Wait and observe\n\n"
            "Respond with JSON in this format:\n"
            '{"action_type": "apply_manifest", "manifest": "...", "namespace": "default"}'
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

{f'Hint: {task_hint}' if task_hint else ''}

Based on the current state, what kubectl action should you take? Respond with JSON."""

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_msg},
        ]

        if self._conversation_history:
            messages = [
                {"role": "system", "content": self._system_prompt},
            ]
            for entry in self._conversation_history[-5:]:
                messages.append({"role": "user", "content": f"Observation: {entry['observation']}"})
                messages.append({"role": "assistant", "content": f"Action: {json.dumps(entry['action'])}"})
            messages.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
        )

        content = response.choices[0].message.content or "{}"

        # Extract JSON from response
        try:
            # Handle cases where JSON is wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            action_data = json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            # Fallback to noop if parsing fails
            action_data = {"action_type": "noop"}

        action = K8sAction(
            action_type=K8sActionType(action_data.get("action_type", "noop")),
            manifest=action_data.get("manifest"),
            resource_kind=action_data.get("resource_kind"),
            resource_name=action_data.get("resource_name"),
            namespace=action_data.get("namespace", "default"),
            command=action_data.get("command"),
            options=action_data.get("options", {}),
        )

        self._conversation_history.append({
            "observation": observation,
            "action": action_data,
        })

        return action

    def reset(self) -> None:
        """Reset conversation history for new episode."""
        self._conversation_history = []


# =============================================================================
# Main inference loop
# =============================================================================

def run_episode(
    env: SREGymEnv,
    agent: K8sAgent,
    max_steps: int | None = None,
) -> tuple[EpisodeLog, str]:
    """Run a single episode with the agent.

    Returns:
        Tuple of (episode_log, initial_prompt) for structured logging.
    """
    episode_log = EpisodeLog(
        task=env.config.task_difficulty,
        difficulty=env.config.task_difficulty,
    )

    task_hint = ""
    initial_prompt = ""
    if hasattr(env._task_instance, "get_hint"):
        task_hint = env._task_instance.get_hint()
        initial_prompt = task_hint

    obs = env.reset()
    agent.reset()
    start_time = time.time()

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

        log_entry = ToolCallLog(
            timestamp=time.time(),
            step=step,
            reasoning=f"Step {step}: health={obs_dict.get('health_score', 0):.2f}, action={action.action_type.value}",
            action=action_dict,
            observation=obs_dict,
            reward=reward,
        )
        episode_log.tool_calls.append(log_entry)
        episode_log.total_reward += reward

    episode_log.success = done and obs.health_score >= 1.0
    episode_log.steps = step
    episode_log.duration = time.time() - start_time

    return episode_log, initial_prompt


def print_episode_log(episode_log: EpisodeLog, initial_prompt: str = "") -> None:
    """Print episode in structured <task>/<tool_call> format per openenv.yaml."""
    # Print task opening tag with metadata
    print(f'<task task_id={episode_log.task} initial_prompt="{initial_prompt}">')

    # Print each tool call in structured format
    for call in episode_log.tool_calls:
        print("<tool_call>")
        print(f"  reasoning: {call.reasoning}")
        print(f"  action: {json.dumps(call.action)}")
        print(f"  observation: {json.dumps(call.observation)}")
        print(f"  reward: {call.reward:.4f}")
        print("</tool_call>")

    # Print task closing tag with summary
    success_str = "true" if episode_log.success else "false"
    print(f"</task> total_reward={episode_log.total_reward:.2f} success={success_str}")


def main():
    """Run SRE Gym inference with LLM agent."""
    import argparse
    parser = argparse.ArgumentParser(description="Run SRE Gym with LLM agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard"], default="easy",
                        help="Task difficulty")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--output", type=str, help="Save logs to JSON file")
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

    env = SREGymEnv(EnvConfig(task_difficulty=args.task))
    agent = K8sAgent(model=args.model)

    print(f"Starting SRE Gym inference | Task: {args.task} | Model: {args.model}")
    print(f"Running {args.episodes} episode(s)...\n")

    all_logs = []
    for i in range(args.episodes):
        print(f"--- Episode {i + 1}/{args.episodes} ---")
        episode_log, initial_prompt = run_episode(env, agent, args.max_steps)
        print_episode_log(episode_log, initial_prompt)
        all_logs.append(episode_log)

    env.close()

    if args.output:
        output_data = [
            {
                "task": log.task,
                "difficulty": log.difficulty,
                "steps": log.steps,
                "total_reward": log.total_reward,
                "success": log.success,
                "duration": log.duration,
                "tool_calls": [
                    {
                        "step": tc.step,
                        "reasoning": tc.reasoning,
                        "action": tc.action,
                        "observation": tc.observation,
                        "reward": tc.reward,
                    }
                    for tc in log.tool_calls
                ],
            }
            for log in all_logs
        ]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Logs saved to {args.output}")


if __name__ == "__main__":
    main()
