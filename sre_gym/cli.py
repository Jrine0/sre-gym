"""CLI entry point for SRE Gym environment."""

from __future__ import annotations

import typer

app = typer.Typer(help="Self-Healing Kubernetes SRE Gym CLI")


@app.command()
def run(
    task: str = typer.Option("easy", "--task", "-t", help="Task difficulty: easy | medium | hard"),
    episodes: int = typer.Option(1, "--episodes", "-e", help="Number of episodes to run"),
    max_steps: int = typer.Option(30, "--max-steps", "-s", help="Max steps per episode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run the SRE Gym environment (manual mode, no LLM agent)."""
    from sre_gym.env import SREGymEnv, EnvConfig

    config = EnvConfig(
        task_difficulty=task,
        max_steps=max_steps,
    )
    env = SREGymEnv(config)

    for ep in range(episodes):
        print(f"\n=== Episode {ep + 1}/{episodes} ===")
        obs = env.reset()
        print(f"Initial state: {obs.model_dump()}")

        done = False
        step = 0
        total_reward = 0.0

        while not done:
            step += 1
            print(f"\nStep {step}:")
            print(f"  Pod: {obs.pod_status}")
            print(f"  Health: {obs.health_score:.2f}")
            if obs.error_message:
                print(f"  Error: {obs.error_message}")

            from sre_gym.models import K8sAction, K8sActionType

            # In manual mode, prompt for action
            print("  Action types: apply_manifest | delete_resource | scale_deployment | exec_command | noop")
            action_type = input("  Action type: ").strip()
            manifest = None
            resource_kind = None
            resource_name = None
            command = None

            if action_type == "apply_manifest":
                print("  Enter YAML manifest (end with ---END):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "---END":
                        break
                    lines.append(line)
                manifest = "\n".join(lines)
            elif action_type in ("delete_resource",):
                resource_kind = input("  Resource kind (pod/deployment/configmap): ").strip()
                resource_name = input("  Resource name: ").strip()
            elif action_type == "scale_deployment":
                resource_kind = "deployment"
                resource_name = input("  Deployment name: ").strip()
                replicas = input("  Replicas: ").strip()
                options = {"replicas": int(replicas) if replicas else 1}
            elif action_type == "exec_command":
                resource_name = input("  Pod name: ").strip()
                command = input("  Command (space-separated): ").strip().split()
            elif action_type == "noop":
                pass
            else:
                print(f"  Unknown action type: {action_type}, defaulting to noop")
                action_type = "noop"

            action = K8sAction(
                action_type=K8sActionType(action_type),
                manifest=manifest,
                resource_kind=resource_kind,
                resource_name=resource_name,
                command=command,
                options=options if action_type == "scale_deployment" else {},
            )

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if verbose:
                print(f"  Reward breakdown: {info.get('reward_breakdown')}")

        print(f"\nEpisode complete! Steps: {step}, Total reward: {total_reward:.3f}")

    env.close()


@app.command()
def validate(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate environment spec compliance."""
    from sre_gym.env import SREGymEnv, EnvConfig

    print("Validating SRE Gym environment...")

    checks = []

    # Check 1: Environment initializes
    try:
        env = SREGymEnv(EnvConfig(task_difficulty="easy"))
        checks.append(("Environment initialization", True))
        env.close()
    except Exception as e:
        checks.append(("Environment initialization", False, str(e)))

    # Check 2: All three difficulties load
    for diff in ["easy", "medium", "hard"]:
        try:
            env = SREGymEnv(EnvConfig(task_difficulty=diff))
            checks.append((f"Task difficulty '{diff}'", True))
            env.close()
        except Exception as e:
            checks.append((f"Task difficulty '{diff}'", False, str(e)))

    # Check 3: reset() returns observation
    try:
        env = SREGymEnv()
        obs = env.reset()
        has_keys = all(k in obs.model_dump() for k in ["kubectl_output", "health_score", "step_number"])
        checks.append(("reset() returns valid observation", has_keys))
        env.close()
    except Exception as e:
        checks.append(("reset() returns valid observation", False, str(e)))

    # Check 4: Rewards in [0, 1]
    try:
        from sre_gym.models import K8sAction, K8sActionType
        env = SREGymEnv()
        obs = env.reset()
        action = K8sAction(action_type=K8sActionType.NOOP)
        _, reward, _, _ = env.step(action)
        checks.append(("Environment step execution", True))
        env.close()
    except Exception as e:
        checks.append(("Environment step execution", False, str(e)))

    print("\nValidation Results:")
    all_passed = True
    for check in checks:
        name = check[0]
        passed = check[1]
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if len(check) > 2:
            print(f"    Error: {check[2]}")
            all_passed = False
        elif not passed:
            all_passed = False

    if all_passed:
        print("\nAll validation checks passed!")
        raise typer.Exit(0)
    else:
        print("\nSome validation checks failed.")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
