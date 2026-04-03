from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from agent import IssueTriageAgent
from envs.GitHubIssueTriageManager.client import GitHubIssueTriageClient
from envs.GitHubIssueTriageManager.models import ActionType


def _structured_print(label: str, payload: Dict[str, Any]) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _observation_snapshot(observation) -> Dict[str, Any]:
    return {
        "step_count": observation.step_count,
        "remaining_steps": observation.remaining_steps,
        "done": observation.done,
        "last_action_message": observation.last_action_message,
        "objective_summary": observation.objective_summary,
        "pending_missing_fields": getattr(observation, "pending_missing_fields", []),
        "provided_fields": getattr(observation, "provided_fields", {}),
    }


def _collect_provided_info(fields: List[str]) -> Dict[str, str]:
    print("\nThe environment requested more information.")
    print("Enter values for the fields below. Leave blank to skip any field.\n")

    values: Dict[str, str] = {}
    for field in fields:
        value = input(f"{field}: ").strip()
        if value:
            values[field] = value
    return values


def _build_provide_info_action(fields: List[str]) -> Dict[str, Any]:
    return {
        "type": ActionType.PROVIDE_INFO.value,
        "fields": _collect_provided_info(fields),
    }


def run_episode(client: GitHubIssueTriageClient, agent: IssueTriageAgent) -> Dict[str, float]:
    observation = client.reset()

    _structured_print(
        "START",
        {
            "task_id": observation.task.task_id,
            "episode_id": observation.episode_id,
            "difficulty": observation.task.difficulty.value,
            "max_steps": observation.task.max_steps,
            "objective_summary": observation.objective_summary,
        },
    )

    step_index = 0

    while True:
        action = agent.next_action(observation.model_dump())
        step_result = client.step(action)

        _structured_print(
            "STEP",
            {
                "task_id": observation.task.task_id,
                "step": step_index,
                "action": action,
                "reward": step_result.reward,
                "action_valid": step_result.done is not None,
                "observation": _observation_snapshot(step_result.observation),
            },
        )

        observation = step_result.observation
        step_index += 1

        if step_result.done:
            break

        if action.get("type") == ActionType.REQUEST_INFO.value:
            requested_fields = action.get("fields") or list(
                getattr(observation, "pending_missing_fields", [])
            )
            if requested_fields:
                provide_action = _build_provide_info_action(list(requested_fields))
                provide_result = client.step(provide_action)

                _structured_print(
                    "STEP",
                    {
                        "task_id": observation.task.task_id,
                        "step": step_index,
                        "action": provide_action,
                        "reward": provide_result.reward,
                        "action_valid": provide_result.done is not None,
                        "observation": _observation_snapshot(provide_result.observation),
                    },
                )

                observation = provide_result.observation
                step_index += 1

                if provide_result.done:
                    break

    final_state = client.state()
    _structured_print(
        "END",
        {
            "task_id": observation.task.task_id,
            "steps_taken": step_index,
            "final_episode_id": final_state.episode_id,
        },
    )

    return {"steps": float(step_index)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run triage against a Docker endpoint.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENENV_BASE_URL", "http://localhost:8000"),
        help="OpenEnv server base URL (Docker endpoint).",
    )
    args = parser.parse_args()

    client = GitHubIssueTriageClient(base_url=args.base_url)
    agent = IssueTriageAgent()

    run_episode(client, agent)


if __name__ == "__main__":
    main()