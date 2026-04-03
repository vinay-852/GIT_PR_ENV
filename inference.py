from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from agent import IssueTriageAgent
from envs.GitHubIssueTriageManager.models import ActionType, Difficulty, GoalType
from envs.GitHubIssueTriageManager.server.environment import GitHubIssueTriageEnvironment
from envs.GitHubIssueTriageManager.server.loader import load_single_episode
from envs.GitHubIssueTriageManager.server.grader import grade_episode


def _structured_print(label: str, payload: Dict) -> None:
    print(f"[{label}] {json.dumps(payload, sort_keys=True)}", flush=True)


def _parse_github_issue_url(issue_url: str) -> tuple[str, str]:
    """
    Returns (repo_id, issue_id) from:
      https://github.com/OWNER/REPO/issues/123
    """
    parsed = urlparse(issue_url)
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) < 4 or parts[2] != "issues":
        raise ValueError(
            f"Unsupported GitHub issue URL: {issue_url}. Expected https://github.com/OWNER/REPO/issues/123"
        )

    owner = parts[0]
    repo = parts[1]
    issue_id = parts[3]
    repo_id = f"{owner}/{repo}"
    return repo_id, issue_id


def _build_task(
    *,
    task_id: str,
    repo_id: str,
    issue_id: str,
    max_steps: int,
) -> Dict:
    return {
        "task_id": task_id,
        "difficulty": Difficulty.EASY,
        "goal_type": GoalType.TRIAGE_ONLY,
        "repo_id": repo_id,
        "issue_id": issue_id,
        "max_steps": max_steps,
        "success_criteria": [],
        "allowed_actions": [
            ActionType.READ_ISSUE,
            ActionType.READ_REPO_RULES,
            ActionType.READ_LABEL_DEFINITIONS,
            ActionType.READ_TEAM_ROUTING,
            ActionType.READ_ASSIGNEE_POOL,
            ActionType.READ_MILESTONES,
            ActionType.SEARCH_SIMILAR_ISSUES,
            ActionType.ADD_LABEL,
            ActionType.REMOVE_LABEL,
            ActionType.ASSIGN_USER,
            ActionType.SET_PRIORITY,
            ActionType.SET_MILESTONE,
            ActionType.COMMENT,
            ActionType.REQUEST_INFO,
            ActionType.PROVIDE_INFO,
            ActionType.MARK_DUPLICATE,
            ActionType.CLOSE_ISSUE,
            ActionType.REOPEN_ISSUE,
            ActionType.NOOP,
        ],
        "hidden_grading_flags": {},
    }


def _observation_snapshot(observation) -> Dict:
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


def _build_provide_info_action(fields: List[str]) -> Dict[str, object]:
    return {
        "type": ActionType.PROVIDE_INFO.value,
        "fields": _collect_provided_info(fields),
    }


def run_episode(
    env: GitHubIssueTriageEnvironment,
    agent: IssueTriageAgent,
    task_id: str,
) -> Dict[str, float]:
    observation = env.reset(task_id=task_id)

    _structured_print(
        "START",
        {
            "task_id": task_id,
            "episode_id": observation.episode_id,
            "difficulty": observation.task.difficulty.value,
            "max_steps": observation.task.max_steps,
            "objective_summary": observation.objective_summary,
        },
    )

    step_index = 0

    while True:
        action = agent.next_action(observation.model_dump())
        step_result = env.step(action)

        _structured_print(
            "STEP",
            {
                "task_id": task_id,
                "step": step_index,
                "action": action,
                "reward": step_result.reward.total,
                "reward_breakdown": step_result.reward.model_dump(),
                "action_valid": step_result.info.action_valid,
                "action_effect": step_result.info.action_effect,
                "grader_notes": step_result.info.grader_notes,
                "observation": _observation_snapshot(step_result.observation),
            },
        )

        observation = step_result.observation
        step_index += 1

        if step_result.done:
            break

        # If the agent asked for more info, pause and collect the details,
        # then send them back as provide_info so they are stored in state.
        if action.get("type") == ActionType.REQUEST_INFO.value:
            requested_fields = action.get("fields") or list(
                getattr(observation, "pending_missing_fields", [])
            )
            if requested_fields:
                provide_action = _build_provide_info_action(list(requested_fields))
                provide_result = env.step(provide_action)

                _structured_print(
                    "STEP",
                    {
                        "task_id": task_id,
                        "step": step_index,
                        "action": provide_action,
                        "reward": provide_result.reward.total,
                        "reward_breakdown": provide_result.reward.model_dump(),
                        "action_valid": provide_result.info.action_valid,
                        "action_effect": provide_result.info.action_effect,
                        "grader_notes": provide_result.info.grader_notes,
                        "observation": _observation_snapshot(provide_result.observation),
                    },
                )

                observation = provide_result.observation
                step_index += 1

                if provide_result.done:
                    break

    final_state = env.state
    grade = grade_episode(final_state)

    result_payload = {
        "task_id": task_id,
        "steps_taken": step_index,
        "score": grade.score,
        "matched_labels": grade.matched_labels,
        "matched_assignee": grade.matched_assignee,
        "matched_priority": grade.matched_priority,
        "matched_milestone": grade.matched_milestone,
        "duplicate_matched": grade.duplicate_matched,
        "missing_fields_requested": grade.missing_fields_requested,
        "closed_correctly": grade.closed_correctly,
        "comment_accepted": grade.comment_accepted,
        "notes": grade.notes,
    }
    _structured_print("END", result_payload)

    return {"score": grade.score, "steps": float(step_index)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GitHub issue triage on a single issue URL.")
    parser.add_argument("--repo-rules", required=True, help="Path to repo_rules.json")
    parser.add_argument("--issue-url", required=True, help="GitHub issue URL")
    parser.add_argument("--task-id", default="triage_single_issue", help="Task ID to use")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps for the episode")
    parser.add_argument(
        "--live-github",
        action="store_true",
        help="Fetch issue data directly from GitHub using the issue URL",
    )
    args = parser.parse_args()

    repo_rules_path = Path(args.repo_rules).resolve()
    if not repo_rules_path.exists():
        raise FileNotFoundError(f"Repo rules file not found: {repo_rules_path}")

    repo_id, issue_id = _parse_github_issue_url(args.issue_url)

    task = _build_task(
        task_id=args.task_id,
        repo_id=repo_id,
        issue_id=issue_id,
        max_steps=args.max_steps,
    )

    episode = load_single_episode(
        repo_rules_path=repo_rules_path,
        task=task,
        issue=args.issue_url,
        live_github=args.live_github,
    )

    env = GitHubIssueTriageEnvironment(
        episodes=[episode],
        strict_mode=True,
        live_github=args.live_github,
    )
    agent = IssueTriageAgent()

    run_episode(env, agent, args.task_id)


if __name__ == "__main__":
    main()