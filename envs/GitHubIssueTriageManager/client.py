# envs/your_env/client.py
from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult as ClientStepResult

from .models import Action, Observation, State, StepResult


class GitHubIssueTriageClient(
    EnvClient[Action, Observation, State]
):
    """
    Client for interacting with the GitHub Issue Triage environment.

    Handles:
    - Converting typed Action -> JSON payload
    - Parsing server StepResult -> typed Observation
    - Parsing server state -> typed State
    """

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        """
        Convert typed Action into JSON payload for server.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> ClientStepResult:
        """
        Parse server response into OpenEnv StepResult.

        Server returns:
        {
            "observation": {...},
            "reward": {...},
            "done": bool,
            "info": {...}
        }
        """
        data = payload.get("result", payload)

        parsed = StepResult.model_validate(data)

        return ClientStepResult(
            observation=parsed.observation,
            reward=parsed.reward.total if parsed.reward else None,
            done=parsed.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse state endpoint response.

        Supports:
        - {"state": {...}}
        - direct state object
        """
        data = payload.get("state", payload)
        return State.model_validate(data)