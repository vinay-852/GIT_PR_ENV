from __future__ import annotations

from pathlib import Path

try:
    from GitHubIssueTriage.models import IssueComment, IssueStatus, Priority
    from GitHubIssueTriage.server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment
    from GitHubIssueTriage.server.grader import grade_episode
    from GitHubIssueTriage.server.loader import load_episode_bundle_from_paths
    from GitHubIssueTriage.server.reward import compute_reward
except ImportError:  # pragma: no cover
    from models import IssueComment, IssueStatus, Priority
    from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment
    from server.grader import grade_episode
    from server.loader import load_episode_bundle_from_paths
    from server.reward import compute_reward


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_first_episode_state():
    episodes = load_episode_bundle_from_paths(DATA_DIR)
    return episodes[0].model_copy(deep=True)


def _make_zero_progress_state():
    state = _load_first_episode_state()
    state.hidden_target = None
    state.issue.labels = []
    state.issue.assignees = []
    state.issue.priority = None
    state.issue.milestone = None
    state.issue.severity = None
    state.issue.component = None
    state.issue.linked_duplicates = []
    state.issue.comments = []
    return state


def _make_perfect_target_state():
    state = _load_first_episode_state()
    target = state.hidden_target
    assert target is not None

    state.issue.labels = list(target.gold_labels)
    state.issue.assignees = [] if target.gold_assignee is None else [target.gold_assignee]
    state.issue.priority = target.gold_priority
    state.issue.milestone = target.gold_milestone
    state.issue.severity = target.gold_severity
    state.issue.component = target.gold_component
    state.issue.linked_duplicates = (
        [] if target.gold_duplicate_issue_id is None else [target.gold_duplicate_issue_id]
    )
    state.requested_fields = list(target.required_missing_fields)

    if target.expected_comment_keywords:
        state.issue.comments = [
            IssueComment(
                comment_id="test-comment-1",
                author="triage-bot",
                body=" ".join(target.expected_comment_keywords),
                created_at="2026-01-01T00:00:00Z",
            )
        ]

    if target.gold_close_reason is None:
        state.issue.status = IssueStatus.OPEN
    else:
        state.issue.status = IssueStatus.CLOSED
        state.issue.metadata["close_reason"] = target.gold_close_reason.value

    state.step_count = 1
    return state


def _make_basic_full_progress_state():
    state = _load_first_episode_state()
    state.hidden_target = None
    state.issue.labels = ["type:bug"]
    state.issue.assignees = ["devon"]
    state.issue.priority = state.issue.priority or Priority.P1
    state.issue.milestone = state.issue.milestone or "v1.0"
    state.issue.comments = [
        IssueComment(
            comment_id="test-comment-2",
            author="triage-bot",
            body="ready for next action",
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    return state


def test_grade_episode_score_is_strict_open_interval():
    result = grade_episode(_load_first_episode_state())
    assert 0.0 < result.score < 1.0


def test_compute_reward_clamps_zero_progress_to_open_interval():
    reward = compute_reward(_make_zero_progress_state())
    assert 0.0 < reward.total < 1.0
    for key, value in reward.model_dump().items():
        if key.endswith("_score"):
            assert 0.0 < value < 1.0


def test_compute_reward_clamps_perfect_state_to_open_interval():
    reward = compute_reward(_make_perfect_target_state())
    assert 0.0 < reward.total < 1.0
    for key, value in reward.model_dump().items():
        if key.endswith("_score"):
            assert 0.0 < value < 1.0


def test_compute_reward_clamps_basic_full_progress_to_open_interval():
    reward = compute_reward(_make_basic_full_progress_state())
    assert 0.0 < reward.total < 1.0
    for key, value in reward.model_dump().items():
        if key.endswith("_score"):
            assert 0.0 < value < 1.0


def test_environment_step_outputs_strict_open_interval_scores():
    episodes = load_episode_bundle_from_paths(DATA_DIR)
    env = GitHubIssueTriageEnvironment(episodes=episodes, strict_mode=True)
    obs = env.reset(task_id=episodes[0].task.task_id)

    step_result = env.step({"type": "read_issue", "issue_id": obs.task.issue_id})

    assert 0.0 < step_result.observation.reward < 1.0
    assert env._state is not None
    assert env._state.internal_score_cache is not None
    assert 0.0 < env._state.internal_score_cache < 1.0


def test_environment_done_branch_outputs_strict_open_interval_scores():
    episodes = load_episode_bundle_from_paths(DATA_DIR)
    env = GitHubIssueTriageEnvironment(episodes=episodes, strict_mode=True)
    env.reset(task_id=episodes[0].task.task_id)
    assert env._state is not None
    env._state.done = True

    step_result = env.step({"type": "noop"})

    assert 0.0 < step_result.reward.total < 1.0
    assert 0.0 < step_result.observation.reward < 1.0
