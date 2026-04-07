from __future__ import annotations

from pathlib import Path

try:
    from GitHubIssueTriage.models import ActionType
    from GitHubIssueTriage.server.loader import load_episode_bundle, load_episode_bundle_from_paths, load_issues
except ImportError:  # pragma: no cover
    from models import ActionType
    from server.loader import load_episode_bundle, load_episode_bundle_from_paths, load_issues


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _copy_data_file(name: str, target_dir: Path) -> Path:
    src = DATA_DIR / name
    dst = target_dir / name
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_load_episode_bundle_from_paths_generates_tasks_when_missing_tasks_file(tmp_path: Path):
    repo_rules_path = _copy_data_file("repo_rules.json", tmp_path)
    issues_path = _copy_data_file("issues.json", tmp_path)
    issues = load_issues(issues_path)

    episodes = load_episode_bundle_from_paths(tmp_path)

    assert len(episodes) == len(issues)
    assert episodes
    assert all(ep.task.task_id.startswith("auto_") for ep in episodes)
    assert all(ActionType.READ_ISSUE in ep.task.allowed_actions for ep in episodes)
    assert all(ep.task.issue_id for ep in episodes)
    assert repo_rules_path.exists()


def test_load_episode_bundle_generates_tasks_when_tasks_path_is_missing(tmp_path: Path):
    repo_rules_path = _copy_data_file("repo_rules.json", tmp_path)
    issues_path = _copy_data_file("issues.json", tmp_path)
    missing_tasks_path = tmp_path / "tasks.json"
    issues = load_issues(issues_path)

    episodes = load_episode_bundle(
        repo_rules_path=repo_rules_path,
        tasks_path=missing_tasks_path,
        issues_path=issues_path,
    )

    assert len(episodes) == len(issues)
    assert all(ep.task.task_id.startswith("auto_") for ep in episodes)
    assert all(ep.task.max_steps == 10 for ep in episodes)
