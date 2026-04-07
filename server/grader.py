# envs/your_env/server/grader.py
from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:
    from GitHubIssueTriage.models import (
        GraderResult,
        HiddenGradingTarget,
        IssueStatus,
        IssueTriageState,
    )
except ImportError:  # pragma: no cover
    from models import GraderResult, HiddenGradingTarget, IssueStatus, IssueTriageState


# Keep a visible safety margin from both 0 and 1.
# This protects against downstream rounding/formatting.
TASK_SCORE_EPSILON = 1e-2


def _normalize_task_score(raw_score: float, *, epsilon: float = TASK_SCORE_EPSILON) -> float:
    """
    Clamp any score into the strict open interval (0, 1).
    """
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.5

    if not math.isfinite(score):
        score = 0.5

    eps = float(epsilon)
    if not math.isfinite(eps):
        eps = TASK_SCORE_EPSILON
    eps = max(1e-6, min(0.49, eps))

    return min(1.0 - eps, max(eps, score))


def _resolve_state(state: Any) -> Any:
    """
    Accepts raw state, callable state providers, or wrappers with an observation field.
    """
    candidate = state
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:
            candidate = None

    if hasattr(candidate, "observation") and not hasattr(candidate, "issue"):
        candidate = getattr(candidate, "observation", None)

    return candidate


def _issue_from_state(state: Any) -> Any:
    return getattr(state, "issue", None)


def _hidden_target_from_state(state: Any) -> Optional[HiddenGradingTarget]:
    target = getattr(state, "hidden_target", None)
    if isinstance(target, HiddenGradingTarget):
        return target
    if isinstance(target, dict):
        validator = getattr(HiddenGradingTarget, "model_validate", None)
        if callable(validator):
            try:
                return validator(target)
            except Exception:
                return None
    if target is not None and hasattr(target, "gold_labels"):
        parser = getattr(HiddenGradingTarget, "model_validate", None)
        if callable(parser):
            try:
                return parser(getattr(target, "__dict__", {}))
            except Exception:
                return None
    return None


def _close_reason(state: Any) -> str:
    issue = _issue_from_state(state)
    if issue is None:
        return ""
    metadata = getattr(issue, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return ""
    return str(metadata.get("close_reason", "")).strip().lower()


def _labels(state: Any) -> List[str]:
    issue = _issue_from_state(state)
    values = getattr(issue, "labels", []) if issue is not None else []
    return [str(v) for v in values if isinstance(v, str) and v.strip()]


def _assignees(state: Any) -> List[str]:
    issue = _issue_from_state(state)
    values = getattr(issue, "assignees", []) if issue is not None else []
    return [str(v) for v in values if isinstance(v, str) and v.strip()]


def _linked_duplicates(state: Any) -> List[str]:
    issue = _issue_from_state(state)
    values = getattr(issue, "linked_duplicates", []) if issue is not None else []
    return [str(v) for v in values if isinstance(v, str) and v.strip()]


def _requested_fields(state: Any) -> List[str]:
    values = getattr(state, "requested_fields", []) or []
    return [str(v) for v in values if isinstance(v, str) and v.strip()]


def _comment_text(state: Any) -> str:
    issue = _issue_from_state(state)
    comments = getattr(issue, "comments", []) if issue is not None else []
    snippets: List[str] = []
    for comment in comments:
        body = getattr(comment, "body", "")
        if isinstance(body, str) and body.strip():
            snippets.append(body.lower())
    return " ".join(snippets)


def _scalar_match(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    return actual == expected


def _exact_or_empty(actual_items: Sequence[str], expected_item: Optional[str]) -> bool:
    if expected_item is None:
        return len(actual_items) == 0
    return list(actual_items) == [expected_item]


def _coverage(required: Iterable[str], present: Iterable[str]) -> Tuple[float, List[str]]:
    req = [str(x) for x in required if isinstance(x, str) and x.strip()]
    if not req:
        return 1.0, []

    present_set = {str(x) for x in present if isinstance(x, str) and str(x).strip()}
    matched = [item for item in req if item in present_set]
    return len(matched) / float(len(req)), matched


def _efficiency_score(state: Any) -> float:
    max_steps = getattr(state, "max_steps", None)
    if max_steps is None and hasattr(state, "task"):
        max_steps = getattr(state.task, "max_steps", None)
    step_count = getattr(state, "step_count", 0)

    try:
        max_steps_value = int(max_steps) if max_steps is not None else 0
    except Exception:
        max_steps_value = 0

    try:
        step_count_value = int(step_count)
    except Exception:
        step_count_value = 0

    if max_steps_value <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(step_count_value) / float(max_steps_value)))
    return 1.0 - ratio


def _matches_comment_keywords(state: Any, keywords: Sequence[str]) -> bool:
    wanted = [k.lower() for k in keywords if isinstance(k, str) and k.strip()]
    if not wanted:
        return True
    text = _comment_text(state)
    return all(word in text for word in wanted)


def _build_result(
    *,
    score: float,
    matched_labels: List[str],
    matched_assignee: bool,
    matched_priority: bool,
    matched_milestone: bool,
    duplicate_matched: bool,
    missing_fields_requested: bool,
    closed_correctly: bool,
    comment_accepted: bool,
    notes: List[str],
) -> GraderResult:
    deduped_notes: List[str] = []
    for note in notes:
        if note and note not in deduped_notes:
            deduped_notes.append(note)

    return GraderResult(
        score=_normalize_task_score(score),
        matched_labels=matched_labels,
        matched_assignee=matched_assignee,
        matched_priority=matched_priority,
        matched_milestone=matched_milestone,
        duplicate_matched=duplicate_matched,
        missing_fields_requested=missing_fields_requested,
        closed_correctly=closed_correctly,
        comment_accepted=comment_accepted,
        notes=deduped_notes,
    )


def _grade_with_hidden_target(state: Any, target: HiddenGradingTarget) -> GraderResult:
    issue = _issue_from_state(state)

    labels_cov, matched_labels = _coverage(target.gold_labels, _labels(state))
    assignee_ok = _exact_or_empty(_assignees(state), target.gold_assignee)
    priority_ok = _scalar_match(getattr(issue, "priority", None), target.gold_priority)
    milestone_ok = _scalar_match(getattr(issue, "milestone", None), target.gold_milestone)
    severity_ok = _scalar_match(getattr(issue, "severity", None), target.gold_severity)
    component_ok = _scalar_match(getattr(issue, "component", None), target.gold_component)

    duplicates = _linked_duplicates(state)
    if target.gold_duplicate_issue_id is None:
        duplicate_ok = len(duplicates) == 0
    else:
        duplicate_ok = target.gold_duplicate_issue_id in duplicates

    missing_cov, _ = _coverage(target.required_missing_fields, _requested_fields(state))
    missing_ok = missing_cov >= 0.999

    issue_status = getattr(issue, "status", None)
    if target.gold_close_reason is None:
        closure_ok = issue_status == IssueStatus.OPEN
    else:
        expected_reason = target.gold_close_reason.value.lower()
        closure_ok = issue_status == IssueStatus.CLOSED and _close_reason(state) == expected_reason

    comment_ok = _matches_comment_keywords(state, target.expected_comment_keywords)
    efficiency = _efficiency_score(state)

    score = 0.0
    score += 0.30 * labels_cov
    score += 0.14 * (1.0 if assignee_ok else 0.0)
    score += 0.08 * (1.0 if priority_ok else 0.0)
    score += 0.07 * (1.0 if milestone_ok else 0.0)
    score += 0.08 * (1.0 if severity_ok else 0.0)
    score += 0.08 * (1.0 if component_ok else 0.0)
    score += 0.07 * (1.0 if duplicate_ok else 0.0)
    score += 0.06 * missing_cov
    score += 0.05 * (1.0 if closure_ok else 0.0)
    score += 0.03 * (1.0 if comment_ok else 0.0)
    score += 0.03 * efficiency

    notes: List[str] = []
    if labels_cov < 0.999:
        notes.append("Label set is incomplete or incorrect.")
    if not assignee_ok:
        notes.append("Assignee does not match target.")
    if not priority_ok:
        notes.append("Priority does not match target.")
    if not milestone_ok:
        notes.append("Milestone does not match target.")
    if not severity_ok:
        notes.append("Severity does not match target.")
    if not component_ok:
        notes.append("Component does not match target.")
    if not duplicate_ok:
        notes.append("Duplicate handling does not match target.")
    if not missing_ok:
        notes.append("Required info fields were not fully requested.")
    if not closure_ok:
        notes.append("Closure state does not match target.")
    if not comment_ok:
        notes.append("Comment keywords did not match target.")

    return _build_result(
        score=score,
        matched_labels=matched_labels,
        matched_assignee=assignee_ok,
        matched_priority=priority_ok,
        matched_milestone=milestone_ok,
        duplicate_matched=duplicate_ok,
        missing_fields_requested=missing_ok,
        closed_correctly=closure_ok,
        comment_accepted=comment_ok,
        notes=notes,
    )


def _grade_without_hidden_target(state: Any) -> GraderResult:
    issue = _issue_from_state(state)
    labels = _labels(state)
    assignees = _assignees(state)
    duplicates = _linked_duplicates(state)
    requested_fields = _requested_fields(state)
    efficiency = _efficiency_score(state)

    priority_ok = getattr(issue, "priority", None) is not None
    milestone_ok = getattr(issue, "milestone", None) is not None
    severity_ok = getattr(issue, "severity", None) is not None
    component_ok = getattr(issue, "component", None) is not None
    comment_ok = bool(getattr(issue, "comments", []))
    duplicate_ok = len(duplicates) > 0
    closed_ok = getattr(issue, "status", None) == IssueStatus.CLOSED

    labels_score = min(1.0, len(labels) / 4.0)
    score = 0.0
    score += 0.28 * labels_score
    score += 0.14 * (1.0 if assignees else 0.0)
    score += 0.10 * (1.0 if priority_ok else 0.0)
    score += 0.10 * (1.0 if milestone_ok else 0.0)
    score += 0.10 * (1.0 if severity_ok else 0.0)
    score += 0.10 * (1.0 if component_ok else 0.0)
    score += 0.06 * (1.0 if duplicate_ok else 0.0)
    score += 0.05 * (1.0 if comment_ok else 0.0)
    score += 0.05 * efficiency

    notes: List[str] = []
    if not labels:
        notes.append("No labels added.")
    if not assignees:
        notes.append("No assignee set.")
    if not priority_ok:
        notes.append("Priority is not set.")
    if not milestone_ok:
        notes.append("Milestone is not set.")
    if not severity_ok:
        notes.append("Severity is not set.")
    if not component_ok:
        notes.append("Component is not set.")
    if not notes:
        notes.append("No hidden_target present; graded on observable state.")

    return _build_result(
        score=score,
        matched_labels=labels,
        matched_assignee=bool(assignees),
        matched_priority=priority_ok,
        matched_milestone=milestone_ok,
        duplicate_matched=duplicate_ok,
        missing_fields_requested=bool(requested_fields),
        closed_correctly=closed_ok,
        comment_accepted=comment_ok,
        notes=notes,
    )


def grade_episode(state: IssueTriageState | Any) -> GraderResult:
    """
    Deterministic task grader.

    Guarantees score is strictly within (0, 1) on every return path.
    """
    resolved = _resolve_state(state)
    issue = _issue_from_state(resolved)
    if resolved is None or issue is None:
        return _build_result(
            score=0.0,
            matched_labels=[],
            matched_assignee=False,
            matched_priority=False,
            matched_milestone=False,
            duplicate_matched=False,
            missing_fields_requested=False,
            closed_correctly=False,
            comment_accepted=False,
            notes=["Invalid state object passed to grader."],
        )

    target = _hidden_target_from_state(resolved)
    if target is None:
        return _grade_without_hidden_target(resolved)
    return _grade_with_hidden_target(resolved, target)


def is_success(state: IssueTriageState) -> bool:
    """
    Strict success check for completed episodes.
    """
    result = grade_episode(state)
    return result.score >= 0.95
