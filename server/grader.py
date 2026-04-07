# envs/your_env/server/grader.py
from __future__ import annotations

import math
from typing import Any, List, Set

try:
    from GitHubIssueTriage.models import GraderResult, HiddenGradingTarget, IssueStatus, IssueTriageState
except ImportError:  # pragma: no cover
    from models import GraderResult, HiddenGradingTarget, IssueStatus, IssueTriageState


def _labels_set(state: IssueTriageState) -> Set[str]:
    return set(state.issue.labels)


def _comment_text(state: IssueTriageState) -> str:
    return " ".join(comment.body.lower() for comment in state.issue.comments)


def _requested_fields_set(state: IssueTriageState) -> Set[str]:
    return set(state.requested_fields)


def _close_reason(state: IssueTriageState) -> str:
    return str(state.issue.metadata.get("close_reason", "")).lower()


def _matched_comment_keywords(state: IssueTriageState, keywords: List[str]) -> bool:
    if not keywords:
        return True
    text = _comment_text(state)
    return all(keyword.lower() in text for keyword in keywords)


def _normalize_task_score(raw_score: float, *, epsilon: float = 1e-6) -> float:
    """
    Normalize score into the strict open interval (0, 1).
    """
    if not math.isfinite(raw_score):
        raw_score = 0.5
    return min(1.0 - epsilon, max(epsilon, raw_score))


def _grade_labels(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str], List[str]]:
    if not target.gold_labels:
        return True, 1.0, [], []

    labels = _labels_set(state)
    matched = [label for label in target.gold_labels if label in labels]
    partial = len(matched) / float(len(target.gold_labels))
    ok = partial == 1.0
    notes = [] if ok else ["Label set incomplete or incorrect."]
    return ok, partial, matched, notes


def _grade_assignee(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_assignee is None:
        ok = len(state.issue.assignees) == 0
    else:
        ok = state.issue.assignees == [target.gold_assignee]
    notes = [] if ok else ["Assignee does not match target."]
    return ok, notes


def _grade_priority(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_priority is None:
        ok = state.issue.priority is None
    else:
        ok = state.issue.priority == target.gold_priority
    notes = [] if ok else ["Priority does not match target."]
    return ok, notes


def _grade_milestone(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_milestone is None:
        ok = state.issue.milestone is None
    else:
        ok = state.issue.milestone == target.gold_milestone
    notes = [] if ok else ["Milestone does not match target."]
    return ok, notes


def _grade_severity(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_severity is None:
        ok = state.issue.severity is None
    else:
        ok = state.issue.severity == target.gold_severity
    notes = [] if ok else ["Severity does not match target."]
    return ok, notes


def _grade_component(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_component is None:
        ok = state.issue.component is None
    else:
        ok = state.issue.component == target.gold_component
    notes = [] if ok else ["Component does not match target."]
    return ok, notes


def _grade_duplicate(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    if target.gold_duplicate_issue_id is None:
        ok = len(state.issue.linked_duplicates) == 0
    else:
        ok = target.gold_duplicate_issue_id in state.issue.linked_duplicates
    notes = [] if ok else ["Duplicate target not linked correctly."]
    return ok, notes


def _grade_missing_info(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str]]:
    if not target.required_missing_fields:
        return True, 1.0, []

    requested = _requested_fields_set(state)
    required = set(target.required_missing_fields)
    matched = requested.intersection(required)
    partial = len(matched) / float(len(required)) if required else 1.0
    ok = partial == 1.0
    notes = [] if ok else ["Required missing fields were not fully requested."]
    return ok, partial, notes


def _grade_closure(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, float, List[str]]:
    if target.gold_close_reason is None:
        ok = state.issue.status == IssueStatus.OPEN
        notes = [] if ok else ["Closure state does not match target."]
        return ok, 0.0, notes

    ok = state.issue.status == IssueStatus.CLOSED
    notes: List[str] = []
    bonus = 0.0
    if ok:
        if _close_reason(state) == target.gold_close_reason.value:
            bonus = 0.02
        else:
            notes.append("Close reason does not match target.")
    else:
        notes.append("Closure state does not match target.")
    return ok, bonus, notes


def _grade_comment(state: IssueTriageState, target: HiddenGradingTarget) -> tuple[bool, List[str]]:
    ok = _matched_comment_keywords(state, target.expected_comment_keywords)
    notes = [] if ok else ["Comment keywords did not match target."]
    return ok, notes


def grade_episode(state: IssueTriageState | Any) -> GraderResult:
    """
    Advanced deterministic grader supporting both IssueTriageState and Observation objects.

    Score range: strictly between 0.0 and 1.0
    
    Scoring breakdown:
      - Labels: 35% (gold label coverage)
      - Assignee: 20% (presence)
      - Priority/Severity/Component: 20% (presence combined)
      - Milestone: 10% (presence)
      - Duplicate handling: 10% (if applicable)
      - Step efficiency bonus: up to 5%
    """
    # Handle callable state providers
    if callable(state):
        try:
            state = state()
        except:
            state = None

    # Try to extract observation from state if it's nested
    if hasattr(state, "observation") and not hasattr(state, "issue"):
        state = state.observation

    # Handle Observation objects (from remote sessions)
    if hasattr(state, "issue") and hasattr(state, "task") and not hasattr(state, "hidden_target"):
        # This is an Observation, build a minimal scoring result from it
        return _grade_observation(state)

    # Validate state has required attributes
    if not hasattr(state, "hidden_target") or not hasattr(state, "issue"):
        return GraderResult(
            score=_normalize_task_score(0.0),
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

    target = state.hidden_target
    if target is None:
        # Fallback grading when no hidden target exists
        return _grade_observation_without_target(state)

    # Full scoring with hidden target
    labels_ok, labels_partial, matched_labels, label_notes = _grade_labels(state, target)
    assignee_ok, assignee_notes = _grade_assignee(state, target)
    priority_ok, priority_notes = _grade_priority(state, target)
    milestone_ok, milestone_notes = _grade_milestone(state, target)
    severity_ok, severity_notes = _grade_severity(state, target)
    component_ok, component_notes = _grade_component(state, target)
    duplicate_ok, duplicate_notes = _grade_duplicate(state, target)
    missing_info_ok, missing_info_partial, missing_notes = _grade_missing_info(state, target)
    closure_ok, closure_bonus, closure_notes = _grade_closure(state, target)
    comment_ok, comment_notes = _grade_comment(state, target)

    # Advanced scoring with weighted components
    score = 0.0
    score += 0.35 * (1.0 if labels_ok else labels_partial)  # 35% for labels
    score += 0.20 if assignee_ok else 0.0  # 20% for assignee
    score += 0.20 * (
        (1.0 if priority_ok else 0.0)
        + (1.0 if severity_ok else 0.0)
        + (1.0 if component_ok else 0.0)
    ) / 3.0  # 20% for priority/severity/component
    score += 0.10 if milestone_ok else 0.0  # 10% for milestone
    score += 0.10 if duplicate_ok else 0.0  # 10% for duplicate handling
    
    # Step efficiency bonus (up to 5%)
    max_steps = getattr(state, "max_steps", 10)
    step_count = getattr(state, "step_count", 0)
    if max_steps > 0 and step_count > 0:
        efficiency = max(0.0, 1.0 - (step_count / max_steps))
        score += 0.05 * efficiency

    score = max(0.0, min(1.0, score))
    score += closure_bonus  # Add closure bonus on top
    score = _normalize_task_score(score)

    notes: List[str] = []
    for bucket in (
        label_notes,
        assignee_notes,
        priority_notes,
        milestone_notes,
        severity_notes,
        component_notes,
        duplicate_notes,
        missing_notes,
        closure_notes,
        comment_notes,
    ):
        for message in bucket:
            if message and message not in notes:
                notes.append(message)

    return GraderResult(
        score=score,
        matched_labels=matched_labels,
        matched_assignee=assignee_ok,
        matched_priority=priority_ok,
        matched_milestone=milestone_ok,
        duplicate_matched=duplicate_ok,
        missing_fields_requested=missing_info_ok,
        closed_correctly=closure_ok,
        comment_accepted=comment_ok,
        notes=notes,
    )


def _grade_observation_without_target(state: Any) -> GraderResult:
    """Grade an IssueTriageState without hidden target using observable state."""
    issue = getattr(state, "issue", None)
    if not issue:
        return GraderResult(
            score=_normalize_task_score(0.0),
            matched_labels=[],
            matched_assignee=False,
            matched_priority=False,
            matched_milestone=False,
            duplicate_matched=False,
            missing_fields_requested=False,
            closed_correctly=False,
            comment_accepted=False,
            notes=["No issue data available for grading."],
        )

    # Score based on observable triage actions taken
    score = 0.0
    notes: List[str] = []

    label_count = len(getattr(issue, "labels", []))
    if label_count > 0:
        score += 0.35 * min(1.0, label_count / 3.0)
    else:
        notes.append("No labels added.")

    assignees = getattr(issue, "assignees", [])
    if assignees:
        score += 0.20
    else:
        notes.append("No assignee set.")

    if getattr(issue, "priority", None):
        score += 0.067
    if getattr(issue, "milestone", None):
        score += 0.067
    if getattr(issue, "severity", None):
        score += 0.067
    if getattr(issue, "component", None):
        score += 0.067
    else:
        notes.append("Missing priority/severity/component/milestone.")

    if getattr(issue, "linked_duplicates", []):
        score += 0.10
    if getattr(issue, "comments", []):
        score += 0.05

    # Efficiency bonus
    max_steps = getattr(state, "max_steps", 10)
    step_count = getattr(state, "step_count", 0)
    if max_steps > 0 and step_count > 0:
        efficiency = max(0.0, 1.0 - (step_count / max_steps))
        score += 0.05 * efficiency

    score = _normalize_task_score(score)
    if not notes:
        notes.append("No hidden_target present; graded on observable state.")

    return GraderResult(
        score=score,
        matched_labels=list(getattr(issue, "labels", [])),
        matched_assignee=bool(assignees),
        matched_priority=getattr(issue, "priority", None) is not None,
        matched_milestone=getattr(issue, "milestone", None) is not None,
        duplicate_matched=bool(getattr(issue, "linked_duplicates", [])),
        missing_fields_requested=False,
        closed_correctly=getattr(issue, "status", None) == IssueStatus.CLOSED,
        comment_accepted=bool(getattr(issue, "comments", [])),
        notes=notes,
    )


def _grade_observation(obs: Any) -> GraderResult:
    """Grade an Observation object (from remote sessions) without hidden target."""
    issue = getattr(obs, "issue", None)
    if not issue:
        return GraderResult(
            score=_normalize_task_score(0.0),
            matched_labels=[],
            matched_assignee=False,
            matched_priority=False,
            matched_milestone=False,
            duplicate_matched=False,
            missing_fields_requested=False,
            closed_correctly=False,
            comment_accepted=False,
            notes=["Invalid observation: no issue data."],
        )

    return _grade_observation_without_target(obs)


def is_success(state: IssueTriageState) -> bool:
    """
    Strict success check for completed episodes.
    """
    result = grade_episode(state)
    return result.score >= 0.95
