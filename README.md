---
title: GitHubIssueTriage Environment Server
emoji: "🧭"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# GitHubIssueTriage Environment

GitHubIssueTriage is an OpenEnv environment for evaluating and training agents that triage GitHub issues.

Instead of an echo loop, each episode models realistic triage work:
- Read issue details and repository policy
- Apply labels, assignment, priority, and milestone
- Request missing information
- Mark duplicates or close/reopen when appropriate
- Receive dense reward and deterministic grading against a hidden target

## What This Environment Simulates

Each episode includes:
- `repo_rules`: canonical triage policy (labels, routing, missing-info rules, templates)
- `issue`: the current issue snapshot
- `task`: allowed actions, max steps, and goal type
- `hidden_target`: gold triage outcome used for reward/grading

Episode termination:
- Max steps reached, or
- Hidden target is fully satisfied

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Server endpoints:
- `/web`: interactive OpenEnv web UI
- `/docs`: FastAPI/OpenAPI docs
- `/health`: health check
- `/ws`: websocket endpoint for session-based stepping

The web UI loads the bundled demo episode by default, so `/web` opens with a working reset state out of the box. To point the server at a different bundle, set `GITHUB_ISSUE_TRIAGE_DATA_DIR` to a folder containing `repo_rules.json`, `tasks.json`, and `issues.json` before starting the app.

### 3. Build and run with Docker (optional)

```bash
docker build -t github-issue-triage-env -f server/Dockerfile .
docker run --rm -p 8000:8000 github-issue-triage-env
```

## Configuration

Create a `.env` file in the repository root with your API and model settings:

```dotenv
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
TEMPERATURE=0.8
MAX_OUTPUT_TOKENS=200
```

The code reads these values automatically via `dotenv` when `agent.py` starts.

## Running Inference

The CLI is designed to run all tasks from `data/tasks.json` by default:

```bash
python inference.py
```

This runs the local environment and prints a final comparison table with:
- `Episode`
- `Task ID`
- `Difficulty`
- `Score`
- `Steps`

### Common CLI options

- `--repo-rules`: path to `repo_rules.json`
- `--tasks-file`: path to `tasks.json` to run all task episodes
- `--issue-file`: path to `issues.json` for single-issue fallback
- `--issue-url`: GitHub issue URL to load a single live issue
- `--live-github`: fetch issue data from GitHub for live URLs
- `--transport`: `local` or `remote` (default: `local`)
- `--base-url`: remote environment base URL when using `--transport remote`
- `--max-steps`: override the maximum steps for generated episodes

### Example: run all local tasks

```bash
python inference.py --transport local
```

### Example: run a single issue URL

```bash
python inference.py --issue-url https://github.com/OWNER/REPO/issues/123 --transport local
```

### Example: use a custom tasks file

```bash
python inference.py --tasks-file data/tasks.json
```

### Baseline scores

The baseline scores below are a reference for the default model and task configuration.

| Task ID              | Difficulty | Baseline Score | Steps |
|----------------------|------------|----------------|-------|
| triage_easy_api_p1   | easy       | 0.488          | 8     |
| needs_info_sso       | medium     | 0.717          | 10    |
| duplicate_ui_crash   | hard       | 0.688          | 10    |

Use these values to compare different model settings, task changes, or prompt updates.

## Data Loading Patterns

The environment supports three common ways to load episodes.

### A. Bundle from JSON files

Use `repo_rules.json`, `tasks.json`, and `issues.json` together.

```python
from server.loader import load_episode_bundle
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

episodes = load_episode_bundle(
    repo_rules_path="data/repo_rules.json",
    tasks_path="data/tasks.json",
    issues_path="data/issues.json",
    live_github=False,
)

env = GitHubIssueTriageEnvironment(episodes=episodes)
obs = env.reset()
```

### B. Folder shortcut

If files are in one folder:

```python
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

env = GitHubIssueTriageEnvironment(data_dir="data")
obs = env.reset()
```

### C. One-off episode from repo rules + issue URL

Useful for inference-style workflows against a single issue.

```python
from server.loader import load_episode_from_source
from server.GitHubIssueTriage_environment import GitHubIssueTriageEnvironment

state = load_episode_from_source(
    repo_rules_path="data/url_repo_rules.json",
    issue_source="https://github.com/owner/repo/issues/123",
    live_github=True,
    max_steps=10,
)

env = GitHubIssueTriageEnvironment(episodes=[state], live_github=True)
obs = env.reset()
```

## Action Space

Key action types include:
- Read actions: `read_issue`, `read_repo_rules`, `read_label_definitions`, `read_team_routing`, `read_assignee_pool`, `read_milestones`, `search_similar_issues`
- Triage actions: `add_label`, `remove_label`, `assign_user`, `set_priority`, `set_milestone`
- Communication actions: `comment`, `request_info`, `provide_info`
- Lifecycle actions: `mark_duplicate`, `close_issue`, `reopen_issue`, `noop`

Action payloads are validated by schema and policy constraints (allowed actions, valid labels, valid assignees, milestone checks, strict-mode label conflicts).

## Reward and Grading

Reward is dense and deterministic.

Main components include:
- Label/type match
- Severity/component match
- Assignee/priority/milestone match
- Missing-info request coverage
- Duplicate handling
- Closure correctness
- Comment keyword quality

Penalties are applied for invalid actions and destructive closures.

Final episode quality can be evaluated with:

```python
from server.grader import grade_episode

result = grade_episode(env._state)  # score in [0, 1]
print(result.score, result.notes)
```

## Hugging Face Spaces Deployment

From the repository root:

```bash
openenv push --repo-id <your-namespace>/GitHubIssueTriageManager
```

`openenv push` validates `openenv.yaml`, builds the Docker Space, and publishes the environment.

## Project Structure

```text
.
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── actions.py
    ├── app.py
    ├── GitHubIssueTriage_environment.py
    ├── grader.py
    ├── loader.py
    ├── observation.py
    ├── reward.py
    ├── termination.py
    ├── transitions.py
    └── Dockerfile
```


