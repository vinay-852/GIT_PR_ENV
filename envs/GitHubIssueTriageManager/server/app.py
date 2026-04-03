# envs/your_env/server/app.py
from __future__ import annotations

from openenv.core.env_server import create_fastapi_app
from .environment import GitHubIssueTriageEnvironment

app = create_fastapi_app(GitHubIssueTriageEnvironment)

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)