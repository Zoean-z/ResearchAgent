"""FastAPI application wiring for the first runnable vertical slice."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from research_agent.api.deps import create_repository_bundle, create_service_bundle, get_app_name
from research_agent.api.routes import memories_router, sessions_router, system_router, task_runs_router
from research_agent.config import load_env_file
from research_agent.utils import reset_request_api_key_override, set_request_api_key_override


def create_app(
    *,
    storage_backend: str | None = None,
    sqlite_path: str | Path | None = None,
) -> FastAPI:
    """Create the API application with mock repositories and stable boundaries."""

    load_env_file()
    app = FastAPI(title=get_app_name())
    @app.middleware("http")
    async def inject_request_api_key_override(request: Request, call_next):
        token = set_request_api_key_override(request.headers.get("X-Research-Agent-Api-Key"))
        try:
            return await call_next(request)
        finally:
            reset_request_api_key_override(token)

    backend_name = storage_backend or os.getenv("RESEARCH_AGENT_STORAGE_BACKEND", "sqlite")
    app.state.repositories = create_repository_bundle(storage_backend=backend_name, sqlite_path=sqlite_path)
    app.state.services = create_service_bundle(app.state.repositories)

    app.include_router(system_router)
    app.include_router(memories_router)
    app.include_router(sessions_router)
    app.include_router(task_runs_router)
    _mount_frontend(app)
    return app

def _mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend when `frontend/dist` is present."""

    frontend_dist = Path(__file__).resolve().parents[4] / "frontend" / "dist"
    assets_dir = frontend_dist / "assets"
    index_file = frontend_dist / "index.html"
    if not assets_dir.exists() or not index_file.exists():
        return

    app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(index_file)

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend_routes(full_path: str) -> FileResponse:
        if full_path.startswith("api/") or full_path == "health":
            raise HTTPException(status_code=404)
        target = frontend_dist / full_path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(index_file)


app = create_app()
