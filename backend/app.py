# backend/app.py
"""
The FastAPI application.

Everything the process needs to serve HTTP is assembled here and nowhere else: logging,
the shared context, CORS, error handlers, the routers, and - when a built frontend exists -
the static files. Keeping assembly in one factory rather than at module scope is what makes
the app testable: a test can call ``create_app(context=AppContext(storage=...))`` and get a
fully wired application pointed at a temporary database, with no environment to patch and no
global to reset afterwards.

Three decisions are worth stating.

**Startup does real work, and failing at it is allowed to be loud - once.** Opening the
database, running migrations and resolving settings happen before the first request, so a
misconfigured deployment is visible immediately rather than as a burst of 500s later. The one
exception is persistence itself: :class:`~backend.config.AppContext` degrades to no database
rather than refusing to boot, and ``/api/health`` reports it.

**Shutdown is bounded.** The job manager holds a thread pool whose workers can be inside a
subprocess. Waiting for them on Ctrl-C would make the server appear hung, so the pool is
cancelled rather than drained - each run's own timeout already caps how long its child can
live, and an interrupted run is recorded as failed rather than left claiming to be running.

**The frontend is served only if it has been built.** A dev setup runs Vite on :5173 and talks
to this server across CORS; a production one builds to ``frontend/dist`` and is served from
here so there is a single origin and no CORS at all. Mounting the directory only when it
exists means the same code covers both without a flag to get wrong.
"""
from __future__ import annotations

import contextlib
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from multi_agent_generator.logging_config import configure_logging, get_logger
from multi_agent_generator.settings import Settings, get_settings

from .api import api_router
from .config import AppContext, reset_context, set_context
from .errors import install_error_handlers
from .services.jobs import reset_job_manager
from .services.playground import reset_playground_manager

__all__ = ["create_app", "app"]

log = get_logger("backend.app")

DESCRIPTION = """
Describe a task in plain English and get a working multi-agent project: analysed, designed,
generated as real files, reviewed, improved, tested, and runnable from the Playground.

Every error this API returns has the same shape - `code`, `message`, `action` - so a client
needs one error component rather than a guess per endpoint. Generation is asynchronous:
`POST /api/projects` returns a `run_id` immediately and progress streams from
`GET /api/runs/{run_id}/events`.
"""

#: Where a production frontend build lands. Checked at startup, not at import, so a build
#: produced after the module was imported (a dev server restart) is still picked up.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app(
    *,
    context: Optional[AppContext] = None,
    settings: Optional[Settings] = None,
    serve_frontend: bool = True,
) -> FastAPI:
    """
    Build the application.

    ``context`` is the seam tests use: pass one built around a temporary database and the app
    never touches the real one. In production it is None and the context is created at
    startup, which is the only time settings are read and the only time the database is
    opened.
    """
    resolved_settings = settings or get_settings()
    configure_logging()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # --- startup ---------------------------------------------------------------
        resolved_settings.ensure_dirs()
        ctx = context or AppContext(settings=resolved_settings)
        set_context(ctx)

        log.info(
            "API ready.",
            extra={
                "persistence": ctx.persistent,
                "default_provider": ctx.settings.default_provider,
            },
        )
        if not ctx.persistent:
            # Worth a line of its own: the API works, but nothing a user generates will
            # survive a refresh, and that is a surprising way to lose work silently.
            log.warning(
                "Running without persistence - projects will not be saved.",
                extra={"error_code": "storage_unavailable", "detail": ctx.storage_error},
            )

        try:
            yield
        finally:
            # --- shutdown ----------------------------------------------------------
            # Order matters: stop accepting and cancel queued work first, then close the
            # database. Closing storage while a worker still holds it would turn a clean
            # stop into an exception in a thread nobody is watching.
            reset_job_manager()
            reset_playground_manager()
            if context is None:
                # Only close what we opened. A context handed in by a test belongs to the
                # test, which may still want to assert against it.
                reset_context()
            log.info("API stopped.")

    app = FastAPI(
        title="Multi-Agent Generator",
        description=DESCRIPTION,
        version=_version(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # CORS is for the dev setup, where Vite serves the UI from a different origin. The list
    # is explicit rather than "*" because credentials are allowed: a wildcard with
    # credentials is both rejected by browsers and the wrong thing to want.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        # Lets the browser read the filename on a project download.
        expose_headers=["Content-Disposition"],
    )

    # Installed before the routers so that a failure raised while resolving a dependency is
    # translated too, not just one raised inside a handler body.
    install_error_handlers(app)
    app.include_router(api_router)

    if serve_frontend:
        _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """
    Serve the built React app, if there is one.

    Mounted at the root *after* the API router, so ``/api/...`` always wins - a static mount
    at ``/`` would otherwise swallow every unmatched path including API typos, turning a
    404 with a useful message into index.html and a very confusing client-side error.
    """
    if not FRONTEND_DIST.is_dir() or not (FRONTEND_DIST / "index.html").is_file():
        return

    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    index = FRONTEND_DIST / "index.html"

    @app.get("/", include_in_schema=False)
    async def _index() -> FileResponse:
        return FileResponse(str(index))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str):
        """
        Client-side routing fallback.

        The React router owns paths like ``/projects/abc``; a browser refresh there asks this
        server for a file that does not exist. Returning index.html lets the app boot and
        route itself. API and docs paths are excluded so a genuine mistake there still gets a
        JSON 404 rather than a page of HTML a fetch() cannot parse.
        """
        if full_path.startswith(("api/", "docs", "openapi.json")):
            return JSONResponse(
                status_code=404,
                content={
                    "code": "not_found",
                    "message": "That endpoint does not exist.",
                    "action": "Check the path against /docs.",
                },
            )
        candidate = (FRONTEND_DIST / full_path).resolve()
        # Resolve and re-check the parent: without it, a request for ``../../.env`` would
        # escape the build directory. This is the same rule the sandbox applies to workspace
        # paths, for the same reason.
        if candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(str(candidate))
        return FileResponse(str(index))


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("multi-agent-generator")
    except PackageNotFoundError:
        return "dev"


#: The ASGI application, for ``uvicorn backend.app:app``. Built at import because that is
#: what uvicorn's string target requires; all the expensive work still happens in lifespan.
app = create_app()


def run(host: str | None = None, port: int | None = None) -> None:
    """
    Start a development server. Backs ``multi-agent-generator --serve``.

    ``host`` and ``port`` override the settings, so the CLI can pass ``--host``/``--port``
    without having to mutate global settings first.

    Reload is off: the pipeline spawns subprocesses and holds a thread pool, and a reloader
    that restarts the parent mid-run leaves orphans. Someone who wants reload can ask uvicorn
    for it directly, having accepted that.
    """
    import uvicorn

    settings = get_settings()
    configure_logging()
    bind_host = host or settings.host
    bind_port = port or settings.port
    if _port_is_occupied(bind_host, bind_port):
        if _existing_backend_is_healthy(bind_host, bind_port):
            print(
                f"The web application is already running at http://{bind_host}:{bind_port}."
            )
            return
        raise RuntimeError(
            f"Port {bind_port} is already in use. Stop the process using it or start the "
            f"backend with --port {bind_port + 1}."
        )
    uvicorn.run(
        "backend.app:app",
        host=bind_host,
        port=bind_port,
        log_config=None,  # keep our formatter; uvicorn's would install its own handlers
    )


def _port_is_occupied(host: str, port: int) -> bool:
    """Check the bind address before handing it to Uvicorn."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) == 0


def _existing_backend_is_healthy(host: str, port: int) -> bool:
    """Distinguish our already-running API from an unrelated process."""
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/health", timeout=0.75
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        return isinstance(body, dict) and body.get("version") == _version()
    except (OSError, ValueError, urllib.error.URLError):
        return False


if __name__ == "__main__":  # pragma: no cover
    run()
