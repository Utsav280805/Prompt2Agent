# backend/api/__init__.py
"""
The HTTP surface, one router per resource.

Routers are assembled here rather than in ``app.py`` so that the application factory has one
import and one ``include_router`` call, and so the URL layout can be read in a single place:

===========================================  ===================================================
``GET  /api/health``                          liveness plus a report of what is configured
``GET  /api/frameworks``                      the frameworks the picker offers
``GET  /api/providers``                       providers, with installed/configured flags
``POST /api/analyze``                         preview the analysis without generating
``POST /api/projects``                        start a generation run; returns a run id at once
``GET  /api/projects``                        the workspace list
``GET  /api/projects/{id}``                   one project, with its generated files
``DELETE /api/projects/{id}``                 delete a project
``GET  /api/projects/{id}/files``             the file tree
``GET  /api/projects/{id}/download``          the project as a zip
``GET  /api/projects/{id}/runs``              run history for a project
``POST /api/projects/{id}/run``               the Playground: execute the agent for real
``GET  /api/runs/{id}``                       run status and result
``GET  /api/runs/{id}/events``                live progress as server-sent events
``GET  /api/settings/llm``                    saved LLM settings, never the key
``PUT  /api/settings/llm``                    save LLM settings
``DELETE /api/settings/llm/key``              remove the stored credential
``POST /api/settings/llm/test``               test a provider configuration
===========================================  ===================================================
"""
from __future__ import annotations

from fastapi import APIRouter

from .analyze import router as analyze_router
from .health import router as health_router
from .meta import router as meta_router
from .playground import router as playground_router
from .projects import router as projects_router
from .runs import router as runs_router
from .settings import router as settings_router

__all__ = ["api_router"]

api_router = APIRouter(prefix="/api")

# Order matters only for documentation grouping; paths do not overlap.
api_router.include_router(health_router)
api_router.include_router(meta_router)
api_router.include_router(analyze_router)
api_router.include_router(projects_router)
api_router.include_router(playground_router)
api_router.include_router(runs_router)
api_router.include_router(settings_router)
