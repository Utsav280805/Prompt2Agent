# backend/deps.py
"""
FastAPI dependencies.

The only place a route obtains a shared object. Routes declare what they need as a
parameter - ``storage: Storage = Depends(require_storage)`` - and receive it, rather than
importing a module-level singleton themselves.

The gain is not ceremony, it is two concrete things. Overriding a dependency in a test is one
line (``app.dependency_overrides[get_service] = ...``), which makes the API testable without a
database or a live provider. And a route that needs persistence declares that by depending on
:func:`require_storage`, which raises a proper :class:`AppError` when the database is
unavailable - so the "we booted without a database" case is handled once here instead of with
a ``if storage is None`` check at the top of a dozen handlers.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Path

from multi_agent_generator.core.service import GeneratorService
from multi_agent_generator.errors import ConfigurationError, NotFoundError
from multi_agent_generator.settings import Settings
from multi_agent_generator.storage import Storage

from .config import AppContext, get_context

__all__ = [
    "context",
    "settings",
    "service",
    "optional_storage",
    "require_storage",
    "project_record",
]


def context() -> AppContext:
    return get_context()


def settings(ctx: AppContext = Depends(context)) -> Settings:
    return ctx.settings


def service(ctx: AppContext = Depends(context)) -> GeneratorService:
    return ctx.service


def optional_storage(ctx: AppContext = Depends(context)) -> Optional[Storage]:
    """Storage if it is available, else None. For endpoints that can degrade."""
    return ctx.storage


def require_storage(ctx: AppContext = Depends(context)) -> Storage:
    """
    Storage, or a clean 400 explaining why there is none.

    Listing saved projects without a database is not a server bug - it is a deployment whose
    data directory is unwritable or misconfigured - so it gets an actionable configuration
    error rather than a 500.
    """
    if ctx.storage is None:
        raise ConfigurationError(
            "This action needs the project database, which could not be opened.",
            action=(
                "Check that DATA_DIR is writable and DATABASE_URL points at a SQLite file, "
                "then restart the server."
            ),
            detail=ctx.storage_error,
        )
    return ctx.storage


def project_record(
    project_id: str = Path(..., min_length=1, max_length=64),
    store: Storage = Depends(require_storage),
) -> dict:
    """
    Load a project by id or raise :class:`NotFoundError`.

    Every ``/api/projects/{id}/...`` route needs this, and putting it in a dependency means
    the 404 is produced identically everywhere - including for the sub-resources, where
    forgetting the check would otherwise surface as an ``AttributeError`` on None.
    """
    record = store.projects.get(project_id)
    if record is None:
        raise NotFoundError(
            "That project does not exist.",
            action="It may have been deleted. Go back to the project list.",
            context={"project_id": project_id},
        )
    return record
