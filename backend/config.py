# backend/config.py
"""
Process-wide objects for the web app, each with exactly one owner.

Three things must exist once per process and be shared: the resolved
:class:`~multi_agent_generator.settings.Settings`, the :class:`~multi_agent_generator.storage.Storage`
handle (which owns the database), and the :class:`~multi_agent_generator.core.GeneratorService`
that runs pipelines. Creating any of them per request would be a bug of a different kind for
each one - re-reading ``.env`` on every call, opening a new SQLite connection per request, and
re-validating the provider (a network round-trip) before every generation.

They live here rather than as module-level globals scattered through the routers so that
there is one place to look for "what is shared" and one place for tests to reset it. Nothing
in :mod:`backend.api` constructs these; routes receive them through :mod:`backend.deps`.
"""
from __future__ import annotations

from typing import Optional

from multi_agent_generator.core.service import GeneratorService, build_service
from multi_agent_generator.logging_config import configure_logging, get_logger
from multi_agent_generator.settings import Settings, get_settings
from multi_agent_generator.storage import Storage

__all__ = ["AppContext", "get_context", "set_context", "reset_context"]

log = get_logger("backend.config")


class AppContext:
    """
    The application's long-lived collaborators.

    Constructed once at startup. ``storage`` is optional: if the database cannot be opened -
    a read-only directory, a bad ``DATABASE_URL`` - the API still serves, generating projects
    that simply are not saved. Refusing to start would turn a degraded feature into a total
    outage, and the failure is reported on ``/api/health`` rather than hidden.
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        storage: Optional[Storage] = None,
        service: Optional[GeneratorService] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.storage_error: Optional[str] = None

        if storage is not None:
            self.storage: Optional[Storage] = storage
        else:
            try:
                self.storage = Storage(settings=self.settings)
            except Exception as exc:  # noqa: BLE001 - degrade, do not refuse to boot
                self.storage = None
                self.storage_error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "Persistence is unavailable; projects will not be saved.",
                    extra={"error_code": "storage_unavailable"},
                )

        if service is not None:
            self.service = service
        elif self.storage is not None:
            # The run repository satisfies the RunStore protocol, so passing it is all the
            # wiring persistence needs - the service itself never learns SQLite exists.
            from multi_agent_generator.execution import run_project_tests

            self.service = GeneratorService(
                settings=self.settings,
                store=self.storage.runs,
                test_runner=run_project_tests,
            )
        else:
            self.service = build_service(persist=False, settings=self.settings)

    @property
    def persistent(self) -> bool:
        return self.storage is not None

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()


_context: Optional[AppContext] = None


def get_context() -> AppContext:
    """The shared context, created on first use."""
    global _context
    if _context is None:
        configure_logging()
        _context = AppContext()
    return _context


def set_context(context: Optional[AppContext]) -> None:
    """Install a context. Used by the app factory at startup and by tests."""
    global _context
    _context = context


def reset_context() -> None:
    """Close and discard the shared context."""
    global _context
    if _context is not None:
        _context.close()
    _context = None
