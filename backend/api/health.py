# backend/api/health.py
"""
Liveness, and an honest report of what is actually configured.

A health check that only ever says ``{"status": "ok"}`` is close to useless - the process
answering means the process is up, which the response itself already proved. This one reports
the three things that decide whether the app can do its job: is persistence working, can
credentials be encrypted at rest, and is a credential resolvable at all.

That makes the common support question self-answering. "Generation fails immediately" is
usually ``credential_configured: false``, and "my projects disappear on restart" is
``persistence: false`` with the reason attached.

No secret is included: ``credential_configured`` is a boolean, derived from whether a key
resolves, and the key itself is never read into the response.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from multi_agent_generator.errors import AppError
from multi_agent_generator.llm import build_provider
from multi_agent_generator.storage import encryption_available

from ..config import AppContext
from ..deps import context
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("multi-agent-generator")
    except Exception:  # noqa: BLE001 - running from a source tree without an install
        return "dev"


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(ctx: AppContext = Depends(context)) -> HealthResponse:
    settings = ctx.settings

    # Whether a credential resolves for the default provider. Building the config is cheap
    # and offline - it layers env over saved settings and does not make a network call, so
    # this stays a fast health check rather than a billed one.
    credential = False
    try:
        saved = None
        if ctx.storage is not None:
            saved = ctx.storage.saved_llm(reveal=True)
        provider = build_provider(settings.default_provider, saved=saved)
        credential = bool(provider.config.api_key) or not provider.needs_credential
    except AppError:
        credential = False
    except Exception:  # noqa: BLE001 - health must not fail because a provider misbehaved
        credential = False

    return HealthResponse(
        status="ok",
        version=_version(),
        persistence=ctx.persistent,
        persistence_error=ctx.storage_error,
        encryption=encryption_available(),
        default_provider=settings.default_provider,
        credential_configured=credential,
    )
