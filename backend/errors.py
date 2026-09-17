# backend/errors.py
"""
Turning exceptions into responses.

Section 21 of the brief: users see friendly errors, never tracebacks. This module is where
that promise is kept for the HTTP layer, and it works because the codebase already
distinguishes two kinds of failure.

An :class:`~multi_agent_generator.errors.AppError` is an *expected* failure - a missing key, an
unknown provider, a run that timed out. It already carries a message, a remediation and a
stable code, so translating it is mechanical: use its ``http_status``, serialise it, return it.
The user gets "Hugging Face API key is missing. Set HF_TOKEN in your environment or .env
file", which is actionable.

Anything else is a *bug in this application*. The user cannot fix it and the internals are not
theirs to see, so the response is a generic 500 with an incident id, while the full traceback
goes to the server log under that same id. That pairing matters: a support conversation
becomes "give me the id" instead of "please paste the whole screen", and no file path,
dependency version or environment detail crosses the wire.

Validation failures get their own handler purely so the message is usable. FastAPI's default
422 body is a nested list of Pydantic error dicts; this flattens it to "requirement: field
required" in the same envelope as every other error, so the frontend has one error shape.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from multi_agent_generator.errors import AppError, redact
from multi_agent_generator.logging_config import get_logger

__all__ = ["install_error_handlers", "error_payload"]

log = get_logger("backend.errors")


def error_payload(
    code: str,
    message: str,
    *,
    action: str | None = None,
    detail: str | None = None,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build an :class:`~backend.schemas.ErrorResponse` body."""
    body: Dict[str, Any] = {"code": code, "message": message}
    if action:
        body["action"] = action
    if detail:
        body["detail"] = detail
    if context:
        body["context"] = redact(context)
    return body


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers. Called once by the app factory."""

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        # An expected failure. Logged at WARNING, not ERROR: a user forgetting an API key is
        # not an incident, and treating it as one trains people to ignore the error log.
        log.warning(
            exc.message,
            extra={
                "error_code": exc.code,
                "path": request.url.path,
                **{k: v for k, v in exc.context.items() if k != "api_key"},
            },
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=redact(exc.to_dict()),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        problems = []
        for error in exc.errors():
            # Drop the leading "body" / "query" segment: it is an implementation detail of
            # where the value arrived, not something the user needs to see.
            location = [str(part) for part in error.get("loc", ()) if part not in ("body", "query")]
            field = ".".join(location) or "request"
            problems.append(f"{field}: {error.get('msg', 'is invalid')}")

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_payload(
                "invalid_request",
                "The request could not be processed.",
                action="Correct the highlighted fields and try again.",
                detail="; ".join(problems) or None,
                context={"problems": problems},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            status.HTTP_404_NOT_FOUND: "not_found",
            status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
            status.HTTP_401_UNAUTHORIZED: "unauthorized",
            status.HTTP_403_FORBIDDEN: "forbidden",
        }
        detail = exc.detail if isinstance(exc.detail, str) else None
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                codes.get(exc.status_code, "http_error"),
                detail or "The request could not be completed.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # A bug in this application. The incident id is the only link between what the user
        # sees and the traceback on the server, which is exactly the point: full diagnostics
        # for whoever can fix it, nothing leaked to whoever cannot.
        incident = uuid.uuid4().hex[:12]
        log.exception(
            "Unhandled error while serving a request.",
            extra={
                "error_code": "internal_error",
                "incident_id": incident,
                "path": request.url.path,
                "method": request.method,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload(
                "internal_error",
                "Something went wrong on our side.",
                action=(
                    "Try again. If it keeps happening, check the server log for incident "
                    f"{incident}."
                ),
                context={"incident_id": incident},
            ),
        )
