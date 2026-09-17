# multi_agent_generator/logging_config.py
"""
Structured logging.

Section 22 of the brief asks for logs carrying ``project_id``, ``run_id``, ``stage``,
``timestamp``, ``status``, ``duration`` and ``error``. The reason those seven fields and not
others: they are exactly what you need to answer "what happened on run X" without reading
the code. A log line that says ``Generation failed`` is useless three days later; one that
says which run, which stage, how long it took and what the error code was can be grepped,
counted and graphed.

Two output modes, because there are two audiences. A developer at a terminal wants aligned,
readable lines. Anything shipping logs to a collector wants one JSON object per line, with
no multi-line tracebacks to reassemble. ``LOG_JSON=true`` switches between them; the fields
carried are identical either way, so nothing is only observable in one mode.

Every record passes through :func:`redact` before it is emitted. That is the last line of
defence for the rule that API keys never reach a log: the providers already wrap credentials
in :class:`~multi_agent_generator.errors.Secret`, but a dict that happens to contain an
``api_key`` key can reach a log line from a dozen places, and catching it here means no
individual call site has to be trusted.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, Mapping, Optional

from .errors import redact

__all__ = [
    "configure_logging",
    "get_logger",
    "StageLogger",
    "log_event",
    "EVENT_FIELDS",
]


#: The context fields a structured record may carry. Anything else a caller passes lands in
#: ``extra`` and is still logged - this list is what gets first-class treatment in both
#: formatters, so these names stay stable for whoever is querying the logs.
EVENT_FIELDS = (
    "project_id",
    "run_id",
    "stage",
    "status",
    "duration_s",
    "iteration",
    "error_code",
)

#: Attributes present on every LogRecord. Used to separate caller-supplied ``extra`` fields
#: from the ones logging itself sets, without hardcoding a version-specific list.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EVENT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and key not in EVENT_FIELDS
        }
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            # The traceback is kept - it is a bug report - but it is a *field*, not the line
            # itself, so a JSON consumer never has to stitch continuation lines together.
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(redact(payload), default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Aligned, human-readable lines with the context appended."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-34s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        bits = []
        for field in EVENT_FIELDS:
            value = getattr(record, field, None)
            if value is None:
                continue
            if field == "duration_s":
                bits.append(f"{float(value):.2f}s")
            elif field == "stage":
                bits.append(f"[{str(value).upper()}]")
            else:
                bits.append(f"{field}={value}")
        return f"{base}  {' '.join(bits)}" if bits else base


_configured = False


def configure_logging(
    level: Optional[str] = None,
    *,
    as_json: Optional[bool] = None,
    force: bool = False,
) -> None:
    """
    Install the root handler. Idempotent unless ``force`` is set.

    Called once at process start - by the API's startup hook and by the CLI - and guarded
    by a flag because configuring twice is how duplicate log lines happen. Settings are read
    here rather than taken as required arguments so that a caller who does not care about
    logging still gets the configured behaviour.
    """
    global _configured
    if _configured and not force:
        return

    from .settings import get_settings

    settings = get_settings()
    resolved_level = (level or settings.log_level or "INFO").upper()
    use_json = settings.log_json if as_json is None else as_json

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if use_json else TextFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved_level, logging.INFO))

    # These are chatty at INFO and drown out our own lines. Silencing a third-party
    # logger's noise is not the same as hiding our own errors.
    for noisy in ("httpx", "httpcore", "urllib3", "LiteLLM", "litellm", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """A logger under the application's namespace."""
    return logging.getLogger(name if name.startswith("magen") else f"magen.{name}")


def log_event(
    logger: logging.Logger,
    event: Any,
    *,
    project_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    """
    Write a :class:`~multi_agent_generator.core.models.PipelineEvent` as one structured line.

    This is the bridge between the pipeline's event stream and the log file: the pipeline
    emits events for the UI, and passing them through here means the same sequence the user
    watches is the sequence recorded on disk, with no second set of log statements to keep
    in sync.
    """
    context = {
        "stage": getattr(getattr(event, "stage", None), "value", None),
        "status": getattr(event, "status", None),
        "duration_s": getattr(event, "duration_s", None),
        "iteration": getattr(event, "iteration", None),
        "project_id": project_id or getattr(event, "project_id", None),
        "run_id": run_id or getattr(event, "run_id", None),
    }
    error = getattr(event, "error", None)
    if isinstance(error, Mapping):
        context["error_code"] = error.get("code")

    level = logging.ERROR if getattr(event, "status", "") == "failed" else logging.INFO
    if getattr(event, "status", "") == "warning":
        level = logging.WARNING
    logger.log(level, getattr(event, "message", ""), extra=context)


class StageLogger:
    """
    A logger bound to one project and run.

    Carrying the ids on the logger rather than passing them to every call is what makes it
    realistic to have them on *every* line. The alternative - remembering to add
    ``extra={"run_id": ...}`` at each of forty call sites - reliably ends up with the lines
    you most want correlated being the ones that lack the id.
    """

    def __init__(
        self,
        name: str = "pipeline",
        *,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self._logger = get_logger(name)
        self._base: Dict[str, Any] = {"project_id": project_id, "run_id": run_id}

    def bind(self, **fields: Any) -> "StageLogger":
        """A copy with extra context. The original is unchanged, so binding is safe to nest."""
        clone = StageLogger.__new__(StageLogger)
        clone._logger = self._logger
        clone._base = {**self._base, **fields}
        return clone

    def _emit(self, level: int, message: str, **fields: Any) -> None:
        merged = {**self._base, **fields}
        stage = merged.get("stage")
        if stage is not None:
            merged["stage"] = getattr(stage, "value", stage)
        self._logger.log(level, message, extra={k: v for k, v in merged.items()})

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, message, **fields)

    def exception(self, message: str, **fields: Any) -> None:
        """Log at ERROR with the active traceback attached."""
        merged = {**self._base, **fields}
        stage = merged.get("stage")
        if stage is not None:
            merged["stage"] = getattr(stage, "value", stage)
        self._logger.error(message, exc_info=True, extra=merged)

    def event(self, event: Any) -> None:
        """Log a pipeline event with this logger's bound context."""
        log_event(
            self._logger,
            event,
            project_id=self._base.get("project_id"),
            run_id=self._base.get("run_id"),
        )
