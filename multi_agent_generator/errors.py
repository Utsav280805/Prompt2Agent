# multi_agent_generator/errors.py
"""
Application-level error taxonomy.

Every failure that a user can plausibly cause - a missing credential, an unknown
provider, a generated agent that will not compile, a run that timed out - gets a class
here. The point is that the layers above (CLI, HTTP API, frontend) can render a useful
message without any of them knowing how to read a Python traceback.

Each error carries four things:

``message``   what went wrong, in one sentence, for a human.
``action``    what that human should do about it.
``detail``    the technical diagnostic (exception text, stderr, offending value).
``context``   structured extras for logs (project_id, provider, stage...).

The traceback is *not* discarded - it stays attached via ``__cause__`` and is written to
the structured log - but it is never the thing shown to a user. That is the whole
difference between "an unhandled exception escaped" and "the product told me what to fix".
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import quote

__all__ = [
    "AppError",
    "ConfigurationError",
    "MissingCredentialError",
    "UnknownProviderError",
    "ProviderNotInstalledError",
    "UnsupportedCombinationError",
    "LLMError",
    "LLMTimeoutError",
    "LLMResponseError",
    "GenerationError",
    "ValidationError",
    "ExecutionError",
    "TimeoutExceededError",
    "NotFoundError",
    "Secret",
    "redact",
    "scrub_values",
]


class AppError(Exception):
    """
    Base class for every expected failure.

    ``code`` is a stable machine-readable string. The frontend switches on it, so
    renaming one is a breaking API change; adding a subclass is not.
    """

    code: str = "internal_error"
    http_status: int = 500
    #: Default remediation, overridable per instance.
    default_action: Optional[str] = None

    def __init__(
        self,
        message: str,
        *,
        action: Optional[str] = None,
        detail: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.action = action or self.default_action
        self.detail = detail
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self, include_detail: bool = True) -> Dict[str, Any]:
        """
        Serialise for an API response.

        ``include_detail=False`` is for responses that cross a trust boundary where even
        the diagnostic might name a path or a header. The message and action are always
        safe: they are written by us, never interpolated from a credential.
        """
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "action": self.action,
        }
        if include_detail and self.detail:
            payload["detail"] = self.detail
        if self.context:
            payload["context"] = {k: v for k, v in self.context.items() if k != "api_key"}
        return payload

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.action:
            return f"{self.message} {self.action}"
        return self.message


# ---------------------------------------------------------------- configuration errors
class ConfigurationError(AppError):
    code = "configuration_error"
    http_status = 400


class MissingCredentialError(ConfigurationError):
    """
    A provider was selected but no API key could be found for it.

    This is the single most common failure in this project, and the one that used to
    surface as ``ValueError: You must provide an api_key to work with auto API`` from
    six frames deep inside huggingface_hub. Raising it *before* the call is made is why
    the message can name the exact environment variables that would fix it.
    """

    code = "missing_credential"
    http_status = 400

    def __init__(
        self,
        provider_label: str,
        env_vars: Iterable[str],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        names = list(env_vars)
        joined = " or ".join(names) if names else "an API key"
        super().__init__(
            f"{provider_label} API key is missing.",
            action=(
                f"Set {joined} in your environment or .env file, save a key on the LLM "
                f"settings page, or select another configured provider."
            ),
            detail=f"Looked for: {', '.join(names) or 'no variables declared'}.",
            context={**(context or {}), "provider": provider_label, "env_vars": names},
        )
        self.env_vars = names


class UnknownProviderError(ConfigurationError):
    code = "unknown_provider"
    http_status = 400


class ProviderNotInstalledError(ConfigurationError):
    """The provider is known, but its Python package is not importable."""

    code = "provider_not_installed"
    http_status = 400


class UnsupportedCombinationError(ConfigurationError):
    """A provider cannot be expressed in the requested framework."""

    code = "unsupported_combination"
    http_status = 400


# ------------------------------------------------------------------------- LLM errors
class LLMError(AppError):
    code = "llm_error"
    http_status = 502
    default_action = "Check the provider status and your model id, then try again."


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    http_status = 504
    default_action = "Raise the request timeout, or pick a smaller/faster model."


class LLMResponseError(LLMError):
    """The provider answered, but not with something usable."""

    code = "llm_bad_response"
    http_status = 502


# ------------------------------------------------------------- pipeline / build errors
class GenerationError(AppError):
    code = "generation_failed"
    http_status = 422


class ValidationError(AppError):
    """Generated (or supplied) content failed validation."""

    code = "validation_failed"
    http_status = 422


class ExecutionError(AppError):
    code = "execution_failed"
    http_status = 500


class TimeoutExceededError(ExecutionError):
    code = "execution_timeout"
    http_status = 504
    default_action = (
        "The process was terminated at its time limit. Raise AGENT_EXECUTION_TIMEOUT "
        "if the work is genuinely slow, or look at the captured output for a hang."
    )


class NotFoundError(AppError):
    code = "not_found"
    http_status = 404


# ------------------------------------------------------------------ secret containment
class Secret:
    """
    A string that refuses to print itself.

    Every accidental credential leak this project could have has the same shape: a
    config object lands in a log line, an error message, a JSON response or a repr, and
    the key goes with it. Wrapping the value means the only way to obtain it is to ask
    for it explicitly via :meth:`reveal`, which is easy to grep for and review.
    """

    __slots__ = ("_value",)

    MASK = "***"

    def __init__(self, value: Optional[str]) -> None:
        self._value = value or None

    def reveal(self) -> Optional[str]:
        """The real value. Call sites should be few and obvious."""
        return self._value

    def __bool__(self) -> bool:
        return bool(self._value)

    def __str__(self) -> str:
        return self.MASK if self._value else ""

    def __repr__(self) -> str:
        return f"Secret({self.MASK!r})" if self._value else "Secret(None)"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Secret):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def hint(self) -> Optional[str]:
        """
        A non-reversible fingerprint, so a UI can show *which* key is saved.

        Last four characters only, and only when the key is long enough that four
        characters cannot meaningfully narrow it down.
        """
        if not self._value or len(self._value) < 12:
            return None
        return f"{self.MASK}{self._value[-4:]}"


_SECRETISH = ("key", "token", "secret", "password", "authorization", "credential")

#: Below this length a "secret" is too generic to search for safely - replacing every
#: occurrence of a four-character string would corrupt unrelated output. A credential that
#: short is not a credential worth protecting by substitution; it needs a different provider.
_MIN_SCRUBBABLE = 8


def scrub_values(text: str, secrets: Iterable[Optional[str]]) -> str:
    """
    Remove literal secret values from free text.

    :func:`redact` works on structure: it masks a value because its *key* was called
    ``api_key``. That is the right approach for configuration objects and it is useless for
    captured output, where a credential appears as bare characters in the middle of a
    traceback - ``AuthenticationError: invalid key sk-abc123...`` has no key name to match.

    Provider SDKs echo the credential into their exception messages routinely, and that
    message reaches the Playground response through the child process's stderr. So captured
    stdout and stderr pass through here before they leave the machine that produced them.

    Substring replacement is deliberate and it is the only thing that works: the key may be
    surrounded by quotes, truncated with an ellipsis, or embedded in a URL, so anchoring on
    word boundaries would miss it. Also scrubs a URL-encoded form, because a credential sent
    as a query parameter comes back that way.
    """
    if not text:
        return text or ""
    scrubbed = str(text)
    seen: set = set()
    for secret in secrets:
        value = str(secret or "")
        if len(value) < _MIN_SCRUBBABLE or value in seen:
            continue
        seen.add(value)
        scrubbed = scrubbed.replace(value, Secret.MASK)
        quoted = quote(value, safe="")
        if quoted != value:
            scrubbed = scrubbed.replace(quoted, Secret.MASK)
    return scrubbed


def redact(data: Any) -> Any:
    """
    Recursively mask anything that looks like a credential.

    Applied at the logging boundary and to every API response that echoes a
    configuration back. Matching is on the *key name* rather than the value, because a
    value-based heuristic either misses short keys or mangles legitimate content.
    """
    if isinstance(data, Secret):
        return str(data)
    if isinstance(data, Mapping):
        out: Dict[Any, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and any(s in key.lower() for s in _SECRETISH):
                out[key] = Secret.MASK if value else None
            else:
                out[key] = redact(value)
        return out
    if isinstance(data, (list, tuple)):
        rebuilt = [redact(v) for v in data]
        return type(data)(rebuilt) if isinstance(data, tuple) else rebuilt
    return data
