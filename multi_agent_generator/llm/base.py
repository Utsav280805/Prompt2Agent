# multi_agent_generator/llm/base.py
"""
The provider-facing contract.

Everything above this module - requirement analysis, the reviewer, the improver, the
CLI, the API - talks to :class:`LLMProvider` and nothing else. That is the whole point:
no other file in the project is allowed to know that Hugging Face needs ``HF_TOKEN``,
that Anthropic wants a ``anthropic/`` route prefix, or that a local transformers
pipeline has to be warmed before first use.

The base class owns the parts that must not vary between providers:

* credentials are validated *before* a request is built, so a missing key is a clean
  :class:`~multi_agent_generator.errors.MissingCredentialError` rather than a traceback
  from inside a vendor SDK;
* every provider exception is translated into the project's own taxonomy;
* timing and token usage are recorded uniformly, because the UI shows them.

Subclasses implement one method, :meth:`_invoke`.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from ..errors import (
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    MissingCredentialError,
    ProviderNotInstalledError,
)
from .config import LLMConfig

__all__ = [
    "ChatMessage",
    "LLMResponse",
    "ConnectionCheck",
    "LLMProvider",
    "MessageLike",
    "normalise_messages",
]

MessageLike = Union["ChatMessage", Mapping[str, Any]]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a conversation."""

    role: str
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls("assistant", content)


def normalise_messages(messages: Sequence[MessageLike]) -> List[Dict[str, str]]:
    """
    Accept dicts or :class:`ChatMessage` and return plain dicts.

    Tolerant on input because callers include user code and JSON from the API; strict on
    output because every backend below expects ``{"role": ..., "content": ...}``.
    """
    out: List[Dict[str, str]] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            out.append(message.as_dict())
            continue
        if isinstance(message, Mapping):
            role = str(message.get("role") or "user")
            content = message.get("content")
            out.append({"role": role, "content": "" if content is None else str(content)})
            continue
        raise TypeError(
            f"Unsupported message type {type(message).__name__!r}; expected a dict or "
            "ChatMessage."
        )
    if not out:
        raise ValueError("At least one message is required.")
    return out


@dataclass
class LLMResponse:
    """A completion plus everything the UI wants to display about it."""

    text: str
    provider: str
    model: str
    duration_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    #: True when the text came from a stub rather than a real model.
    simulated: bool = False

    def usage(self) -> Dict[str, Optional[int]]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "duration_s": round(self.duration_s, 3),
            "finish_reason": self.finish_reason,
            "simulated": self.simulated,
            "usage": self.usage(),
        }


@dataclass
class ConnectionCheck:
    """Result of the "Test Connection" button."""

    ok: bool
    provider: str
    model: str
    message: str
    duration_s: float = 0.0
    detail: Optional[str] = None
    code: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "duration_s": round(self.duration_s, 3),
            "detail": self.detail,
            "code": self.code,
        }


class LLMProvider(ABC):
    """
    Base class for every LLM backend.

    Subclasses set the class attributes and implement :meth:`_invoke`. They never raise
    vendor exceptions outward: :meth:`complete` funnels everything through
    :meth:`_translate_error`.
    """

    #: Registry key. Must match the key in ``multi_agent_generator.providers.PROVIDERS``
    #: so that runtime behaviour and generated-code snippets can never disagree.
    name: str = "base"
    #: Human label for messages and the settings UI.
    label: str = "Base provider"
    #: Whether a credential is required for this provider to work at all.
    needs_credential: bool = True
    #: Import names that must be present. Checked lazily, reported as a clean error.
    required_packages: tuple = ()
    #: Whether the provider runs entirely on the user's machine.
    local: bool = False
    #: Sampling parameters this backend rejects outright, dropped before the request.
    drop_params: tuple = ()
    #: True for stubs, so callers can refuse to present fake output as real.
    simulated: bool = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    # ---------------------------------------------------------------- introspection
    @property
    def model(self) -> str:
        return self.config.model

    def describe(self) -> Dict[str, Any]:
        """Safe-to-serialise summary. Never includes the key itself."""
        return {
            "provider": self.name,
            "label": self.label,
            "model": self.config.model,
            "local": self.local,
            "needs_credential": self.needs_credential,
            "credential_configured": bool(self.config.api_key),
            "credential_hint": self.config.api_key.hint(),
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "simulated": self.simulated,
        }

    # -------------------------------------------------------------------- validation
    def validate(self) -> None:
        """
        Check everything that can be checked without a network call.

        Called by :meth:`complete`, and separately by the API so that the settings page
        can red-flag a broken configuration before the user starts a generation run.
        """
        self._check_packages()
        if self.needs_credential and not self.config.api_key:
            raise MissingCredentialError(
                self.label,
                self.config.credential_env,
                context={"provider": self.name, "model": self.config.model},
            )

    def _check_packages(self) -> None:
        import importlib.util

        missing = [
            pkg
            for pkg in self.required_packages
            if importlib.util.find_spec(pkg) is None
        ]
        if missing:
            raise ProviderNotInstalledError(
                f"{self.label} needs {', '.join(missing)}, which is not installed.",
                action=(
                    f"Install it with: pip install "
                    f"\"multi-agent-generator[{self.name}]\""
                ),
                context={"provider": self.name, "missing": missing},
            )

    # ------------------------------------------------------------------- the call
    def complete(
        self,
        messages: Sequence[MessageLike],
        **overrides: Any,
    ) -> LLMResponse:
        """
        Run one completion.

        Raises only :class:`~multi_agent_generator.errors.AppError` subclasses.
        """
        self.validate()
        payload = normalise_messages(messages)
        params = self._request_params(overrides)

        started = time.perf_counter()
        try:
            response = self._invoke(payload, params)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, then translated
            raise self._translate_error(exc) from exc
        response.duration_s = time.perf_counter() - started

        if not (response.text or "").strip():
            raise LLMResponseError(
                f"{self.label} returned an empty response.",
                action="Try a different model, or raise max_tokens.",
                context={"provider": self.name, "model": self.config.model},
            )
        return response

    def complete_text(self, prompt: str, system: Optional[str] = None, **overrides: Any) -> str:
        """Convenience wrapper for the common single-turn case."""
        messages: List[ChatMessage] = []
        if system:
            messages.append(ChatMessage.system(system))
        messages.append(ChatMessage.user(prompt))
        return self.complete(messages, **overrides).text

    def test_connection(self) -> ConnectionCheck:
        """
        Make the smallest possible real request.

        Returns a result object instead of raising, because the caller is a UI button
        whose entire job is to display failure legibly.
        """
        started = time.perf_counter()
        try:
            self.validate()
            response = self.complete(
                [ChatMessage.user("Reply with the single word: ok")],
                max_tokens=16,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            err = exc if isinstance(exc, LLMError) else self._as_app_error(exc)
            return ConnectionCheck(
                ok=False,
                provider=self.name,
                model=self.config.model,
                message=getattr(err, "message", str(err)),
                detail=getattr(err, "detail", None) or getattr(err, "action", None),
                code=getattr(err, "code", "llm_error"),
                duration_s=time.perf_counter() - started,
            )
        return ConnectionCheck(
            ok=True,
            provider=self.name,
            model=self.config.model,
            message=f"Connected to {self.label} using {self.config.model}.",
            detail=response.text.strip()[:200] or None,
            duration_s=response.duration_s,
        )

    # ----------------------------------------------------------------- subclass hooks
    @abstractmethod
    def _invoke(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> LLMResponse:
        """Perform the request. May raise anything; the base class translates it."""

    def _request_params(self, overrides: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Merge configured defaults with per-call overrides and drop unsupported keys.

        ``drop_params`` matters more than it looks: Hugging Face's inference API returns
        a 422 for the OpenAI penalty parameters rather than ignoring them, so passing
        them through would make every HF call fail for a reason unrelated to the prompt.
        """
        params: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        params.update(self.config.extra)
        # `is not None` rather than truthiness: temperature=0 and max_tokens=0 are
        # legitimate values that `or`-style merging silently replaced with the default.
        params.update({k: v for k, v in overrides.items() if v is not None})
        for key in self.drop_params:
            params.pop(key, None)
        return params

    # --------------------------------------------------------------- error translation
    def _translate_error(self, exc: Exception) -> Exception:
        """Map a backend exception onto the project taxonomy."""
        from ..errors import AppError

        if isinstance(exc, AppError):
            return exc
        return self._as_app_error(exc)

    def _as_app_error(self, exc: Exception) -> LLMError:
        text = str(exc)
        lowered = text.lower()
        ctx = {"provider": self.name, "model": self.config.model}

        # Authentication comes first: several SDKs phrase a missing key as a generic
        # ValueError, and the useful response is always "set this variable".
        auth_markers = (
            "api_key",
            "api key",
            "unauthorized",
            "authentication",
            "401",
            "invalid token",
            "hf auth login",
            "must provide",
        )
        if any(marker in lowered for marker in auth_markers):
            return MissingCredentialError(
                self.label, self.config.credential_env, context=ctx
            )

        if "timeout" in lowered or "timed out" in lowered:
            return LLMTimeoutError(
                f"{self.label} did not respond in time.",
                detail=text,
                context=ctx,
            )

        if "not found" in lowered or "404" in lowered or "does not exist" in lowered:
            return LLMError(
                f"{self.label} does not have a model called {self.config.model!r}.",
                action="Pick a different model on the LLM settings page.",
                detail=text,
                context=ctx,
            )

        if "rate limit" in lowered or "429" in lowered or "quota" in lowered:
            return LLMError(
                f"{self.label} rate-limited the request.",
                action="Wait a moment and retry, or switch provider.",
                detail=text,
                context=ctx,
            )

        if "connection" in lowered or "refused" in lowered or "unreachable" in lowered:
            action = (
                "Start the local model server and check its base URL."
                if self.local
                else "Check your network connection and the provider's status page."
            )
            return LLMError(
                f"Could not reach {self.label}.",
                action=action,
                detail=text,
                context=ctx,
            )

        return LLMError(
            f"{self.label} request failed.",
            detail=text,
            context=ctx,
        )

    # ------------------------------------------------------------------------ helpers
    @staticmethod
    def _usage_from(obj: Any) -> Dict[str, Optional[int]]:
        """
        Pull token counts out of whatever shape the backend used.

        LiteLLM, the OpenAI SDK and huggingface_hub all report usage slightly
        differently, and every one of them is allowed to omit it entirely.
        """
        usage = getattr(obj, "usage", None) or (
            obj.get("usage") if isinstance(obj, Mapping) else None
        )
        if not usage:
            return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

        def pick(*names: str) -> Optional[int]:
            for candidate in names:
                value = (
                    usage.get(candidate)
                    if isinstance(usage, Mapping)
                    else getattr(usage, candidate, None)
                )
                if isinstance(value, (int, float)):
                    return int(value)
            return None

        return {
            "prompt_tokens": pick("prompt_tokens", "input_tokens"),
            "completion_tokens": pick("completion_tokens", "output_tokens"),
            "total_tokens": pick("total_tokens"),
        }
