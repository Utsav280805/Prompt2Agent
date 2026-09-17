# multi_agent_generator/model_inference.py
"""
Backwards-compatible façade over :mod:`multi_agent_generator.llm`.

This module used to *be* the inference layer: two hand-rolled backend classes, their own
credential lookup, their own parameter merging, and their own error strings. All of that
now lives in :mod:`multi_agent_generator.llm`, where it is shared by the CLI, the REST API
and the pipeline. Keeping a second implementation here would mean two places to fix every
provider bug, so there is exactly one left, and this file adapts the old call shape onto
it.

What survives, and why:

* :class:`Message` - callers construct these, and it costs nothing to keep accepting them.
* :func:`build_inference` - the documented entry point; now returns an
  :class:`InferenceAdapter` wrapping a real provider.
* :class:`ModelInference` / :class:`LocalTransformersInference` - re-exported names so
  ``from multi_agent_generator import ModelInference`` keeps working. They construct
  provider-backed adapters and warn.

Two behavioural notes that are improvements rather than accidents:

*Importing this module no longer touches the environment.* There used to be a
``load_dotenv()`` at module scope. That made ``import multi_agent_generator`` mutate
``os.environ`` as a side effect, and - worse - meant that any code path which did *not*
import this module never loaded ``.env`` at all. A generated standalone script is exactly
such a path, which is why "I put HF_TOKEN in .env" kept ending in ``You must provide an
api_key``. ``.env`` is now loaded once, lazily, by
:func:`multi_agent_generator.settings.get_settings`, which every configuration path goes
through.

*Failures raise* :class:`~multi_agent_generator.errors.AppError` *subclasses* rather than a
flattened ``RuntimeError(f"Model inference failed: {e}")``. Callers that catch
``Exception`` are unaffected; callers that want to tell "no credential" apart from "model
does not exist" can now do so.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import BaseModel

from .llm import ChatMessage, LLMConfig, LLMProvider, LLMResponse, build_provider

__all__ = [
    "Message",
    "InferenceAdapter",
    "ModelInference",
    "LocalTransformersInference",
    "build_inference",
]


class Message(BaseModel):
    """One conversation turn. Kept for callers that already build these."""

    role: str
    content: str


#: LiteLLM route prefix -> provider registry key. Used only to interpret a legacy
#: pre-prefixed model id such as ``"huggingface/Qwen/Qwen2.5-7B-Instruct"``; new code
#: passes a provider name and a plain model id instead.
_PREFIX_TO_PROVIDER: Dict[str, str] = {
    "anthropic": "anthropic",
    "groq": "groq",
    "gemini": "google",
    "google": "google",
    "watsonx": "watsonx",
    "ollama": "ollama",
    "huggingface": "huggingface",
}

MessageInput = Union[Message, ChatMessage, Mapping[str, Any]]


def _to_chat_messages(messages: Sequence[MessageInput]) -> List[ChatMessage]:
    """
    Accept every message shape this project has ever passed around.

    :func:`multi_agent_generator.llm.base.normalise_messages` already handles dicts and
    :class:`ChatMessage`, but not the pydantic :class:`Message` above, and doing the
    conversion here keeps that knowledge of a legacy type out of the new layer.
    """
    converted: List[ChatMessage] = []
    for message in messages:
        if isinstance(message, Message):
            converted.append(ChatMessage(role=message.role, content=message.content))
        elif isinstance(message, ChatMessage):
            converted.append(message)
        elif isinstance(message, Mapping):
            content = message.get("content")
            converted.append(
                ChatMessage(
                    role=str(message.get("role") or "user"),
                    content="" if content is None else str(content),
                )
            )
        else:
            raise TypeError(
                f"Unsupported message type {type(message).__name__!r}; expected a dict, "
                "a ChatMessage or a Message."
            )
    return converted


class InferenceAdapter:
    """
    The old ``generate_text`` surface, backed by a real
    :class:`~multi_agent_generator.llm.base.LLMProvider`.

    Deliberately thin. It converts message types, forwards keyword overrides and returns
    the response text. Everything that used to make this class interesting - credential
    resolution, dropping parameters a vendor rejects, error translation, token accounting -
    is now inherited behaviour of the provider it wraps.
    """

    def __init__(self, provider: LLMProvider, **default_params: Any):
        self.provider = provider
        # None means "not specified"; keeping such entries would override a configured
        # value with nothing, which is how `temperature=None` used to reach litellm.
        self.default_params = {k: v for k, v in default_params.items() if v is not None}

    # ------------------------------------------------------------------ introspection
    @property
    def model(self) -> str:
        """The plain model id, without any route prefix."""
        return self.provider.model

    @property
    def backend(self) -> str:
        """The provider registry key, e.g. ``"huggingface"``."""
        return self.provider.name

    @property
    def config(self) -> LLMConfig:
        return self.provider.config

    def describe(self) -> Dict[str, Any]:
        """Safe-to-log summary. Never contains the credential."""
        return self.provider.describe()

    # ------------------------------------------------------------------------- calling
    def generate_text(self, messages: Sequence[MessageInput], **override_params: Any) -> str:
        """Run one completion and return just the text."""
        return self.complete(messages, **override_params).text

    def complete(self, messages: Sequence[MessageInput], **override_params: Any) -> LLMResponse:
        """Run one completion and return the full response, including token usage."""
        params = {**self.default_params, **override_params}
        return self.provider.complete(_to_chat_messages(messages), **params)

    def complete_text(self, prompt: str, system: Optional[str] = None, **overrides: Any) -> str:
        """Single-turn convenience wrapper."""
        messages: List[ChatMessage] = []
        if system:
            messages.append(ChatMessage.system(system))
        messages.append(ChatMessage.user(prompt))
        return self.complete(messages, **overrides).text

    def validate(self) -> None:
        """Raise if this configuration cannot possibly work. No network call."""
        self.provider.validate()

    def test_connection(self):
        """Smallest possible real request. Returns a result object, never raises."""
        return self.provider.test_connection()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"InferenceAdapter(provider={self.provider.name!r}, model={self.model!r})"


def build_inference(
    provider: str = "openai",
    model: Optional[str] = None,
    **params: Any,
) -> InferenceAdapter:
    """
    Construct an inference adapter for ``provider``.

    Sampling parameters (``temperature``, ``max_tokens``, ``top_p``, ``timeout``) are
    understood; anything else is forwarded to the provider as extra request state. A
    provider that rejects a given parameter declares it in ``drop_params`` and it is
    stripped before the request is built, so passing OpenAI's penalty parameters to
    Hugging Face is harmless rather than a 422.
    """
    known = ("api_key", "base_url", "temperature", "max_tokens", "timeout", "top_p", "saved")
    build_kwargs = {key: params.pop(key) for key in known if key in params}
    # `api_base` was the old name for `base_url`; accept it so existing callers keep
    # working instead of silently losing their endpoint override.
    if "api_base" in params:
        build_kwargs.setdefault("base_url", params.pop("api_base"))
    # Only meaningful to the retired implementation: which parameters to strip. The
    # provider classes now declare that themselves, per vendor, so honouring the argument
    # would let a caller re-enable a parameter we know breaks the request.
    params.pop("drop_params", None)

    llm = build_provider(provider, model, **build_kwargs, **params)
    return InferenceAdapter(llm)


class ModelInference(InferenceAdapter):
    """
    Deprecated. Use :func:`build_inference` or
    :func:`multi_agent_generator.llm.build_provider`.

    Retained because it is exported from the package root. The old signature took a
    *routed* model id (``"huggingface/Qwen/Qwen2.5-7B-Instruct"``), so the prefix is read
    back off to work out which provider was meant.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        drop_params: Sequence[str] = (),
        **default_params: Any,
    ):
        warnings.warn(
            "ModelInference is deprecated; use multi_agent_generator.llm.build_provider() "
            "or model_inference.build_inference() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        provider_name, plain_model = _split_routed_model(model)
        llm = build_provider(
            provider_name,
            plain_model,
            api_key=api_key,
            base_url=api_base,
            **{k: v for k, v in default_params.items() if v is not None},
        )
        super().__init__(llm)


class LocalTransformersInference(InferenceAdapter):
    """
    Deprecated. Use ``build_inference("huggingface-local", model)``.

    ``transformers`` is still imported only on the first call, not here, so constructing
    this remains cheap.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        device: Optional[str] = None,
        **_ignored: Any,
    ):
        warnings.warn(
            "LocalTransformersInference is deprecated; use "
            "build_inference('huggingface-local', model) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        extra = {"device": device} if device is not None else {}
        llm = build_provider(
            "huggingface-local",
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )
        super().__init__(llm)


def _split_routed_model(model: str) -> tuple:
    """
    Split a LiteLLM-routed model id into ``(provider, plain_model)``.

    Only recognised prefixes are treated as prefixes. That restriction is the whole
    point: a Hugging Face repo id (``Qwen/Qwen2.5-7B-Instruct``) is indistinguishable in
    shape from a routed id, so a generic "split on the first slash" rule would turn the
    organisation name into a provider and mangle the model.
    """
    text = (model or "").strip()
    if "/" in text:
        head, _, tail = text.partition("/")
        provider = _PREFIX_TO_PROVIDER.get(head.lower())
        if provider and tail:
            return provider, tail
    return "openai", text
