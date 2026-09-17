# multi_agent_generator/llm/__init__.py
"""
The LLM layer.

Public surface:

    from multi_agent_generator.llm import build_provider, LLMConfig, ChatMessage

``build_provider`` is the one function the rest of the application calls. Give it a
provider name (and optionally a model, key, saved config...) and it returns something
that satisfies :class:`LLMProvider`. Nothing outside this package should import a
concrete provider class or reference an environment-variable name.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Type

from ..errors import UnknownProviderError
from .base import ChatMessage, ConnectionCheck, LLMProvider, LLMResponse, normalise_messages
from .config import LLMConfig
from .providers import (
    AnthropicProvider,
    GoogleProvider,
    GroqProvider,
    HuggingFaceProvider,
    LiteLLMProvider,
    LocalTransformersProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    WatsonxProvider,
)

__all__ = [
    "ChatMessage",
    "ConnectionCheck",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "build_provider",
    "provider_classes",
    "available_provider_names",
    "normalise_messages",
]

#: The registry. Adding a provider is one line here plus its class - and, if it should
#: appear in generated code, one snippet entry in ``multi_agent_generator.providers``.
_REGISTRY: Dict[str, Type[LLMProvider]] = {
    OpenAIProvider.name: OpenAIProvider,
    AnthropicProvider.name: AnthropicProvider,
    GroqProvider.name: GroqProvider,
    GoogleProvider.name: GoogleProvider,
    WatsonxProvider.name: WatsonxProvider,
    OllamaProvider.name: OllamaProvider,
    HuggingFaceProvider.name: HuggingFaceProvider,
    LocalTransformersProvider.name: LocalTransformersProvider,
    MockProvider.name: MockProvider,
}

#: Aliases users actually type. Kept in step with ``providers.get_provider``.
_ALIASES = {
    "hf": "huggingface",
    "huggingface-hosted": "huggingface",
    "hf-local": "huggingface-local",
    "hf_local": "huggingface-local",
    "huggingface_local": "huggingface-local",
    "transformers": "huggingface-local",
    "local": "huggingface-local",
    "gemini": "google",
    "google-gemini": "google",
    "claude": "anthropic",
    "ibm": "watsonx",
    "watsonx-ai": "watsonx",
    "stub": "mock",
    "offline": "mock",
    "fake": "mock",
}


def _canonical(name: Optional[str]) -> str:
    if not name:
        from ..settings import get_settings

        name = get_settings().default_provider
    key = str(name).strip().lower()
    return _ALIASES.get(key, key)


def provider_classes() -> Dict[str, Type[LLMProvider]]:
    return dict(_REGISTRY)


def available_provider_names() -> List[str]:
    return list(_REGISTRY)


def build_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    top_p: Optional[float] = None,
    saved: Optional[Mapping[str, Any]] = None,
    config: Optional[LLMConfig] = None,
    **extra: Any,
) -> LLMProvider:
    """
    Construct a provider instance.

    Either pass a fully-built ``config``, or pass the loose fields and let
    :meth:`LLMConfig.resolve` layer them over saved state and the environment.
    """
    if config is not None:
        key = config.provider
    else:
        key = _canonical(provider or (saved or {}).get("provider"))

    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise UnknownProviderError(
            f"Unknown LLM provider {provider!r}.",
            action=f"Choose one of: {known}.",
            context={"provider": provider},
        )

    if config is None:
        config = LLMConfig.resolve(
            provider=key,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            top_p=top_p,
            saved=saved,
            extra=extra or None,
        )
    return _REGISTRY[key](config)
