# multi_agent_generator/llm/config.py
"""
Resolved LLM configuration.

:class:`LLMConfig` is the only object that ever holds a credential, and it holds it
wrapped in :class:`~multi_agent_generator.errors.Secret` so that logging it, putting it
in an error message or serialising it to JSON cannot leak the value.

Resolution order, highest priority first:

1. an explicit argument (``--api-key``, a field in the API request);
2. the configuration the user saved on the LLM settings page;
3. the provider's declared environment variables, in the order the registry lists them;
4. the built-in default.

The model id stored here is always the **plain** id - ``Qwen/Qwen2.5-7B-Instruct``, not
``huggingface/Qwen/Qwen2.5-7B-Instruct``. Route prefixes are a detail of one particular
backend (LiteLLM) and are added by that backend's provider class. Storing a prefixed id
here is what produced ``huggingface/huggingface/Qwen/...`` when a user passed ``--model``
with a prefix already attached.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from ..errors import Secret

__all__ = ["LLMConfig"]


def _spec(provider: Optional[str]):
    """
    Look up the registry entry.

    ``get_provider`` already raises :class:`UnknownProviderError` with the list of valid
    names, so there is nothing to translate here. This stays as a named function because the
    import has to be deferred - ``providers`` imports ``errors``, and doing it at module
    scope would close a cycle through the package root.
    """
    from ..providers import get_provider

    return get_provider(provider)


def _strip_route_prefix(model: str, prefixes: Tuple[str, ...]) -> str:
    """
    Remove a LiteLLM-style route prefix a user may have typed.

    Only known prefixes are stripped. That restriction is essential: Hugging Face repo
    ids look exactly like prefixed model ids (``Qwen/Qwen2.5-7B-Instruct``), so a
    generic "drop everything before the first slash" rule would mangle them.
    """
    for prefix in prefixes:
        if prefix and model.startswith(f"{prefix}/"):
            return model[len(prefix) + 1 :]
    return model


def _first_env(names: Tuple[str, ...], env: Mapping[str, str]) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _same_provider(saved: Optional[str], resolved: str) -> bool:
    """
    Whether a saved settings row describes the provider being resolved.

    Compared through the registry rather than by string equality, so a row saved as ``"hf"``
    still matches when the caller asks for ``"huggingface"``. An unknown saved name simply
    does not match, which fails safe: the worst case is falling back to the environment.
    """
    if not saved:
        return False
    try:
        return _spec(saved).name == resolved
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class LLMConfig:
    """Everything one provider instance needs, with the key contained."""

    provider: str
    model: str
    api_key: Secret = field(default_factory=lambda: Secret(None))
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 120
    top_p: Optional[float] = None
    #: Environment variables that would satisfy this provider, for error messages.
    credential_env: Tuple[str, ...] = ()
    #: Extra provider-specific keyword arguments passed straight through.
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------- construction
    @classmethod
    def resolve(
        cls,
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
        env: Optional[Mapping[str, str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> "LLMConfig":
        """
        Build a configuration by layering explicit arguments over saved state and env.

        ``saved`` is the row from the LLM-settings table (already decrypted by the
        repository) and is deliberately passed in rather than looked up here, so this
        module stays free of a dependency on the database.
        """
        from ..settings import get_settings

        environ = env if env is not None else os.environ
        saved = dict(saved or {})
        settings = get_settings()

        provider_name = provider or saved.get("provider") or settings.default_provider
        spec = _spec(provider_name)

        # Saved settings describe one provider. When the caller asks for a *different* one,
        # the provider-specific fields of that row do not apply: its key belongs to another
        # vendor, its model id is not valid here, and its base URL points somewhere else.
        # Honouring them anyway is how one vendor's credential ends up being sent to
        # another's endpoint, so each is gated on the row actually being for this provider.
        # Sampling parameters (temperature, max_tokens, timeout) are provider-independent
        # and stay in effect either way.
        saved_matches = _same_provider(saved.get("provider"), spec.name)

        # --- model -----------------------------------------------------------------
        raw_model = (
            model
            or (saved.get("model") if saved_matches else None)
            or (
                _first_env(("HF_MODEL_ID",), environ)
                if spec.name in ("huggingface", "huggingface-local")
                else None
            )
            or _first_env(("DEFAULT_MODEL",), environ)
            or spec.runtime_model
        )
        known_prefixes = (spec.name,)
        if "/" in (spec.default_model or "") and spec.backend == "litellm":
            known_prefixes = known_prefixes + (spec.default_model.split("/", 1)[0],)
        resolved_model = _strip_route_prefix(str(raw_model).strip(), known_prefixes)

        # --- credential ------------------------------------------------------------
        saved_key = saved.get("api_key") if saved_matches else None
        key = api_key or saved_key or _first_env(spec.credential_env, environ)

        # --- base url --------------------------------------------------------------
        saved_url = saved.get("base_url") if saved_matches else None
        url = base_url or saved_url or _default_base_url(spec.name, environ)

        def pick(explicit, saved_key, default):
            if explicit is not None:
                return explicit
            value = saved.get(saved_key)
            return default if value is None else value

        return cls(
            provider=spec.name,
            model=resolved_model,
            api_key=Secret(key),
            base_url=url,
            temperature=float(pick(temperature, "temperature", settings.temperature)),
            max_tokens=int(pick(max_tokens, "max_tokens", settings.max_tokens)),
            timeout=int(pick(timeout, "timeout", settings.request_timeout)),
            top_p=pick(top_p, "top_p", None),
            credential_env=tuple(spec.credential_env),
            # `extra` carries provider-specific arguments (a watsonx project id, say), so it
            # is gated on the row matching for the same reason the key is.
            extra={
                **((saved.get("extra") or {}) if saved_matches else {}),
                **(extra or {}),
            },
        )

    # ---------------------------------------------------------------------- mutation
    def with_overrides(self, **changes: Any) -> "LLMConfig":
        """A copy with fields replaced. ``api_key`` accepts a plain string."""
        if "api_key" in changes and not isinstance(changes["api_key"], Secret):
            changes["api_key"] = Secret(changes["api_key"])
        return replace(self, **changes)

    # ------------------------------------------------------------------ serialisation
    def redacted(self) -> Dict[str, Any]:
        """
        Safe for logs, API responses and the settings UI.

        There is no non-redacted serialiser on purpose. If a caller needs the key it has
        to reach for ``config.api_key.reveal()``, which is one grep away from a reviewer.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "top_p": self.top_p,
            "credential_configured": bool(self.api_key),
            "credential_hint": self.api_key.hint(),
            "credential_env": list(self.credential_env),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic, but load-bearing
        # dataclass' generated repr would print the Secret's repr, which is already
        # masked - but being explicit here means a future field addition cannot
        # accidentally start printing a credential.
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={self.api_key!r}, base_url={self.base_url!r})"
        )


def _default_base_url(provider: str, env: Mapping[str, str]) -> Optional[str]:
    """
    Base URL a provider needs when it is not the vendor's own default.

    Only the providers that genuinely require one are listed. Returning None for the
    rest lets each SDK use its own default, which is more robust than us hardcoding a
    hostname that may change.
    """
    from ..providers import HF_LOCAL_BASE_URL_DEFAULT, HF_ROUTER_BASE_URL

    if provider == "huggingface":
        return env.get("HF_ROUTER_BASE_URL") or HF_ROUTER_BASE_URL
    if provider == "huggingface-local":
        return env.get("HF_LOCAL_BASE_URL") or HF_LOCAL_BASE_URL_DEFAULT
    if provider == "ollama":
        return env.get("OLLAMA_URL") or env.get("OLLAMA_HOST") or "http://localhost:11434"
    if provider == "watsonx":
        return env.get("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com"
    return None
