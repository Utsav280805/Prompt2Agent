# backend/api/settings.py
"""
LLM settings: read, save, test, and remove the credential.

This router handles the application's most sensitive data, so the asymmetry between writing
and reading a credential is built into the endpoints rather than left to discipline:

**A key goes in but never comes out.** ``PUT`` accepts ``api_key``; the response model has no
such field. The most the read endpoint will say is ``credential_configured: true`` plus the
last four characters, which is enough for a user to recognise *which* key is saved and useless
to anyone who intercepts it.

**Omitting the key preserves it.** Changing the temperature must not silently wipe the
credential, so an absent ``api_key`` means "leave it alone". Deleting one is therefore never
accidental - it needs ``DELETE /api/settings/llm/key``, which exists precisely so that
clearing a credential is an explicit act.

**Testing does not require saving.** ``POST /api/settings/llm/test`` accepts a key inline so a
user can verify it *before* committing it, and that key is used for one request and never
stored. Because the check reports a bad credential as a rendered result rather than an
exception, the settings page can show "that key was rejected" without a failed request.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Response, status

from multi_agent_generator.core.service import GeneratorService
from multi_agent_generator.errors import ConfigurationError
from multi_agent_generator.llm.config import LLMConfig
from multi_agent_generator.providers import get_provider
from multi_agent_generator.storage import Storage, encryption_available

from ..config import AppContext
from ..deps import context, require_storage, service
from ..schemas import LLMSettingsRequest, LLMSettingsResponse, TestConnectionRequest

router = APIRouter(tags=["settings"])


@router.get("/settings/llm", response_model=LLMSettingsResponse, summary="Saved LLM settings")
def get_llm_settings(ctx: AppContext = Depends(context)) -> LLMSettingsResponse:
    saved: Optional[Dict[str, Any]] = None
    if ctx.storage is not None:
        saved = ctx.storage.saved_llm()  # reveal=False: the key is not needed to render

    settings = ctx.settings
    provider = (saved or {}).get("provider") or settings.default_provider

    # Resolve with reveal so the response can report *whether* a key exists - including one
    # that comes only from the environment, which the saved row knows nothing about. Only
    # booleans and a four-character hint derived from it leave this function.
    revealed = ctx.storage.saved_llm(reveal=True) if ctx.storage is not None else None
    try:
        resolved = LLMConfig.resolve(provider=provider, saved=revealed)
    except Exception:  # noqa: BLE001 - a bad saved provider must not break the page
        resolved = None

    stored_key = bool((saved or {}).get("credential_configured"))
    live_key = bool(resolved.api_key) if resolved is not None else False

    return LLMSettingsResponse(
        provider=provider,
        model=(saved or {}).get("model") or (resolved.model if resolved else None),
        base_url=(saved or {}).get("base_url") or (resolved.base_url if resolved else None),
        temperature=(saved or {}).get("temperature"),
        max_tokens=(saved or {}).get("max_tokens"),
        timeout=(saved or {}).get("timeout"),
        top_p=(saved or {}).get("top_p"),
        extra=(saved or {}).get("extra") or {},
        credential_configured=live_key,
        credential_hint=(resolved.api_key.hint() if resolved else None),
        # True when a key resolves but none is stored - i.e. it came from the environment.
        # The UI uses this to explain why a key it never saved is nonetheless working.
        credential_from_env=live_key and not stored_key,
        can_store_credentials=encryption_available(),
        source=("saved" if saved else "default"),
    )


@router.put("/settings/llm", response_model=LLMSettingsResponse, summary="Save LLM settings")
def save_llm_settings(
    payload: LLMSettingsRequest,
    ctx: AppContext = Depends(context),
    store: Storage = Depends(require_storage),
) -> LLMSettingsResponse:
    # Reject an unknown provider before writing. get_provider raises UnknownProviderError,
    # which is a 400 listing the valid names.
    spec = get_provider(payload.provider)

    if payload.api_key and not encryption_available():
        # Storing a key in plaintext is not an acceptable fallback. Refusing keeps the
        # promise that a saved credential is encrypted at rest.
        raise ConfigurationError(
            "This installation cannot encrypt saved credentials, so the key was not saved.",
            action=(
                "Install the encryption extra with `pip install multi-agent-generator[secure]`"
                f" and save again, or set {spec.credential_env[0] if spec.credential_env else 'the provider API key'}"
                " in your environment instead."
            ),
        )

    store.llm_config.save(
        {
            "provider": spec.name,
            "model": payload.model,
            "base_url": payload.base_url,
            "temperature": payload.temperature,
            "max_tokens": payload.max_tokens,
            "timeout": payload.timeout,
            "top_p": payload.top_p,
            # Absent means "keep the stored key"; the repository COALESCEs on NULL.
            "api_key": payload.api_key,
            "extra": payload.extra,
        }
    )
    return get_llm_settings(ctx)


@router.delete(
    "/settings/llm/key",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove the stored API key",
)
def delete_llm_key(store: Storage = Depends(require_storage)) -> Response:
    store.llm_config.clear_key()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/settings/llm/test", summary="Test a provider configuration")
def test_llm_connection(
    payload: TestConnectionRequest,
    ctx: AppContext = Depends(context),
    svc: GeneratorService = Depends(service),
) -> Dict[str, Any]:
    saved = None
    if payload.use_saved and ctx.storage is not None:
        saved = ctx.storage.saved_llm(reveal=True)

    check = svc.check_connection(
        provider=payload.provider,
        model=payload.model,
        api_key=payload.api_key,
        base_url=payload.base_url,
        saved=saved,
    )
    # ConnectionCheck.as_dict carries ok/message/latency/code - no credential.
    return check.as_dict()
