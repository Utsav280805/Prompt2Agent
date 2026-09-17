# backend/api/meta.py
"""
What the app can build, and what it can build it with.

These two endpoints back the create form's framework picker and the settings page's provider
list. Both are computed from the same registries the generator uses -
:data:`FRAMEWORK_PROFILES` and :data:`PROVIDERS` - rather than from a list maintained in the
frontend. A hard-coded list in the UI is how a picker ends up offering a framework the backend
cannot generate, or omitting one it can; deriving it means adding a framework is one registry
entry and the UI updates itself.

Each entry carries two flags the UI needs in order to be useful rather than merely pretty:
``installed`` (are the Python packages importable) and, for providers, ``configured`` (is a
credential resolvable). Together they let the frontend explain *why* an option will not work
and show the exact ``pip install`` command to fix it, instead of letting the user pick it and
hit a failure three stages into a run.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends

from multi_agent_generator.core.selection import FRAMEWORK_PROFILES
from multi_agent_generator.dependencies import (
    framework_import_name,
    install_command,
    is_installed,
    requirements_for,
)
from multi_agent_generator.llm import available_provider_names
from multi_agent_generator.llm.config import LLMConfig
from multi_agent_generator.providers import PROVIDERS, get_provider

from ..config import AppContext
from ..deps import context
from ..schemas import FrameworkInfo, ProviderInfo

router = APIRouter(tags=["meta"])


@router.get("/frameworks", response_model=List[FrameworkInfo], summary="Available frameworks")
def list_frameworks() -> List[FrameworkInfo]:
    infos: List[FrameworkInfo] = []
    for profile in FRAMEWORK_PROFILES.values():
        import_name = framework_import_name(profile.name)
        installed = is_installed(import_name) if import_name else True
        command: Optional[str] = None
        if not installed:
            try:
                command = install_command(requirements_for(framework=profile.name))
            except Exception:  # noqa: BLE001 - a missing hint must not break the list
                command = None

        infos.append(
            FrameworkInfo(
                name=profile.name,
                label=profile.label,
                description=profile.summary,
                strengths=list(profile.strengths),
                # The selector's "weaknesses" are the honest form of "best for": knowing
                # LangGraph handles branching and CrewAI does not is the actual decision.
                best_for=[s for s in profile.strengths if s not in profile.weaknesses],
                installed=installed,
                install_command=command,
            )
        )
    return infos


@router.get("/providers", response_model=List[ProviderInfo], summary="Available LLM providers")
def list_llm_providers(ctx: AppContext = Depends(context)) -> List[ProviderInfo]:
    saved = None
    if ctx.storage is not None:
        # reveal=True so "is a key configured" is answered from the real stored value.
        # Only the boolean derived from it leaves this function.
        saved = ctx.storage.saved_llm(reveal=True)

    runnable = set(available_provider_names())
    infos: List[ProviderInfo] = []

    for spec in PROVIDERS.values():
        if spec.name not in runnable:
            # Present in the code-generation registry but with no runtime implementation.
            # Offering it would let a user configure something that cannot be called.
            continue

        configured = False
        try:
            resolved = LLMConfig.resolve(provider=spec.name, saved=saved)
            configured = bool(resolved.api_key)
        except Exception:  # noqa: BLE001
            configured = False

        missing: List[str] = []
        try:
            for requirement in requirements_for(provider=spec.name):
                if not is_installed(requirement.import_name):
                    missing.append(requirement.spec)
        except Exception:  # noqa: BLE001
            missing = []

        infos.append(
            ProviderInfo(
                name=spec.name,
                label=spec.label,
                description=spec.description,
                default_model=spec.default_model,
                runtime_model=spec.runtime_model,
                credential_env=list(spec.credential_env),
                # A local provider needs no key, so it is "configured" as soon as its
                # packages are importable - reporting it as unconfigured would be a lie
                # that sends users hunting for a key they do not need.
                configured=configured or spec.local or not spec.credential_env,
                installed=not missing,
                install_command=(f"pip install {' '.join(missing)}" if missing else None),
                local=spec.local,
                notes=spec.notes or None,
            )
        )
    return infos


@router.get(
    "/providers/{name}",
    response_model=ProviderInfo,
    summary="One provider",
)
def get_llm_provider(name: str, ctx: AppContext = Depends(context)) -> ProviderInfo:
    # get_provider raises UnknownProviderError (a 400 with the valid names) for a bad name.
    spec = get_provider(name)
    for info in list_llm_providers(ctx):
        if info.name == spec.name:
            return info
    # Registered for code generation but not runnable; describe it without the live flags.
    return ProviderInfo(
        name=spec.name,
        label=spec.label,
        description=spec.description,
        default_model=spec.default_model,
        runtime_model=spec.runtime_model,
        credential_env=list(spec.credential_env),
        local=spec.local,
        notes=spec.notes or None,
    )
