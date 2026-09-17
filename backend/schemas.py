# backend/schemas.py
"""
Request and response models.

These exist to make the API's contract explicit and self-documenting: FastAPI turns them
into the OpenAPI schema the frontend's types are written against, and Pydantic rejects a
malformed request before any of it reaches the pipeline.

Two conventions worth stating.

*Requests are validated, responses are not modelled exhaustively.* Input needs strict
validation because it is untrusted; output is produced by ``as_dict()`` methods on the domain
objects, which already know their own shape. Re-declaring every field of a
:class:`PipelineResult` here would create two definitions of the same thing that could drift,
and the domain object is the one that must win.

*No secret ever appears in a response model.* :class:`LLMSettingsResponse` carries
``credential_configured`` - a boolean - and never the key. The write model accepts a key and
the read model cannot return one, which makes the asymmetry structural rather than a rule
someone has to remember.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AnalyzeRequest",
    "CreateProjectRequest",
    "RunAgentRequest",
    "LLMSettingsRequest",
    "TestConnectionRequest",
    "ErrorResponse",
    "HealthResponse",
    "ProjectSummary",
    "LLMSettingsResponse",
    "ProviderInfo",
    "FrameworkInfo",
]

#: Upper bound on a requirement. Long enough for a detailed brief, short enough that a
#: pasted book cannot become a model bill.
MAX_REQUIREMENT_CHARS = 8000


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ------------------------------------------------------------------------------ requests
class AnalyzeRequest(_Base):
    """Preview what the pipeline understands, without generating anything."""

    requirement: str = Field(..., min_length=3, max_length=MAX_REQUIREMENT_CHARS)
    provider: Optional[str] = None
    model: Optional[str] = None
    #: False runs the offline heuristic only - instant, free, no credential.
    use_model: bool = True

    @field_validator("requirement")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Describe what you want the agents to do.")
        return value


class CreateProjectRequest(_Base):
    """
    Start a generation run.

    ``framework`` left as None means "you choose" and the selection stage decides with a
    stated reason and confidence. Naming one is an instruction, not a hint: an explicit
    choice is always respected.
    """

    requirement: str = Field(..., min_length=3, max_length=MAX_REQUIREMENT_CHARS)
    name: Optional[str] = Field(default=None, max_length=200)
    framework: Optional[str] = None

    #: The provider that *designs* the project.
    provider: Optional[str] = None
    model: Optional[str] = None

    #: The provider baked into the generated code, if different from the designer. Someone
    #: may well design with a strong model and ship code that runs on a cheap one.
    runtime_provider: Optional[str] = None
    runtime_model: Optional[str] = None

    # The web product promises a verified generated project by default. Users can still
    # explicitly disable this for a fast draft through the API, but the normal path must
    # generate and execute the proof suite before Playground use.
    include_tests: bool = True
    run_tests: bool = True
    use_model: bool = True
    max_iterations: Optional[int] = Field(default=None, ge=1, le=10)

    @field_validator("requirement")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Describe what you want the agents to do.")
        return value


class RunAgentRequest(_Base):
    """Execute a generated agent for real, from the Playground."""

    query: str = Field(..., min_length=1, max_length=MAX_REQUIREMENT_CHARS)
    #: Install the project's requirements first. Needed for a genuine run, since the framework
    #: has to be importable. The install goes into a content-keyed cache rather than into the
    #: server's own interpreter, so the cost is paid once per requirement set - which is why
    #: this defaults to on rather than being left to the caller to remember.
    install: bool = True
    timeout: Optional[int] = Field(default=None, ge=5, le=1800)


class LLMSettingsRequest(_Base):
    """
    Save LLM settings.

    ``api_key`` is write-only. Omitting it keeps whatever key is already stored, so changing
    the temperature does not require re-entering the credential - and sending ``""`` is not
    a way to blank it either. Use ``DELETE /api/settings/llm/key`` for that, so removing a
    credential is always a deliberate act.
    """

    provider: str = Field(..., min_length=1, max_length=64)
    model: Optional[str] = Field(default=None, max_length=200)
    api_key: Optional[str] = Field(default=None, max_length=500)
    base_url: Optional[str] = Field(default=None, max_length=500)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=200_000)
    timeout: Optional[int] = Field(default=None, ge=1, le=3600)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_key")
    @classmethod
    def _blank_is_absent(cls, value: Optional[str]) -> Optional[str]:
        return value or None


class TestConnectionRequest(_Base):
    """
    Test a provider configuration.

    The key is optional: with it, an unsaved key can be verified before committing it; without
    it, the check uses the saved key or the environment. Either way the key is used for one
    request and never echoed back.
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = Field(default=None, max_length=500)
    base_url: Optional[str] = Field(default=None, max_length=500)
    #: Use the key already saved on the server rather than supplying one.
    use_saved: bool = True


# ----------------------------------------------------------------------------- responses
class ErrorResponse(BaseModel):
    """
    The shape of *every* error this API returns.

    One shape for all failures is what lets the frontend have a single error component
    instead of a per-endpoint guess at what went wrong. ``code`` is stable and switchable;
    ``action`` is the sentence a user can act on, and is why a 400 from here reads as
    "set OPENAI_API_KEY in your .env file" rather than "Bad Request".
    """

    code: str
    message: str
    action: Optional[str] = None
    detail: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    persistence: bool
    #: Present only when persistence failed, so a degraded deployment is visible.
    persistence_error: Optional[str] = None
    encryption: bool
    default_provider: str
    credential_configured: bool


class ProjectSummary(BaseModel):
    """A project as the workspace list shows it - no generated source."""

    id: str
    name: str
    requirement: str
    framework: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status: str
    created_at: str
    updated_at: str


class LLMSettingsResponse(BaseModel):
    """
    Saved LLM settings, readable by the frontend.

    There is deliberately no ``api_key`` field. ``credential_configured`` says whether a key
    exists and ``credential_hint`` shows its last four characters so a user can tell *which*
    key is saved, which is the only thing the UI actually needs.
    """

    provider: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    top_p: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    credential_configured: bool = False
    credential_hint: Optional[str] = None
    #: True when the key comes from the environment rather than saved storage.
    credential_from_env: bool = False
    #: False when `cryptography` is missing, so the UI can explain why saving a key is refused.
    can_store_credentials: bool = True
    source: str = "default"


class ProviderInfo(BaseModel):
    """One selectable LLM provider, with everything the settings page needs to render it."""

    name: str
    label: str
    description: str = ""
    default_model: Optional[str] = None
    runtime_model: Optional[str] = None
    credential_env: List[str] = Field(default_factory=list)
    #: Whether a credential for it is currently resolvable.
    configured: bool = False
    #: Whether its Python packages are importable.
    installed: bool = True
    install_command: Optional[str] = None
    local: bool = False
    notes: Optional[str] = None


class FrameworkInfo(BaseModel):
    """One target framework, for the picker on the create form."""

    name: str
    label: str
    description: str = ""
    strengths: List[str] = Field(default_factory=list)
    best_for: List[str] = Field(default_factory=list)
    installed: bool = True
    install_command: Optional[str] = None
