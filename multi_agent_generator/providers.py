# multi_agent_generator/providers.py
"""
Central provider registry.

This module is the single source of truth for "what does a provider mean". It answers
three different questions that used to be answered in six different places:

1. Which model should the *generator* use to analyse a prompt? -> ``ProviderSpec.default_model``
2. Which model should the *generated agents* use at runtime?    -> ``ProviderSpec.runtime_model``
3. How do I construct that provider's LLM object inside emitted
   code for a given framework?                                 -> :func:`llm_snippet`

Keeping these together is what stops the generator and its own output from disagreeing
about a provider, which is exactly what happened before: you could generate agents with
a local Ollama model and still get a file back that demanded an OPENAI_API_KEY.

Adding a provider should be a matter of adding one entry to ``PROVIDERS`` plus one entry
per framework family in the snippet builders below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .errors import UnknownProviderError

# --------------------------------------------------------------------------------------
# Framework families
# --------------------------------------------------------------------------------------
# Several frameworks share an LLM construction style, so snippets are written per family
# rather than per framework.
FRAMEWORK_FAMILY: Dict[str, str] = {
    "crewai": "crewai",
    "crewai-flow": "crewai",
    "langgraph": "langchain",
    "react": "langchain",
    "react-lcel": "langchain",
    "agno": "agno",
}

#: Base URL used for Hugging Face's OpenAI-compatible inference router.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

#: Default base URL assumed for a locally served, OpenAI-compatible model server
#: (vLLM, text-generation-inference or llama.cpp).
HF_LOCAL_BASE_URL_DEFAULT = "http://localhost:8000/v1"


class UnsupportedCombinationError(ValueError):
    """Raised when a provider cannot be expressed in a given framework."""


# A self-contained preamble emitted into generated files that need a Hugging Face token.
# It does three things the old one-liner did not: soft-loads a local .env so a standalone
# script sees the same token the package does, accepts all three variable names the HF
# ecosystem uses, and fails *early* with a one-line instruction instead of a ValueError
# thrown six frames deep inside huggingface_hub. This is the fix for
# "You must provide an api_key to work with auto API".
_HF_TOKEN_PREAMBLE = (
    "try:  # Load a local .env if python-dotenv is available; harmless if it is not.\n"
    "    from dotenv import load_dotenv\n"
    "    load_dotenv()\n"
    "except Exception:\n"
    "    pass\n"
    "\n"
    "\n"
    "def _resolve_hf_token():\n"
    '    """First Hugging Face token found across the accepted variable names."""\n'
    "    for _var in (\"HF_TOKEN\", \"HUGGINGFACEHUB_API_TOKEN\", \"HUGGINGFACE_API_KEY\"):\n"
    "        _value = os.environ.get(_var)\n"
    "        if _value:\n"
    "            return _value\n"
    "    raise RuntimeError(\n"
    '        "Hugging Face API key is missing. Set HF_TOKEN (or "\n'
    '        "HUGGINGFACEHUB_API_TOKEN / HUGGINGFACE_API_KEY) in your environment or a "\n'
    '        ".env file. Get a free token at https://huggingface.co/settings/tokens."\n'
    "    )\n"
    "\n"
    "\n"
    "HF_TOKEN = _resolve_hf_token()\n"
)


@dataclass(frozen=True)
class ProviderSpec:
    """Everything the project needs to know about one LLM provider."""

    name: str
    label: str
    #: LiteLLM-style model id used by the generator itself.
    default_model: str
    #: Model / repo id written into generated agent code.
    runtime_model: str
    #: Environment variables that may carry credentials, in priority order.
    credential_env: Tuple[str, ...] = ()
    #: pyproject extras a user needs installed for this provider.
    extras: Tuple[str, ...] = ()
    #: "litellm" for anything LiteLLM can reach, "transformers" for in-process execution.
    backend: str = "litellm"
    #: Sampling parameters this provider rejects, dropped before the call is made.
    drop_params: Tuple[str, ...] = ()
    #: True when the provider runs entirely on the user's machine.
    local: bool = False
    description: str = ""
    notes: str = ""

    def resolve_credential(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Return the first credential found in ``credential_env``, or None."""
        import os

        source = env if env is not None else os.environ
        for key in self.credential_env:
            value = source.get(key)
            if value:
                return value
        return None


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------
PROVIDERS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        label="OpenAI",
        default_model="gpt-4o-mini",
        runtime_model="gpt-4o-mini",
        credential_env=("OPENAI_API_KEY", "API_KEY"),
        description="Hosted GPT models. Requires a paid API key.",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        label="Anthropic",
        default_model="anthropic/claude-3-5-sonnet-20241022",
        runtime_model="claude-3-5-sonnet-20241022",
        credential_env=("ANTHROPIC_API_KEY", "API_KEY"),
        description="Hosted Claude models. Requires a paid API key.",
    ),
    "watsonx": ProviderSpec(
        name="watsonx",
        label="IBM WatsonX",
        default_model="watsonx/meta-llama/llama-3-3-70b-instruct",
        runtime_model="meta-llama/llama-3-3-70b-instruct",
        credential_env=("WATSONX_API_KEY", "WATSONX_APIKEY", "API_KEY"),
        extras=("watsonx",),
        description="Enterprise access to Llama and other foundation models.",
    ),
    "ollama": ProviderSpec(
        name="ollama",
        label="Ollama (local)",
        default_model="ollama/llama3.2:3b",
        runtime_model="llama3.2:3b",
        extras=("ollama",),
        local=True,
        description="Open-weight models served locally by an Ollama daemon.",
        notes="Requires `ollama serve` to be running. No API key needed.",
    ),
    # ---- Hugging Face: hosted -----------------------------------------------------
    "huggingface": ProviderSpec(
        name="huggingface",
        label="Hugging Face (hosted inference)",
        # The old 7B id is no longer exposed by Hugging Face's routed providers. Keep the
        # model configurable through HF_MODEL_ID so an existing generated project can switch
        # models without being regenerated.
        default_model="huggingface/Qwen/Qwen2.5-72B-Instruct",
        runtime_model="Qwen/Qwen2.5-72B-Instruct",
        credential_env=("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"),
        extras=("huggingface",),
        # HF's inference API does not implement the OpenAI penalty parameters; sending
        # them results in a 422 rather than being ignored.
        drop_params=("frequency_penalty", "presence_penalty"),
        description=(
            "Open-weight models run on Hugging Face's inference service. No GPU or "
            "model download required; needs a free HF_TOKEN."
        ),
        notes="Get a token at https://huggingface.co/settings/tokens",
    ),
    # ---- Hugging Face: on-device --------------------------------------------------
    "huggingface-local": ProviderSpec(
        name="huggingface-local",
        label="Hugging Face (local transformers)",
        # Deliberately small: a first run should finish on a laptop rather than fill
        # the disk. Override with HF_MODEL_ID or --model for something larger.
        default_model="Qwen/Qwen2.5-0.5B-Instruct",
        runtime_model="Qwen/Qwen2.5-0.5B-Instruct",
        extras=("huggingface-local",),
        backend="transformers",
        drop_params=("frequency_penalty", "presence_penalty", "top_p"),
        local=True,
        description=(
            "Open-weight models downloaded from the Hub and executed in-process with "
            "transformers. Fully offline and free, but needs disk space and CPU/GPU."
        ),
        notes=(
            "First run downloads weights. Set HF_MODEL_ID to pick a different model. "
            "CrewAI and Agno cannot execute a transformers pipeline in-process; for "
            "those frameworks this provider targets a local OpenAI-compatible server."
        ),
    ),
    "groq": ProviderSpec(
        name="groq",
        label="Groq",
        default_model="groq/llama-3.3-70b-versatile",
        runtime_model="llama-3.3-70b-versatile",
        credential_env=("GROQ_API_KEY", "API_KEY"),
        description="Fast hosted inference for open-weight Llama models.",
    ),
    "google": ProviderSpec(
        name="google",
        label="Google Gemini",
        # `gemini/` is LiteLLM's route for the public Gemini API. The `vertex_ai/` route
        # is a different product needing service-account credentials, not an API key.
        default_model="gemini/gemini-1.5-flash",
        runtime_model="gemini-1.5-flash",
        credential_env=("GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"),
        extras=("google",),
        description="Hosted Gemini models. Free tier available with an API key.",
        notes="Get a key at https://aistudio.google.com/app/apikey",
    ),
    # ---- Offline stub ----------------------------------------------------------------
    # Not a real provider. It exists so the pipeline, the API and the frontend can be
    # driven end to end with no credentials, no network and no cost, and so this
    # project's own tests can assert on pipeline behaviour instead of model behaviour.
    # Responses are marked `simulated` all the way up to the UI, so stub output is never
    # presented as a model's work, and it is never selected as a silent fallback.
    "mock": ProviderSpec(
        name="mock",
        label="Mock (offline stub)",
        default_model="mock-stub",
        runtime_model="mock-stub",
        backend="stub",
        local=True,
        description=(
            "Deterministic offline stub. No network, no key, no cost. Use it to try the "
            "pipeline or to run tests; it does not produce real model output."
        ),
        notes="Select explicitly with --provider mock.",
    ),
}

#: Providers that serve open-weight models, for documentation and CLI grouping.
OPEN_SOURCE_PROVIDERS = ("huggingface", "huggingface-local", "ollama", "groq")

DEFAULT_PROVIDER = "openai"


def list_providers() -> List[ProviderSpec]:
    """All registered providers, in registry order."""
    return list(PROVIDERS.values())


def provider_names() -> List[str]:
    """Provider names, suitable for an argparse ``choices`` list."""
    return list(PROVIDERS.keys())


def get_provider(name: Optional[str]) -> ProviderSpec:
    """
    Look up a provider, tolerating the aliases users actually type.

    Unknown names raise rather than silently falling through to a branch that treats
    the provider name as a model id, which is what the previous code did.
    """
    if not name:
        return PROVIDERS[DEFAULT_PROVIDER]

    key = name.strip().lower()
    aliases = {
        "hf": "huggingface",
        "hf-local": "huggingface-local",
        "huggingface_local": "huggingface-local",
        "hf_local": "huggingface-local",
        "local": "huggingface-local",
        "transformers": "huggingface-local",
        "openai-compatible": "openai",
        "ibm": "watsonx",
        "watsonx-ai": "watsonx",
        "claude": "anthropic",
        "gemini": "google",
        "google-gemini": "google",
        "stub": "mock",
        "offline": "mock",
        "fake": "mock",
    }
    key = aliases.get(key, key)

    if key not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise UnknownProviderError(
            f"Unknown provider {name!r}.",
            action=f"Choose one of: {known}.",
            context={"provider": name},
        )
    return PROVIDERS[key]


def _apply_provider_prefix(spec: ProviderSpec, model: str) -> str:
    """
    Give ``model`` the LiteLLM route prefix its provider needs.

    LiteLLM decides which backend to call from the ``provider/`` prefix on the model id,
    so a bare ``Qwen/Qwen2.5-7B-Instruct`` or ``llama3.2:3b`` is routed to OpenAI and
    fails with a confusing authentication error. The prefix is derived from the registry
    default rather than hardcoded, so it stays correct as providers are added.
    """
    if spec.backend != "litellm":
        # The local transformers backend loads a bare Hub repo id.
        return model
    if "/" not in spec.default_model:
        # openai: LiteLLM takes unprefixed ids.
        return model
    prefix = spec.default_model.split("/", 1)[0]
    if model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def resolve_generator_model(provider: str, override: Optional[str] = None) -> str:
    """
    Model id for the generator's own inference call.

    Precedence: explicit override, then a provider-specific env var, then
    ``DEFAULT_MODEL``, then the registry default. Every path except the registry default
    is run through :func:`_apply_provider_prefix`, because a user passing
    ``--model Qwen/Qwen2.5-7B-Instruct`` means "that model on the provider I selected",
    not "that model on OpenAI".
    """
    import os

    spec = get_provider(provider)

    candidate = override
    if not candidate and spec.name in ("huggingface", "huggingface-local"):
        candidate = os.getenv("HF_MODEL_ID")
    if not candidate:
        candidate = os.getenv("DEFAULT_MODEL")

    if not candidate:
        return spec.default_model
    return _apply_provider_prefix(spec, candidate)


def _strip_provider_prefix(spec: ProviderSpec, model: str) -> str:
    """
    Remove a redundant provider route prefix from ``model``.

    Runtime model ids must stay *unprefixed*, because the snippet builders add whatever
    routing prefix their framework needs. Without this, ``--model huggingface/Qwen/...``
    produced ``LLM(model="huggingface/huggingface/Qwen/...")``, which LiteLLM cannot
    route.

    Only the provider's own prefix is stripped, and the LiteLLM route prefix is only
    considered for LiteLLM-backed providers. That matters: ``huggingface-local`` defaults
    to ``Qwen/Qwen2.5-0.5B-Instruct``, so treating its first path segment as a route
    prefix would mangle every Hub repo id starting with ``Qwen/``.
    """
    prefixes = {spec.name}
    if spec.backend == "litellm" and "/" in spec.default_model:
        prefixes.add(spec.default_model.split("/", 1)[0])
    for prefix in prefixes:
        if model.startswith(f"{prefix}/"):
            return model[len(prefix) + 1 :]
    return model


def resolve_runtime_model(provider: str, override: Optional[str] = None) -> str:
    """
    Model id to write into generated agent code.

    Unlike the generator's own model this is usually *unprefixed*, because the generated
    code names its provider through a native class (``ChatAnthropic``, ``ChatGroq``) that
    expects a plain model id. Only the CrewAI family, which routes through LiteLLM, needs
    the prefix, and its snippet builder adds it.
    """
    spec = get_provider(provider)
    if override:
        return _strip_provider_prefix(spec, override)
    return spec.runtime_model


# --------------------------------------------------------------------------------------
# Code generation snippets
# --------------------------------------------------------------------------------------
@dataclass
class LLMSnippet:
    """Import lines plus a setup block that defines ``llm`` in generated code."""

    imports: List[str] = field(default_factory=list)
    setup: str = ""
    #: Module-level code that must run *before* the setup block or the expression - for
    #: example credential resolution. Kept separate from ``setup`` because Agno consumes
    #: only ``expression`` (inside a function body), so it needs the module-level part
    #: emitted independently.
    preamble: str = ""
    #: Expression form, for frameworks that take the model inline (Agno).
    expression: str = ""
    comment: str = ""
    #: The model id this snippet actually writes into the generated code. Generators use
    #: it for the file header, so the banner can never disagree with the code below it.
    model: str = ""

    def import_block(self) -> str:
        return "\n".join(self.imports)


def _langchain_snippet(spec: ProviderSpec, model: str) -> LLMSnippet:
    """LLM construction for LangGraph / ReAct / ReAct-LCEL output."""
    if spec.name == "openai":
        return LLMSnippet(
            imports=["from langchain_openai import ChatOpenAI"],
            setup=f'llm = ChatOpenAI(model="{model}", temperature=0.7)',
        )

    if spec.name == "anthropic":
        return LLMSnippet(
            imports=["from langchain_anthropic import ChatAnthropic"],
            setup=f'llm = ChatAnthropic(model="{model}", temperature=0.7)',
        )

    if spec.name == "groq":
        return LLMSnippet(
            imports=["from langchain_groq import ChatGroq"],
            setup=f'llm = ChatGroq(model="{model}", temperature=0.7)',
        )

    if spec.name == "ollama":
        return LLMSnippet(
            imports=["from langchain_ollama import ChatOllama"],
            setup=f'llm = ChatOllama(model="{model}", temperature=0.7)',
            comment="Requires a running `ollama serve` and `ollama pull` for the model.",
        )

    if spec.name == "watsonx":
        return LLMSnippet(
            imports=["import os", "from langchain_ibm import ChatWatsonx"],
            setup=(
                "llm = ChatWatsonx(\n"
                f'    model_id="{model}",\n'
                '    url=os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),\n'
                '    project_id=os.environ.get("WATSONX_PROJECT_ID"),\n'
                ")"
            ),
        )

    if spec.name == "google":
        return LLMSnippet(
            imports=["from langchain_google_genai import ChatGoogleGenerativeAI"],
            setup=f'llm = ChatGoogleGenerativeAI(model="{model}", temperature=0.7)',
            comment="Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment.",
        )

    if spec.name == "mock":
        return LLMSnippet(
            imports=["from langchain_core.language_models import FakeListChatModel"],
            setup=(
                f"# Offline stub for model={model}: no network, no key, no cost. Replace\n"
                "# this with a real provider before using the agents for anything that matters.\n"
                'llm = FakeListChatModel(responses=["[offline stub response]"] * 64)'
            ),
            comment="Generated with --provider mock. Output is simulated, not real.",
        )

    if spec.name == "huggingface":
        return LLMSnippet(
            imports=[
                "import os",
                "from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint",
            ],
            preamble=_HF_TOKEN_PREAMBLE,
            setup=(
                "# Open-weight model served by Hugging Face's hosted inference API.\n"
                "_hf_endpoint = HuggingFaceEndpoint(\n"
                f'    repo_id="{model}",\n'
                '    task="text-generation",\n'
                "    max_new_tokens=512,\n"
                "    temperature=0.7,\n"
                "    huggingfacehub_api_token=HF_TOKEN,\n"
                ")\n"
                "llm = ChatHuggingFace(llm=_hf_endpoint)"
            ),
            comment="Set HF_TOKEN in your environment (free at huggingface.co/settings/tokens).",
        )

    if spec.name == "huggingface-local":
        return LLMSnippet(
            imports=[
                "from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline",
            ],
            setup=(
                "# Open weights downloaded from the Hub and executed in this process.\n"
                "# The first run downloads the model; afterwards it works fully offline.\n"
                "_hf_pipeline = HuggingFacePipeline.from_model_id(\n"
                f'    model_id="{model}",\n'
                '    task="text-generation",\n'
                '    pipeline_kwargs={"max_new_tokens": 512, "do_sample": True, "temperature": 0.7},\n'
                ")\n"
                "llm = ChatHuggingFace(llm=_hf_pipeline)"
            ),
            comment="Needs the huggingface-local extra (transformers + torch).",
        )

    raise UnsupportedCombinationError(
        f"Provider {spec.name!r} has no LangChain integration configured."
    )


def _crewai_snippet(spec: ProviderSpec, model: str) -> LLMSnippet:
    """
    LLM construction for CrewAI / CrewAI-Flow output.

    CrewAI's own ``LLM`` class routes through LiteLLM, so a prefixed model string is
    the native way to reach any provider.
    """
    imports = ["import os", "from crewai import LLM"]

    if spec.name == "openai":
        return LLMSnippet(
            imports=["from crewai import LLM"],
            setup=f'llm = LLM(model="{model}")',
        )

    if spec.name == "huggingface":
        return LLMSnippet(
            imports=imports,
            preamble=_HF_TOKEN_PREAMBLE,
            setup=(
                "# Open-weight model via Hugging Face's OpenAI-compatible router.\n"
                "# Override HF_MODEL_ID when the selected model is not enabled for your token.\n"
                "llm = LLM(\n"
                f'    model="openai/" + os.environ.get("HF_MODEL_ID", {model!r}),\n'
                '    base_url="https://router.huggingface.co/v1",\n'
                "    api_key=HF_TOKEN,\n"
                ")"
            ),
            comment="Set HF_TOKEN in your environment.",
        )

    if spec.name == "huggingface-local":
        return LLMSnippet(
            imports=imports,
            setup=(
                "# CrewAI cannot execute a transformers pipeline in-process, so local\n"
                "# weights must be served behind an OpenAI-compatible endpoint. Start one\n"
                "# with vLLM, text-generation-inference or llama.cpp, for example:\n"
                f"#   vllm serve {model} --port 8000\n"
                "llm = LLM(\n"
                f'    model="openai/{model}",\n'
                f'    base_url=os.environ.get("HF_LOCAL_BASE_URL", "{HF_LOCAL_BASE_URL_DEFAULT}"),\n'
                '    api_key="not-needed",\n'
                ")"
            ),
            comment="Requires a local OpenAI-compatible model server.",
        )

    if spec.name == "ollama":
        return LLMSnippet(
            imports=imports,
            setup=(
                "llm = LLM(\n"
                f'    model="ollama/{model}",\n'
                '    base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),\n'
                ")"
            ),
            comment="Requires a running `ollama serve`.",
        )

    if spec.name in ("anthropic", "groq", "watsonx"):
        return LLMSnippet(
            imports=["from crewai import LLM"],
            setup=f'llm = LLM(model="{spec.name}/{model}")',
        )

    if spec.name == "google":
        return LLMSnippet(
            imports=["from crewai import LLM"],
            setup=f'llm = LLM(model="gemini/{model}")',
            comment="Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment.",
        )

    if spec.name == "mock":
        return LLMSnippet(
            imports=["from crewai import LLM"],
            setup=(
                "# Offline stub. CrewAI always routes through a real endpoint, so there is\n"
                "# no genuine offline mode here: this points at a local OpenAI-compatible\n"
                "# server so the file is runnable once you start one. Regenerate with a\n"
                "# real --provider before relying on the output.\n"
                'llm = LLM(model="openai/mock-stub", '
                f'base_url="{HF_LOCAL_BASE_URL_DEFAULT}", api_key="not-needed")'
            ),
            comment="Generated with --provider mock. Not a real model.",
        )

    raise UnsupportedCombinationError(
        f"Provider {spec.name!r} has no CrewAI integration configured."
    )


def _agno_snippet(spec: ProviderSpec, model: str) -> LLMSnippet:
    """
    Model construction for Agno output.

    Every provider here is expressed through Agno's ``OpenAIChat`` class with a
    ``base_url`` override rather than a provider-specific Agno class. That is a
    deliberate choice: Hugging Face's router and Ollama both implement the OpenAI
    chat-completions API, so the requests are correct, and relying only on
    ``OpenAIChat`` means the emitted import is one already known to exist at the
    pinned Agno version. Guessing at provider-specific Agno class names would risk
    trading a working integration for a plausible-looking ImportError.
    """
    if spec.name == "openai":
        return LLMSnippet(
            imports=["from agno.models.openai import OpenAIChat"],
            expression=f'OpenAIChat(id="{model}")',
        )

    if spec.name == "huggingface":
        return LLMSnippet(
            imports=["import os", "from agno.models.openai import OpenAIChat"],
            preamble=_HF_TOKEN_PREAMBLE,
            expression=(
                f'OpenAIChat(\n'
                f'        id="{model}",\n'
                f'        base_url="{HF_ROUTER_BASE_URL}",\n'
                f'        api_key=HF_TOKEN,\n'
                f'    )'
            ),
            comment=(
                "Hugging Face's inference router is OpenAI-compatible, so Agno's "
                "OpenAIChat can talk to open-weight models directly. Set HF_TOKEN."
            ),
        )

    if spec.name == "huggingface-local":
        return LLMSnippet(
            imports=["import os", "from agno.models.openai import OpenAIChat"],
            expression=(
                f'OpenAIChat(\n'
                f'        id="{model}",\n'
                f'        base_url=os.environ.get("HF_LOCAL_BASE_URL", "{HF_LOCAL_BASE_URL_DEFAULT}"),\n'
                f'        api_key="not-needed",\n'
                f'    )'
            ),
            comment=(
                "Agno cannot execute a transformers pipeline in-process. Serve the "
                f"weights locally first, e.g. `vllm serve {model} --port 8000`."
            ),
        )

    if spec.name == "ollama":
        return LLMSnippet(
            imports=["import os", "from agno.models.openai import OpenAIChat"],
            expression=(
                f'OpenAIChat(\n'
                f'        id="{model}",\n'
                f'        base_url=os.environ.get("OLLAMA_OPENAI_URL", "http://localhost:11434/v1"),\n'
                f'        api_key="ollama",\n'
                f'    )'
            ),
            comment="Ollama exposes an OpenAI-compatible API on /v1.",
        )

    if spec.name == "mock":
        return LLMSnippet(
            imports=["from agno.models.openai import OpenAIChat"],
            expression=(
                f'OpenAIChat(\n'
                f'        id="mock-stub",\n'
                f'        base_url="{HF_LOCAL_BASE_URL_DEFAULT}",\n'
                f'        api_key="not-needed",\n'
                f'    )'
            ),
            comment=(
                "Generated with --provider mock. Agno always calls a real endpoint, so "
                "this points at a local OpenAI-compatible server. Regenerate with a real "
                "provider before relying on the output."
            ),
        )

    raise UnsupportedCombinationError(
        f"Agno output does not support the {spec.name!r} provider. Agno reaches models "
        "through an OpenAI-compatible endpoint, so use one of: openai, huggingface, "
        "huggingface-local, ollama - or pick a different --framework."
    )


_SNIPPET_BUILDERS = {
    "langchain": _langchain_snippet,
    "crewai": _crewai_snippet,
    "agno": _agno_snippet,
}


def llm_snippet(
    framework: str,
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
) -> LLMSnippet:
    """
    Build the LLM setup code for ``framework`` using ``provider``.

    Raises:
        UnsupportedCombinationError: if the provider cannot be expressed in that
            framework, with a message naming the providers that can.
    """
    spec = get_provider(provider)
    family = FRAMEWORK_FAMILY.get(framework)
    if family is None:
        known = ", ".join(sorted(FRAMEWORK_FAMILY))
        raise ValueError(f"Unknown framework {framework!r}. Known frameworks: {known}")

    resolved = resolve_runtime_model(spec.name, model)
    snippet = _SNIPPET_BUILDERS[family](spec, resolved)
    # Record what was actually used, so a generator's header banner reports the same
    # model the code below it constructs.
    snippet.model = resolved
    return snippet


def provider_summary(spec: ProviderSpec) -> str:
    """One-paragraph human description, used by ``--list-providers``."""
    bits = [spec.description]
    if spec.credential_env:
        bits.append(f"Credentials: {' or '.join(spec.credential_env)}.")
    else:
        bits.append("No API key required.")
    if spec.extras:
        extras = ",".join(spec.extras)
        bits.append(f"Install extra: pip install 'multi-agent-generator[{extras}]'")
    if spec.notes:
        bits.append(spec.notes)
    return " ".join(b for b in bits if b)
