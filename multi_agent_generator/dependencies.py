# multi_agent_generator/dependencies.py
"""
Dependency resolution and preflight checking for generated projects.

This module exists because of a specific failure: the tool generated CrewAI code, and a
CrewAI test suite, while ``crewai`` was not a dependency of the tool and was never
mentioned to the user. On a clean machine the generated tests died at collection with
``ModuleNotFoundError: No module named 'crewai'``.

The fix is to write the requirements down in exactly one place. Everything that needs to
know what a generated project depends on reads this module:

* ``--check-deps`` reports what is missing before you hit it.
* Generated ``requirements-test.txt`` files are rendered from here.
* Generated ``conftest.py`` files use these import names for their skip guards.
* ``pyproject.toml`` extras mirror these groupings.

Because they share a source, the checker, the extras and the generated manifest cannot
drift apart.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, List, Optional, Sequence, Tuple

from .providers import get_provider

__all__ = [
    "Requirement",
    "DependencyReport",
    "BASE_REQUIREMENTS",
    "requirements_for",
    "check_dependencies",
    "install_command",
    "is_installed",
    "framework_import_name",
    "screen_requirements",
    "distribution_name",
]


@dataclass(frozen=True)
class Requirement:
    """A pip requirement together with the module name used to detect it."""

    #: The pip requirement specifier, e.g. ``"crewai>=0.80.0"``.
    spec: str
    #: The top-level module to import when checking whether it is installed.
    import_name: str
    #: Why this is needed, shown in ``--check-deps`` output.
    reason: str = ""

    @property
    def package(self) -> str:
        """
        The bare distribution name, without version specifier, extras or direct reference.

        The *earliest* separator wins rather than whichever one this list happens to mention
        first. ``crewai[tools]>=0.80.0`` contains both ``[`` and ``>=``, and testing ``>=``
        first left the extras bracket attached, yielding ``crewai[tools]`` as a distribution
        name. Nothing in the registry below uses extras, so it never showed up here - but
        requirements contributed by a tool spec or by model output can, and a mangled name
        propagates into the import validator as a false "undeclared dependency" report against
        code that is perfectly correct.
        """
        earliest = len(self.spec)
        for sep in (">=", "<=", "==", "~=", "!=", ">", "<", "[", "@", " "):
            index = self.spec.find(sep)
            if index != -1:
                earliest = min(earliest, index)
        return self.spec[:earliest].strip()


# --------------------------------------------------------------------------------------
# Requirements of every generated project
# --------------------------------------------------------------------------------------
#: Packages the emitted code imports whatever framework or provider it targets.
#:
#: The generated ``config`` module calls ``load_dotenv(override=False)`` inside
#: ``get_settings()`` so a local ``.env`` works without exporting anything by hand, and it
#: imports ``dotenv`` at module level to do it. That makes python-dotenv a hard requirement of
#: the code this tool writes, and it was previously not declared anywhere: it happened to be
#: importable because the *generator* depends on it, and because litellm - pulled in by most
#: providers - depends on it too. Neither of those helps the case that matters. Export a
#: project, install its own ``requirements.txt`` on a clean machine, and its very first import
#: raises ``ModuleNotFoundError: No module named 'dotenv'``. The huggingface-local provider
#: does not route through litellm at all, so that path failed even in place.
BASE_REQUIREMENTS: Tuple[Requirement, ...] = (
    Requirement(
        "python-dotenv>=1.0.0",
        "dotenv",
        "Reads .env in the generated config module",
    ),
)

# --------------------------------------------------------------------------------------
# Framework requirements
# --------------------------------------------------------------------------------------
#: Packages each generated framework needs in order to import and run.
FRAMEWORK_REQUIREMENTS: Dict[str, Tuple[Requirement, ...]] = {
    "crewai": (
        Requirement("crewai>=0.70.0", "crewai", "CrewAI agents, tasks and crews"),
    ),
    "crewai-flow": (
        Requirement("crewai>=0.70.0", "crewai", "CrewAI Flow event-driven workflows"),
        # The generated flow entry point declares its state as a pydantic model, so it
        # imports pydantic directly rather than through crewai. crewai does depend on
        # pydantic, but an undeclared transitive dependency is a working project by luck.
        Requirement("pydantic>=2.0.0", "pydantic", "Typed Flow state model"),
    ),
    "langgraph": (
        Requirement("langgraph>=0.2.0", "langgraph", "Stateful multi-actor graphs"),
        Requirement("langchain-core>=0.3.0", "langchain_core", "Messages and tools"),
    ),
    "react": (
        Requirement("langchain>=0.3.0", "langchain", "AgentExecutor and create_react_agent"),
        Requirement("langchain-core>=0.3.0", "langchain_core", "Prompts and tools"),
    ),
    "react-lcel": (
        Requirement("langchain-core>=0.3.0", "langchain_core", "LCEL runnables and prompts"),
    ),
    "agno": (
        Requirement("agno>=2.0.0", "agno", "Agno agents and teams"),
    ),
}

#: The module a generated test suite should guard on with ``pytest.importorskip``.
FRAMEWORK_IMPORT_NAME: Dict[str, str] = {
    "crewai": "crewai",
    "crewai-flow": "crewai",
    "langgraph": "langgraph",
    "react": "langchain",
    "react-lcel": "langchain_core",
    "agno": "agno",
}

# --------------------------------------------------------------------------------------
# Provider requirements
# --------------------------------------------------------------------------------------
#: Extra packages a provider needs, keyed by (provider, framework family).
PROVIDER_REQUIREMENTS: Dict[str, Tuple[Requirement, ...]] = {
    "openai": (
        Requirement("litellm>=1.0.0", "litellm", "Provider-agnostic routing"),
    ),
    "anthropic": (
        Requirement("litellm>=1.0.0", "litellm", "Provider-agnostic routing"),
    ),
    "groq": (
        Requirement("litellm>=1.0.0", "litellm", "Provider-agnostic routing"),
    ),
    "google": (
        Requirement("litellm>=1.0.0", "litellm", "Routes gemini/ model ids"),
    ),
    "watsonx": (
        Requirement("litellm>=1.0.0", "litellm", "Provider-agnostic routing"),
        Requirement("ibm-watsonx-ai>=0.2.0", "ibm_watsonx_ai", "IBM WatsonX SDK"),
    ),
    "ollama": (
        Requirement("litellm>=1.0.0", "litellm", "Provider-agnostic routing"),
    ),
    "huggingface": (
        Requirement("litellm>=1.0.0", "litellm", "Routes huggingface/ model ids"),
        Requirement("huggingface-hub>=0.24.0", "huggingface_hub", "Hub API and auth"),
    ),
    "huggingface-local": (
        Requirement("transformers>=4.40.0", "transformers", "Local model execution"),
        Requirement("torch>=2.0.0", "torch", "Tensor backend for transformers"),
        Requirement("accelerate>=0.30.0", "accelerate", "Device placement for local models"),
    ),
}

#: Additional packages needed by *generated code* for a provider/framework family pair.
#: The generated LangChain code imports partner packages that the provider registry
#: emits; those are only needed when that combination is actually generated.
GENERATED_CODE_REQUIREMENTS: Dict[Tuple[str, str], Tuple[Requirement, ...]] = {
    ("openai", "langchain"): (
        Requirement("langchain-openai>=0.2.0", "langchain_openai", "ChatOpenAI"),
    ),
    ("anthropic", "langchain"): (
        Requirement("langchain-anthropic>=0.2.0", "langchain_anthropic", "ChatAnthropic"),
    ),
    ("groq", "langchain"): (
        Requirement("langchain-groq>=0.2.0", "langchain_groq", "ChatGroq"),
    ),
    ("google", "langchain"): (
        Requirement(
            "langchain-google-genai>=2.0.0",
            "langchain_google_genai",
            "ChatGoogleGenerativeAI",
        ),
    ),
    ("ollama", "langchain"): (
        Requirement("langchain-ollama>=0.2.0", "langchain_ollama", "ChatOllama"),
    ),
    ("watsonx", "langchain"): (
        Requirement("langchain-ibm>=0.3.0", "langchain_ibm", "ChatWatsonx"),
    ),
    ("huggingface", "langchain"): (
        Requirement(
            "langchain-huggingface>=0.1.0",
            "langchain_huggingface",
            "ChatHuggingFace and HuggingFaceEndpoint",
        ),
        Requirement("huggingface-hub>=0.24.0", "huggingface_hub", "Hub API and auth"),
    ),
    ("huggingface-local", "langchain"): (
        Requirement(
            "langchain-huggingface>=0.1.0",
            "langchain_huggingface",
            "ChatHuggingFace and HuggingFacePipeline",
        ),
        Requirement("transformers>=4.40.0", "transformers", "Local model execution"),
        Requirement("torch>=2.0.0", "torch", "Tensor backend for transformers"),
    ),
}

#: Packages needed to *run* a generated test suite.
TEST_REQUIREMENTS: Tuple[Requirement, ...] = (
    Requirement("pytest>=7.0.0", "pytest", "Test runner"),
    # Generated tests are decorated with @pytest.mark.timeout(...). Without this plugin
    # the mark is inert, and under --strict-markers it is a hard error. It was not
    # declared anywhere before, not even in the dev extra.
    Requirement("pytest-timeout>=2.1.0", "pytest_timeout", "Enables @pytest.mark.timeout"),
    # The offline reliability tests build a small prompt -> model -> parser chain, so they
    # need langchain-core even in a CrewAI or Agno suite, which would not otherwise pull
    # it in. Note that the stand-in model itself is NOT from here: it lives in the
    # generated _helpers.py and imports only the standard library, because reaching for
    # langchain_core.language_models drags in transformers and stalls for minutes.
    Requirement("langchain-core>=0.3.0", "langchain_core", "Prompts and parsers for offline tests"),
)


def framework_import_name(framework: str) -> str:
    """The module a generated suite for ``framework`` should guard on."""
    return FRAMEWORK_IMPORT_NAME.get(framework, framework.replace("-", "_"))


# --------------------------------------------------------------------------------------
# Screening requirement lines
# --------------------------------------------------------------------------------------
#: Requirement lines that would install from somewhere other than the configured index.
#: Refused rather than sanitised: a generated project has no legitimate reason to install from
#: a URL or a local path, and quietly stripping the line would install a *different* set of
#: packages than the file says, which is worse than refusing.
UNSAFE_REQUIREMENT_LINE = re.compile(
    r"^\s*(?:-e\b|--editable\b|--index-url\b|--extra-index-url\b|--find-links\b|-f\b|"
    r"--trusted-host\b|--pre\b|https?://|git\+|file:|\.{1,2}/)",
    re.IGNORECASE,
)

#: A PEP 508 *direct reference*: ``name @ url``. This needs its own pattern because the one
#: above anchors on the line start, and a direct reference starts with an ordinary package
#: name - so ``crewai @ git+https://example.invalid/repo`` sailed through a screen whose whole
#: stated purpose is "only named packages from the configured index". The URL is the second
#: token, not the first, and pip honours it. The space around ``@`` is optional because pip
#: accepts ``crewai@https://...`` too, and ``@`` has no other meaning in a requirement line.
DIRECT_REFERENCE = re.compile(r"^\s*[A-Za-z0-9][A-Za-z0-9._+-]*(?:\[[^\]]*\])?\s*@")

#: Appended to every refusal so the user is told the rule, not just the verdict.
_REFUSAL_REASON = (
    "refused: only named packages from the configured index are installed for a "
    "generated project."
)


def distribution_name(line: str) -> str:
    """
    The bare distribution name from one ``requirements.txt`` line, or ``""``.

    Handles the four things that actually appear in a generated requirements file: a version
    specifier, an extras bracket (``crewai[tools]>=0.80.0``), an environment marker
    (``uvloop; sys_platform != "win32"``) and a trailing comment. Option lines such as
    ``--index-url`` return ``""``, because they are not distributions at all.
    """
    text = str(line or "").split("#", 1)[0].strip()
    if not text or text.startswith("-"):
        return ""
    text = text.split(";", 1)[0].strip()
    if not text:
        return ""
    return Requirement(spec=text, import_name="").package.strip()


def screen_requirements(text: str) -> Tuple[List[str], List[str]]:
    """
    Split requirement lines into the ones that may be installed and the ones that may not.

    Returns ``(allowed, rejected)`` where each rejected entry already carries its reason, so
    a caller can show the user which line was refused and why rather than a generic
    "dependencies were rejected".

    This lives here, next to the requirement registry, rather than in the installer, because
    two callers need the same answer: the sandbox decides what to install with it, and
    validation warns *before* a run that a project's own requirements file contains a line
    the sandbox will refuse. Two copies of this rule would eventually disagree, and the
    failure mode of that disagreement is a project validated as installable that then fails
    to install.
    """
    allowed: List[str] = []
    rejected: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if UNSAFE_REQUIREMENT_LINE.match(line) or DIRECT_REFERENCE.match(line):
            rejected.append(f"{line}  -> {_REFUSAL_REASON}")
            continue
        allowed.append(line)
    return allowed, rejected


def is_installed(import_name: str) -> bool:
    """Whether ``import_name`` can be imported in the current interpreter."""
    try:
        import_module(import_name)
        return True
    except Exception:
        # Deliberately broad: a package can be present but raise on import (a broken
        # native extension, say), and for our purposes that is just as unusable.
        return False


def _dedupe(requirements: Sequence[Requirement]) -> List[Requirement]:
    """Drop duplicate packages, keeping the first mention of each."""
    seen = {}
    for req in requirements:
        seen.setdefault(req.package, req)
    return list(seen.values())


def requirements_for(
    framework: Optional[str] = None,
    provider: Optional[str] = None,
    include_tests: bool = False,
) -> List[Requirement]:
    """
    Everything a generated project needs.

    Args:
        framework: Target framework, or None to skip framework packages.
        provider: LLM provider, or None to skip provider packages.
        include_tests: Also include the packages needed to run generated tests.
    """
    from .providers import FRAMEWORK_FAMILY

    # Always first: every emitted project imports these, so they are not conditional on
    # anything the caller passes in.
    collected: List[Requirement] = list(BASE_REQUIREMENTS)

    if framework:
        collected.extend(FRAMEWORK_REQUIREMENTS.get(framework, ()))

    if provider:
        spec = get_provider(provider)
        collected.extend(PROVIDER_REQUIREMENTS.get(spec.name, ()))
        if framework:
            family = FRAMEWORK_FAMILY.get(framework)
            if family:
                collected.extend(
                    GENERATED_CODE_REQUIREMENTS.get((spec.name, family), ())
                )

    if include_tests:
        collected.extend(TEST_REQUIREMENTS)

    return _dedupe(collected)


@dataclass
class DependencyReport:
    """The outcome of a preflight dependency check."""

    framework: Optional[str]
    provider: Optional[str]
    installed: List[Requirement]
    missing: List[Requirement]

    @property
    def ok(self) -> bool:
        return not self.missing

    def install_command(self) -> str:
        return install_command(self.missing)

    def render(self) -> str:
        """A human-readable report for the CLI."""
        lines: List[str] = []
        target = " / ".join(p for p in (self.framework, self.provider) if p)
        lines.append(f"Dependency check{f' for {target}' if target else ''}")
        lines.append("=" * 52)

        if self.installed:
            lines.append("")
            lines.append("Installed:")
            for req in self.installed:
                lines.append(f"  [ok]      {req.package:<28} {req.reason}")

        if self.missing:
            lines.append("")
            lines.append("Missing:")
            for req in self.missing:
                lines.append(f"  [MISSING] {req.package:<28} {req.reason}")
            lines.append("")
            lines.append("Install the missing packages with:")
            lines.append(f"  {self.install_command()}")
        else:
            lines.append("")
            lines.append("All required packages are installed.")

        spec = get_provider(self.provider) if self.provider else None
        if spec and spec.credential_env and not spec.resolve_credential():
            lines.append("")
            lines.append(
                f"Note: no credentials found for {spec.label}. Set one of: "
                f"{', '.join(spec.credential_env)}."
            )
            if spec.notes:
                lines.append(f"      {spec.notes}")

        return "\n".join(lines)


def check_dependencies(
    framework: Optional[str] = None,
    provider: Optional[str] = None,
    include_tests: bool = True,
) -> DependencyReport:
    """Check which required packages are present and which are missing."""
    requirements = requirements_for(framework, provider, include_tests=include_tests)
    installed = [r for r in requirements if is_installed(r.import_name)]
    missing = [r for r in requirements if not is_installed(r.import_name)]
    return DependencyReport(
        framework=framework,
        provider=provider,
        installed=installed,
        missing=missing,
    )


def install_command(requirements: Sequence[Requirement]) -> str:
    """Render a copy-pasteable pip install command for ``requirements``."""
    if not requirements:
        return "# nothing to install"
    specs = " ".join(f"'{r.spec}'" for r in requirements)
    return f"pip install {specs}"


def requirements_txt(
    framework: Optional[str] = None,
    provider: Optional[str] = None,
    include_tests: bool = True,
) -> str:
    """
    Render a ``requirements.txt`` for a generated project.

    Emitted alongside generated code so that a handed-over project states its own
    dependencies instead of leaving the recipient to guess.
    """
    reqs = requirements_for(framework, provider, include_tests=include_tests)
    header = [
        "# Auto-generated by multi-agent-generator.",
    ]
    if framework:
        header.append(f"# Framework: {framework}")
    if provider:
        header.append(f"# Provider:  {provider}")
    header.append("#")
    header.append("# Install with:  pip install -r requirements-test.txt")
    header.append("")

    body = []
    for req in reqs:
        if req.reason:
            body.append(f"# {req.reason}")
        body.append(req.spec)
    return "\n".join(header + body) + "\n"
