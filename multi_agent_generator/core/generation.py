# multi_agent_generator/core/generation.py
"""
Stage 3: turn a chosen framework and an analysis into a runnable project.

The output is a :class:`GeneratedProject` - a directory of files - not a single Python
string. That distinction is the whole reason this module exists. The old flow generated
one ``.py`` blob and printed it, which meant the thing handed to the user could not be
run: it had no ``requirements.txt`` to install from, no ``.env.example`` to fill in, no
tests to prove it worked and no README to explain any of it. A project that states its own
dependencies, ships its own offline tests and documents its own entry point is the
difference between "here is some code" and "here is something you can run".

The agent *configuration* still comes from a model (via :class:`AgentGenerator`), because
deciding which agents and tools a requirement implies is genuinely a language task. But
rendering that configuration into framework code is deterministic - the framework
generators are pure string builders - so this stage is reproducible given the same config.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..dependencies import requirements_txt
from ..errors import AppError, GenerationError
from ..frameworks import FRAMEWORKS, generate_code
from ..generator import AgentGenerator
from ..llm.base import LLMProvider
from ..providers import get_provider, resolve_runtime_model
from .models import FrameworkChoice, GeneratedProject, RequirementAnalysis

__all__ = ["generate_project", "build_agent_config"]


#: The single source file the framework generators emit is written here. Every generated
#: project uses the same entry-point name so the runner, the tests and the README never
#: have to agree on it separately.
ENTRYPOINT = "agent.py"


def build_agent_config(
    analysis: RequirementAnalysis,
    framework: str,
    provider: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask the configured model for the agent/task configuration.

    This is the one place a model shapes the *content* of the project. It runs through the
    existing :class:`AgentGenerator`, which already knows how to prompt per framework and
    how to fall back to a sane default when the model returns something unusable - so a
    small open-weight model that fumbles the JSON degrades to a working default rather than
    failing the run.

    The requirement passed to the model is the analysis summary when we have one, because
    the summary is the cleaned-up statement of intent; the raw requirement is the fallback.
    """
    generator = AgentGenerator(provider=provider, model=model)
    prompt = analysis.summary or analysis.requirement
    return generator.analyze_prompt(prompt, framework)


def generate_project(
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    provider: str,
    model: Optional[str] = None,
    *,
    config: Optional[Dict[str, Any]] = None,
    include_tests: bool = True,
    llm_provider: Optional[LLMProvider] = None,  # accepted for symmetry; unused here
) -> GeneratedProject:
    """
    Build a complete project for ``choice.framework``.

    Args:
        analysis: The structural reading of the requirement.
        choice: The selected framework.
        provider: LLM provider the *generated code* will call at run time.
        model: Optional explicit model id for the generated code.
        config: A pre-built agent configuration. When None, one is generated from a model.
            Passing it in is what lets the improvement stage re-render without re-prompting.
        include_tests: Whether to bundle an offline test suite. On by default because a
            project without tests cannot be verified, and verification is the point.

    Raises:
        GenerationError: if the framework is unknown or code rendering fails. This one does
            propagate - unlike analysis and selection, there is no non-model answer to fall
            back to, so a failure here is a real dead end the caller must see.
    """
    framework = choice.framework
    if framework not in FRAMEWORKS:
        raise GenerationError(
            f"Cannot generate code for unknown framework {framework!r}.",
            action=f"Choose one of: {', '.join(FRAMEWORKS)}.",
            context={"framework": framework},
        )

    spec = get_provider(provider)
    runtime_model = resolve_runtime_model(spec.name, model)

    if config is None:
        config = build_agent_config(analysis, framework, spec.name, model)

    try:
        source = generate_code(config, framework, provider=spec.name, model=model)
    except AppError:
        raise
    except Exception as exc:  # noqa: BLE001 - framework generators raise plain ValueErrors
        raise GenerationError(
            f"Failed to render {framework} code.",
            detail=str(exc),
            context={"framework": framework, "provider": spec.name},
        ) from exc

    project = GeneratedProject(
        framework=framework,
        provider=spec.name,
        model=runtime_model,
        config=config,
    )
    project.add_file(
        ENTRYPOINT,
        source,
        is_entrypoint=True,
        description=f"{framework} multi-agent system, ready to run.",
    )

    dependencies = requirements_txt(framework, spec.name, include_tests=include_tests)
    project.add_file(
        "requirements.txt",
        dependencies,
        description="Everything this project needs, pinned loosely for pip.",
    )
    project.dependencies = _requirement_lines(dependencies)

    project.add_file(
        ".env.example",
        _env_example(spec),
        description="Copy to .env and fill in your credential. Never commit the .env.",
    )

    if include_tests:
        _attach_tests(project, config, framework, spec.name)

    project.add_file(
        "README.md",
        _readme(analysis, choice, project),
        description="What this project is, how to run it, and how it was chosen.",
    )

    project.run_instructions = _run_instructions(spec, project)
    project.notes = _notes(analysis, choice, spec)
    return project


# ----------------------------------------------------------------------------- helpers
def _attach_tests(
    project: GeneratedProject,
    config: Dict[str, Any],
    framework: str,
    provider: str,
) -> None:
    """
    Add the offline test bundle under ``tests/``.

    The bundle is generated by the existing :class:`TestGenerator`, whose whole design goal
    was that the tests run with no network and no credentials - a ``FakeChatModel`` stands
    in for the provider. Nesting them under ``tests/`` (rather than the project root) keeps
    the generated agent and its tests visually separate in the Files tab, and matches what
    pytest discovers by default.

    Test generation must never sink the whole project: a project with agent code but no
    tests is degraded, not dead. A failure here is recorded as a note and the tests are
    skipped, so the user still gets runnable code.
    """
    # Imported lazily: the evaluation package pulls in heavier modules, and generation
    # should not pay that import cost when tests are disabled.
    from ..evaluation.test_generator import TestGenerator

    try:
        bundle = TestGenerator().generate_bundle(config, framework, provider)
    except Exception as exc:  # noqa: BLE001 - degrade, don't fail
        project.notes.append(
            f"Test suite could not be generated ({exc}). The agent code is complete; "
            "add tests by hand or re-run generation."
        )
        return

    for filename, content in bundle.files.items():
        # requirements-test.txt is folded into the top-level requirements.txt already, and
        # a second requirements file under tests/ only invites drift.
        if filename == "requirements-test.txt":
            continue
        project.add_file(
            f"tests/{filename}",
            content,
            description=_test_file_description(filename),
        )


def _test_file_description(filename: str) -> str:
    return {
        "conftest.py": "Pytest fixtures that keep the suite offline (no network, no keys).",
        "_helpers.py": "A zero-dependency fake chat model the tests run against.",
        "pytest.ini": "Registers the unit/integration/live markers the suite uses.",
        "README.md": "How to run this project's tests.",
    }.get(filename, "Generated test module.")


def _requirement_lines(requirements_text: str) -> list:
    """Pull the actual requirement specifiers out of a rendered requirements.txt."""
    lines = []
    for raw in requirements_text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _env_example(spec) -> str:
    """
    A ``.env.example`` naming the credential this provider needs and nothing else.

    Only the *names* of the variables appear - never a value. This file is meant to be
    committed, so writing a real key into it would be exactly the secret leak the brief
    forbids. The user copies it to ``.env`` and fills in their own key locally.
    """
    lines = [
        "# Copy this file to .env and fill in your credential.",
        "# .env is git-ignored; never commit real keys.",
        "",
    ]
    if spec.credential_env:
        primary = spec.credential_env[0]
        lines.append(f"# Credential for {spec.label}:")
        lines.append(f"{primary}=")
        if spec.notes:
            lines.append(f"# {spec.notes}")
    elif spec.local:
        lines.append(f"# {spec.label} runs locally and needs no API key.")
        if spec.notes:
            lines.append(f"# {spec.notes}")
    else:
        lines.append("# This provider needs no credential.")
    lines.append("")
    return "\n".join(lines)


def _run_instructions(spec, project: GeneratedProject) -> str:
    entry = project.entrypoint or ENTRYPOINT
    steps = [
        "pip install -r requirements.txt",
    ]
    if spec.credential_env:
        steps.append("cp .env.example .env    # then edit .env and add your key")
    elif spec.local and "ollama" in spec.name:
        steps.append("ollama serve            # start the local model server")
    steps.append(f"python {entry}")
    return "\n".join(steps)


def _notes(analysis: RequirementAnalysis, choice: FrameworkChoice, spec) -> list:
    notes = []
    if not analysis.from_model:
        notes.append(
            "The requirement was analysed with the offline heuristic, not a model. "
            "Configure an LLM provider for a sharper analysis."
        )
    if choice.confidence < 0.5 and not choice.user_specified:
        notes.append(
            f"Framework confidence is low ({choice.confidence:.2f}); the requirement did "
            "not clearly favour one framework. Review the choice on the Overview tab."
        )
    if spec.local:
        notes.append(f"{spec.label} runs on your machine; no data leaves it.")
    return notes


def _readme(
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    project: GeneratedProject,
) -> str:
    """A README that explains what the project is and why this framework was chosen."""
    lines = [
        f"# {choice.framework} multi-agent system",
        "",
        "Generated by **multi-agent-generator** from the requirement:",
        "",
        f"> {analysis.requirement}",
        "",
        "## Why this framework",
        "",
        choice.reason,
        "",
        f"Selection confidence: **{choice.confidence:.0%}**"
        + ("  (you chose this explicitly)" if choice.user_specified else ""),
        "",
    ]

    if choice.alternatives:
        lines.append("Alternatives considered:")
        lines.append("")
        for alt in choice.alternatives:
            name = alt.get("framework")
            if not name:
                continue
            note = alt.get("note", "")
            lines.append(f"- **{name}** - {note}")
        lines.append("")

    lines += [
        "## Run it",
        "",
        "```bash",
        project.run_instructions or "pip install -r requirements.txt\npython agent.py",
        "```",
        "",
        "## Test it",
        "",
        "The tests are offline: they run against a fake model, so they need no API key "
        "and make no network calls.",
        "",
        "```bash",
        "pip install -r requirements.txt",
        "pytest tests/ -m 'not live'",
        "```",
        "",
        "## Files",
        "",
    ]
    for entry in project.tree():
        marker = "  *(entry point)*" if entry["is_entrypoint"] else ""
        lines.append(f"- `{entry['path']}` - {entry['lines']} lines{marker}")
    lines.append("")

    if project.notes:
        lines.append("## Notes")
        lines.append("")
        for note in project.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
