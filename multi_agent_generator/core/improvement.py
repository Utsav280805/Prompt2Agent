# multi_agent_generator/core/improvement.py
"""
Stage 5: act on the review.

The loop this module serves is the difference between a generator and a system that gets
something right: generate, review, fix, review again, up to ``MAX_GENERATION_ITERATIONS``.

Improvement works on the *configuration*, not the rendered text, wherever it can. The
framework generators are deterministic string builders, so a config change re-renders into
correct, idiomatic framework code, whereas asking a model to patch generated Python invites
it to invent a different structure and break the parts that were already right. Direct
source repair exists only for the one class of problem a config cannot express - a file that
does not parse - and it is tightly guarded.

Two rules keep the loop from making things worse, which is the failure mode of every naive
retry loop:

1. A revision is only accepted if it still renders. The render is a real test, not a
   proxy - if the revised config cannot produce code, it is rejected outright.
2. A revision is only accepted if it does not add blocking issues. Static review runs on
   the candidate before it is returned, and a candidate that regresses is discarded and
   reported as such.

Nothing here fabricates progress. If the model cannot improve the project, the outcome says
so and the pipeline stops iterating instead of burning three rounds pretending.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..errors import AppError
from ..llm.base import LLMProvider
from .models import (
    FrameworkChoice,
    GeneratedProject,
    RequirementAnalysis,
    ReviewResult,
)
from .parsing import extract_json_object, strip_code_fences
from .review import static_review

if TYPE_CHECKING:  # pragma: no cover - annotation only, see the lazy imports below
    from .manifest import ProjectManifest

__all__ = ["improve_project", "ImprovementOutcome", "IMPROVEMENT_SYSTEM_PROMPT"]


@dataclass
class ImprovementOutcome:
    """
    What one improvement attempt achieved.

    ``project`` is None when nothing could be safely applied. That is deliberately distinct
    from "an unchanged project was returned": the pipeline needs to know that another
    iteration would be pointless so it can stop early rather than spend the remaining
    budget re-running an identical round.
    """

    project: Optional[GeneratedProject] = None
    changes_applied: List[str] = field(default_factory=list)
    #: Why nothing was applied, or what was skipped. Surfaced to the user, so it is written
    #: for a person rather than a log parser.
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.project is not None and bool(self.changes_applied)


IMPROVEMENT_SYSTEM_PROMPT = """\
You revise the JSON configuration of a multi-agent system so that a reviewer's required
changes are addressed.

Reply with ONLY the complete revised JSON configuration object - no prose, no code fence,
no commentary. It must have the same top-level keys as the configuration you were given.

Rules:
- Keep everything that was not criticised exactly as it was. You are making targeted edits,
  not rewriting from scratch.
- Address every required change you can express in the configuration.
- Do not remove a required top-level key, even if you have nothing to put in it.
- Do not add keys the original did not have; the code generator will ignore them.
- Do not put API keys, tokens or any credential value anywhere in the configuration.
"""


REPAIR_SYSTEM_PROMPT = """\
You fix a single Python file that fails to parse.

Reply with ONLY the corrected, complete contents of the file - no prose, no explanation, no
markdown fence. Preserve the existing structure, imports, names and behaviour; change only
what is needed to make the file valid Python. Do not add credentials or hardcoded keys.
"""


def improve_project(
    project: GeneratedProject,
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    review: ReviewResult,
    provider: Optional[LLMProvider] = None,
    *,
    runtime_provider: Optional[str] = None,
    model: Optional[str] = None,
    include_tests: bool = True,
    manifest: Optional["ProjectManifest"] = None,
) -> ImprovementOutcome:
    """
    Try to fix what ``review`` complained about.

    Args:
        project: The project as generated.
        analysis: The requirement analysis, for context in the prompt.
        choice: The framework choice; the framework cannot change during improvement.
        review: The verdict to act on. Its ``required_changes`` drive the work.
        provider: The LLM doing the revising. Without one, only the deterministic
            repairs are attempted.
        runtime_provider: Provider name for the regenerated code. Defaults to the
            project's own, so improvement never silently switches providers.
        model: Explicit model id for the regenerated code.
        include_tests: Whether the re-rendered project keeps its test bundle.
        manifest: The plan this project was built from, when it has one. Passing it is what
            keeps a revision multi-file: without it the re-render falls back to the
            single-file generator, and a project that arrived as a package would come back
            from its first improvement as one long module.

    Returns:
        An :class:`ImprovementOutcome`. Never raises for model-side problems - a failed
        improvement attempt is a normal, expected outcome that the loop must survive.
    """
    if not review.required_changes and not review.blocking_issues:
        return ImprovementOutcome(note="The review raised nothing that needs fixing.")

    if provider is None:
        return ImprovementOutcome(
            note=(
                "No LLM is configured, so the required changes could not be applied. "
                "Configure a provider to enable the improvement loop."
            )
        )

    candidate = _revise_configuration(
        project,
        analysis,
        choice,
        review,
        provider,
        runtime_provider=runtime_provider or project.provider,
        model=model,
        include_tests=include_tests,
        manifest=manifest,
    )

    # Whatever we have now - revised or original - a file that does not parse is worth one
    # targeted repair attempt. Config revision cannot fix a syntax error in rendered output.
    target = candidate.project or project
    repaired = _repair_broken_files(target, provider)

    changes = list(candidate.changes_applied) + repaired.changes_applied
    notes = " ".join(part for part in (candidate.note, repaired.note) if part).strip()

    if not changes:
        return ImprovementOutcome(note=notes or "No change could be applied safely.")

    return ImprovementOutcome(
        project=repaired.project or candidate.project,
        changes_applied=changes,
        note=notes,
    )


# ------------------------------------------------------------------ configuration revision
def _revise_configuration(
    project: GeneratedProject,
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    review: ReviewResult,
    provider: LLMProvider,
    *,
    runtime_provider: str,
    model: Optional[str],
    include_tests: bool,
    manifest: Optional["ProjectManifest"] = None,
) -> ImprovementOutcome:
    """Ask the model for a revised config, then prove the revision is not a regression."""
    if not project.config:
        return ImprovementOutcome(
            note="The project has no configuration to revise."
        )

    try:
        raw = provider.complete_text(
            _revision_prompt(project, analysis, choice, review),
            system=IMPROVEMENT_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=3000,
        )
    except AppError as exc:
        return ImprovementOutcome(
            note=f"The improvement step could not reach the model: {exc.message}"
        )

    revised = extract_json_object(raw)
    if not revised:
        return ImprovementOutcome(
            note="The model did not return a usable configuration, so nothing was changed."
        )

    missing = [key for key in project.config if key not in revised]
    if missing:
        # A revision that dropped `agents` or `tasks` would render into code missing whole
        # sections. Restoring the originals is safe and keeps a partially-good revision
        # usable instead of throwing the whole thing away.
        for key in missing:
            revised[key] = project.config[key]

    if revised == project.config:
        return ImprovementOutcome(
            note="The model returned the configuration unchanged; no fix was applied."
        )

    if _has_credential_literal(revised):
        # Refusing outright rather than stripping: a config that came back containing a
        # key-shaped string is not a config to trust the rest of.
        return ImprovementOutcome(
            note=(
                "The revised configuration contained something that looks like a "
                "credential, so it was rejected."
            )
        )

    # Lazy imports: generation, architecture and emit all import this module's siblings, and
    # importing them at module scope would make core.improvement and core.generation import
    # each other.
    try:
        candidate = _render(
            revised,
            analysis,
            choice,
            runtime_provider,
            model,
            include_tests=include_tests,
            manifest=manifest,
        )
    except AppError as exc:
        return ImprovementOutcome(
            note=f"The revised configuration could not be rendered ({exc.message}); kept the original."
        )

    before = static_review(project, analysis)
    after = static_review(candidate, analysis)
    if len(after.blocking_issues) > len(before.blocking_issues):
        return ImprovementOutcome(
            note=(
                "The revised configuration introduced new problems "
                f"({len(after.blocking_issues)} vs {len(before.blocking_issues)}), "
                "so the original was kept."
            )
        )

    return ImprovementOutcome(
        project=candidate,
        changes_applied=_describe_config_changes(project.config, revised),
    )


def _render(
    config: Dict[str, Any],
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    runtime_provider: str,
    model: Optional[str],
    *,
    include_tests: bool,
    manifest: Optional["ProjectManifest"],
) -> GeneratedProject:
    """
    Turn a revised configuration back into a project, by the same route it came in by.

    A project that was planned and assembled is re-planned and re-assembled; one that came
    from the single-file generator goes back through it. Routing both through the multi-file
    path regardless would be worse than it sounds - the caller compares the candidate against
    the original with :func:`static_review`, and a layout change would swamp that comparison
    with differences that have nothing to do with the review's findings.
    """
    if manifest is not None:
        from .architecture import plan_architecture

        from ..emit import assemble_project

        replanned = plan_architecture(
            analysis,
            choice,
            runtime_provider,
            model,
            config=config,
            include_tests=include_tests,
            project_name=manifest.project_name,
        )
        return assemble_project(replanned)

    from .generation import generate_project

    return generate_project(
        analysis,
        choice,
        runtime_provider,
        model,
        config=config,
        include_tests=include_tests,
    )


def _revision_prompt(
    project: GeneratedProject,
    analysis: RequirementAnalysis,
    choice: FrameworkChoice,
    review: ReviewResult,
) -> str:
    """
    Give the model the requirement, the config, and precisely what to fix.

    The generated *code* is deliberately left out. The model is editing configuration, and
    showing it the rendered output invites it to reply with code instead - which is the
    single most common way this step fails.
    """
    lines = [
        f"Original requirement: {analysis.requirement}",
        f"Framework: {choice.framework}",
        "",
        "Current configuration:",
        "```json",
        json.dumps(project.config, indent=2, ensure_ascii=False),
        "```",
        "",
        f"The reviewer scored this {review.score:.1f}/10.",
    ]
    if review.summary:
        lines.append(f"Reviewer summary: {review.summary}")
    lines.append("")

    if review.required_changes:
        lines.append("Required changes:")
        for i, change in enumerate(review.required_changes, 1):
            lines.append(f"{i}. {change}")
        lines.append("")

    blocking = review.blocking_issues
    if blocking:
        lines.append("Issues behind those changes:")
        for issue in blocking[:10]:
            where = f" [{issue.file}]" if issue.file else ""
            lines.append(f"- ({issue.severity.value}){where} {issue.message}")
        lines.append("")

    lines.append(
        "Return the complete revised configuration as JSON, keeping every top-level key."
    )
    return "\n".join(lines)


def _describe_config_changes(before: Dict[str, Any], after: Dict[str, Any]) -> List[str]:
    """
    Describe what actually changed, key by key.

    The reviewer's ``required_changes`` are what was *asked for*; this is what was *done*.
    Reporting the request as though it were the outcome is how a loop ends up claiming
    fixes it never made, so the two are kept separate all the way to the UI.
    """
    changes: List[str] = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if key not in before:
            changes.append(f"Added configuration key '{key}'.")
        elif key not in after:
            changes.append(f"Removed configuration key '{key}'.")
        elif isinstance(old, list) and isinstance(new, list) and len(old) != len(new):
            changes.append(f"Changed '{key}' from {len(old)} to {len(new)} entries.")
        else:
            changes.append(f"Revised '{key}'.")
    return changes or ["Revised the agent configuration."]


def _has_credential_literal(config: Any) -> bool:
    """Look for a key-shaped string anywhere in a nested config."""
    if isinstance(config, dict):
        return any(_has_credential_literal(v) for v in config.values())
    if isinstance(config, list):
        return any(_has_credential_literal(v) for v in config)
    if isinstance(config, str):
        text = config.strip()
        return text.startswith(("sk-", "hf_", "gsk_")) and len(text) > 20
    return False


# ------------------------------------------------------------------------- source repair
def _repair_broken_files(
    project: GeneratedProject,
    provider: LLMProvider,
) -> ImprovementOutcome:
    """
    Attempt to fix Python files that do not parse.

    The guard is absolute: a repaired file replaces the original only if it now parses.
    A model that returns something equally broken, or prose, changes nothing. Since the
    accept condition is checked with :func:`ast.parse` rather than taken on trust, this
    step cannot make the project less valid than it already was.
    """
    broken = []
    for file in project.source_files():
        try:
            ast.parse(file.content, filename=file.path)
        except SyntaxError as exc:
            broken.append((file, exc))

    if not broken:
        return ImprovementOutcome()

    changes: List[str] = []
    failures: List[str] = []
    for file, error in broken[:3]:  # a project with more than three broken files is a bug
        fixed = _repair_one(file.path, file.content, error, provider)
        if fixed is None:
            failures.append(file.path)
            continue
        project.add_file(
            file.path,
            fixed,
            is_entrypoint=file.is_entrypoint,
            description=file.description,
        )
        changes.append(f"Fixed a syntax error in {file.path}.")

    note = ""
    if failures:
        note = (
            f"Could not repair {', '.join(failures)}; the file still does not parse. "
            "This is a bug in code generation rather than in your request."
        )
    return ImprovementOutcome(
        project=project if changes else None,
        changes_applied=changes,
        note=note,
    )


def _repair_one(
    path: str,
    content: str,
    error: SyntaxError,
    provider: LLMProvider,
) -> Optional[str]:
    """Return repaired source that parses, or None."""
    prompt = "\n".join(
        [
            f"File: {path}",
            f"Python reports: {error.msg} at line {error.lineno}.",
            "",
            "```python",
            content,
            "```",
            "",
            "Return the corrected file contents.",
        ]
    )
    try:
        raw = provider.complete_text(
            prompt, system=REPAIR_SYSTEM_PROMPT, temperature=0.0, max_tokens=4000
        )
    except AppError:
        return None

    candidate = strip_code_fences(raw).strip()
    if not candidate:
        return None
    try:
        ast.parse(candidate, filename=path)
    except SyntaxError:
        return None
    # A "repair" that threw most of the file away is not a repair.
    if len(candidate) < len(content) * 0.5:
        return None
    return candidate + ("\n" if not candidate.endswith("\n") else "")
