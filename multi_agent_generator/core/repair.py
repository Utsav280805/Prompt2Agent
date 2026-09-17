# multi_agent_generator/core/repair.py
"""
Stage 9: fix what validation found, or say plainly that it could not be fixed.

This is the module that separates "the system generated code" from "the system delivered a
working agent". Validation produces a list of blocking findings; this module tries to turn
each of them back into a passing check, re-validates to prove it worked, and reports every
attempt - including the ones that achieved nothing.

Three rules shape the whole design, and each one exists because the obvious alternative is
worse:

**Repair means rewriting the code, never relaxing the check.** There is no path through this
module that edits a threshold, deletes a test, widens a pattern or downgrades a severity. A
finding is cleared by making it untrue. Section 18 spells this out for test failures and the
same logic holds for every other gate: a check that can be argued away is not a check.

**The plan is the repair.** For any file the architecture plan covers, the fix is to re-emit
that file from the manifest. The emitters are deterministic string builders, so re-emission
produces the file the architecture intended - which is a genuine fix for a truncated module, a
file that never got written, a syntax error introduced downstream, or a literal that should
never have been in generated source. Asking a model to patch broken generated Python is the
last resort, not the first, because a model handed a broken file will happily return a
differently-structured one that breaks two other files' imports.

**Bounded, and honest about the boundary.** ``MAX_REPAIR_ITERATIONS`` is a hard stop, and a
round that applies nothing - or that applies something without reducing the number of blocking
findings - ends the loop immediately. Three identical rounds is not persistence. When the loop
stops with findings outstanding, :class:`RepairOutcome` carries them with a root cause per
finding, because section 77 is explicit that the system must not claim it can fix everything.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..llm.base import LLMProvider
from .manifest import ProjectManifest
from .models import GeneratedProject
from .validation import Check, ValidationReport, static_checks

__all__ = [
    "MAX_REPAIR_ITERATIONS",
    "RepairAction",
    "RepairOutcome",
    "repair_project",
    "root_cause_of",
]


#: The ceiling section 19 names. A hard stop, not a hint: :func:`repair_project` will not run a
#: fourth round even if the third one was making progress, because an unbounded self-healing
#: loop is indistinguishable from a hang to the person waiting on it.
MAX_REPAIR_ITERATIONS = 3

#: Why each check tends to fail, in one sentence a non-technical user can read.
#:
#: Keyed by check id. This is the *root cause*, which is not the same thing as the check's
#: ``summary`` (what was observed) or its ``fix`` (what to do). Section 19 asks for the cause,
#: and stating it separately is what stops the repair log from being a second copy of the
#: validation report.
_ROOT_CAUSES: Dict[str, str] = {
    "configuration.secrets": (
        "A credential-shaped string was written into the project's own source instead of "
        "being read from the environment."
    ),
    "configuration.settings_module": (
        "No module defines get_settings(), so nothing in the project has a single place to "
        "read its configuration from."
    ),
    "imports.internal": (
        "A file imports another file in this project that was never written, so the project "
        "cannot be imported at all."
    ),
    "imports.third_party": (
        "The code imports a package that requirements.txt does not declare, so a clean "
        "install produces a project that cannot start."
    ),
    "imports.syntax": "A generated file is not valid Python, so nothing that imports it can load.",
    "structure.parses": "A generated file is not valid Python, so nothing that imports it can load.",
    "entrypoint.declared": "The project has no file the runner knows how to start.",
    "entrypoint.runnable": (
        "The entry point does not expose one of the function names the runner calls, so the "
        "project cannot be launched even though the code is present."
    ),
    "entrypoint.result_contract": (
        "The entry point prints its answer instead of returning it, so the Playground has no "
        "value to display."
    ),
    "agents.emitted": "An agent named in the plan has no factory function in the code.",
    "tools.emitted": "A tool named in the plan has no implementation in the code.",
    "workflow.emitted": "The plan describes a workflow that the code does not build.",
    "tests.present": "The project has no test suite, so nothing about it can be verified.",
}

#: File extensions the re-emit strategy will accept a replacement for.
#:
#: Restricted deliberately. Re-emitting ``requirements.txt`` or ``.env.example`` from the plan
#: is safe and often the actual fix for a dependency finding; re-emitting a README over a
#: hand-edited one would be destructive for no gain, so documentation is left alone.
_REEMITTABLE = (".py", ".txt", ".example", ".cfg", ".toml", ".json", ".yml", ".yaml")

#: Check categories whose failure can be *caused by a file simply not being there*.
#:
#: This list exists because of a mismatch that would otherwise make the re-emit strategy
#: useless for the most common breakage. When ``app/main.py`` imports ``app/agents/writer.py``
#: and that file was never written, the dependency graph attaches the problem to the *importing*
#: file - so ``imports.internal`` reports ``app/main.py`` in its ``paths``, and rewriting
#: ``app/main.py`` from the plan changes nothing at all. The file that needs writing is the one
#: nobody named. For findings in these categories the round therefore also offers every planned
#: file the project does not currently contain, which is precisely the fix.
_ABSENCE_SENSITIVE = (
    "imports",
    "structure",
    "entrypoint",
    "agents",
    "tools",
    "workflow",
    "tests",
)


@dataclass(frozen=True)
class RepairAction:
    """
    One attempt at one finding.

    ``applied=False`` entries are kept and reported rather than dropped. A repair log that
    only lists successes reads as "everything was fixed", which is the fake-green outcome the
    brief opens by rejecting - and the unfixed findings are precisely the ones the user needs
    to see.
    """

    check_id: str
    #: Why the check failed, in plain language. See :data:`_ROOT_CAUSES`.
    root_cause: str
    #: What this repair did, or would have done. Written for a person.
    action: str
    applied: bool
    paths: Tuple[str, ...] = ()
    #: The strategy that ran: ``"replan"``, ``"model"`` or ``"none"``.
    strategy: str = "none"
    #: Why it was not applied, when it was not. Empty on success.
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "root_cause": self.root_cause,
            "action": self.action,
            "applied": self.applied,
            "paths": list(self.paths),
            "strategy": self.strategy,
            "note": self.note,
        }


@dataclass
class RepairOutcome:
    """
    Everything the repair loop did, and what it left behind.

    ``project`` is None when nothing was applied, which the pipeline reads as "another round
    would be pointless". That is deliberately distinct from returning an unchanged project:
    the caller has to be able to tell "I tried and could not" from "I tried and it is now
    fine", and a project object alone cannot express the difference.
    """

    project: Optional[GeneratedProject] = None
    actions: List[RepairAction] = field(default_factory=list)
    #: How many rounds ran. Never above :data:`MAX_REPAIR_ITERATIONS`.
    iterations: int = 0
    #: Blocking check ids still failing when the loop stopped.
    unresolved: List[str] = field(default_factory=list)
    #: Why the loop stopped, for the event trace. Written for a person.
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.project is not None and any(a.applied for a in self.actions)

    @property
    def applied(self) -> List[RepairAction]:
        return [a for a in self.actions if a.applied]

    def changes_applied(self) -> List[str]:
        """The applied actions as plain sentences, for :class:`IterationRecord`."""
        return [a.action for a in self.applied]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "iterations": self.iterations,
            "applied": len(self.applied),
            "attempted": len(self.actions),
            "unresolved": list(self.unresolved),
            "actions": [a.as_dict() for a in self.actions],
            "note": self.note,
        }


def root_cause_of(check: Check) -> str:
    """
    The root cause for one finding.

    Falls back to the check's own summary rather than to a generic sentence: an unmapped check
    still deserves a real explanation, and "an unknown problem occurred" is exactly the kind of
    text section 85 rules out.
    """
    mapped = _ROOT_CAUSES.get(check.id)
    if mapped:
        return mapped
    prefix = check.id.split(".")[0]
    for key, value in _ROOT_CAUSES.items():
        if key.split(".")[0] == prefix and check.category == prefix:
            return value
    return check.summary


def repair_project(
    project: GeneratedProject,
    report: ValidationReport,
    *,
    manifest: Optional[ProjectManifest] = None,
    provider: Optional[LLMProvider] = None,
    secrets: Sequence[str] = (),
    expect_tests: bool = True,
    max_iterations: Optional[int] = None,
) -> RepairOutcome:
    """
    Try to clear ``report``'s blocking findings by rewriting the project.

    Args:
        project: The project to repair. Never mutated - a deep copy is worked on, so a repair
            that turns out to be a regression can be discarded without the caller having lost
            the version that at least got this far.
        report: The validation report to act on. Only its blocking findings are addressed;
            warnings are left for the improvement stage, which can afford to be speculative.
        manifest: The architecture plan. Without it the re-emit strategy is unavailable and
            most findings can only be reported, which is why the pipeline keeps the plan
            alongside the project rather than discarding it after generation.
        provider: An LLM, for the one strategy that needs it. Optional: every other strategy
            here is deterministic, which is what makes the repair loop testable offline.
        secrets: Platform-held credential values, forwarded to re-validation so the leak check
            keeps proving none of them reached the source.
        expect_tests: Forwarded to re-validation.
        max_iterations: Override the ceiling downward for a caller that wants a single round.
            Values above :data:`MAX_REPAIR_ITERATIONS` are clamped to it.

    Returns:
        A :class:`RepairOutcome`. Does not raise: a repair engine that throws converts a
        diagnosable problem into an opaque one, and this runs on model-authored source.
    """
    budget = MAX_REPAIR_ITERATIONS if max_iterations is None else max_iterations
    budget = max(0, min(int(budget), MAX_REPAIR_ITERATIONS))
    if budget == 0:
        return RepairOutcome(note="Repair was disabled for this run.")

    blocking = report.blocking()
    if not blocking:
        return RepairOutcome(note="Nothing was blocking, so there was nothing to repair.")

    working = copy.deepcopy(project)
    actions: List[RepairAction] = []
    outstanding = list(blocking)
    rounds = 0
    stop_note = ""

    while rounds < budget and outstanding:
        rounds += 1
        before = len(outstanding)
        round_actions = _one_round(working, outstanding, manifest, provider)
        actions.extend(round_actions)

        if not any(a.applied for a in round_actions):
            stop_note = (
                "Repair stopped because this round could not change anything. "
                "The remaining findings need a person."
            )
            break

        # Re-checked rather than assumed. A strategy that "succeeded" without clearing the
        # finding it targeted has not repaired anything, and the only way to know which of
        # those two happened is to run the gates again.
        checks = static_checks(
            working,
            manifest,
            secrets=secrets,
            expect_tests=expect_tests,
        )
        outstanding = [c for c in checks if c.blocking]

        if len(outstanding) >= before:
            stop_note = (
                f"Repair stopped after {rounds} round(s): the changes it made did not reduce "
                "the number of blocking problems."
            )
            break
    else:
        if outstanding:
            stop_note = (
                f"Repair stopped after {rounds} round(s), the maximum. "
                f"{len(outstanding)} problem(s) still need attention."
            )

    applied_any = any(a.applied for a in actions)
    if not stop_note:
        stop_note = (
            "Every blocking problem was repaired."
            if not outstanding
            else f"{len(outstanding)} problem(s) still need attention."
        )

    return RepairOutcome(
        project=working if applied_any else None,
        actions=actions,
        iterations=rounds,
        unresolved=[c.id for c in outstanding],
        note=stop_note,
    )


# ------------------------------------------------------------------------------- one round
def _one_round(
    project: GeneratedProject,
    blocking: Sequence[Check],
    manifest: Optional[ProjectManifest],
    provider: Optional[LLMProvider],
) -> List[RepairAction]:
    """
    Attempt every blocking finding once, cheapest strategy first.

    The plan is consulted once per round, not once per finding: re-emitting is a whole-project
    operation, so doing it per finding would run the emitters five times to fix five files in
    the same module.
    """
    planned = _planned_files(manifest)
    # Computed once, before anything is written: a file the plan requires and the project does
    # not contain is the single most repairable defect there is, and no check names it directly.
    absent = [path for path in planned if project.get(path) is None]
    actions: List[RepairAction] = []
    touched: set = set()

    for check in blocking:
        cause = root_cause_of(check)
        targets = [p for p in check.paths if p]
        offered = list(absent) + targets if check.category in _ABSENCE_SENSITIVE else targets

        # --- the plan covers these files: re-emit them --------------------------------
        replanned = [
            p
            for i, p in enumerate(offered)
            if p in planned and p not in touched and p not in offered[:i]
        ]
        if replanned:
            written, failure = _restore_from_plan(project, planned, replanned)
            # Every path attempted is marked, not just the ones written. A file the plan already
            # matches will be refused for the same reason on every subsequent finding in this
            # round, and re-offering it would fill the log with the same refusal five times.
            touched.update(replanned)
            actions.append(
                RepairAction(
                    check_id=check.id,
                    root_cause=cause,
                    action=(
                        f"Rewrote {', '.join(written)} from the architecture plan."
                        if written
                        else f"Could not rewrite {', '.join(replanned)} from the plan."
                    ),
                    applied=bool(written),
                    paths=tuple(written or replanned),
                    strategy="replan",
                    note="" if written else failure,
                )
            )
            continue

        # --- a file the plan does not cover, and it does not parse ---------------------
        broken = [p for p in targets if _does_not_parse(project, p)]
        if broken and provider is not None:
            fixed = _repair_with_model(project, broken[0], provider)
            actions.append(
                RepairAction(
                    check_id=check.id,
                    root_cause=cause,
                    action=(
                        f"Rewrote {broken[0]} so that it parses."
                        if fixed
                        else f"Could not produce a version of {broken[0]} that parses."
                    ),
                    applied=fixed,
                    paths=(broken[0],),
                    strategy="model",
                    note=(
                        ""
                        if fixed
                        else "The model did not return valid Python for this file."
                    ),
                )
            )
            if fixed:
                touched.add(broken[0])
            continue

        # --- nothing safe to do: say so, and say what would fix it --------------------
        actions.append(
            RepairAction(
                check_id=check.id,
                root_cause=cause,
                action=check.fix or "This needs a change the repair engine cannot make safely.",
                applied=False,
                paths=tuple(targets),
                strategy="none",
                note=_why_not(check, manifest, provider),
            )
        )

    return actions


def _why_not(
    check: Check,
    manifest: Optional[ProjectManifest],
    provider: Optional[LLMProvider],
) -> str:
    """Name the missing capability, not just the failure. Vague is useless here."""
    if manifest is None:
        return (
            "There is no architecture plan stored for this project, so the file could not be "
            "regenerated from it."
        )
    if not check.paths:
        return (
            "This finding is about the project as a whole rather than one file, so there is "
            "no single file to rewrite."
        )
    if provider is None:
        return (
            "The plan does not cover this file and no model is configured, so there was no "
            "safe way to rewrite it."
        )
    return "The plan does not cover this file, and it parses, so rewriting it would be a guess."


# ------------------------------------------------------------------------ replan strategy
def _planned_files(manifest: Optional[ProjectManifest]) -> Dict[str, Tuple[str, str, bool]]:
    """
    Re-emit the plan and index the result by path.

    Returns ``{path: (content, description, is_entrypoint)}``, empty when there is no plan or
    when assembly fails. Assembly failure is swallowed *here specifically*: it means the plan
    itself is unbuildable, which is a generation bug the validator will already be reporting,
    and it must not become an exception raised from the repair engine.
    """
    if manifest is None:
        return {}
    # Imported here rather than at module scope: :mod:`multi_agent_generator.emit` imports
    # ``core.manifest`` and ``core.models``, so a top-level import would have this module and
    # that package referencing each other while ``core`` is still initialising.
    try:
        from ..emit import assemble_project
    except Exception:  # noqa: BLE001 - optional at import time, reportable at use time
        return {}

    try:
        rebuilt = assemble_project(manifest)
    except Exception:  # noqa: BLE001 - see docstring
        return {}

    return {
        f.path: (f.content, f.description, f.is_entrypoint)
        for f in rebuilt.files
        if f.path.endswith(_REEMITTABLE)
    }


def _restore_from_plan(
    project: GeneratedProject,
    planned: Dict[str, Tuple[str, str, bool]],
    paths: Sequence[str],
) -> Tuple[List[str], str]:
    """
    Replace ``paths`` with the plan's version. Returns the paths written and any refusal.

    A replacement is accepted only when it is non-empty, parses if it is Python, and differs
    from what is already there. The last condition is what stops the loop spinning: rewriting
    a file with byte-identical content is not progress, and reporting it as a change applied
    would make the loop believe it was getting somewhere.
    """
    written: List[str] = []
    refusals: List[str] = []

    for path in paths:
        content, description, is_entrypoint = planned[path]
        if not content.strip():
            refusals.append(f"the plan produced an empty {path}")
            continue
        if path.endswith(".py"):
            try:
                ast.parse(content, filename=path)
            except SyntaxError as exc:
                refusals.append(f"the plan's {path} does not parse ({exc.msg})")
                continue
        existing = project.get(path)
        if existing is not None and existing.content == content:
            refusals.append(f"{path} already matches the plan")
            continue

        project.add_file(
            path,
            content,
            is_entrypoint=is_entrypoint or (existing.is_entrypoint if existing else False),
            description=description or (existing.description if existing else ""),
        )
        written.append(path)

    return written, "; ".join(refusals)


# ------------------------------------------------------------------------- model strategy
def _does_not_parse(project: GeneratedProject, path: str) -> bool:
    file = project.get(path)
    if file is None or not path.endswith(".py"):
        return False
    try:
        ast.parse(file.content, filename=path)
    except SyntaxError:
        return True
    return False


def _repair_with_model(
    project: GeneratedProject,
    path: str,
    provider: LLMProvider,
) -> bool:
    """
    Ask the model for a version of one file that parses. True when one was accepted.

    Reuses :func:`multi_agent_generator.core.improvement._repair_one` rather than writing a
    second prompt for the same job. It is private to that module, and importing it here is a
    deliberate choice: two independently-worded prompts for "fix this file" would drift, and
    the guards on the accept condition - it must parse, and it must not have thrown half the
    file away - are the part that matters and should exist once.
    """
    file = project.get(path)
    if file is None:
        return False
    try:
        ast.parse(file.content, filename=path)
    except SyntaxError as error:
        from .improvement import _repair_one

        fixed = _repair_one(path, file.content, error, provider)
        if fixed is None:
            return False
        project.add_file(
            path,
            fixed,
            is_entrypoint=file.is_entrypoint,
            description=file.description,
        )
        return True
    return False
