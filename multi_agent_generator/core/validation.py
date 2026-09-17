# multi_agent_generator/core/validation.py
"""
Deciding whether a generated project is actually ready.

This module exists because of one sentence in section 17 of the brief: *do not say "Agent
Ready" simply because code generation succeeded*. Generation succeeding means a language model
returned text and the emitters wrote files. It does not mean the imports resolve, the framework
is declared as a dependency, the entry point exists, the workflow can be built, or that no API
key ended up inside the source. Every one of those has failed in practice while the pipeline
reported success, and each failure surfaced later as a traceback in front of a user.

So readiness is a *conjunction of named properties*, each with its own evidence, and this
module is where those properties are stated. The output is never a boolean: a bare ``False``
tells the user nothing, tells the repair engine nothing, and cannot be rendered as the Testing
dashboard that section 17 asks for. It is a :class:`ValidationReport` of :class:`Check`
records, each carrying a plain-language ``summary`` for the person and a technical ``detail``
for the log - the two-level error model of sections 85 and 86.

Three rules govern everything below.

**No check may fabricate its verdict.** Every count comes from a ``len()`` over parsed source
or over a real report. There is no scoring formula and no confidence number, because section 87
forbids inventing them and because a number nobody can derive is a number nobody can act on.

**Absence of evidence is not a pass.** A check with nothing to look at is ``SKIPPED``, and a
``SKIPPED`` check that was *required* blocks readiness exactly as a failure would. This is the
whole reason the enum has four members instead of two: "we did not run the tests" and "the
tests passed" must not collapse into the same green tick.

**Nothing here executes anything.** No subprocess, no import of generated code, no network.
The static gates have to be cheap enough to run inside the repair loop after every rewrite, and
they have to be safe to run on model output that has not been screened yet. Execution belongs
to :mod:`multi_agent_generator.execution`, and its results arrive here as arguments.
"""
from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..dependencies import FRAMEWORK_REQUIREMENTS, distribution_name, screen_requirements
from .depgraph import (
    CODE_AMBIGUOUS_MODULE,
    CODE_BAD_RELATIVE_IMPORT,
    CODE_CIRCULAR_IMPORT,
    CODE_MISSING_SYMBOL,
    CODE_ORPHAN_FILE,
    CODE_SYNTAX_ERROR,
    CODE_TEST_IMPORTED,
    CODE_UNDECLARED_DEPENDENCY,
    CODE_UNPLANNED_EDGE,
    CODE_UNREALISED_EDGE,
    CODE_UNRESOLVED_INTERNAL,
    DependencyGraph,
    GraphProblem,
    build_graph,
)
from .manifest import ProjectManifest
from .models import GeneratedFile, GeneratedProject, ReviewIssue, RunResult, Severity, TestReport

__all__ = [
    "CATEGORIES",
    "CATEGORY_LABELS",
    "RUN_ENTRY_NAMES",
    "MAX_REASONABLE_FILE_LINES",
    "CheckStatus",
    "Check",
    "ValidationReport",
    "static_checks",
    "validate_project",
]


# ============================================================================== vocabulary
#: The validation categories, in the order the Testing tab shows them.
#:
#: This is section 17's list, with two additions. ``structure`` covers "is this a project or is
#: it one enormous file", which sections 3 and 4 make a hard requirement rather than a matter of
#: taste. ``requirements`` is populated by the requirement-coverage checks passed in as
#: ``extra_checks`` - it lives here so the ordering is stated once, in one place, instead of
#: being re-invented by whichever caller renders the list.
CATEGORIES: Tuple[str, ...] = (
    "configuration",
    "dependencies",
    "imports",
    "structure",
    "entrypoint",
    "agents",
    "tools",
    "workflow",
    "error_handling",
    "tests",
    "runtime",
    "requirements",
)

#: Human labels for the categories. The UI must not invent its own wording for these, because
#: the same category name then reads differently in the dashboard, the log and the API.
CATEGORY_LABELS: Dict[str, str] = {
    "configuration": "Configuration",
    "dependencies": "Dependencies",
    "imports": "Imports",
    "structure": "Project structure",
    "entrypoint": "Entry point",
    "agents": "Agent construction",
    "tools": "Tool construction",
    "workflow": "Workflow",
    "error_handling": "Error handling",
    "tests": "Generated tests",
    "runtime": "Runtime",
    "requirements": "Requirement coverage",
}

#: The function names the execution layer looks for when it runs a generated project, in the
#: order it prefers them.
#:
#: Defined here rather than in :mod:`multi_agent_generator.execution.runner` so that the
#: validator asserts *the same* contract the runner depends on. The alternative - a private
#: tuple in the runner and a second guess in here - is how a project comes to pass validation
#: and then fail at launch with "no runnable entry point found", which is precisely the class of
#: contradiction this module exists to prevent. The runner imports this name.
RUN_ENTRY_NAMES: Tuple[str, ...] = ("run_workflow", "run_agent", "run_flow", "run", "main")

#: The names that return the agent's answer, as opposed to merely printing it.
#:
#: The playground shows a *final response*, and it can only do that when the entry function
#: returns a value the harness can capture. ``main`` usually prints and returns ``None``, so a
#: project that offers only ``main`` still runs but degrades to stdout scraping.
RESULT_ENTRY_NAMES: Tuple[str, ...] = ("run_workflow", "run_agent", "run_flow", "run")

#: Above this many lines, one Python file is doing too much.
#:
#: Not a style preference. Section 3 forbids emitting a single 1,000-to-2,000-line ``agent.py``,
#: and section 4 forbids the opposite over-correction of thirty files for a trivial agent. This
#: number is the point at which a file stops being reviewable in one sitting; it is a warning
#: rather than a blocker, for the reason given in :func:`_structure_checks`.
MAX_REASONABLE_FILE_LINES = 400


class CheckStatus(str, Enum):
    """
    The outcome of one check.

    Four states, because the two obvious ones are not enough to be honest with.

    ``PASSED``
        The property holds, and the check looked at something real to decide that.

    ``FAILED``
        The property does not hold. Whether that stops the project being called ready is
        decided by the check's ``severity``, not by this status - a stub tool and a missing
        framework dependency are both failures, and only one of them means the agent cannot
        run at all.

    ``WARNING``
        The property holds, but with a caveat the user should see.

    ``SKIPPED``
        There was nothing to look at. This is *not* a pass. A skipped check that was required
        blocks readiness, because "the test suite never ran" and "the test suite passed" are
        different facts and section 59 forbids presenting the first as the second.
    """

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


#: Ordered weakest-to-strongest, for folding several findings into one verdict.
_SEVERITY_ORDER: Tuple[Severity, ...] = (
    Severity.INFO,
    Severity.MINOR,
    Severity.MAJOR,
    Severity.BLOCKER,
)


def _worst(severities: Iterable[Severity]) -> Severity:
    """The most serious severity in ``severities``, or ``MINOR`` when there are none."""
    found = [s for s in severities if s in _SEVERITY_ORDER]
    if not found:
        return Severity.MINOR
    return max(found, key=_SEVERITY_ORDER.index)


# =================================================================================== check
@dataclass(frozen=True)
class Check:
    """
    One named property of a generated project, and what we found when we looked.

    Frozen because a check is a *finding*: something that was true of one version of one
    project at one moment. Mutating it after the fact would make the report disagree with the
    events already emitted for it.

    The three text fields carry the two-level error model of sections 85 and 86, and they are
    not interchangeable:

    ``summary``
        One sentence, plain language, no jargon, no traceback. This is what a non-technical
        user reads. It states what was found, not what a developer should type.

    ``detail``
        The technical text - the exception, the offending line, the pip output. Shown only
        behind "View technical details".

    ``fix``
        What to do about it, or what the system is about to do about it. Empty when the check
        passed, because a passing check has nothing to fix and a filled-in field there is the
        kind of decorative text section 10 rules out.
    """

    id: str
    category: str
    title: str
    status: CheckStatus
    summary: str
    severity: Severity = Severity.MAJOR
    detail: str = ""
    fix: str = ""
    #: Files this finding is about. The repair engine rewrites these; the UI links to them.
    paths: Tuple[str, ...] = ()
    #: Short factual lines supporting the verdict, e.g. "4 agents, 4 factories emitted".
    evidence: Tuple[str, ...] = ()
    #: False for checks whose absence is acceptable - a live provider call, for instance.
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status in (CheckStatus.PASSED, CheckStatus.WARNING)

    @property
    def blocking(self) -> bool:
        """
        Whether this finding must stop the project being reported ready.

        Two ways to block, and the second is the one that matters. A ``FAILED`` check blocks
        when its severity blocks a release, which is the ordinary case. A ``SKIPPED`` check
        blocks when it was *required*, because a property nobody verified has not been
        established - and reporting an unverified project as ready is the exact behaviour the
        brief opens by rejecting.
        """
        if self.status is CheckStatus.FAILED:
            return self.required and self.severity.blocks_release
        if self.status is CheckStatus.SKIPPED:
            return self.required
        return False

    def to_issue(self) -> ReviewIssue:
        """As a :class:`ReviewIssue`, so review, repair and validation share one issue type."""
        return ReviewIssue(
            category=self.category,
            severity=self.severity,
            message=self.summary,
            file=self.paths[0] if self.paths else None,
            line=None,
            suggestion=self.fix,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "title": self.title,
            "status": self.status.value,
            "summary": self.summary,
            "severity": self.severity.value,
            "detail": self.detail,
            "fix": self.fix,
            "paths": list(self.paths),
            "evidence": list(self.evidence),
            "required": self.required,
            "blocking": self.blocking,
        }


# ================================================================================== report
@dataclass
class ValidationReport:
    """
    Every check that was run, and the single verdict that follows from them.

    ``ready`` is derived, never assigned. There is deliberately no setter and no override:
    the only way to make a project ready is to make its checks pass, which is what stops
    "ready" from drifting back into meaning "generation returned without raising".
    """

    checks: List[Check] = field(default_factory=list)
    #: Assumptions and caveats worth showing, e.g. "no manifest, plan checks were skipped".
    notes: List[str] = field(default_factory=list)
    #: The file the execution layer will run, as determined by the entry-point checks.
    entrypoint: Optional[str] = None
    duration_s: float = 0.0
    #: The parsed import graph, kept so callers do not have to re-parse the project. Not
    #: serialised - it has its own ``as_dict`` and belongs under its own key in an API payload.
    graph: Optional[DependencyGraph] = None

    # ------------------------------------------------------------------------------ queries
    @property
    def ready(self) -> bool:
        """True when no check blocks. Also False when there are no checks at all."""
        if not self.checks:
            return False
        return not any(check.blocking for check in self.checks)

    def blocking(self) -> List[Check]:
        return [check for check in self.checks if check.blocking]

    def failed(self) -> List[Check]:
        return [check for check in self.checks if check.status is CheckStatus.FAILED]

    def warnings(self) -> List[Check]:
        return [check for check in self.checks if check.status is CheckStatus.WARNING]

    def get(self, check_id: str) -> Optional[Check]:
        for check in self.checks:
            if check.id == check_id:
                return check
        return None

    def by_category(self) -> List[Dict[str, Any]]:
        """
        The checks grouped for the Testing tab, in :data:`CATEGORIES` order.

        Empty categories are omitted rather than rendered as an empty section. A category
        header with nothing under it reads as a feature that has not been built, and section 58
        rules out showing those.
        """
        grouped: List[Dict[str, Any]] = []
        for category in CATEGORIES:
            members = [c for c in self.checks if c.category == category]
            if not members:
                continue
            grouped.append(
                {
                    "category": category,
                    "label": CATEGORY_LABELS.get(category, category),
                    "status": _fold_status(members).value,
                    "blocking": any(m.blocking for m in members),
                    "checks": [m.as_dict() for m in members],
                }
            )
        return grouped

    def status_counts(self) -> Dict[str, int]:
        counts = {status.value: 0 for status in CheckStatus}
        for check in self.checks:
            counts[check.status.value] += 1
        return counts

    def unmet(self) -> List[str]:
        """
        Plain sentences naming what is not satisfied, for the user and for the event trace.

        Replaces the hand-written criteria list that used to live on ``PipelineResult``: the
        sentences are now generated from the same checks the dashboard shows, so the summary
        line and the detail view cannot disagree about whether something passed.
        """
        lines: List[str] = []
        for check in self.blocking():
            label = CATEGORY_LABELS.get(check.category, check.category)
            lines.append(f"{label}: {check.summary}")
        return lines

    def to_issues(self) -> List[ReviewIssue]:
        """Failures as review issues, so the repair engine has one input type."""
        return [c.to_issue() for c in self.checks if c.status is CheckStatus.FAILED]

    def summary(self) -> Dict[str, Any]:
        counts = self.status_counts()
        return {
            "ready": self.ready,
            "checks": len(self.checks),
            "passed": counts[CheckStatus.PASSED.value],
            "failed": counts[CheckStatus.FAILED.value],
            "warnings": counts[CheckStatus.WARNING.value],
            "skipped": counts[CheckStatus.SKIPPED.value],
            "blocking": len(self.blocking()),
            "entrypoint": self.entrypoint,
            "duration_s": round(self.duration_s, 3),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.summary(),
            "categories": self.by_category(),
            "unmet": self.unmet(),
            "notes": list(self.notes),
        }


def _fold_status(checks: Sequence[Check]) -> CheckStatus:
    """
    One status for a group.

    Worst wins, with ``SKIPPED`` ranked above ``WARNING`` when it was required: an unverified
    required property is a more serious thing to show a user than a verified one with a caveat.
    """
    if any(c.status is CheckStatus.FAILED for c in checks):
        return CheckStatus.FAILED
    if any(c.status is CheckStatus.SKIPPED and c.required for c in checks):
        return CheckStatus.SKIPPED
    if any(c.status is CheckStatus.WARNING for c in checks):
        return CheckStatus.WARNING
    if any(c.status is CheckStatus.SKIPPED for c in checks):
        return CheckStatus.SKIPPED
    return CheckStatus.PASSED


# ================================================================================= helpers
def _passed(
    check_id: str,
    category: str,
    title: str,
    summary: str,
    *,
    evidence: Sequence[str] = (),
    paths: Sequence[str] = (),
) -> Check:
    """A passing check. Separate constructor purely so the call sites stay readable."""
    return Check(
        id=check_id,
        category=category,
        title=title,
        status=CheckStatus.PASSED,
        summary=summary,
        severity=Severity.INFO,
        evidence=tuple(evidence),
        paths=tuple(paths),
    )


def _app_files(project: GeneratedProject) -> List[GeneratedFile]:
    """
    The project's own Python modules, excluding its tests.

    ``GeneratedProject.source_files()`` is every ``.py`` file and ``test_files()`` is a *subset*
    of it, not a disjoint partition. Every structural question here - is this one enormous
    module, does the entry point handle its errors, which environment variables must be
    documented - is about application code, so mixing the tests in would count ``tests/`` toward
    the module split and would demand that a variable only a test reads appear in
    ``.env.example``. Both would be findings about nothing.
    """
    tests = {f.path for f in project.test_files()}
    return [f for f in project.source_files() if f.path not in tests]


def _app_lines(project: GeneratedProject) -> int:
    """
    Lines of application Python.

    Not ``GeneratedProject.total_lines``, which sums *every* file - the README, the Dockerfile
    and ``requirements.txt`` included. A project whose README is long is not a project whose
    code needs splitting up.
    """
    return sum(f.line_count for f in _app_files(project))


def _parse(source: str) -> Optional[ast.AST]:
    """Parse Python source, or ``None`` if it does not parse. Never raises."""
    try:
        return ast.parse(source or "")
    except (SyntaxError, ValueError, RecursionError):
        # A syntax error is already reported as its own check by the import graph, so there is
        # nothing to add here beyond "we could not inspect this file".
        return None


def _dotted(node: ast.AST) -> str:
    """``os.environ`` for an attribute chain, ``os`` for a name, ``""`` for anything else."""
    parts: List[str] = []
    current: Any = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return ""


def _string(node: Optional[ast.AST]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _env_reads(tree: ast.AST) -> Set[str]:
    """
    Environment variable names read by literal in this module.

    Only literals, deliberately. ``os.getenv(name)`` where ``name`` is a variable cannot be
    resolved without executing the module, and guessing would produce a check that reports
    undocumented variables that do not exist.
    """
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _dotted(node.func)
            if target in ("os.getenv", "getenv", "os.environ.get", "environ.get") and node.args:
                name = _string(node.args[0])
                if name:
                    found.add(name)
        elif isinstance(node, ast.Subscript):
            if _dotted(node.value) in ("os.environ", "environ"):
                name = _string(node.slice)
                if name:
                    found.add(name)
    return found


def _silent_handlers(tree: ast.AST) -> List[Tuple[int, str]]:
    """
    ``except`` blocks whose entire body is ``pass`` or ``...``, with line and exception text.

    Section 80's rule is that a failure must never disappear. A handler that does nothing at
    all is the mechanical form of swallowing it: the run continues, the user sees a plausible
    empty answer, and there is no record anywhere of what went wrong. Handlers that log, or
    ``continue`` past one bad item, are not reported - those are deliberate and visible.
    """
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = [
            stmt
            for stmt in node.body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        if len(body) != 1:
            continue
        only = body[0]
        empty = isinstance(only, ast.Pass) or (
            isinstance(only, ast.Expr)
            and isinstance(only.value, ast.Constant)
            and only.value.value is Ellipsis
        )
        if empty:
            caught = _dotted(node.type) if node.type is not None else "everything"
            found.append((node.lineno, caught or "everything"))
    return found


def _has_try(tree: ast.AST) -> bool:
    return any(isinstance(node, ast.Try) for node in ast.walk(tree))


def _called_names(tree: ast.AST) -> Set[str]:
    """Every function name called in this module, by last component (``mod.run`` -> ``run``)."""
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted:
                names.add(dotted.rsplit(".", 1)[-1])
    return names


#: Shapes that look like a real credential sitting in source. Each pattern is anchored on a
#: vendor prefix rather than on entropy, because an entropy heuristic on generated code fires on
#: hashes, UUIDs and base64 test fixtures, and a leak detector nobody trusts gets switched off.
_CREDENTIAL_SHAPES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("an OpenAI-style secret key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("a Hugging Face access token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("a Google API key", re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}")),
    ("a Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}")),
    ("an Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("a Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}")),
    (
        "a credential assigned as a literal",
        # ``[^"'\s]`` rather than ``[^"']`` is doing real work: a credential never contains a
        # space, and prose does. Without it this fires on any docstring or comment of the form
        # ``API_KEY: "the key used to authenticate"``, which appears in generated configuration
        # modules and would put a spurious blocker on the project.
        re.compile(
            r"(?:API_?KEY|TOKEN|SECRET|PASSWORD)[\"']?\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']",
            re.IGNORECASE,
        ),
    ),
)

#: Substrings that mark a matched value as an obvious placeholder rather than a real credential.
#:
#: Tested against *the matched text*, not the whole line. Matching on the line would be far too
#: coarse - one occurrence of the word "example" anywhere in a file's line would excuse a real
#: key sitting next to it, and in a check whose whole job is to catch leaked credentials a
#: false negative is the expensive direction.
#:
#: ``test-`` and ``test_`` are here for a specific reason. The generated ``tests/conftest.py``
#: sets deliberately fake credentials of the form ``test-openai-api-key``, which matches the
#: generic assignment shape below. Without this entry every single project would be reported as
#: leaking a credential - and a check that fires on every project is one whose real findings
#: stop being read.
_PLACEHOLDER_MARKERS: Tuple[str, ...] = (
    "your-",
    "your_",
    "yourkey",
    "test-",
    "test_",
    "xxx",
    "changeme",
    "change-me",
    "replace",
    "placeholder",
    "example",
    "dummy",
    "fake",
    "not-real",
    "notreal",
    "do-not",
    "print-me",
    "abc123",
    "...",
    "<",
    "${",
)


def _looks_like_placeholder(text: str) -> bool:
    return any(marker in text.lower() for marker in _PLACEHOLDER_MARKERS)


def _secret_findings(
    project: GeneratedProject,
    secrets: Sequence[str],
) -> List[Tuple[str, int, str]]:
    """
    Places where a credential appears to be written into the project, as (path, line, kind).

    The value itself is never returned, never logged and never rendered. Section 34's rule is
    that a complete key is never printed, and a leak report that quotes the leaked key to prove
    it found one has simply moved the leak somewhere else.

    Two passes with different confidence, and the order matters. The first compares against the
    credential values the platform actually holds: an exact substring match is proof, so it runs
    on every file unconditionally and is never filtered as a placeholder. The second looks for
    the *shape* of a credential, which is a guess, so its matches are discarded when the matched
    text is visibly a stand-in.

    ``secrets`` is passed in rather than read from the environment here. That keeps this module
    free of ambient state, and it means the caller decides what counts as a secret - validation
    never goes looking for credentials of its own accord.
    """
    findings: List[Tuple[str, int, str]] = []
    needles = sorted(
        {str(value or "").strip() for value in secrets if len(str(value or "").strip()) >= 8}
    )
    for generated in project.files:
        for lineno, line in enumerate((generated.content or "").splitlines(), start=1):
            if any(needle in line for needle in needles):
                findings.append((generated.path, lineno, "a configured credential value"))
                continue
            for kind, pattern in _CREDENTIAL_SHAPES:
                match = pattern.search(line)
                if match and not _looks_like_placeholder(match.group(0)):
                    findings.append((generated.path, lineno, kind))
                    break
    return findings


def _providers(graph: DependencyGraph, *, include_tests: bool = False) -> Dict[str, List[str]]:
    """
    Module-level name -> the project files that *define* it.

    Definitions only, not re-exports. Every caller here is asking "which file is the
    configuration module / the agent factory / the entry point?", and a file that says
    ``from config import get_settings`` answers none of those - the name is importable from it,
    but the thing itself lives elsewhere. Reading ``node.provides`` (which includes imported
    bindings, by design, so that ``from x import y`` can be checked) let a workflow module stand
    in for a config module that had never been generated: the settings check passed and its
    summary named the importer as the owner. Both are section 9 fabrication, so this reads
    ``node.defines``.

    Tests are excluded by default. A test module that happens to define ``main`` must not
    satisfy "the project has an entry point", because the runner will never call it.
    """
    found: Dict[str, List[str]] = {}
    for path, node in graph.nodes.items():
        if node.is_test and not include_tests:
            continue
        for name in node.defines:
            found.setdefault(name, []).append(path)
    return {name: sorted(paths) for name, paths in found.items()}


def _declared_distributions(items: Iterable[str]) -> Set[str]:
    """The distribution names in a list of requirement specifiers, lowercased."""
    names: Set[str] = set()
    for item in items:
        name = distribution_name(item)
        if name:
            names.add(name.lower().replace("_", "-"))
    return names


def _sorted(checks: Iterable[Check]) -> List[Check]:
    order = {category: index for index, category in enumerate(CATEGORIES)}
    return sorted(checks, key=lambda c: (order.get(c.category, len(order)), c.id))


# ====================================================================== configuration gates
def _configuration_checks(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest],
    graph: DependencyGraph,
    secrets: Sequence[str],
) -> List[Check]:
    checks: List[Check] = []
    provides = _providers(graph)

    # --- the settings module exists ------------------------------------------------------
    owners = provides.get("get_settings", [])
    if owners:
        checks.append(
            _passed(
                "configuration.settings_module",
                "configuration",
                "Configuration is loaded from one place",
                "The project reads its settings and credentials through a single "
                f"configuration module ({owners[0]}).",
                paths=owners,
            )
        )
    else:
        checks.append(
            Check(
                id="configuration.settings_module",
                category="configuration",
                title="Configuration is loaded from one place",
                status=CheckStatus.FAILED,
                summary=(
                    "Nothing in this project loads its configuration, so it has no way to find "
                    "the API key it needs to run."
                ),
                severity=Severity.MAJOR,
                detail="No generated module defines get_settings().",
                fix="Add a configuration module that exposes get_settings().",
            )
        )

    # --- every variable the code reads is written down -----------------------------------
    documented: Set[str] = set()
    env_example = project.get(".env.example")
    if env_example is not None:
        for line in (env_example.content or "").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            documented.add(text.split("=", 1)[0].strip())

    planned = {var.name for var in manifest.env_vars} if manifest else set()

    read: Dict[str, List[str]] = {}
    for generated in _app_files(project):
        tree = _parse(generated.content)
        if tree is None:
            continue
        for name in _env_reads(tree):
            read.setdefault(name, []).append(generated.path)

    if env_example is None:
        checks.append(
            Check(
                id="configuration.env_documented",
                category="configuration",
                title="Required settings are documented",
                status=CheckStatus.FAILED,
                summary=(
                    "There is no .env.example, so nobody receiving this project can tell which "
                    "settings it needs before running it."
                ),
                severity=Severity.MAJOR,
                detail=f"Variables read by the code: {', '.join(sorted(read)) or 'none'}.",
                fix="Emit a .env.example listing every variable the project reads.",
            )
        )
    else:
        undocumented = sorted(name for name in read if name not in documented)
        missing_planned = sorted(name for name in planned if name not in documented)
        if missing_planned:
            checks.append(
                Check(
                    id="configuration.env_documented",
                    category="configuration",
                    title="Required settings are documented",
                    status=CheckStatus.FAILED,
                    summary=(
                        f"{len(missing_planned)} setting"
                        + ("s are" if len(missing_planned) != 1 else " is")
                        + " part of this project's plan but missing from .env.example: "
                        + ", ".join(missing_planned)
                        + "."
                    ),
                    severity=Severity.MAJOR,
                    detail="Planned but undocumented: " + ", ".join(missing_planned),
                    fix="Add the missing variables to .env.example.",
                    paths=(".env.example",),
                )
            )
        elif undocumented:
            checks.append(
                Check(
                    id="configuration.env_documented",
                    category="configuration",
                    title="Required settings are documented",
                    status=CheckStatus.WARNING,
                    summary=(
                        "The code reads "
                        + ", ".join(undocumented)
                        + " from the environment, and .env.example does not mention "
                        + ("them." if len(undocumented) != 1 else "it.")
                    ),
                    severity=Severity.MINOR,
                    detail="\n".join(
                        f"{name}: read in {', '.join(read[name])}" for name in undocumented
                    ),
                    fix="Add these variables to .env.example, or stop reading them.",
                    paths=(".env.example",),
                    evidence=(f"{len(documented)} documented, {len(read)} read in code",),
                )
            )
        else:
            checks.append(
                _passed(
                    "configuration.env_documented",
                    "configuration",
                    "Required settings are documented",
                    f"All {len(documented)} setting"
                    + ("s" if len(documented) != 1 else "")
                    + " the project reads are listed in .env.example.",
                    evidence=(
                        f"{len(documented)} documented",
                        f"{len(read)} read by literal name in code",
                    ),
                    paths=(".env.example",),
                )
            )

    # --- no credential is written into the project ---------------------------------------
    leaks = _secret_findings(project, secrets)
    if leaks:
        checks.append(
            Check(
                id="configuration.secrets",
                category="configuration",
                title="No credentials in the generated code",
                status=CheckStatus.FAILED,
                summary=(
                    f"Something that looks like a credential is written into "
                    f"{len({path for path, _, _ in leaks})} file"
                    + ("s" if len({p for p, _, _ in leaks}) != 1 else "")
                    + " of this project. Credentials must come from the environment, never "
                    "from the source."
                ),
                severity=Severity.BLOCKER,
                detail="\n".join(f"{path}:{line} contains {kind}" for path, line, kind in leaks),
                fix=(
                    "Replace the literal with a read from the configuration module, and rotate "
                    "the credential if it was a real one."
                ),
                paths=tuple(sorted({path for path, _, _ in leaks})),
                evidence=(f"{len(leaks)} occurrence(s)",),
            )
        )
    else:
        checks.append(
            _passed(
                "configuration.secrets",
                "configuration",
                "No credentials in the generated code",
                "No API key or token appears anywhere in the generated files.",
                evidence=(f"{len(project.files)} files scanned",),
            )
        )

    return checks


# ======================================================================== dependency gates
def _dependency_checks(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest],
    graph: DependencyGraph,
) -> List[Check]:
    checks: List[Check] = []
    requirements = project.get("requirements.txt")

    if requirements is None:
        checks.append(
            Check(
                id="dependencies.file",
                category="dependencies",
                title="The project declares what it needs",
                status=CheckStatus.FAILED,
                summary=(
                    "There is no requirements.txt, so this project cannot be installed by "
                    "anyone who receives it."
                ),
                severity=Severity.BLOCKER,
                detail=f"Declared dependencies in the plan: {', '.join(project.dependencies)}.",
                fix="Emit a requirements.txt from the project's dependency list.",
            )
        )
        return checks

    text = requirements.content or ""
    in_file = _declared_distributions(text.splitlines())
    expected = _declared_distributions(project.dependencies)
    absent = sorted(expected - in_file)

    if absent:
        checks.append(
            Check(
                id="dependencies.file",
                category="dependencies",
                title="The project declares what it needs",
                status=CheckStatus.FAILED,
                summary=(
                    f"requirements.txt is missing {len(absent)} package"
                    + ("s" if len(absent) != 1 else "")
                    + " this project depends on: "
                    + ", ".join(absent)
                    + "."
                ),
                severity=Severity.MAJOR,
                detail=f"In requirements.txt: {', '.join(sorted(in_file)) or 'nothing'}.",
                fix="Add the missing packages to requirements.txt.",
                paths=("requirements.txt",),
            )
        )
    else:
        checks.append(
            _passed(
                "dependencies.file",
                "dependencies",
                "The project declares what it needs",
                f"requirements.txt lists all {len(in_file)} package"
                + ("s" if len(in_file) != 1 else "")
                + " the project depends on.",
                evidence=(f"{len(in_file)} declared",),
                paths=("requirements.txt",),
            )
        )

    # --- the framework itself is declared -------------------------------------------------
    framework_reqs = FRAMEWORK_REQUIREMENTS.get(project.framework, ())
    if not framework_reqs:
        checks.append(
            Check(
                id="dependencies.framework",
                category="dependencies",
                title="The chosen framework is installed with the project",
                status=CheckStatus.SKIPPED,
                summary=(
                    f"There is no recorded package list for the {project.framework} framework, "
                    "so its dependencies could not be checked."
                ),
                severity=Severity.MINOR,
                detail=f"framework={project.framework!r} is not in FRAMEWORK_REQUIREMENTS.",
                fix="Register this framework's packages in multi_agent_generator/dependencies.py.",
                required=False,
            )
        )
    else:
        wanted = {r.package.lower().replace("_", "-"): r for r in framework_reqs}
        missing = sorted(name for name in wanted if name not in in_file)
        if missing:
            checks.append(
                Check(
                    id="dependencies.framework",
                    category="dependencies",
                    title="The chosen framework is installed with the project",
                    status=CheckStatus.FAILED,
                    summary=(
                        f"This project is built on {project.framework}, but "
                        + ", ".join(missing)
                        + " is not in requirements.txt. Nothing will import on a clean machine."
                    ),
                    severity=Severity.BLOCKER,
                    detail="\n".join(f"{name}: {wanted[name].reason}" for name in missing),
                    fix="Add the framework's packages to requirements.txt.",
                    paths=("requirements.txt",),
                )
            )
        else:
            checks.append(
                _passed(
                    "dependencies.framework",
                    "dependencies",
                    "The chosen framework is installed with the project",
                    f"requirements.txt includes the {project.framework} packages the generated "
                    "code imports.",
                    evidence=tuple(sorted(wanted)),
                    paths=("requirements.txt",),
                )
            )

    # --- every line can actually be installed ---------------------------------------------
    allowed, rejected = screen_requirements(text)
    if rejected:
        checks.append(
            Check(
                id="dependencies.installable",
                category="dependencies",
                title="Every requirement can be installed",
                status=CheckStatus.FAILED,
                summary=(
                    f"{len(rejected)} line"
                    + ("s" if len(rejected) != 1 else "")
                    + " in requirements.txt asks to install from a URL, a local path or an "
                    "alternative index. Those are refused when this project is run, so it "
                    "would fail to install."
                ),
                severity=Severity.MAJOR,
                detail="\n".join(rejected),
                fix="Replace these with ordinary package names and version specifiers.",
                paths=("requirements.txt",),
            )
        )
    else:
        checks.append(
            _passed(
                "dependencies.installable",
                "dependencies",
                "Every requirement can be installed",
                f"All {len(allowed)} requirement line"
                + ("s" if len(allowed) != 1 else "")
                + " name a package from the standard index.",
                evidence=(f"{len(allowed)} installable lines",),
                paths=("requirements.txt",),
            )
        )

    return checks


# ============================================================================ import gates
#: One check per import-graph problem code: (code, id, category, title, message when clean).
#:
#: Folded mechanically so that every code the graph can report becomes a visible row, and a
#: code that reported nothing shows as a pass rather than vanishing. A dashboard that only
#: lists what went wrong cannot be read as "we checked this" - which is the difference between
#: a report and a pile of errors.
_GRAPH_CHECKS: Tuple[Tuple[str, str, str, str, str], ...] = (
    (
        CODE_SYNTAX_ERROR,
        "structure.syntax",
        "structure",
        "Every file is valid Python",
        "All generated Python files parse.",
    ),
    (
        CODE_UNRESOLVED_INTERNAL,
        "imports.internal",
        "imports",
        "Files can find each other",
        "Every import between the project's own files resolves.",
    ),
    (
        CODE_UNDECLARED_DEPENDENCY,
        "imports.third_party",
        "imports",
        "Every imported package is declared",
        "Every third-party package the code imports is in requirements.txt.",
    ),
    (
        CODE_MISSING_SYMBOL,
        "imports.symbols",
        "imports",
        "Imported names exist",
        "Every name imported from another file is actually defined there.",
    ),
    (
        CODE_CIRCULAR_IMPORT,
        "imports.cycles",
        "imports",
        "No circular imports",
        "No file depends on itself through a chain of imports.",
    ),
    (
        CODE_BAD_RELATIVE_IMPORT,
        "imports.relative",
        "imports",
        "Relative imports stay inside the project",
        "Every relative import points somewhere inside the project.",
    ),
    (
        CODE_AMBIGUOUS_MODULE,
        "imports.ambiguous",
        "imports",
        "No two files claim the same module name",
        "Every module name maps to exactly one file.",
    ),
    (
        CODE_TEST_IMPORTED,
        "imports.test_isolation",
        "imports",
        "Application code does not import its tests",
        "No source file imports a test module.",
    ),
    (
        CODE_ORPHAN_FILE,
        "structure.orphans",
        "structure",
        "Every file is used",
        "Every generated file is imported by something or is an entry point.",
    ),
    (
        CODE_UNPLANNED_EDGE,
        "structure.unplanned_imports",
        "structure",
        "Imports match the plan",
        "The files import each other the way the architecture said they would.",
    ),
    (
        CODE_UNREALISED_EDGE,
        "structure.unrealised_plan",
        "structure",
        "The plan was carried out",
        "Every dependency the architecture planned exists in the code.",
    ),
)


def _graph_checks(graph: DependencyGraph) -> List[Check]:
    checks: List[Check] = []
    by_code: Dict[str, List[GraphProblem]] = {}
    for problem in graph.problems:
        by_code.setdefault(problem.code, []).append(problem)

    for code, check_id, category, title, clean in _GRAPH_CHECKS:
        problems = by_code.get(code, [])
        if not problems:
            checks.append(_passed(check_id, category, title, clean))
            continue
        severity = _worst(p.severity for p in problems)
        paths = tuple(sorted({p.path for p in problems if p.path}))
        first = problems[0]
        extra = (
            f" ({len(problems) - 1} more like it)"
            if len(problems) > 1
            else ""
        )
        checks.append(
            Check(
                id=check_id,
                category=category,
                title=title,
                status=CheckStatus.FAILED,
                summary=first.message + extra,
                severity=severity,
                detail="\n".join(
                    f"{p.path or '-'}"
                    + (f":{p.line}" if p.line else "")
                    + f" {p.message}"
                    + (f" | {'; '.join(p.detail)}" if p.detail else "")
                    for p in problems
                ),
                fix=first.suggestion,
                paths=paths,
                evidence=(f"{len(problems)} finding(s)",),
            )
        )

    # Codes the graph grew after this list was written would otherwise be silently dropped.
    known = {code for code, _, _, _, _ in _GRAPH_CHECKS}
    unknown = sorted(set(by_code) - known)
    for code in unknown:
        problems = by_code[code]
        checks.append(
            Check(
                id=f"imports.{code}",
                category="imports",
                title=code.replace("_", " ").capitalize(),
                status=CheckStatus.FAILED,
                summary=problems[0].message,
                severity=_worst(p.severity for p in problems),
                detail="\n".join(p.message for p in problems),
                fix=problems[0].suggestion,
                paths=tuple(sorted({p.path for p in problems if p.path})),
            )
        )

    return checks


# ========================================================================= structure gates
def _structure_checks(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest],
    graph: DependencyGraph,
) -> List[Check]:
    checks: List[Check] = []
    sources = _app_files(project)
    total = _app_lines(project)

    # --- the planned files were emitted ---------------------------------------------------
    if manifest is None:
        checks.append(
            Check(
                id="structure.planned_files",
                category="structure",
                title="Every planned file was written",
                status=CheckStatus.SKIPPED,
                summary=(
                    "This project has no stored architecture plan, so there is nothing to "
                    "compare its files against."
                ),
                severity=Severity.MINOR,
                detail="No ProjectManifest was supplied.",
                required=False,
            )
        )
    else:
        emitted = {f.path for f in project.files}
        missing = sorted(spec.path for spec in manifest.files if spec.path not in emitted)
        if missing:
            checks.append(
                Check(
                    id="structure.planned_files",
                    category="structure",
                    title="Every planned file was written",
                    status=CheckStatus.FAILED,
                    summary=(
                        f"The architecture planned {len(manifest.files)} files and "
                        f"{len(missing)} of them "
                        + ("was" if len(missing) == 1 else "were")
                        + " never written: "
                        + ", ".join(missing[:5])
                        + ("..." if len(missing) > 5 else "")
                        + "."
                    ),
                    severity=Severity.MAJOR,
                    detail="Missing:\n" + "\n".join(missing),
                    fix="Generate the missing files, or remove them from the plan.",
                    paths=tuple(missing),
                    evidence=(
                        f"{len(manifest.files)} planned",
                        f"{len(emitted)} emitted",
                    ),
                )
            )
        else:
            checks.append(
                _passed(
                    "structure.planned_files",
                    "structure",
                    "Every planned file was written",
                    f"All {len(manifest.files)} files in the architecture plan exist.",
                    evidence=(
                        f"{len(manifest.files)} planned",
                        f"{len(emitted)} emitted",
                    ),
                )
            )

    # --- the project is a project, not one huge file --------------------------------------
    #
    # A warning rather than a failure, on purpose. Sections 3 and 4 make multi-file structure a
    # hard requirement of *generation*, and it is enforced there, by the architecture planner
    # choosing a tier and a file list. By the time a project reaches validation the code is
    # already written, and blocking readiness on file size would take a project that imports,
    # tests and runs correctly and report it as failed - for a reason no repair iteration can
    # fix, since splitting a module is a re-architecture rather than a repair. Reporting a
    # working agent as broken is the same category of dishonesty as the reverse, so this states
    # the problem loudly and lets the verdict rest on whether the thing works.
    oversized = [f for f in sources if f.line_count > MAX_REASONABLE_FILE_LINES]
    if len(sources) <= 1 and total > MAX_REASONABLE_FILE_LINES:
        checks.append(
            Check(
                id="structure.multi_file",
                category="structure",
                title="Work is split across modules",
                status=CheckStatus.FAILED,
                summary=(
                    f"The whole project is a single {total}-line Python file. It should be split "
                    "into modules for the agents, the tools and the workflow."
                ),
                severity=Severity.MAJOR,
                detail=f"Application source files: {[f.path for f in sources]}",
                fix="Re-plan the architecture so each responsibility gets its own module.",
                paths=tuple(f.path for f in sources),
                evidence=(f"{len(sources)} source file", f"{total} lines"),
            )
        )
    elif oversized:
        worst = max(oversized, key=lambda f: f.line_count)
        checks.append(
            Check(
                id="structure.multi_file",
                category="structure",
                title="Work is split across modules",
                status=CheckStatus.WARNING,
                summary=(
                    f"{worst.path} is {worst.line_count} lines long, which is more than one "
                    "module should carry."
                ),
                severity=Severity.MAJOR
                if worst.line_count > 2 * MAX_REASONABLE_FILE_LINES
                else Severity.MINOR,
                detail="\n".join(f"{f.path}: {f.line_count} lines" for f in oversized),
                fix="Split the largest modules along their responsibilities.",
                paths=tuple(f.path for f in oversized),
                evidence=(
                    f"{len(sources)} source files",
                    f"{total} lines of application code",
                    f"threshold {MAX_REASONABLE_FILE_LINES} lines",
                ),
            )
        )
    else:
        checks.append(
            _passed(
                "structure.multi_file",
                "structure",
                "Work is split across modules",
                f"{len(sources)} source file"
                + ("s" if len(sources) != 1 else "")
                + f" totalling {total} lines, none of them oversized.",
                evidence=(
                    f"{len(sources)} source files",
                    f"{total} lines of application code",
                ),
            )
        )

    return checks


# ======================================================================== entry-point gates
def _entrypoint_checks(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest],
    graph: DependencyGraph,
) -> Tuple[List[Check], Optional[str]]:
    """
    The entry-point gates, and the path the execution layer will actually run.

    These checks deliberately assert the properties
    :func:`multi_agent_generator.execution.runner._run_target` relies on, rather than deciding
    for themselves which file runs. Two independent answers to "what is the entry point" is how
    a project passes validation against ``main.py`` and is then launched from ``agent.py``.
    """
    checks: List[Check] = []
    provides = _providers(graph)

    declared = (project.entrypoint or "").strip()
    if not declared and manifest is not None:
        declared = (manifest.entrypoint or "").strip()

    entry_file = project.get(declared) if declared else None
    if entry_file is None:
        marked = [f for f in project.files if f.is_entrypoint]
        entry_file = marked[0] if marked else None
        declared = entry_file.path if entry_file else declared

    if entry_file is None:
        checks.append(
            Check(
                id="entrypoint.declared",
                category="entrypoint",
                title="There is a file to run",
                status=CheckStatus.FAILED,
                summary=(
                    "This project does not say which file to run, so it cannot be started."
                    if not declared
                    else f"The project says to run {declared}, but that file was not generated."
                ),
                severity=Severity.BLOCKER,
                detail=f"entrypoint={project.entrypoint!r}; files={[f.path for f in project.files]}",
                fix="Mark one generated file as the entry point.",
            )
        )
    else:
        checks.append(
            _passed(
                "entrypoint.declared",
                "entrypoint",
                "There is a file to run",
                f"The project starts from {entry_file.path}.",
                paths=(entry_file.path,),
            )
        )

    # --- something is callable ------------------------------------------------------------
    runnable = [name for name in RUN_ENTRY_NAMES if provides.get(name)]
    if runnable:
        first = runnable[0]
        checks.append(
            _passed(
                "entrypoint.runnable",
                "entrypoint",
                "The agent can be started",
                f"The agent is started by calling {first}() in {provides[first][0]}.",
                evidence=tuple(f"{name}() in {provides[name][0]}" for name in runnable),
                paths=tuple(provides[first]),
            )
        )
    else:
        checks.append(
            Check(
                id="entrypoint.runnable",
                category="entrypoint",
                title="The agent can be started",
                status=CheckStatus.FAILED,
                summary=(
                    "None of the generated files defines a function that starts the agent, so "
                    "there is nothing for the playground to call."
                ),
                severity=Severity.BLOCKER,
                detail="Looked for: " + ", ".join(f"{n}()" for n in RUN_ENTRY_NAMES),
                fix="Add a run_workflow(query) function that runs the workflow and returns its result.",
                paths=(entry_file.path,) if entry_file else (),
            )
        )

    # --- and it returns the answer rather than only printing it ---------------------------
    returning = [name for name in RESULT_ENTRY_NAMES if provides.get(name)]
    if returning:
        checks.append(
            _passed(
                "entrypoint.result_contract",
                "entrypoint",
                "The agent returns its answer",
                f"{returning[0]}() returns the agent's result, so it can be shown in the "
                "playground rather than scraped from console output.",
                paths=tuple(provides[returning[0]]),
            )
        )
    elif runnable:
        checks.append(
            Check(
                id="entrypoint.result_contract",
                category="entrypoint",
                title="The agent returns its answer",
                status=CheckStatus.FAILED,
                summary=(
                    "The agent can be started, but no function returns its answer - the result "
                    "can only be read from console output, which is less reliable."
                ),
                severity=Severity.MINOR,
                detail="Found only: " + ", ".join(f"{n}()" for n in runnable),
                fix="Have run_workflow(query) return the final response.",
            )
        )
    else:
        checks.append(
            Check(
                id="entrypoint.result_contract",
                category="entrypoint",
                title="The agent returns its answer",
                status=CheckStatus.SKIPPED,
                summary="Not checked, because the project has no way to start the agent at all.",
                severity=Severity.MINOR,
                required=False,
            )
        )

    return checks, (entry_file.path if entry_file else None)


# ============================================================== agent / tool / workflow gates
def _agent_checks(manifest: Optional[ProjectManifest], graph: DependencyGraph) -> List[Check]:
    if manifest is None:
        return []
    checks: List[Check] = []
    provides = _providers(graph)

    if not manifest.agents:
        checks.append(
            Check(
                id="agents.emitted",
                category="agents",
                title="Every planned agent exists in the code",
                status=CheckStatus.FAILED,
                summary="This project has no agents, so there is nothing to run.",
                severity=Severity.BLOCKER,
                detail="manifest.agents is empty.",
                fix="Plan at least one agent for the requirement.",
            )
        )
        return checks

    missing = [a for a in manifest.agents if not provides.get(a.factory_name)]
    if missing:
        checks.append(
            Check(
                id="agents.emitted",
                category="agents",
                title="Every planned agent exists in the code",
                status=CheckStatus.FAILED,
                summary=(
                    f"{len(missing)} of {len(manifest.agents)} planned agent"
                    + ("s" if len(missing) != 1 else "")
                    + " "
                    + ("were" if len(missing) != 1 else "was")
                    + " never generated: "
                    + ", ".join(a.label for a in missing)
                    + "."
                ),
                severity=Severity.BLOCKER,
                detail="\n".join(f"{a.name}: expected {a.factory_name}()" for a in missing),
                fix="Generate a builder function for each planned agent.",
                paths=tuple(sorted({a.module.replace(".", "/") + ".py" for a in missing if a.module})),
                evidence=(
                    f"{len(manifest.agents)} planned",
                    f"{len(manifest.agents) - len(missing)} emitted",
                ),
            )
        )
    else:
        checks.append(
            _passed(
                "agents.emitted",
                "agents",
                "Every planned agent exists in the code",
                f"All {len(manifest.agents)} agent"
                + ("s" if len(manifest.agents) != 1 else "")
                + " in the plan have a builder in the generated code.",
                evidence=tuple(f"{a.label} -> {a.factory_name}()" for a in manifest.agents),
            )
        )

    # --- the tools they are given exist ---------------------------------------------------
    dangling: List[str] = []
    for agent in manifest.agents:
        for name in agent.tools:
            if manifest.tool(name) is None:
                dangling.append(f"{agent.label} -> {name}")
    if dangling:
        checks.append(
            Check(
                id="agents.tools_exist",
                category="agents",
                title="Agents only reference tools that exist",
                status=CheckStatus.FAILED,
                summary=(
                    f"{len(dangling)} agent-to-tool reference"
                    + ("s point" if len(dangling) != 1 else " points")
                    + " at a tool this project does not have."
                ),
                severity=Severity.MAJOR,
                detail="\n".join(dangling),
                fix="Either add the missing tool or remove it from the agent.",
            )
        )
    else:
        assigned = sum(len(a.tools) for a in manifest.agents)
        checks.append(
            _passed(
                "agents.tools_exist",
                "agents",
                "Agents only reference tools that exist",
                f"All {assigned} tool assignment"
                + ("s" if assigned != 1 else "")
                + " point at a tool in this project."
                if assigned
                else "No agent needs a tool, so there is nothing to mis-wire.",
                evidence=(f"{assigned} assignments", f"{len(manifest.tools)} tools"),
            )
        )

    return checks


def _tool_checks(manifest: Optional[ProjectManifest], graph: DependencyGraph) -> List[Check]:
    if manifest is None or not manifest.tools:
        return []
    checks: List[Check] = []
    provides = _providers(graph)

    missing = [t for t in manifest.tools if not provides.get(manifest.tool_symbol(t))]
    if missing:
        checks.append(
            Check(
                id="tools.emitted",
                category="tools",
                title="Every planned tool exists in the code",
                status=CheckStatus.FAILED,
                summary=(
                    f"{len(missing)} of {len(manifest.tools)} planned tool"
                    + ("s" if len(missing) != 1 else "")
                    + " "
                    + ("were" if len(missing) != 1 else "was")
                    + " never generated: "
                    + ", ".join(t.label for t in missing)
                    + "."
                ),
                severity=Severity.MAJOR,
                detail="\n".join(
                    f"{t.name}: expected {manifest.tool_symbol(t)} in {t.module or '?'}"
                    for t in missing
                ),
                fix="Generate each planned tool, or remove it from the plan.",
                evidence=(
                    f"{len(manifest.tools)} planned",
                    f"{len(manifest.tools) - len(missing)} emitted",
                ),
            )
        )
    else:
        checks.append(
            _passed(
                "tools.emitted",
                "tools",
                "Every planned tool exists in the code",
                f"All {len(manifest.tools)} tool"
                + ("s" if len(manifest.tools) != 1 else "")
                + " in the plan exist in the generated code.",
                evidence=tuple(
                    f"{t.label} -> {manifest.tool_symbol(t)}" for t in manifest.tools
                ),
            )
        )

    # --- the registry the agents are wired through ----------------------------------------
    if provides.get("TOOLS") or provides.get("all_tools"):
        checks.append(
            _passed(
                "tools.registry",
                "tools",
                "Tools are registered where agents can find them",
                "The project exposes its tools through a registry the agent builders read.",
                paths=tuple(provides.get("TOOLS") or provides.get("all_tools") or ()),
            )
        )
    else:
        checks.append(
            Check(
                id="tools.registry",
                category="tools",
                title="Tools are registered where agents can find them",
                status=CheckStatus.FAILED,
                summary=(
                    "The tools exist but nothing collects them, so the agents cannot be given "
                    "them."
                ),
                severity=Severity.MAJOR,
                detail="No module defines TOOLS or all_tools().",
                fix="Emit a tool registry exposing TOOLS and all_tools().",
            )
        )

    # --- and which of them are real ------------------------------------------------------
    stubs = [t for t in manifest.tools if not t.implemented]
    if stubs:
        checks.append(
            Check(
                id="tools.implemented",
                category="tools",
                title="Tools do real work",
                status=CheckStatus.WARNING,
                summary=(
                    f"{len(stubs)} of {len(manifest.tools)} tool"
                    + ("s are" if len(stubs) != 1 else " is")
                    + " a placeholder: "
                    + ", ".join(t.label for t in stubs)
                    + ". "
                    + ("They run" if len(stubs) != 1 else "It runs")
                    + " and "
                    + ("return" if len(stubs) != 1 else "returns")
                    + " a clearly-labelled stand-in answer instead of doing the real thing."
                ),
                severity=Severity.MINOR,
                detail="\n".join(f"{t.name}: {t.purpose}" for t in stubs),
                fix=(
                    "Replace the marked function bodies with a real implementation. The agent "
                    "runs either way, but these answers are not real."
                ),
                paths=tuple(sorted({t.module.replace(".", "/") + ".py" for t in stubs if t.module})),
                evidence=(
                    f"{len(manifest.tools) - len(stubs)} implemented",
                    f"{len(stubs)} stubs",
                ),
            )
        )
    else:
        checks.append(
            _passed(
                "tools.implemented",
                "tools",
                "Tools do real work",
                f"All {len(manifest.tools)} tools have a real implementation.",
            )
        )

    return checks


def _workflow_checks(manifest: Optional[ProjectManifest], graph: DependencyGraph) -> List[Check]:
    if manifest is None:
        return []
    checks: List[Check] = []
    provides = _providers(graph)
    workflow = manifest.primary_workflow

    if workflow is None or not workflow.nodes:
        checks.append(
            Check(
                id="workflow.spec",
                category="workflow",
                title="The workflow is fully described",
                status=CheckStatus.FAILED,
                summary=(
                    "This project has no workflow, so there is nothing to show on the workflow "
                    "diagram and no defined order for the agents to run in."
                ),
                severity=Severity.MAJOR,
                detail="manifest.workflows is empty or has no nodes.",
                fix="Plan a workflow with at least one agent step.",
            )
        )
    else:
        faults: List[str] = []
        if not workflow.entry:
            faults.append("no entry node is set")
        elif workflow.node(workflow.entry) is None:
            faults.append(f"the entry node {workflow.entry!r} does not exist")
        dangling = workflow.dangling_edges()
        if dangling:
            faults.append(
                f"{len(dangling)} connection(s) point at a step that does not exist: "
                + ", ".join(f"{e.source}->{e.target}" for e in dangling)
            )
        if not workflow.agent_nodes():
            faults.append("no step is handled by an agent")

        if faults:
            checks.append(
                Check(
                    id="workflow.spec",
                    category="workflow",
                    title="The workflow is fully described",
                    status=CheckStatus.FAILED,
                    summary="The workflow is incomplete: " + "; ".join(faults) + ".",
                    severity=Severity.MAJOR,
                    detail="\n".join(faults),
                    fix="Repair the workflow plan so every connection points at a real step.",
                    evidence=(
                        f"{len(workflow.nodes)} steps",
                        f"{len(workflow.edges)} connections",
                    ),
                )
            )
        else:
            checks.append(
                _passed(
                    "workflow.spec",
                    "workflow",
                    "The workflow is fully described",
                    f"{len(workflow.agent_nodes())} agent step"
                    + ("s" if len(workflow.agent_nodes()) != 1 else "")
                    + f" connected by {len(workflow.edges)} link"
                    + ("s" if len(workflow.edges) != 1 else "")
                    + f", starting at {workflow.entry}.",
                    evidence=(
                        f"{len(workflow.nodes)} steps",
                        f"{len(workflow.edges)} connections",
                        "branching" if workflow.has_branching else "linear",
                    ),
                )
            )

    # --- and it exists in the code --------------------------------------------------------
    absent = [name for name in ("build_workflow", "run_workflow") if not provides.get(name)]
    if absent:
        checks.append(
            Check(
                id="workflow.emitted",
                category="workflow",
                title="The workflow exists in the code",
                status=CheckStatus.FAILED,
                summary=(
                    "The generated code does not build the workflow the plan describes, so the "
                    "agents have no defined order to run in."
                ),
                severity=Severity.BLOCKER,
                detail="Missing: " + ", ".join(f"{n}()" for n in absent),
                fix="Emit build_workflow() and run_workflow() in the workflow module.",
            )
        )
    else:
        checks.append(
            _passed(
                "workflow.emitted",
                "workflow",
                "The workflow exists in the code",
                f"build_workflow() and run_workflow() are defined in "
                f"{provides['run_workflow'][0]}.",
                paths=tuple(provides["run_workflow"]),
            )
        )

    return checks


# ==================================================================== error-handling gates
def _error_handling_checks(
    project: GeneratedProject,
    graph: DependencyGraph,
    entrypoint: Optional[str],
) -> List[Check]:
    checks: List[Check] = []

    # --- failures are not swallowed -------------------------------------------------------
    silent: List[Tuple[str, int, str]] = []
    for generated in _app_files(project):
        tree = _parse(generated.content)
        if tree is None:
            continue
        for line, caught in _silent_handlers(tree):
            silent.append((generated.path, line, caught))

    if silent:
        checks.append(
            Check(
                id="error_handling.no_silent_failures",
                category="error_handling",
                title="Failures are never silently discarded",
                status=CheckStatus.FAILED,
                summary=(
                    f"{len(silent)} place"
                    + ("s" if len(silent) != 1 else "")
                    + " in the code catch"
                    + ("" if len(silent) != 1 else "es")
                    + " an error and do nothing with it. A run can fail there and still look "
                    "like it worked."
                ),
                severity=Severity.MINOR,
                detail="\n".join(
                    f"{path}:{line} catches {caught} and does nothing"
                    for path, line, caught in silent
                ),
                fix="Log the error or re-raise it, so the failure reaches the user.",
                paths=tuple(sorted({path for path, _, _ in silent})),
            )
        )
    else:
        checks.append(
            _passed(
                "error_handling.no_silent_failures",
                "error_handling",
                "Failures are never silently discarded",
                "No part of the code catches an error and ignores it.",
            )
        )

    # --- the entry point reports its own failures ------------------------------------------
    entry = project.get(entrypoint) if entrypoint else None
    tree = _parse(entry.content) if entry is not None else None
    if entry is None or tree is None:
        checks.append(
            Check(
                id="error_handling.entrypoint_guard",
                category="error_handling",
                title="A failed run is explained rather than dumped",
                status=CheckStatus.SKIPPED,
                summary="Not checked, because the entry point could not be read.",
                severity=Severity.MINOR,
                required=False,
            )
        )
    elif _has_try(tree):
        checks.append(
            _passed(
                "error_handling.entrypoint_guard",
                "error_handling",
                "A failed run is explained rather than dumped",
                f"{entry.path} handles its own failures, so a missing key or a provider error "
                "is reported as a message rather than a raw crash.",
                paths=(entry.path,),
            )
        )
    else:
        checks.append(
            Check(
                id="error_handling.entrypoint_guard",
                category="error_handling",
                title="A failed run is explained rather than dumped",
                status=CheckStatus.FAILED,
                summary=(
                    f"{entry.path} does not handle failures, so anything that goes wrong "
                    "reaches the user as a raw Python traceback."
                ),
                severity=Severity.MINOR,
                detail=f"No try/except anywhere in {entry.path}.",
                fix="Wrap the run in a try/except that prints what went wrong and why.",
                paths=(entry.path,),
            )
        )

    return checks


# ============================================================================== test gates
def _test_presence_checks(project: GeneratedProject, expect_tests: bool) -> List[Check]:
    tests = project.test_files()
    if tests:
        return [
            _passed(
                "tests.present",
                "tests",
                "The project ships tests",
                f"{len(tests)} test file"
                + ("s" if len(tests) != 1 else "")
                + " were generated alongside the code.",
                evidence=tuple(f.path for f in tests),
                paths=tuple(f.path for f in tests),
            )
        ]
    if not expect_tests:
        return [
            Check(
                id="tests.present",
                category="tests",
                title="The project ships tests",
                status=CheckStatus.SKIPPED,
                summary="Tests were not requested for this project.",
                severity=Severity.MINOR,
                required=False,
            )
        ]
    return [
        Check(
            id="tests.present",
            category="tests",
            title="The project ships tests",
            status=CheckStatus.FAILED,
            summary=(
                "No tests were generated, so there is no way to tell whether this project "
                "works without running it by hand."
            ),
            severity=Severity.MAJOR,
            detail=f"{len(project.files)} files, none of them under tests/.",
            fix="Generate the test suite for this project.",
        )
    ]


def _test_result_checks(project: GeneratedProject, report: Optional[TestReport]) -> List[Check]:
    """
    What the generated test suite actually did.

    Required, and required in a specific way: when there is no report the checks are
    ``SKIPPED`` *and* required, which blocks readiness. This is section 17's central demand -
    a project is not ready because it was generated, it is ready because its tests ran and
    passed - expressed as data rather than as a comment.
    """
    if report is None:
        return [
            Check(
                id="tests.executed",
                category="tests",
                title="The tests were run",
                status=CheckStatus.SKIPPED,
                summary="The test suite has not been run yet, so nothing about this project "
                "has been verified by running it.",
                severity=Severity.MAJOR,
                detail="No TestReport was supplied to validation.",
                fix="Run the generated test suite.",
                required=True,
            )
        ]

    checks: List[Check] = []
    counts = (
        f"{report.passed_count} passed",
        f"{report.failed_count} failed",
        f"{report.skipped_count} skipped",
        f"{report.error_count} errors",
    )

    if not report.ran:
        checks.append(
            Check(
                id="tests.executed",
                category="tests",
                title="The tests were run",
                status=CheckStatus.SKIPPED,
                summary=(
                    report.unavailable_reason
                    or "The test suite could not be run, so nothing has been verified by "
                    "running it."
                ),
                severity=Severity.MAJOR,
                detail=(report.stderr or report.stdout or "")[-4000:],
                fix=(
                    "Fix what stops the suite from running - usually a missing dependency or a "
                    "collection error - and run it again."
                ),
                evidence=(f"status={report.status}", f"exit_code={report.exit_code}"),
            )
        )
        return checks

    checks.append(
        _passed(
            "tests.executed",
            "tests",
            "The tests were run",
            f"The generated test suite ran in {report.duration_s:.1f}s.",
            evidence=counts,
        )
    )

    if report.ok:
        checks.append(
            _passed(
                "tests.passed",
                "tests",
                "The tests pass",
                f"All {report.passed_count} test"
                + ("s" if report.passed_count != 1 else "")
                + " that ran passed.",
                evidence=counts,
            )
        )
    else:
        checks.append(
            Check(
                id="tests.passed",
                category="tests",
                title="The tests pass",
                status=CheckStatus.FAILED,
                summary=(
                    f"{report.failed_count + report.error_count} of "
                    f"{report.passed_count + report.failed_count + report.error_count} tests "
                    "failed, so this project does not work yet."
                    if not report.timed_out
                    else f"The test suite was still running after "
                    f"{report.duration_s:.0f}s and had to be stopped - something in it hangs."
                ),
                severity=Severity.BLOCKER,
                detail="\n".join(report.failed_tests[:20])
                + ("\n\n" + report.stdout[-6000:] if report.stdout else ""),
                fix=(
                    "Find the cause of each failure and fix it. Do not raise the timeout or "
                    "remove the test."
                ),
                evidence=counts,
            )
        )

    # --- a suite where everything skipped has verified nothing -----------------------------
    if report.passed_count >= 1:
        checks.append(
            _passed(
                "tests.meaningful",
                "tests",
                "The tests actually checked something",
                f"{report.passed_count} test"
                + ("s" if report.passed_count != 1 else "")
                + " ran to completion.",
                evidence=counts,
            )
        )
    else:
        checks.append(
            Check(
                id="tests.meaningful",
                category="tests",
                title="The tests actually checked something",
                status=CheckStatus.SKIPPED,
                summary=(
                    f"The suite finished without failing, but every one of its "
                    f"{report.skipped_count} tests was skipped - nothing was actually checked."
                ),
                severity=Severity.MAJOR,
                detail="\n".join(report.notes) or report.stdout[-2000:],
                fix=(
                    "Find why the tests skip - usually a missing framework package - and make "
                    "them runnable."
                ),
                evidence=counts,
            )
        )

    return checks


# =========================================================================== runtime gates
def _runtime_checks(
    project: GeneratedProject,
    report: Optional[TestReport],
    runtime: Optional[RunResult],
    *,
    require_offline: bool = True,
) -> List[Check]:
    checks: List[Check] = []

    # --- was the workflow itself exercised, not just imported? -----------------------------
    exercising: List[str] = []
    for generated in project.test_files():
        tree = _parse(generated.content)
        if tree is None:
            continue
        called = _called_names(tree)
        if called & {*RUN_ENTRY_NAMES, "build_workflow"}:
            exercising.append(generated.path)

    if not require_offline:
        checks.append(
            Check(
                id="runtime.offline",
                category="runtime",
                title="The workflow was run without a provider",
                status=CheckStatus.SKIPPED,
                summary="Offline workflow tests were skipped for this run.",
                severity=Severity.INFO,
                required=False,
            )
        )
    elif not exercising:
        checks.append(
            Check(
                id="runtime.offline",
                category="runtime",
                title="The workflow was run without a provider",
                status=CheckStatus.FAILED,
                summary=(
                    "No generated test runs the workflow, so passing tests would only prove "
                    "the files import - not that the agent works."
                ),
                severity=Severity.MAJOR,
                detail="No test file calls run_workflow(), run(), main() or build_workflow().",
                fix="Add a test that runs the workflow against the offline stand-in model.",
            )
        )
    elif report is None or not report.ran:
        checks.append(
            Check(
                id="runtime.offline",
                category="runtime",
                title="The workflow was run without a provider",
                status=CheckStatus.SKIPPED,
                summary=(
                    "A test that runs the workflow exists, but the suite has not been run, so "
                    "the workflow has not actually been executed."
                ),
                severity=Severity.MAJOR,
                detail="Tests that exercise the workflow: " + ", ".join(exercising),
                fix="Run the generated test suite.",
                paths=tuple(exercising),
            )
        )
    elif report.ok and report.passed_count >= 1:
        checks.append(
            _passed(
                "runtime.offline",
                "runtime",
                "The workflow was run without a provider",
                "The workflow was built and executed against a stand-in model, with no network "
                "and no API key, and it completed.",
                evidence=tuple(exercising),
                paths=tuple(exercising),
            )
        )
    elif report.ok:
        # Ran, nothing failed, and nothing passed either - every test skipped. Reporting that as
        # "the workflow did not complete" would be inventing a failure that was never observed,
        # which is the same dishonesty as reporting it as a pass, only in the other direction.
        # The truthful statement is that the workflow still has not been executed.
        checks.append(
            Check(
                id="runtime.offline",
                category="runtime",
                title="The workflow was run without a provider",
                status=CheckStatus.SKIPPED,
                summary=(
                    "The test that runs the workflow was skipped rather than executed, so the "
                    "workflow still has not been run even once."
                ),
                severity=Severity.MAJOR,
                detail=(
                    f"{report.skipped_count} skipped, {report.passed_count} passed. "
                    "Tests that would exercise the workflow: " + ", ".join(exercising)
                ),
                fix=(
                    "Find why the test skips - usually a missing framework package - and make it "
                    "runnable."
                ),
                paths=tuple(exercising),
            )
        )
    else:
        checks.append(
            Check(
                id="runtime.offline",
                category="runtime",
                title="The workflow was run without a provider",
                status=CheckStatus.FAILED,
                summary="The workflow was executed by the test suite and did not complete.",
                severity=Severity.BLOCKER,
                detail="\n".join(report.failed_tests[:20]),
                fix="Fix the failing workflow test.",
                paths=tuple(exercising),
            )
        )

    # --- a real, provider-backed run -------------------------------------------------------
    #
    # Not required, and the title says so. A live call costs money and needs a credential this
    # machine may not have; gating readiness behind it would mean no project is ever ready in
    # CI. But a project that *has* had a real run is in a different, stronger state than one
    # that has not, and flattening the two would be the sort of overclaim section 59 rules out.
    if runtime is None:
        checks.append(
            Check(
                id="runtime.live",
                category="runtime",
                title="A real run against the provider (optional)",
                status=CheckStatus.SKIPPED,
                summary=(
                    "The agent has not been run against a real model yet. Use the playground to "
                    "try it - this is not required for the project to be ready."
                ),
                severity=Severity.INFO,
                required=False,
            )
        )
    elif runtime.ok:
        checks.append(
            _passed(
                "runtime.live",
                "runtime",
                "A real run against the provider (optional)",
                f"The agent ran against the real model in {runtime.duration_s:.1f}s and "
                "returned an answer.",
                evidence=(
                    f"{len(runtime.steps)} steps",
                    f"{len(runtime.tool_calls)} tool calls",
                    f"entry={runtime.entry or '-'}",
                ),
            )
        )
    else:
        message = (runtime.error or {}).get("message") if runtime.error else ""
        checks.append(
            Check(
                id="runtime.live",
                category="runtime",
                title="A real run against the provider (optional)",
                status=CheckStatus.FAILED,
                summary=(
                    "The last real run of this agent did not succeed: "
                    + (message or f"it ended with status {runtime.status.value}.")
                ),
                # MINOR, not MAJOR, and ``required=False`` is not what does the work here.
                # Check.blocking asks severity.blocks_release for a FAILED check, and that is
                # true for BLOCKER and MAJOR regardless of ``required`` - which meant a rejected
                # API key dropped a fully validated project out of READY, exactly the outcome
                # the comment above this branch says must not happen. MINOR keeps the failure
                # visible in failed() and out of blocking() and unmet().
                severity=Severity.MINOR,
                detail=(runtime.stderr or runtime.stdout or "")[-6000:],
                fix="Read the technical details for the underlying error and fix it.",
                required=False,
            )
        )

    return checks


# =========================================================================== public surface
def static_checks(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest] = None,
    graph: Optional[DependencyGraph] = None,
    *,
    secrets: Sequence[str] = (),
    expect_tests: bool = True,
) -> List[Check]:
    """
    Every gate that can be decided by reading the project, with nothing executed.

    Split out from :func:`validate_project` because the repair loop runs this after every
    rewrite: it has to be cheap, it has to be safe on code that has not been screened, and it
    must not need a test report to say something useful.
    """
    if graph is None:
        graph = _safe_graph(project, manifest)[0]

    checks: List[Check] = []
    checks.extend(_configuration_checks(project, manifest, graph, secrets))
    checks.extend(_dependency_checks(project, manifest, graph))
    checks.extend(_graph_checks(graph))
    checks.extend(_structure_checks(project, manifest, graph))
    entry_checks, entrypoint = _entrypoint_checks(project, manifest, graph)
    checks.extend(entry_checks)
    checks.extend(_agent_checks(manifest, graph))
    checks.extend(_tool_checks(manifest, graph))
    checks.extend(_workflow_checks(manifest, graph))
    checks.extend(_error_handling_checks(project, graph, entrypoint))
    checks.extend(_test_presence_checks(project, expect_tests))
    return _sorted(checks)


def _safe_graph(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest],
) -> Tuple[DependencyGraph, Optional[str]]:
    """
    Build the import graph, converting a crash into a reportable fact.

    :func:`build_graph` is not supposed to raise - it catches syntax errors itself - but it
    walks model-authored source, and validation is the last thing that should ever be the
    reason a pipeline run dies. If it does raise, the caller gets an empty graph and the text
    of the exception to show.
    """
    try:
        return build_graph(project, manifest), None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return DependencyGraph(), f"{type(exc).__name__}: {exc}"


def validate_project(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest] = None,
    *,
    graph: Optional[DependencyGraph] = None,
    tests: Optional[TestReport] = None,
    runtime: Optional[RunResult] = None,
    extra_checks: Sequence[Check] = (),
    secrets: Sequence[str] = (),
    expect_tests: bool = True,
    require_test_run: bool = True,
) -> ValidationReport:
    """
    Decide whether this project is ready, and say exactly why.

    Args:
        project: The generated project.
        manifest: The architecture plan, when there is one. Without it the plan-comparison
            checks report ``SKIPPED`` and do not block - a project restored from an older
            record should not be declared broken because its plan was not stored.
        graph: A previously built import graph, to avoid parsing every file twice.
        tests: The result of running the generated suite. ``None`` blocks readiness.
        runtime: A real provider-backed run, if one has happened. Never blocks.
        extra_checks: Additional checks from elsewhere - requirement coverage, in particular,
            which needs the original request and so cannot be computed from the project alone.
        secrets: Credential values the platform holds, so the leak check can prove none of
            them reached the source. Never stored on the report.
        expect_tests: False for a project generated deliberately without tests.
        require_test_run: False when the caller skipped executing the suite. Presence of
            tests is still checked via ``expect_tests``; execution and offline-runtime
            gates are waived so they cannot block readiness.

    Returns:
        A :class:`ValidationReport`. This function does not raise: a validator that throws
        turns a diagnosable problem into an opaque one.
    """
    started = time.perf_counter()
    notes: List[str] = []

    if graph is None:
        graph, failure = _safe_graph(project, manifest)
        if failure:
            notes.append(f"The import graph could not be built: {failure}")
    if manifest is None:
        notes.append(
            "No architecture plan was available, so the plan-comparison checks were skipped."
        )
    notes.extend(graph.notes)

    checks = static_checks(
        project,
        manifest,
        graph,
        secrets=secrets,
        expect_tests=expect_tests,
    )
    if require_test_run:
        checks.extend(_test_result_checks(project, tests))
        checks.extend(_runtime_checks(project, tests, runtime))
    else:
        checks.extend(_runtime_checks(project, tests, runtime, require_offline=False))
    checks.extend(extra_checks)

    entrypoint_check = next((c for c in checks if c.id == "entrypoint.declared"), None)
    entrypoint = entrypoint_check.paths[0] if entrypoint_check and entrypoint_check.paths else None

    return ValidationReport(
        checks=_sorted(checks),
        notes=notes,
        entrypoint=entrypoint,
        duration_s=time.perf_counter() - started,
        graph=graph,
    )
