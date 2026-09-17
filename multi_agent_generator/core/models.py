# multi_agent_generator/core/models.py
"""
Domain types for the generation pipeline.

Everything the pipeline produces is a structured object defined here, not a dict passed
between functions and not a single blob of source text. Two reasons, both of which were
real problems:

*A generated project is a directory, not a string.* The old flow produced one Python
string and printed it. That works only while the answer to "what should this agent be" is
"one file", and it stops working the moment the honest answer includes a
``requirements.txt``, a ``README``, a test suite and a package layout. :class:`GeneratedFile`
and :class:`GeneratedProject` model that shape, so the CLI can write it to disk, the API
can serialise it, and the frontend's Files tab can show a real tree.

*A stage either succeeded or it did not.* Review, test and execution results all carry an
explicit outcome and, when they failed, the specific reasons. A stage is never allowed to
report success with an empty payload - :class:`ReviewResult` and :class:`TestReport` are
constructed from parsed data or an error, never defaulted into looking fine.
"""
from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - imported for annotations only
    # Both of these import this module at run time, so importing them back would be a cycle.
    # The pipeline result needs to *hold* them, not to know anything about them, which is
    # exactly the case a deferred annotation is for.
    from .manifest import ProjectManifest
    from .repair import RepairOutcome
    from .validation import ValidationReport

__all__ = [
    "Stage",
    "ProjectStatus",
    "RunStatus",
    "Severity",
    "GeneratedFile",
    "GeneratedProject",
    "RequirementAnalysis",
    "FrameworkChoice",
    "ReviewIssue",
    "ReviewResult",
    "TestReport",
    "RunResult",
    "IterationRecord",
    "PipelineEvent",
    "PipelineResult",
    "relative_path",
    "utcnow",
]


def utcnow() -> datetime:
    """Timezone-aware UTC now. Naive datetimes in a database are a bug in waiting."""
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    """
    Pipeline stages, in the order they run.

    The values are the tags that appear in structured logs (``[GENERATION]``,
    ``[REVIEW]`` ...) and the keys the frontend uses to light up its progress list, so
    these strings are part of the API contract.
    """

    ANALYSIS = "analysis"
    FRAMEWORK_SELECTION = "framework_selection"
    #: Deciding what files exist before writing any of them. Section 5 makes the plan a
    #: first-class artefact rather than something implied by the output, so it gets a stage
    #: of its own: the user can see the architecture was decided, and when generation fails
    #: the trace says whether it failed planning or emission.
    PLANNING = "planning"
    GENERATION = "generation"
    REVIEW = "review"
    IMPROVEMENT = "improvement"
    TEST = "test"
    #: Rewriting the project to clear what validation found. Distinct from IMPROVEMENT:
    #: improvement acts on a reviewer's opinion and may make things worse, repair acts on a
    #: failed gate and is re-validated before it is accepted.
    REPAIR = "repair"
    EXECUTION = "execution"
    VALIDATION = "validation"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def log_tag(self) -> str:
        return f"[{self.value.upper()}]"


class ProjectStatus(str, Enum):
    """Lifecycle of a project as the UI understands it."""

    DRAFT = "draft"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    IMPROVING = "improving"
    TESTING = "testing"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (ProjectStatus.READY, ProjectStatus.FAILED)


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self is not RunStatus.PENDING and self is not RunStatus.RUNNING


class Severity(str, Enum):
    """How badly a review issue matters."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"

    @property
    def blocks_release(self) -> bool:
        return self in (Severity.BLOCKER, Severity.MAJOR)

    @classmethod
    def parse(cls, value: Any) -> "Severity":
        """
        Coerce whatever a model wrote into a known severity.

        Models are asked for one of these four words and mostly comply, but "critical",
        "high" and "error" all turn up. Mapping them is better than either crashing or
        silently downgrading an unrecognised severity to ``info``, which would let a
        blocker through the release gate.
        """
        text = str(value or "").strip().lower()
        aliases = {
            "critical": cls.BLOCKER,
            "fatal": cls.BLOCKER,
            "high": cls.MAJOR,
            "error": cls.MAJOR,
            "medium": cls.MAJOR,
            "moderate": cls.MINOR,
            "low": cls.MINOR,
            "warning": cls.MINOR,
            "nit": cls.MINOR,
            "note": cls.INFO,
            "suggestion": cls.INFO,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError:
            # Unknown severity is treated as MAJOR, not INFO. An unreadable severity on a
            # real problem must not be the reason a broken project ships.
            return cls.MAJOR


# ===================================================================== generated project
@dataclass
class GeneratedFile:
    """
    One file in a generated project.

    ``path`` is always POSIX-style and relative to the project root. Storing it that way
    means the same project record produces identical trees on Windows and Linux, and that
    the API never leaks a host path.
    """

    path: str
    content: str
    #: Marks the entry point, so "Run Agent" knows what to execute.
    is_entrypoint: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        self.path = relative_path(self.path)

    @property
    def language(self) -> str:
        """Best-guess language tag for syntax highlighting in the Files tab."""
        suffix = posixpath.splitext(self.path)[1].lower()
        return {
            ".py": "python",
            ".md": "markdown",
            ".txt": "text",
            ".json": "json",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".ini": "ini",
            ".cfg": "ini",
            ".env": "bash",
            ".sh": "bash",
        }.get(suffix, "text")

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1 if self.content else 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "language": self.language,
            "lines": self.line_count,
            "is_entrypoint": self.is_entrypoint,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratedFile":
        """
        Rebuild from :meth:`as_dict`.

        ``language`` and ``lines`` are derived properties and are deliberately ignored on the
        way back in - recomputing them keeps a hand-edited or older record from carrying a
        stale value that disagrees with its own content.
        """
        return cls(
            path=str(data.get("path") or ""),
            content=str(data.get("content") or ""),
            is_entrypoint=bool(data.get("is_entrypoint", False)),
            description=str(data.get("description") or ""),
        )


def relative_path(raw: str) -> str:
    """
    Normalise a generated path and refuse to let it escape the project root.

    This is a security boundary, not tidiness. Generated file paths come from model
    output, and a model that emits ``../../.ssh/authorized_keys`` must not be able to
    reach outside the workspace when the project is written to disk. Absolute paths,
    drive letters and ``..`` segments are all stripped here, once, so no writer
    downstream has to remember to check.

    Note what is *not* stripped: a leading dot that belongs to a filename. ``.env.example``
    and ``.dockerignore`` are ordinary files a project needs, and the obvious spelling of
    this function - ``path.lstrip("./")`` - eats their names, because ``lstrip`` takes a set
    of characters rather than a prefix. :class:`~multi_agent_generator.core.manifest.FileSpec`
    was written that way and turned every planned ``.env.example`` into ``env.example``,
    which made the assembler reject its own plan for every framework. Both layers now share
    this one function so the plan and the emitted file cannot disagree about a path again.
    """
    text = str(raw or "").strip().replace("\\", "/")
    # Drop a Windows drive letter or a leading slash: both make the path absolute.
    if len(text) > 1 and text[1] == ":":
        text = text[2:]
    text = text.lstrip("/")
    parts = [p for p in text.split("/") if p not in ("", ".", "..")]
    if not parts:
        return "unnamed.txt"
    return posixpath.join(*parts)


@dataclass
class GeneratedProject:
    """
    A complete, runnable project: files, dependencies and how to run it.

    This is what the reviewer inspects, what the sandbox writes to disk and what the
    frontend renders. It deliberately knows nothing about *where* it will be written.
    """

    framework: str
    provider: str
    model: str
    files: List[GeneratedFile] = field(default_factory=list)
    #: pip requirement specifiers, as strings.
    dependencies: List[str] = field(default_factory=list)
    #: The agent configuration the code was rendered from, kept for the Architecture tab.
    config: Dict[str, Any] = field(default_factory=dict)
    entrypoint: Optional[str] = None
    run_instructions: str = ""
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------------ accessors
    def add_file(
        self,
        path: str,
        content: str,
        *,
        is_entrypoint: bool = False,
        description: str = "",
    ) -> GeneratedFile:
        """Add or replace a file. Replacing matters: the improver rewrites files by path."""
        generated = GeneratedFile(
            path=path,
            content=content,
            is_entrypoint=is_entrypoint,
            description=description,
        )
        for index, existing in enumerate(self.files):
            if existing.path == generated.path:
                self.files[index] = generated
                break
        else:
            self.files.append(generated)
        if is_entrypoint:
            self.entrypoint = generated.path
        return generated

    def get(self, path: str) -> Optional[GeneratedFile]:
        wanted = relative_path(path)
        return next((f for f in self.files if f.path == wanted), None)

    def source_files(self) -> List[GeneratedFile]:
        return [f for f in self.files if f.path.endswith(".py")]

    def test_files(self) -> List[GeneratedFile]:
        """Python files that pytest would collect."""
        return [
            f
            for f in self.source_files()
            if posixpath.basename(f.path).startswith("test_")
            or f.path.startswith("tests/")
        ]

    @property
    def entrypoint_file(self) -> Optional[GeneratedFile]:
        if self.entrypoint:
            return self.get(self.entrypoint)
        return next((f for f in self.files if f.is_entrypoint), None)

    @property
    def total_lines(self) -> int:
        return sum(f.line_count for f in self.files)

    def tree(self) -> List[Dict[str, Any]]:
        """Flat, sorted listing for the Files tab. The UI builds the tree from paths."""
        return sorted(
            (
                {
                    "path": f.path,
                    "language": f.language,
                    "lines": f.line_count,
                    "is_entrypoint": f.is_entrypoint,
                }
                for f in self.files
            ),
            key=lambda item: item["path"],
        )

    def as_dict(self, include_content: bool = True) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "provider": self.provider,
            "model": self.model,
            "entrypoint": self.entrypoint or (
                self.entrypoint_file.path if self.entrypoint_file else None
            ),
            "dependencies": list(self.dependencies),
            "run_instructions": self.run_instructions,
            "notes": list(self.notes),
            "config": self.config,
            "file_count": len(self.files),
            "total_lines": self.total_lines,
            "files": (
                [f.as_dict() for f in self.files] if include_content else self.tree()
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratedProject":
        """
        Rebuild from :meth:`as_dict`.

        This is what makes a *saved* project runnable. The Playground executes a project the
        user generated in an earlier session, which means the stored JSON has to round-trip
        back into the same object the sandbox would have received live - otherwise "Run
        Agent" would only ever work in the session that created the project.

        Only records written with ``include_content=True`` can be run: the summary form
        stores a file tree without contents, so there is no source to execute. That case
        surfaces as a project with empty files rather than a crash here, and the caller
        reports it as "nothing to run".
        """
        files = [
            GeneratedFile.from_dict(item)
            for item in (data.get("files") or [])
            if isinstance(item, Mapping) and "content" in item
        ]
        return cls(
            framework=str(data.get("framework") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            files=files,
            dependencies=[str(d) for d in (data.get("dependencies") or [])],
            config=dict(data.get("config") or {}),
            entrypoint=(str(data["entrypoint"]) if data.get("entrypoint") else None),
            run_instructions=str(data.get("run_instructions") or ""),
            notes=[str(n) for n in (data.get("notes") or [])],
        )


# ==================================================================== analysis/selection
@dataclass
class RequirementAnalysis:
    """
    What the user actually asked for, made explicit.

    Produced before any framework is chosen, because the choice depends on this. Keeping
    it as its own record means the Overview tab can show *why* the pipeline did what it
    did, rather than presenting generated code as if it arrived by magic.
    """

    requirement: str
    summary: str = ""
    goals: List[str] = field(default_factory=list)
    #: Distinct roles the work implies, e.g. "researcher", "writer".
    suggested_roles: List[str] = field(default_factory=list)
    suggested_tools: List[str] = field(default_factory=list)
    #: True when the work has steps that must happen in order.
    needs_sequential_steps: bool = False
    #: True when a step's outcome should change what happens next.
    needs_branching: bool = False
    #: True when a coordinator must delegate to specialists.
    needs_delegation: bool = False
    #: True when state has to persist across turns.
    needs_state: bool = False
    needs_human_input: bool = False
    complexity: str = "moderate"
    risks: List[str] = field(default_factory=list)
    #: False when the analysis came from the offline heuristic rather than a model.
    from_model: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requirement": self.requirement,
            "summary": self.summary,
            "goals": list(self.goals),
            "suggested_roles": list(self.suggested_roles),
            "suggested_tools": list(self.suggested_tools),
            "needs_sequential_steps": self.needs_sequential_steps,
            "needs_branching": self.needs_branching,
            "needs_delegation": self.needs_delegation,
            "needs_state": self.needs_state,
            "needs_human_input": self.needs_human_input,
            "complexity": self.complexity,
            "risks": list(self.risks),
            "from_model": self.from_model,
        }


@dataclass
class FrameworkChoice:
    """
    The selected framework, with its justification.

    The brief asks for exactly this shape - ``{"framework", "reason", "confidence"}`` -
    because a recommendation the user cannot interrogate is not a recommendation. When the
    user named a framework explicitly, ``user_specified`` is True and confidence is 1.0:
    respecting an explicit choice is not a judgement call.
    """

    framework: str
    reason: str
    confidence: float = 0.5
    user_specified: bool = False
    #: Frameworks that were plausible but not chosen, with a one-line note each.
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Clamp rather than validate: a model that returns 1.4 or -0.2 has expressed
        # "certain" and "no idea", and refusing the whole selection over it would be
        # pedantic in a way that costs the user a run.
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "user_specified": self.user_specified,
            "alternatives": list(self.alternatives),
        }


# =============================================================================== review
@dataclass
class ReviewIssue:
    """One problem the reviewer found."""

    #: One of the review dimensions, e.g. "security", "requirements_coverage".
    category: str
    severity: Severity
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: str = ""

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["ReviewIssue"]:
        """
        Build from whatever the reviewer emitted, or None if there is nothing usable.

        Returning None for an unusable entry - rather than a placeholder issue - keeps
        junk out of the issue list without inventing a problem that was not reported.
        """
        if isinstance(raw, str):
            text = raw.strip()
            return cls(category="general", severity=Severity.MAJOR, message=text) if text else None
        if not isinstance(raw, dict):
            return None
        message = str(
            raw.get("message") or raw.get("issue") or raw.get("description") or ""
        ).strip()
        if not message:
            return None
        line = raw.get("line")
        try:
            line_no = int(line) if line is not None else None
        except (TypeError, ValueError):
            line_no = None
        return cls(
            category=str(raw.get("category") or raw.get("dimension") or "general").strip(),
            severity=Severity.parse(raw.get("severity")),
            message=message,
            file=(str(raw["file"]).strip() or None) if raw.get("file") else None,
            line=line_no,
            suggestion=str(raw.get("suggestion") or raw.get("fix") or "").strip(),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "file": self.file,
            "line": self.line,
            "suggestion": self.suggestion,
        }


@dataclass
class ReviewResult:
    """
    The judge agent's verdict.

    ``passed`` is computed, never taken on trust from the model. A reviewer that says
    ``"passed": true`` while also listing a blocker is contradicting itself, and the safe
    reading is the blocker. :meth:`decide` applies that rule in one place.
    """

    score: float
    passed: bool
    summary: str = ""
    issues: List[ReviewIssue] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    #: Per-dimension scores, e.g. {"security": 9.0, "test_coverage": 6.5}.
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    #: False when the verdict came from static checks rather than a model.
    from_model: bool = True

    def __post_init__(self) -> None:
        self.score = max(0.0, min(10.0, float(self.score)))

    @property
    def blockers(self) -> List[ReviewIssue]:
        return [i for i in self.issues if i.severity is Severity.BLOCKER]

    @property
    def blocking_issues(self) -> List[ReviewIssue]:
        return [i for i in self.issues if i.severity.blocks_release]

    def decide(self, pass_score: float) -> bool:
        """
        Apply the release gate: a good score *and* no blocker.

        Both halves matter. Score alone lets a single catastrophic problem average away
        against nine tidy dimensions; blockers alone would pass code that is uniformly
        mediocre with nothing individually damning.
        """
        self.passed = self.score >= pass_score and not self.blockers
        return self.passed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "passed": self.passed,
            "summary": self.summary,
            "issues": [i.as_dict() for i in self.issues],
            "required_changes": list(self.required_changes),
            "dimension_scores": {k: round(v, 2) for k, v in self.dimension_scores.items()},
            "blocker_count": len(self.blockers),
            "from_model": self.from_model,
        }


# ====================================================================== test / execution
@dataclass
class TestReport:
    """
    Result of running a generated project's own test suite in the sandbox.

    ``timed_out`` is a first-class field rather than an inference from the exit code,
    because the two mean different things to the improver: a failing assertion is a bug in
    generated logic, while a timeout is usually a hang, and telling them apart is what
    stops the pipeline from "fixing" a test that was never wrong.
    """

    status: str  # "passed" | "failed" | "error" | "skipped" | "timeout"
    exit_code: Optional[int] = None
    timed_out: bool = False
    duration_s: float = 0.0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    failed_tests: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    #: Set when the suite could not be run at all (no tests, install failure...).
    unavailable_reason: Optional[str] = None
    #: Context about the run that is not a pass/fail verdict - most usefully, requirement lines
    #: that were refused before installing. Kept separate from ``unavailable_reason`` because a
    #: note must not make a suite that genuinely ran look like one that did not: an import error
    #: caused by a refused ``git+https://...`` line is far easier to understand with the refusal
    #: shown next to it, and silently dropping that context is how a clear cause becomes a
    #: mysterious ``ModuleNotFoundError``.
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "passed" and not self.timed_out

    @property
    def ran(self) -> bool:
        """Whether the suite actually executed. A skipped suite is not a green suite."""
        return self.status in ("passed", "failed") and self.unavailable_reason is None

    def as_dict(self, max_output: int = 20000) -> Dict[str, Any]:
        return {
            "status": self.status,
            # ``ok`` and ``ran`` are both serialised because a client cannot derive them
            # safely: "did not run" and "ran and failed" are different outcomes with
            # different fixes, and re-implementing that rule in the UI is how the two end up
            # disagreeing about whether a suite is green.
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "ran": self.ran,
            "counts": {
                "passed": self.passed_count,
                "failed": self.failed_count,
                "skipped": self.skipped_count,
                "errors": self.error_count,
            },
            "failed_tests": list(self.failed_tests),
            "stdout": _tail(self.stdout, max_output),
            "stderr": _tail(self.stderr, max_output),
            "unavailable_reason": self.unavailable_reason,
            "notes": list(self.notes),
        }


@dataclass
class RunResult:
    """One execution of a generated agent, from the Playground or a validation run."""

    status: RunStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    duration_s: float = 0.0
    #: The agent's answer, extracted from stdout when the harness could isolate it.
    output: Optional[str] = None
    #: What the agent actually did, one entry per step, taken from the framework's own record
    #: of the run. Empty when the project's run function returned only an answer - which is
    #: reported as "no steps recorded" rather than being filled in with a plausible guess.
    steps: List[Dict[str, Any]] = field(default_factory=list)
    #: Tool invocations the run recorded: name, arguments and result per call.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    #: Which function in the generated project the harness called. Useful when a run produced
    #: nothing, because it distinguishes "no run function found" from "the run function failed".
    entry: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    usage: Dict[str, Optional[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is RunStatus.SUCCEEDED

    def as_dict(self, max_output: int = 20000) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "ok": self.ok,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "output": self.output,
            "steps": [dict(s) for s in self.steps],
            "tool_calls": [dict(c) for c in self.tool_calls],
            "entry": self.entry,
            "stdout": _tail(self.stdout, max_output),
            "stderr": _tail(self.stderr, max_output),
            "error": self.error,
            "usage": dict(self.usage),
        }


def _tail(text: str, limit: int) -> str:
    """
    Keep the end of a long output, with a marker saying what was dropped.

    The end is the useful half: tracebacks, assertion diffs and pytest summaries all live
    there. Silently truncating from the front would hide the fact that anything was cut,
    which is how a "clean" log ends up missing the actual failure.
    """
    if not text or len(text) <= limit:
        return text or ""
    dropped = len(text) - limit
    return f"... [{dropped} earlier characters omitted] ...\n{text[-limit:]}"


# ============================================================================= pipeline
@dataclass
class IterationRecord:
    """
    One pass of the review -> test -> validate -> repair loop.

    All four results are optional and independently so, because a pass can legitimately stop
    at any of them: a project whose imports do not resolve never reaches the test runner, and
    recording an absent stage as a passing one is the fake-green outcome the brief rejects.
    """

    index: int
    review: Optional[ReviewResult] = None
    tests: Optional[TestReport] = None
    #: The gate report for this pass. Holds the object, not a summary, so the API and the
    #: Testing tab render the same checks the loop actually decided on.
    validation: Optional["ValidationReport"] = None
    #: What the repair engine did this pass, when it ran.
    repair: Optional["RepairOutcome"] = None
    #: A real provider-backed run of the project, when one was attempted.
    runtime: Optional[RunResult] = None
    changes_applied: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=utcnow)
    completed_at: Optional[datetime] = None

    @property
    def duration_s(self) -> float:
        end = self.completed_at or utcnow()
        return max(0.0, (end - self.started_at).total_seconds())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "review": self.review.as_dict() if self.review else None,
            "tests": self.tests.as_dict() if self.tests else None,
            "validation": self.validation.as_dict() if self.validation else None,
            "repair": self.repair.as_dict() if self.repair else None,
            "runtime": self.runtime.as_dict() if self.runtime else None,
            "changes_applied": list(self.changes_applied),
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class PipelineEvent:
    """
    One structured log line.

    The fields are the ones section 22 of the brief asks for. It is a dataclass rather
    than a formatted string so the same event can be written to a log, stored on the run
    and streamed to the frontend without being re-parsed at each hop.
    """

    stage: Stage
    status: str  # "started" | "succeeded" | "failed" | "info" | "warning"
    message: str
    timestamp: datetime = field(default_factory=utcnow)
    duration_s: Optional[float] = None
    project_id: Optional[str] = None
    run_id: Optional[str] = None
    iteration: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        return {"failed": "ERROR", "warning": "WARNING"}.get(self.status, "INFO")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "status": self.status,
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "duration_s": None if self.duration_s is None else round(self.duration_s, 3),
            "project_id": self.project_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "error": self.error,
            "data": self.data,
        }

    def render(self) -> str:
        """Human-readable single line, for a terminal or a plain log file."""
        head = f"{self.timestamp.strftime('%H:%M:%S')} {self.stage.log_tag}"
        took = f" ({self.duration_s:.2f}s)" if self.duration_s is not None else ""
        return f"{head} {self.message}{took}"


@dataclass
class PipelineResult:
    """
    Everything one end-to-end run produced.

    ``succeeded`` is not "the pipeline finished". It means the project passed review and
    its tests actually ran and passed. A run that produced code nobody verified reports
    ``succeeded=False`` with a reason, because the alternative - calling that a success -
    is precisely the fake-green outcome the brief forbids.
    """

    requirement: str
    status: ProjectStatus
    analysis: Optional[RequirementAnalysis] = None
    framework_choice: Optional[FrameworkChoice] = None
    #: The architecture plan the project was built from.
    #:
    #: Kept on the result, not discarded after generation, for two reasons that both turned out
    #: to matter: the repair engine regenerates a broken file *from the plan*, and the
    #: validator's plan-comparison checks skip themselves entirely without it - so a run that
    #: threw the plan away could never report whether the code matched the architecture.
    manifest: Optional["ProjectManifest"] = None
    project: Optional[GeneratedProject] = None
    iterations: List[IterationRecord] = field(default_factory=list)
    events: List[PipelineEvent] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None
    started_at: datetime = field(default_factory=utcnow)
    completed_at: Optional[datetime] = None

    @property
    def final_review(self) -> Optional[ReviewResult]:
        for record in reversed(self.iterations):
            if record.review is not None:
                return record.review
        return None

    @property
    def final_validation(self) -> Optional["ValidationReport"]:
        """The last gate report produced. This is what decides ``READY``."""
        for record in reversed(self.iterations):
            if record.validation is not None:
                return record.validation
        return None

    @property
    def final_runtime(self) -> Optional[RunResult]:
        for record in reversed(self.iterations):
            if record.runtime is not None:
                return record.runtime
        return None

    @property
    def final_tests(self) -> Optional[TestReport]:
        for record in reversed(self.iterations):
            if record.tests is not None:
                return record.tests
        return None

    @property
    def duration_s(self) -> float:
        end = self.completed_at or utcnow()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def succeeded(self) -> bool:
        return self.status is ProjectStatus.READY

    def unmet_criteria(self) -> List[str]:
        """
        Why this run is not a success. Empty list means it is.

        Used by the pipeline to set the final status and by the UI to explain a project
        that has code but is not marked ready.

        When a :class:`~multi_agent_generator.core.validation.ValidationReport` exists, its
        blocking findings are the answer and the review/test heuristics below are not
        consulted. That ordering is the point: the gate report is computed from the code
        itself and covers configuration, imports, entry point, agents, tools and workflow,
        whereas a review score is an opinion about the code. Keeping both and taking the
        union would let a passing opinion be reported alongside a failing gate as though the
        two were comparable evidence.
        """
        reasons: List[str] = []
        if self.project is None or not self.project.files:
            reasons.append("No project was generated.")
            return reasons

        report = self.final_validation
        if report is not None:
            reasons.extend(report.unmet())
            # The suite still has to have run. Validation reports an unrun suite itself, but
            # only when it was handed one to look at; a run configured with ``run_tests=False``
            # never produces a report for it, and that must not read as a pass.
            if self.final_tests is None:
                reasons.append("The generated test suite was never run.")
            return reasons

        review = self.final_review
        if review is None:
            reasons.append("The generated project was never reviewed.")
        elif not review.passed:
            blockers = len(review.blockers)
            detail = f", {blockers} blocking issue(s)" if blockers else ""
            reasons.append(f"Review did not pass (score {review.score:.1f}{detail}).")
        tests = self.final_tests
        if tests is None:
            reasons.append("The generated test suite was never run.")
        elif tests.timed_out:
            reasons.append("The generated test suite timed out.")
        elif not tests.ran:
            reasons.append(
                f"The generated test suite did not run: "
                f"{tests.unavailable_reason or tests.status}."
            )
        elif not tests.ok:
            reasons.append(f"{tests.failed_count} generated test(s) failed.")
        return reasons

    def as_dict(self, include_content: bool = True) -> Dict[str, Any]:
        return {
            "requirement": self.requirement,
            "status": self.status.value,
            "succeeded": self.succeeded,
            "unmet_criteria": self.unmet_criteria(),
            "analysis": self.analysis.as_dict() if self.analysis else None,
            "framework_choice": (
                self.framework_choice.as_dict() if self.framework_choice else None
            ),
            "manifest": self.manifest.as_dict() if self.manifest else None,
            "architecture": (
                self.manifest.architecture_summary() if self.manifest else None
            ),
            "project": (
                self.project.as_dict(include_content=include_content)
                if self.project
                else None
            ),
            "iterations": [i.as_dict() for i in self.iterations],
            "review": self.final_review.as_dict() if self.final_review else None,
            "tests": self.final_tests.as_dict() if self.final_tests else None,
            "validation": (
                self.final_validation.as_dict() if self.final_validation else None
            ),
            "runtime": self.final_runtime.as_dict() if self.final_runtime else None,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_s": round(self.duration_s, 3),
            "events": [e.as_dict() for e in self.events],
        }
