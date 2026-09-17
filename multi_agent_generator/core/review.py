# multi_agent_generator/core/review.py
"""
Stage 4: judge the generated project.

Two reviewers run, and they are not interchangeable.

:func:`static_review` parses the code. It compiles every Python file, looks for a real
entry point, hunts for hardcoded credentials and checks that the roles the analysis asked
for actually appear. Everything it reports is a fact about the file rather than an opinion
about it, so it runs on every review regardless of whether a model is configured - and it
is the reason the pipeline can review code with no API key at all.

:func:`model_review` asks an LLM. It catches the things static analysis cannot: whether the
agents are sensibly separated, whether the tasks actually serve the requirement, whether a
prompt is likely to behave. Its opinions are valuable and its facts are unreliable.

When both run, the results are merged with the *lower* score winning and the issue lists
unioned. A model that scores syntactically-broken code 9/10 does not get to overrule the
compiler, and ``passed`` is always recomputed from the merged evidence by
:meth:`ReviewResult.decide` - never read from the model's own ``passed`` field.
"""
from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Tuple

from ..errors import AppError
from ..llm.base import LLMProvider
from ..settings import get_settings
from .models import (
    GeneratedProject,
    RequirementAnalysis,
    ReviewIssue,
    ReviewResult,
    Severity,
)
from .parsing import as_float, as_str_list, extract_json_object

__all__ = ["review_project", "static_review", "model_review", "REVIEW_SYSTEM_PROMPT"]


REVIEW_SYSTEM_PROMPT = """\
You are a senior engineer reviewing generated multi-agent code before it is handed to a
user. Be specific and be strict: a vague complaint cannot be acted on.

Reply with ONLY a JSON object, no prose and no code fence:

{
  "score": 0.0-10.0,
  "summary": "one or two sentences on the overall state",
  "dimension_scores": {
    "requirement_coverage": 0-10,
    "correctness": 0-10,
    "framework_usage": 0-10,
    "error_handling": 0-10,
    "security": 0-10,
    "clarity": 0-10
  },
  "issues": [
    {
      "category": "one of the dimension names",
      "severity": "blocker" | "major" | "minor" | "info",
      "file": "path/to/file.py",
      "message": "what is wrong",
      "suggestion": "the concrete change that fixes it"
    }
  ],
  "required_changes": [
    "an imperative instruction, e.g. 'Add a tools=[] argument to the writer Agent'"
  ]
}

Severity means:
- blocker: the code cannot work - it will not import, or it leaks a credential.
- major: it runs but does not do what the requirement asked.
- minor: it works but is unclear, unidiomatic or missing error handling.
- info: an observation, not a defect.

Rules:
- Only list "required_changes" for blocker and major issues. Each one must name the file
  and the change. Do not restate a problem as an instruction ("fix the agent") - say what
  the new code should be.
- Do not invent problems to look thorough. An empty issues list with a high score is a
  valid review of good code.
- Do not comment on the TODO bodies of tool stubs unless the requirement needed that tool
  to be real; stubs are intentional placeholders in generated projects.
"""


#: Patterns that look like a credential pasted into source. Deliberately narrow - matching
#: on the *shape of an assignment to a secret-ish name* rather than on any long string,
#: because generated code is full of long model ids and prompt text.
_SECRET_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"""(?i)\b(api_key|apikey|secret|token|password)\s*=\s*["'][A-Za-z0-9_\-]{16,}["']""",
     "A credential appears to be hardcoded in the source."),
    (r"""\bsk-[A-Za-z0-9]{20,}\b""", "What looks like an OpenAI key is embedded in the source."),
    (r"""\bhf_[A-Za-z0-9]{20,}\b""", "What looks like a Hugging Face token is embedded in the source."),
)

#: Function names that count as an entry point in generated code.
_ENTRY_FUNCTIONS = ("run_workflow", "run_agent", "run_flow", "main", "run")


def review_project(
    project: GeneratedProject,
    analysis: Optional[RequirementAnalysis] = None,
    provider: Optional[LLMProvider] = None,
    pass_score: Optional[float] = None,
) -> ReviewResult:
    """
    Review ``project`` and decide whether it may ship.

    Args:
        project: The generated project.
        analysis: Used to check requirement coverage. Optional but strongly recommended.
        provider: LLM for the judgement pass. When None, only static checks run.
        pass_score: Gate threshold. Defaults to ``REVIEW_PASS_SCORE`` from settings.

    Never raises for model-side problems: an unreachable reviewer degrades the review to
    static-only and says so in the summary. The static half is always real, so the pipeline
    always has an honest verdict to act on.
    """
    threshold = (
        pass_score if pass_score is not None else get_settings().review_pass_score
    )

    static = static_review(project, analysis)
    if provider is None:
        static.decide(threshold)
        return static

    try:
        judged = model_review(project, analysis, provider)
    except AppError as exc:
        static.summary = (
            f"{static.summary} Model review was skipped: {exc.message}"
        ).strip()
        static.decide(threshold)
        return static

    if judged is None:
        static.summary = (
            f"{static.summary} The reviewing model did not return a usable verdict, so "
            "this score reflects static checks only."
        ).strip()
        static.decide(threshold)
        return static

    merged = _merge(static, judged)
    merged.decide(threshold)
    return merged


# ============================================================================== static
def static_review(
    project: GeneratedProject,
    analysis: Optional[RequirementAnalysis] = None,
) -> ReviewResult:
    """
    Check the things that can be checked by reading the code.

    Every issue raised here is verifiable: the file either compiles or it does not, the
    entry point either exists or it does not. That is what makes this safe to run without a
    model and safe to trust over one.
    """
    issues: List[ReviewIssue] = []
    dimensions: Dict[str, float] = {}

    issues.extend(_check_syntax(project))
    issues.extend(_check_secrets(project))
    issues.extend(_check_entrypoint(project))
    issues.extend(_check_tests(project))
    issues.extend(_check_coverage(project, analysis))
    issues.extend(_check_dependencies(project))

    # Dimension scores are derived from the issues that landed in each category, so the
    # per-dimension display and the overall score can never disagree.
    for dimension in ("correctness", "security", "requirement_coverage", "test_coverage"):
        penalty = sum(
            _penalty(issue.severity)
            for issue in issues
            if issue.category == dimension
        )
        dimensions[dimension] = max(0.0, 10.0 - penalty)

    score = max(0.0, 10.0 - sum(_penalty(issue.severity) for issue in issues))
    summary = _static_summary(issues)

    return ReviewResult(
        score=score,
        passed=False,  # recomputed by decide(); never assumed
        summary=summary,
        issues=issues,
        required_changes=[
            issue.suggestion or issue.message
            for issue in issues
            if issue.severity.blocks_release
        ],
        dimension_scores=dimensions,
        from_model=False,
    )


def _penalty(severity: Severity) -> float:
    """
    Score cost per severity.

    A blocker costs 6 rather than 10: the score should still distinguish "one blocker" from
    "a blocker and eight other problems", which a saturating penalty would flatten. The
    release gate does not depend on this arithmetic anyway - a blocker fails
    :meth:`ReviewResult.decide` outright, whatever the score works out to.
    """
    return {
        Severity.BLOCKER: 6.0,
        Severity.MAJOR: 2.0,
        Severity.MINOR: 0.5,
        Severity.INFO: 0.0,
    }[severity]


def _check_syntax(project: GeneratedProject) -> List[ReviewIssue]:
    """
    Compile every Python file.

    This is the check that matters most and the cheapest one available. A generated file
    with a syntax error is worthless, and the failure is completely unambiguous - which is
    why it is a blocker rather than a deduction.
    """
    issues: List[ReviewIssue] = []
    for file in project.source_files():
        try:
            ast.parse(file.content, filename=file.path)
        except SyntaxError as exc:
            issues.append(
                ReviewIssue(
                    category="correctness",
                    severity=Severity.BLOCKER,
                    message=f"{file.path} is not valid Python: {exc.msg} (line {exc.lineno}).",
                    file=file.path,
                    line=exc.lineno,
                    suggestion=(
                        f"Rewrite {file.path} so that it parses. The error is at line "
                        f"{exc.lineno}: {exc.msg}."
                    ),
                )
            )
    return issues


def _check_secrets(project: GeneratedProject) -> List[ReviewIssue]:
    """
    Refuse to ship a project with a credential baked into it.

    A blocker, without exception. Generated code is meant to be committed, and a key in a
    committed file is a key that has to be rotated. Every provider snippet reads its
    credential from the environment, so a literal here means something went wrong.
    """
    issues: List[ReviewIssue] = []
    for file in project.files:
        if file.path.endswith((".md", ".txt")):
            continue
        for pattern, message in _SECRET_PATTERNS:
            match = re.search(pattern, file.content)
            if not match:
                continue
            line_no = file.content[: match.start()].count("\n") + 1
            issues.append(
                ReviewIssue(
                    category="security",
                    severity=Severity.BLOCKER,
                    message=f"{file.path}: {message}",
                    file=file.path,
                    line=line_no,
                    suggestion=(
                        f"Remove the literal credential from {file.path} line {line_no} and "
                        "read it from an environment variable instead, e.g. "
                        "os.environ['API_KEY']."
                    ),
                )
            )
            break  # one secret finding per file is enough to fail it
    return issues


def _check_entrypoint(project: GeneratedProject) -> List[ReviewIssue]:
    """Check there is a file to run and a function in it that does the running."""
    entry = project.entrypoint_file
    if entry is None:
        return [
            ReviewIssue(
                category="correctness",
                severity=Severity.BLOCKER,
                message="The project has no entry point, so there is nothing to run.",
                suggestion="Add an agent.py with a run_workflow(query) function.",
            )
        ]

    try:
        tree = ast.parse(entry.content, filename=entry.path)
    except SyntaxError:
        return []  # already reported by _check_syntax; don't double-count

    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if not functions & set(_ENTRY_FUNCTIONS):
        return [
            ReviewIssue(
                category="correctness",
                severity=Severity.MAJOR,
                message=(
                    f"{entry.path} defines no callable entry point "
                    f"(expected one of: {', '.join(_ENTRY_FUNCTIONS)})."
                ),
                file=entry.path,
                suggestion=(
                    f"Add a `def run_workflow(query: str)` to {entry.path} that executes "
                    "the agents and returns the result."
                ),
            )
        ]
    return []


def _check_tests(project: GeneratedProject) -> List[ReviewIssue]:
    """A project with no tests cannot be verified, which is a real defect."""
    if project.test_files():
        return []
    return [
        ReviewIssue(
            category="test_coverage",
            severity=Severity.MAJOR,
            message="The project has no tests, so nothing verifies that it works.",
            suggestion="Generate the offline test bundle alongside the agent code.",
        )
    ]


def _check_coverage(
    project: GeneratedProject,
    analysis: Optional[RequirementAnalysis],
) -> List[ReviewIssue]:
    """
    Check the roles the analysis identified actually show up in the code.

    Deliberately lenient - it only fires when *none* of the expected roles appear, because
    role names are paraphrased freely ("researcher" becoming "research_specialist") and a
    stricter match would produce constant false positives. Firing only on a total miss
    catches the case that matters: a config that ignored the requirement entirely.
    """
    if analysis is None or not analysis.suggested_roles:
        return []

    entry = project.entrypoint_file
    haystack = (entry.content if entry else "").lower()
    if not haystack:
        return []

    # Compare on word stems so "research" matches "research_specialist" and "researcher".
    stems = [re.split(r"[_\s]", role.lower())[0][:6] for role in analysis.suggested_roles]
    stems = [s for s in stems if len(s) >= 4]
    if stems and not any(stem in haystack for stem in stems):
        return [
            ReviewIssue(
                category="requirement_coverage",
                severity=Severity.MAJOR,
                message=(
                    "None of the roles the requirement implied "
                    f"({', '.join(analysis.suggested_roles)}) appear in the generated code."
                ),
                file=entry.path if entry else None,
                suggestion=(
                    "Regenerate the agent configuration so it includes agents for: "
                    f"{', '.join(analysis.suggested_roles)}."
                ),
            )
        ]
    return []


def _check_dependencies(project: GeneratedProject) -> List[ReviewIssue]:
    """The project must say what it needs installed."""
    if project.get("requirements.txt") and project.dependencies:
        return []
    return [
        ReviewIssue(
            category="correctness",
            severity=Severity.MINOR,
            message="The project does not declare its dependencies.",
            suggestion="Add a requirements.txt listing the framework and provider packages.",
        )
    ]


def _static_summary(issues: List[ReviewIssue]) -> str:
    if not issues:
        return "Static checks passed: the code parses, has an entry point and declares its dependencies."
    blockers = sum(1 for i in issues if i.severity is Severity.BLOCKER)
    majors = sum(1 for i in issues if i.severity is Severity.MAJOR)
    minors = len(issues) - blockers - majors
    parts = []
    if blockers:
        parts.append(f"{blockers} blocking")

    if majors:
        parts.append(f"{majors} major")
    if minors:
        parts.append(f"{minors} minor")
    return f"Static checks found {', '.join(parts)} issue(s)."


# =============================================================================== model
def model_review(
    project: GeneratedProject,
    analysis: Optional[RequirementAnalysis],
    provider: LLMProvider,
) -> Optional[ReviewResult]:
    """
    Ask a model to judge the project. Returns None when it did not answer usably.

    None is distinct from a bad score: "the reviewer is broken" and "the code is bad" call
    for opposite responses from the pipeline, and collapsing them would make it improve
    code in response to a parsing failure.
    """
    raw = provider.complete_text(
        _review_prompt(project, analysis),
        system=REVIEW_SYSTEM_PROMPT,
        temperature=0.2,
        max_tokens=2000,
    )
    payload = extract_json_object(raw)
    if not payload:
        return None

    issues = [
        issue
        for issue in (ReviewIssue.from_raw(raw_issue) for raw_issue in payload.get("issues") or [])
        if issue is not None
    ]
    dimensions = {}
    for key, value in (payload.get("dimension_scores") or {}).items():
        dimensions[str(key)] = max(0.0, min(10.0, as_float(value, 0.0)))

    score = payload.get("score")
    if score is None and dimensions:
        # A model that filled in the dimensions but forgot the total has still told us
        # what it thinks; averaging is more faithful than defaulting to zero.
        score = sum(dimensions.values()) / len(dimensions)

    return ReviewResult(
        score=as_float(score, 5.0),
        passed=False,  # recomputed by decide()
        summary=str(payload.get("summary") or "").strip(),
        issues=issues,
        required_changes=as_str_list(payload.get("required_changes"), limit=12),
        dimension_scores=dimensions,
        from_model=True,
    )


def _review_prompt(
    project: GeneratedProject,
    analysis: Optional[RequirementAnalysis],
) -> str:
    """
    Build the review prompt.

    Only source files are included, and each is truncated. Sending the README and the
    generated test helpers would spend most of the context window on text the reviewer was
    not asked about, and on a small model that crowds out the code itself.
    """
    lines = []
    if analysis is not None:
        lines += [
            f"Requirement: {analysis.requirement}",
            f"Roles the requirement implies: {', '.join(analysis.suggested_roles) or 'none'}",
            f"Tools the requirement implies: {', '.join(analysis.suggested_tools) or 'none'}",
            "",
        ]
    lines += [
        f"Framework: {project.framework}",
        f"Provider: {project.provider} ({project.model})",
        "",
        "Files under review:",
        "",
    ]

    for file in project.source_files():
        if file.path.startswith("tests/"):
            continue
        content = file.content
        if len(content) > 12000:
            content = content[:12000] + "\n# ... truncated for review ...\n"
        lines += [f"--- {file.path} ---", "```python", content, "```", ""]

    test_files = project.test_files()
    if test_files:
        names = ", ".join(f.path for f in test_files)
        lines.append(f"The project also ships tests ({names}); assume they exist.")
    return "\n".join(lines)


# =============================================================================== merge
def _merge(static: ReviewResult, judged: ReviewResult) -> ReviewResult:
    """
    Combine a static verdict with a model verdict.

    The score is the lower of the two and the issues are unioned. Taking the minimum is the
    conservative reading and the correct one: each reviewer sees things the other cannot, so
    a problem found by either is a real problem, and averaging would let a generous model
    dilute a blocker into a passing grade.
    """
    seen = set()
    issues: List[ReviewIssue] = []
    for issue in list(static.issues) + list(judged.issues):
        # Dedupe on (file, message) so the same finding reported by both is listed once.
        key = (issue.file or "", issue.message.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        issues.append(issue)

    changes: List[str] = []
    for change in list(static.required_changes) + list(judged.required_changes):
        text = change.strip()
        if text and text.lower() not in {c.lower() for c in changes}:
            changes.append(text)

    dimensions = dict(judged.dimension_scores)
    for key, value in static.dimension_scores.items():
        # Static wins on any dimension it measured directly: it is measuring, not judging.
        dimensions[key] = min(value, dimensions.get(key, value))

    summary = judged.summary or static.summary
    if static.issues and judged.score > static.score:
        summary = (
            f"{summary} Static checks were stricter than the model review: "
            f"{static.summary}"
        ).strip()

    return ReviewResult(
        score=min(static.score, judged.score),
        passed=False,  # recomputed by decide()
        summary=summary,
        issues=issues,
        required_changes=changes[:12],
        dimension_scores=dimensions,
        from_model=judged.from_model,
    )
