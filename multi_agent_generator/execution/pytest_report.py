# multi_agent_generator/execution/pytest_report.py
"""
Turning pytest's console output into a structured :class:`TestReport`.

Parsing console text is not the elegant way to do this - ``--report-log`` or a plugin would
be cleaner - but it is the *robust* way here, because the sandbox may not have pytest's JSON
reporting available and must never depend on a plugin being installed inside a generated
project. The summary line pytest prints has been stable for years, so it is a safer contract
than an optional dependency.

What matters far more than the counts is the distinction the exit code makes. Pytest's exit
codes are meaningful and the naive reading of them is wrong: exit code 5 means "no tests were
collected", which is *not* a pass, and treating a non-zero exit as "tests failed" turns a
collection error into a phantom test failure the improver will then try to fix. So the exit
code decides ``status`` and the parsed counts only ever add detail.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..core.models import TestReport
from .process import ProcessResult

__all__ = ["parse_pytest_result", "EXIT_CODES"]


#: Pytest's documented exit codes.
EXIT_CODES = {
    0: "passed",
    1: "failed",
    2: "interrupted",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests_collected",
}

#: The trailing summary line, e.g. "=== 3 passed, 1 failed, 2 warnings in 0.41s ===".
_SUMMARY_RE = re.compile(r"^=+\s(.*?)\sin\s[\d.]+s.*?=+$", re.MULTILINE)
#: One "<n> <outcome>" pair inside that line.
_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|error|errors|xfailed|xpassed)")
#: A failing test's node id from the short summary block.
_FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


def parse_pytest_result(result: ProcessResult) -> TestReport:
    """
    Build a :class:`TestReport` from a finished pytest process.

    The report is honest about the difference between "ran and passed", "ran and failed" and
    "did not run". Only the first is a green suite, and only the first two set ``ran``, which
    is what the pipeline's acceptance check keys off - so a suite that could not be collected
    can never be mistaken for a suite that passed.
    """
    output = result.combined_output

    if result.timed_out:
        return TestReport(
            status="timeout",
            exit_code=result.exit_code,
            timed_out=True,
            duration_s=result.duration_s,
            stdout=result.stdout,
            stderr=result.stderr,
            failed_tests=_failed_tests(output),
            unavailable_reason=None,
            **_counts(output),
        )

    if result.start_error is not None:
        return TestReport(
            status="error",
            exit_code=None,
            duration_s=result.duration_s,
            stdout=result.stdout,
            stderr=result.stderr,
            unavailable_reason=f"pytest could not be started: {result.start_error}",
        )

    counts = _counts(output)
    code = result.exit_code

    if code == 0:
        status, unavailable = "passed", None
    elif code == 1:
        status, unavailable = "failed", None
    elif code == 5:
        # Collected nothing. The suite did not fail - it did not exist.
        status, unavailable = "skipped", "pytest collected no tests."
    elif code in (2, 3, 4):
        status, unavailable = "error", (
            f"pytest exited with {code} ({EXIT_CODES.get(code, 'unknown')}): "
            f"{_first_error_line(output) or 'see the captured output'}"
        )
    else:
        # An unrecognised code is treated as an error rather than a failure. Guessing
        # "failed" here would send the improver after tests that may be perfectly fine.
        status, unavailable = "error", (
            f"pytest exited with an unexpected code {code}. "
            f"{_first_error_line(output) or ''}".strip()
        )

    # A zero exit with zero collected tests is the subtle case: pytest can exit 0 having run
    # nothing at all if everything was deselected. That is not a pass.
    if status == "passed" and counts["passed_count"] == 0 and counts["skipped_count"] == 0:
        status, unavailable = "skipped", "pytest reported success but ran no tests."

    return TestReport(
        status=status,
        exit_code=code,
        timed_out=False,
        duration_s=result.duration_s,
        stdout=result.stdout,
        stderr=result.stderr,
        failed_tests=_failed_tests(output),
        unavailable_reason=unavailable,
        **counts,
    )


def _counts(output: str) -> dict:
    """
    Pull the outcome counts out of pytest's summary line.

    The *last* summary line is used. A run that reports collection errors and then a summary
    prints more than one, and the final one is the authoritative total.
    """
    totals = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    matches = _SUMMARY_RE.findall(output or "")
    if matches:
        for count, outcome in _COUNT_RE.findall(matches[-1]):
            key = "error" if outcome.startswith("error") else outcome
            if key == "xfailed":
                key = "skipped"
            elif key == "xpassed":
                key = "passed"
            if key in totals:
                totals[key] += int(count)
    return {
        "passed_count": totals["passed"],
        "failed_count": totals["failed"],
        "skipped_count": totals["skipped"],
        "error_count": totals["error"],
    }


def _failed_tests(output: str) -> List[str]:
    """Node ids of failing tests, deduplicated, so the improver knows what to look at."""
    seen: List[str] = []
    for node in _FAILED_RE.findall(output or ""):
        cleaned = node.strip().rstrip(":")
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen[:25]


def _first_error_line(output: str) -> Optional[str]:
    """
    The most explanatory single line from a broken run.

    Collection errors and usage errors both put the real reason on a line starting with
    ``ERROR`` or ``ImportError``; surfacing it means the user sees "no module named crewai"
    instead of "pytest exited with 4".
    """
    for line in (output or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(("ERROR", "ImportError", "ModuleNotFoundError", "E   ")):
            return stripped[:300]
    return None
