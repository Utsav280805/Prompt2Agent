"""
Unit tests for the acceptance gate.

Section 28 of the brief forbids a green result that was not earned, and the gate that enforces
it is small enough to test exhaustively: ``PipelineResult.unmet_criteria`` plus
``TestReport.ran``/``ok`` plus ``ReviewResult.decide``. If those three agree that a project is
ready, the whole system says so - the API marks it verified, the UI shows a green badge and the
CLI exits zero.

So the interesting cases here are the ones that *look* like success:

* a suite that reported "skipped" (nothing ran, but nothing failed either)
* a suite that could not be installed (``unavailable_reason`` set, status still nominal)
* a suite that timed out (no assertion failed)
* a review with a high score and one blocker

Each of those must be a refusal. A regression in any of them is how "we generated something"
starts being reported as "it works".
"""
from __future__ import annotations

import pytest

from multi_agent_generator.core.models import (
    GeneratedProject,
    IterationRecord,
    PipelineResult,
    ProjectStatus,
    ReviewIssue,
    ReviewResult,
    Severity,
    Stage,
    TestReport,
)

pytestmark = pytest.mark.unit


def _project() -> GeneratedProject:
    project = GeneratedProject(framework="crewai", provider="huggingface", model="m")
    project.add_file("main.py", "print('hi')\n", is_entrypoint=True)
    project.add_file("tests/test_main.py", "def test_ok():\n    assert True\n")
    return project


def _result(*, review=None, tests=None, project=True) -> PipelineResult:
    result = PipelineResult(
        requirement="Create a research assistant",
        status=ProjectStatus.DRAFT,
        project=_project() if project else None,
    )
    if review is not None or tests is not None:
        result.iterations.append(IterationRecord(index=1, review=review, tests=tests))
    return result


def _passing_review() -> ReviewResult:
    return ReviewResult(score=9.0, passed=True)


def _passing_tests() -> TestReport:
    return TestReport(status="passed", exit_code=0, passed_count=3)


class TestTestReportHonesty:
    def test_passing_suite_ran_and_is_ok(self):
        report = _passing_tests()
        assert report.ran and report.ok

    def test_failing_suite_ran_but_is_not_ok(self):
        # The distinction the UI renders in three tones: this one is a real red, not a warning.
        report = TestReport(status="failed", exit_code=1, failed_count=2)
        assert report.ran
        assert not report.ok

    @pytest.mark.parametrize("status", ["skipped", "error"])
    def test_a_suite_that_never_executed_did_not_run(self, status):
        report = TestReport(status=status)
        assert not report.ran
        assert not report.ok

    def test_an_unavailable_reason_means_it_did_not_run_whatever_the_status(self):
        # The realistic accident: install fails, the runner reports "passed" for zero tests.
        report = TestReport(status="passed", unavailable_reason="pip install failed")
        assert not report.ran

    def test_a_timeout_is_not_a_pass(self):
        report = TestReport(status="timeout", timed_out=True)
        assert not report.ok
        assert not report.ran

    def test_serialised_report_carries_both_flags(self):
        payload = TestReport(status="skipped").as_dict()
        assert payload["ok"] is False
        assert payload["ran"] is False


class TestReviewGate:
    def test_high_score_without_blockers_passes(self):
        review = ReviewResult(score=8.5, passed=False)
        assert review.decide(7.0) is True

    def test_one_blocker_fails_a_high_score(self):
        review = ReviewResult(
            score=9.8,
            passed=False,
            issues=[ReviewIssue(category="security", severity=Severity.BLOCKER, message="hardcoded key")],
        )
        assert review.decide(7.0) is False
        assert review.passed is False

    def test_low_score_without_blockers_fails(self):
        assert ReviewResult(score=4.0, passed=True).decide(7.0) is False

    def test_score_is_clamped_to_the_scale(self):
        # A model asked for 0-10 will occasionally answer 95. Clamping stops that from
        # silently satisfying any threshold.
        assert ReviewResult(score=95, passed=False).score == 10.0
        assert ReviewResult(score=-3, passed=False).score == 0.0

    def test_unknown_severity_is_treated_as_major_not_info(self):
        # Downgrading an unrecognised severity to "info" would let a model invent a label and
        # bypass the gate. Defaulting upward is the safe direction.
        assert Severity.parse("catastrophic") is Severity.MAJOR
        assert Severity.parse("blocker") is Severity.BLOCKER
        assert Severity.parse(None) is Severity.MAJOR


class TestUnmetCriteria:
    def test_review_and_tests_both_green_means_no_unmet_criteria(self):
        result = _result(review=_passing_review(), tests=_passing_tests())
        assert result.unmet_criteria() == []

    def test_no_project_is_the_only_reason_reported(self):
        reasons = _result(project=False).unmet_criteria()
        assert len(reasons) == 1
        assert "No project" in reasons[0]

    def test_missing_review_is_unmet(self):
        reasons = _result(tests=_passing_tests()).unmet_criteria()
        assert any("never reviewed" in r for r in reasons)

    def test_missing_test_run_is_unmet(self):
        reasons = _result(review=_passing_review()).unmet_criteria()
        assert any("never run" in r for r in reasons)

    def test_skipped_suite_is_unmet_even_though_nothing_failed(self):
        reasons = _result(review=_passing_review(), tests=TestReport(status="skipped")).unmet_criteria()
        assert any("did not run" in r for r in reasons)

    def test_timed_out_suite_is_unmet(self):
        reasons = _result(
            review=_passing_review(), tests=TestReport(status="timeout", timed_out=True)
        ).unmet_criteria()
        assert any("timed out" in r for r in reasons)

    def test_failing_review_reports_its_score(self):
        reasons = _result(
            review=ReviewResult(score=3.2, passed=False), tests=_passing_tests()
        ).unmet_criteria()
        assert any("3.2" in r for r in reasons)

    def test_succeeded_tracks_the_status_not_the_wish(self):
        result = _result(review=_passing_review(), tests=_passing_tests())
        assert result.succeeded is False  # status is still DRAFT
        result.status = ProjectStatus.READY
        assert result.succeeded is True

    def test_serialised_result_exposes_the_reasons(self):
        payload = _result(review=_passing_review()).as_dict(include_content=False)
        assert payload["succeeded"] is False
        assert payload["unmet_criteria"]
        assert payload["status"] == ProjectStatus.DRAFT.value


class TestProjectRoundTrip:
    def test_from_dict_restores_what_as_dict_produced(self):
        # The API stores a project as JSON and rebuilds it to run the playground, so a lossy
        # round-trip would mean the code you can download is not the code that gets executed.
        original = _project()
        original.dependencies = ["crewai>=0.80.0"]
        original.notes = ["needs HF_TOKEN"]
        original.run_instructions = "python main.py"

        restored = GeneratedProject.from_dict(original.as_dict())

        assert restored.framework == original.framework
        assert restored.entrypoint == "main.py"
        assert [f.path for f in restored.files] == [f.path for f in original.files]
        assert restored.get("main.py").content == "print('hi')\n"
        assert restored.dependencies == original.dependencies
        assert restored.notes == original.notes
        assert restored.total_lines == original.total_lines

    def test_test_files_are_recognised(self):
        assert [f.path for f in _project().test_files()] == ["tests/test_main.py"]

    def test_adding_a_file_twice_replaces_it(self):
        project = _project()
        project.add_file("main.py", "print('second')\n", is_entrypoint=True)
        assert len(project.files) == 2
        assert project.entrypoint_file.content == "print('second')\n"


class TestStageContract:
    def test_stage_values_match_the_api_contract(self):
        # The frontend's STAGE_ORDER is written against these strings.
        assert [s.value for s in Stage] == [
            "analysis",
            "framework_selection",
            "planning",
            "generation",
            "review",
            "improvement",
            "test",
            "repair",
            "execution",
            "validation",
        ]
