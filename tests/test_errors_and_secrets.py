"""
Unit tests for the error taxonomy and credential handling.

These are the two things the rest of the system trusts blindly, so they are tested first and
tested for the *unhappy* properties: that a secret does not turn back into a string by accident
and that an error carries something the user can act on.

The second half of the file covers the two boundaries where a credential and a model-authored
dependency list actually leave the application: the scrubbing of a sandbox's captured output on
the way back, and the screening of a generated ``requirements.txt`` on the way in. Both are
security properties, so they are asserted rather than assumed.

No network and no real credentials, so all of it runs in the default suite; a couple of tests
use ``tmp_path`` because a dependency directory is a directory. Nothing here is allowed to
require a key - that is what the ``live`` marker is for - and nothing here installs anything.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from multi_agent_generator.core.models import RunStatus, TestReport
from multi_agent_generator.errors import (
    AppError,
    ConfigurationError,
    ExecutionError,
    LLMTimeoutError,
    MissingCredentialError,
    NotFoundError,
    Secret,
    UnknownProviderError,
    redact,
    scrub_values,
)
from multi_agent_generator.execution.deps import (
    DependencySet,
    _cache_key,
    _screen,
    ensure_dependencies,
    python_path_for,
)
from multi_agent_generator.execution.process import ProcessResult
from multi_agent_generator.execution.runner import _BEGIN, _END, _to_run_result
from multi_agent_generator.execution.workspace import Workspace
from multi_agent_generator.settings import reload_settings

pytestmark = pytest.mark.unit

#: Shaped like a real key and long enough to be scrubbable, but not one. Used everywhere below
#: so a grep for it in a failure message unambiguously means the scrub did not happen.
FAKE_KEY = "sk-live-notarealkey-0123456789abcdef"


@pytest.fixture()
def isolated_settings(tmp_path, monkeypatch):
    """
    Point the settings singleton at ``tmp_path`` for one test, then put it back.

    Needed because :func:`~multi_agent_generator.execution.deps.ensure_dependencies` reads the
    configured cache directory, and a test must not create or read directories under the
    developer's real data dir.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    yield reload_settings()
    monkeypatch.delenv("DATA_DIR", raising=False)
    reload_settings()


class TestAppError:
    def test_carries_message_action_and_detail(self):
        exc = AppError("It broke.", action="Try this.", detail="line 3", context={"stage": "review"})
        payload = exc.to_dict()
        assert payload["message"] == "It broke."
        assert payload["action"] == "Try this."
        assert payload["detail"] == "line 3"
        assert payload["code"] == "internal_error"

    def test_detail_can_be_withheld(self):
        # The API omits detail for a 500 so an internal path or SDK message does not leak to a
        # browser; the log keeps it. If this ever returns detail anyway, that promise is broken.
        exc = AppError("It broke.", detail="/home/someone/secret/path.py")
        assert exc.to_dict(include_detail=False).get("detail") in (None, "")

    @pytest.mark.parametrize(
        "exc, code, status",
        [
            (ConfigurationError("x"), "configuration_error", 400),
            (UnknownProviderError("x"), "unknown_provider", 400),
            (LLMTimeoutError("x"), "llm_timeout", 504),
            (ExecutionError("x"), "execution_failed", 500),
            (NotFoundError("x"), "not_found", 404),
        ],
    )
    def test_status_codes_are_stable(self, exc, code, status):
        # These pairs are the API contract. A change here changes HTTP behaviour, so it should
        # have to be deliberate enough to edit a test.
        assert exc.code == code
        assert exc.http_status == status

    def test_missing_credential_names_the_variable(self):
        exc = MissingCredentialError("Hugging Face", ["HF_TOKEN", "HUGGINGFACE_API_KEY"])
        rendered = f"{exc.message} {exc.action or ''}"
        assert "HF_TOKEN" in rendered
        assert exc.http_status == 400
        assert exc.env_vars == ["HF_TOKEN", "HUGGINGFACE_API_KEY"]


class TestSecret:
    def test_str_and_repr_never_reveal(self):
        secret = Secret("sk-abcdefghijklmnop")
        assert "abcdefghijklmnop" not in str(secret)
        assert "abcdefghijklmnop" not in repr(secret)
        assert str(secret) == Secret.MASK

    def test_reveal_is_the_only_way_out(self):
        assert Secret("sk-1234567890ab").reveal() == "sk-1234567890ab"

    def test_empty_secret_is_falsey_and_renders_empty(self):
        blank = Secret(None)
        assert not blank
        assert str(blank) == ""
        assert blank.reveal() is None

    def test_hint_shows_last_four_of_a_long_key(self):
        assert Secret("sk-proj-abcdefgh1234").hint().endswith("1234")

    def test_hint_refuses_a_short_key(self):
        # Four of eight characters is not a fingerprint, it is a third of the key.
        assert Secret("short123").hint() is None

    def test_formatting_a_secret_into_a_log_line_masks_it(self):
        # The realistic accident: someone interpolates the config into a log message.
        assert "hunter2hunter2" not in f"key={Secret('hunter2hunter2')}"


class TestRedact:
    def test_masks_credential_shaped_keys(self):
        out = redact(
            {
                "provider": "openai",
                "api_key": "sk-live-123456",
                "auth_token": "t-123",
                "password": "pw",
                "AUTHORIZATION": "Bearer x",
                "model": "gpt-4o-mini",
            }
        )
        assert out["provider"] == "openai"
        assert out["model"] == "gpt-4o-mini"
        for field in ("api_key", "auth_token", "password", "AUTHORIZATION"):
            assert out[field] == Secret.MASK

    def test_recurses_into_nested_structures(self):
        out = redact({"outer": {"items": [{"api_key": "sk-1"}, {"safe": "ok"}]}})
        assert out["outer"]["items"][0]["api_key"] == Secret.MASK
        assert out["outer"]["items"][1]["safe"] == "ok"

    def test_absent_credential_stays_none_rather_than_masked(self):
        # Masking an empty key would claim a key exists. The settings page reads exactly this
        # distinction to decide whether to say "no key configured".
        assert redact({"api_key": None})["api_key"] is None

    def test_unwraps_secret_objects(self):
        assert redact({"nested": Secret("sk-abcdefghijkl")})["nested"] == Secret.MASK

    def test_leaves_tuples_as_tuples(self):
        out = redact({"pair": ("a", "b")})
        assert isinstance(out["pair"], tuple)


class TestLoggingBoundary:
    def test_a_redacted_config_is_safe_to_log(self, caplog):
        config = {"provider": "openai", "api_key": "sk-live-should-not-appear"}
        with caplog.at_level(logging.INFO):
            logging.getLogger("test").info("config=%s", redact(config))
        assert "sk-live-should-not-appear" not in caplog.text


class TestScrubValues:
    """
    ``redact`` works on structure; ``scrub_values`` works on text.

    The distinction is the whole reason the second function exists. ``redact`` masks a value
    because its *key* was called ``api_key``, which is useless for a traceback - there is no key
    name in ``AuthenticationError: Incorrect API key provided: sk-live-...``, just characters.
    """

    def test_removes_a_key_embedded_in_a_traceback(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "agent.py", line 12, in run_workflow\n'
            f"openai.AuthenticationError: Incorrect API key provided: {FAKE_KEY}. "
            "You can find your API key at https://platform.openai.com/account/api-keys"
        )
        out = scrub_values(text, [FAKE_KEY])
        assert FAKE_KEY not in out
        assert Secret.MASK in out
        # The surrounding diagnostic survives - a scrub that destroyed the error message would
        # make the failure unreadable, which is the opposite of the point.
        assert "AuthenticationError" in out
        assert "line 12" in out

    def test_removes_every_occurrence(self):
        text = f"{FAKE_KEY} then again {FAKE_KEY}"
        assert FAKE_KEY not in scrub_values(text, [FAKE_KEY])

    def test_removes_a_quoted_and_truncated_form_by_substring(self):
        # A key surrounded by quotes, and a key cut short by an ellipsis, are both shapes that
        # word-boundary anchoring would miss. Plain substring replacement catches both.
        text = f"headers={{'Authorization': 'Bearer {FAKE_KEY}'}} key={FAKE_KEY[:20]}…"
        out = scrub_values(text, [FAKE_KEY, FAKE_KEY[:20]])
        assert FAKE_KEY not in out
        assert FAKE_KEY[:20] not in out

    def test_removes_the_url_encoded_form(self):
        # A credential sent as a query parameter comes back percent-encoded, so the raw
        # replacement alone would miss it.
        token = "hf_ab/cd+ef=="
        encoded = quote(token, safe="")
        assert encoded != token, "test is pointless unless the value actually re-encodes"
        out = scrub_values(f"GET https://api.example.com/v1?token={encoded} -> 401", [token])
        assert encoded not in out
        assert Secret.MASK in out

    def test_leaves_a_short_value_alone(self):
        # Replacing every occurrence of a four-character string would corrupt unrelated output,
        # and a credential that short is not protectable by substitution anyway.
        text = "the abc module raised an error in abcdef"
        assert scrub_values(text, ["abc"]) == text

    def test_tolerates_none_and_blank_inputs(self):
        assert scrub_values("", [FAKE_KEY]) == ""
        assert scrub_values(None, [FAKE_KEY]) == ""  # type: ignore[arg-type]
        assert scrub_values("nothing to hide", [None, ""]) == "nothing to hide"

    def test_scrubs_several_secrets_at_once(self):
        other = "gsk_secondkey_9876543210"
        out = scrub_values(f"a={FAKE_KEY} b={other}", [FAKE_KEY, other, FAKE_KEY])
        assert FAKE_KEY not in out and other not in out


def _harness_stdout(payload: dict, *, noise: str = "") -> str:
    """stdout shaped the way the real sandbox harness produces it, delimiters included."""
    return f"{noise}\n{_BEGIN}\n{json.dumps(payload)}\n{_END}\n"


class TestRunResultScrubbing:
    """
    The Playground response is the one place a live credential could get back to a browser.

    ``run_agent`` puts the real key into the child's environment, and the child's stdout and
    stderr are returned in the response body. ``_to_run_result`` is the last layer that still
    knows which key was used, so it is the layer that has to scrub - and these tests assert the
    key is absent from the *serialised* result, not merely from one field, because the API
    returns ``as_dict()`` wholesale.
    """

    def test_key_is_absent_from_the_whole_serialised_failure(self):
        payload = {
            "error": "The agent raised an exception.",
            "detail": f"openai.AuthenticationError: Incorrect API key provided: {FAKE_KEY}",
            "entry": "run",
            "duration_s": 1.25,
        }
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=1,
            stdout=_harness_stdout(payload, noise="loading crew..."),
            stderr=f"WARNING auth failed for key {FAKE_KEY}",
            duration_s=3.0,
        )
        run = _to_run_result(result, 60, secrets=[FAKE_KEY])

        assert run.status is RunStatus.FAILED
        assert FAKE_KEY not in json.dumps(run.as_dict())
        # And the failure is still diagnosable afterwards.
        assert "AuthenticationError" in (run.error or {}).get("detail", "")
        assert run.entry == "run"

    def test_key_is_absent_when_the_process_died_before_printing_anything(self):
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=1,
            stdout="",
            stderr=f"ValueError: bad credential {FAKE_KEY}\n",
            duration_s=0.4,
        )
        run = _to_run_result(result, 60, secrets=[FAKE_KEY])
        assert run.status is RunStatus.FAILED
        assert FAKE_KEY not in json.dumps(run.as_dict())
        assert Secret.MASK in run.stderr

    def test_key_is_absent_from_a_timeout_result(self):
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=None,
            stdout=f"calling api with {FAKE_KEY}",
            stderr="",
            duration_s=60.0,
            timed_out=True,
        )
        run = _to_run_result(result, 60, secrets=[FAKE_KEY])
        assert run.status is RunStatus.TIMED_OUT
        assert FAKE_KEY not in json.dumps(run.as_dict())

    def test_key_is_absent_from_a_start_error(self):
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=None,
            start_error=f"could not exec with env OPENAI_API_KEY={FAKE_KEY}",
            duration_s=0.0,
        )
        run = _to_run_result(result, 60, secrets=[FAKE_KEY])
        assert FAKE_KEY not in json.dumps(run.as_dict())

    def test_no_secret_passed_is_not_an_error(self):
        # A local provider needs no credential, and the run must still be reported normally.
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=0,
            stdout=_harness_stdout({"output": "hello", "entry": "run_workflow"}),
            duration_s=1.0,
        )
        run = _to_run_result(result, 60)
        assert run.status is RunStatus.SUCCEEDED
        assert run.output == "hello"

    def test_unsupported_huggingface_model_has_an_actionable_error(self):
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=1,
            stdout="",
            stderr="model_not_supported: not supported by any provider",
            duration_s=1.0,
        )
        run = _to_run_result(result, 60)
        assert run.error["code"] == "model_not_supported"
        assert "HF_MODEL_ID" in run.error["action"]


class TestRunResultDetail:
    """
    The step and tool detail the Playground displays has to be the run's own.

    Section 87 of the brief forbids hard-coded execution detail, and the previous harness threw
    this away by stringifying the result, which left the UI with nothing real to show.
    """

    def test_steps_tool_calls_and_entry_survive_the_round_trip(self):
        payload = {
            "output": "Rome was founded in 753 BC.",
            "steps": [
                {"agent": "researcher", "action": "search", "status": "succeeded"},
                {"agent": "writer", "action": "summarise", "status": "succeeded"},
            ],
            "tool_calls": [{"name": "web_search", "args": {"q": "Rome"}, "result": "..."}],
            "duration_s": 2.5,
            "entry": "run",
        }
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=0,
            stdout=_harness_stdout(payload),
            duration_s=9.9,
        )
        run = _to_run_result(result, 60, secrets=[FAKE_KEY])

        assert run.status is RunStatus.SUCCEEDED
        assert [s["agent"] for s in run.steps] == ["researcher", "writer"]
        assert run.tool_calls[0]["name"] == "web_search"
        assert run.entry == "run"
        # The harness timed the agent call; the process duration also includes interpreter
        # startup and framework import, which would make every run look slower than it was.
        assert run.duration_s == pytest.approx(2.5)

    def test_missing_detail_stays_empty_rather_than_being_invented(self):
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=0,
            stdout=_harness_stdout({"output": "an answer", "entry": "run_workflow"}),
            duration_s=1.0,
        )
        run = _to_run_result(result, 60)
        assert run.steps == []
        assert run.tool_calls == []

    def test_non_dict_entries_are_dropped_not_coerced(self):
        payload = {"output": "x", "steps": ["not a step", {"agent": "a"}], "tool_calls": "nope"}
        result = ProcessResult(
            command=["python", "_run_agent.py"],
            exit_code=0,
            stdout=_harness_stdout(payload),
            duration_s=1.0,
        )
        run = _to_run_result(result, 60)
        assert run.steps == [{"agent": "a"}]
        assert run.tool_calls == []


class TestRequirementScreening:
    """
    A generated ``requirements.txt`` is model output, so its lines are input to be validated.

    ``_screen`` is private and tested directly on purpose: it is the whole of the policy, and a
    test that went through ``ensure_dependencies`` would have to run pip to reach it.
    """

    @pytest.mark.parametrize(
        "line",
        [
            "-e .",
            "--editable ./local-pkg",
            "--index-url https://pypi.example.invalid/simple",
            "--extra-index-url https://pypi.example.invalid/simple",
            "--find-links /tmp/wheels",
            "-f /tmp/wheels",
            "--trusted-host pypi.example.invalid",
            "--pre",
            "https://example.invalid/evil-1.0-py3-none-any.whl",
            "git+https://github.com/someone/something.git",
            "file:///tmp/evil",
            "./local-pkg",
            "../sibling-pkg",
        ],
    )
    def test_refuses_anything_that_is_not_a_named_package(self, line):
        allowed, rejected = _screen(f"crewai>=0.80.0\n{line}\n")
        assert allowed == ["crewai>=0.80.0"]
        assert len(rejected) == 1
        # The refused line is quoted back with a reason, so the user can see what was dropped.
        # Silently stripping it would install a different set of packages than the file names.
        assert line in rejected[0]
        assert "refused" in rejected[0]

    def test_keeps_ordinary_pinned_and_ranged_requirements(self):
        text = "crewai>=0.80.0\npython-dotenv==1.0.1\nlangchain-openai~=0.2\npydantic\n"
        allowed, rejected = _screen(text)
        assert allowed == [
            "crewai>=0.80.0",
            "python-dotenv==1.0.1",
            "langchain-openai~=0.2",
            "pydantic",
        ]
        assert rejected == []

    def test_skips_comments_and_blank_lines(self):
        allowed, rejected = _screen("# needed by the crew\n\n  crewai>=0.80.0  \n\n")
        assert allowed == ["crewai>=0.80.0"]
        assert rejected == []

    def test_extras_and_environment_markers_are_allowed(self):
        # These are ordinary index installs and refusing them would break real projects.
        allowed, rejected = _screen('crewai[tools]>=0.80.0\nuvloop; sys_platform != "win32"\n')
        assert len(allowed) == 2
        assert rejected == []


class TestDependencyCacheKey:
    def test_key_ignores_ordering(self):
        assert _cache_key(["a==1", "b==2"]) == _cache_key(["b==2", "a==1"])

    def test_key_changes_when_a_requirement_changes(self):
        assert _cache_key(["a==1"]) != _cache_key(["a==2"])

    def test_key_is_short_enough_to_be_a_directory_name(self):
        key = _cache_key(["crewai>=0.80.0"])
        assert key.isalnum() and 8 <= len(key) <= 40


class TestDependencySetContract:
    def test_a_project_with_no_dependencies_is_a_success(self, tmp_path, isolated_settings):
        # "Nothing to install" must not be reported as an install failure, or every
        # dependency-free project would be marked broken.
        workspace = Workspace(root=tmp_path, entrypoint=None, files=[], owns_dir=False)
        deps = ensure_dependencies(workspace, cache_dir=tmp_path / "cache")
        assert deps.ok is True
        assert deps.installed is False
        assert deps.path is None
        assert "no installable dependencies" in deps.reason

    def test_a_file_of_only_refused_lines_is_a_failure_not_a_silent_success(
        self, tmp_path, isolated_settings
    ):
        # Every line refused means the project's imports will fail. Returning ok=True here
        # would hand the caller a green install followed by an unexplained
        # ModuleNotFoundError - the same class of dishonesty as a green tick on a failed run.
        (tmp_path / "requirements.txt").write_text("-e .\ngit+https://x/y.git\n", encoding="utf-8")
        workspace = Workspace(root=tmp_path, entrypoint=None, files=[], owns_dir=False)
        deps = ensure_dependencies(workspace, cache_dir=tmp_path / "cache")

        assert deps.ok is False
        assert deps.installed is False
        assert len(deps.rejected) == 2
        assert "refused" in deps.as_dict()["rejected"][0]
        # No pip run happened, so there is no exit code to report - and reporting 0 here would
        # look like a successful install.
        assert deps.as_dict()["exit_code"] is None
        # The cache directory is not created for a set that was never installed.
        assert not (tmp_path / "cache").exists()

    def test_python_path_puts_the_project_first(self, tmp_path):
        # A generated project containing `tools.py` must import its own, not a third-party
        # package that happens to be called `tools`.
        installed = tmp_path / "deps"
        installed.mkdir()
        workspace = Workspace(root=tmp_path / "proj", entrypoint=None, files=[], owns_dir=False)
        path = python_path_for(workspace, DependencySet(path=installed, ok=True))
        assert path.split(os.pathsep) == [str(tmp_path / "proj"), str(installed)]

    def test_python_path_omits_a_failed_install(self, tmp_path):
        workspace = Workspace(root=tmp_path, entrypoint=None, files=[], owns_dir=False)
        failed = DependencySet(path=Path(tmp_path / "never-created"), ok=False)
        assert python_path_for(workspace, failed) == str(tmp_path)
        assert python_path_for(workspace, None) == str(tmp_path)


class TestRefusalVisibility:
    """
    A refused requirement line must reach the user without changing the test verdict.

    ``run_project_tests`` puts refusals in ``TestReport.notes``. The invariant that matters is
    that a note is context, not a verdict: a suite that ran and failed must still report
    ``ran=True``, or the pipeline would treat a real failure as "never executed" and skip the
    repair it should be attempting.
    """

    def test_a_note_does_not_make_a_suite_that_ran_look_unavailable(self):
        report = TestReport(
            status="failed",
            exit_code=1,
            failed_count=1,
            notes=["git+https://x/y.git  -> refused: only named packages ..."],
        )
        assert report.ran is True
        assert report.ok is False
        assert report.as_dict()["notes"] == report.notes

    def test_an_install_failure_is_unavailable_rather_than_failed(self):
        # "Could not install" is not "the tests failed". Conflating them would make the
        # improver try to fix generated logic that was never run.
        report = TestReport(
            status="error",
            unavailable_reason="The project's dependencies could not be installed...",
            notes=["-e .  -> refused: only named packages ..."],
        )
        assert report.ran is False
        assert report.ok is False
        assert report.as_dict()["notes"]

    def test_notes_default_to_empty(self):
        assert TestReport(status="passed").as_dict()["notes"] == []
