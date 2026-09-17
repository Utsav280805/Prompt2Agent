"""
Shared pytest fixtures for the generated crewai test suite.

Everything here exists to make the suite runnable without credentials and without
network access. Tests marked `live` opt out of the offline guards.
"""

import importlib
import os
import socket

import pytest


# ---------------------------------------------------------------------------- markers
def pytest_configure(config):
    """Register markers so `--strict-markers` does not reject the generated suite."""
    for marker, description in [
        ("contract", "validates the agent configuration; no dependencies needed"),
        ("unit", "builds individual agents"),
        ("integration", "exercises agent and task wiring"),
        ("end_to_end", "runs the full workflow"),
        ("performance", "measures timing and memory"),
        ("reliability", "checks error handling"),
        ("quality", "scores output quality"),
        ("live", "requires real credentials and network access"),
        # Registered here as well as by pytest-timeout, so the suite still collects
        # under --strict-markers if that plugin happens to be missing.
        ("timeout", "per-test time limit; honoured when pytest-timeout is installed"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {description}")


# ------------------------------------------------------------------- offline guards
@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch):
    """
    Switch off framework telemetry before anything is imported.

    This is not tidiness, it is a correctness fix. CrewAI ships an OpenTelemetry
    exporter that tries to phone home, and chromadb (pulled in transitively) does the
    same. With the socket guard below active, that background traffic surfaces as
    confusing errors in tests that have nothing to do with the network. Opting out is
    also the right default for a generated suite: nobody expects running tests to emit
    usage data.

    Applies to live tests too - a real model call still works with telemetry off.

    The values are spelled out per variable rather than derived, because they do not
    agree on polarity. `LANGCHAIN_TRACING_V2` is the trap: it is an *enable* flag, so
    setting it to "true" here would switch LangSmith tracing on, and the tracer would
    then try to POST every chain run straight into the socket guard below. chromadb's
    `ANONYMIZED_TELEMETRY` is likewise an enable flag and wants "False".
    """
    for var, value in (
        ("CREWAI_TELEMETRY_OPT_OUT", "true"),   # opt-out flag: true means "do not send"
        ("OTEL_SDK_DISABLED", "true"),          # disable flag
        ("ANONYMIZED_TELEMETRY", "False"),      # chromadb enable flag
        ("LANGCHAIN_TRACING_V2", "false"),      # LangSmith enable flag
        ("LANGCHAIN_TRACING", "false"),         # its pre-v2 spelling
        ("LANGSMITH_TRACING", "false"),         # its current spelling
        ("HF_HUB_DISABLE_TELEMETRY", "1"),      # disable flag
    ):
        monkeypatch.setenv(var, value)


@pytest.fixture(autouse=True)
def stub_credentials(request, monkeypatch):
    """
    Replace provider credentials with obviously-fake values.

    The previous generated suite did `os.environ.setdefault("OPENAI_API_KEY", "test-key")`
    and then made real calls, so it returned 401s. Here the fake key is paired with a
    network block, so nothing tries to authenticate in the first place.
    """
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-api-key")


@pytest.fixture(autouse=True)
def block_network(request, monkeypatch):
    """
    Fail fast if a non-live test tries to open a socket.

    This turns "these tests do not need the network" from an assumption into an
    enforced property. Without it, a refactor could silently reintroduce billed calls.

    Deliberately not a yield fixture: an early `return` in a generator fixture makes
    pytest raise "fixture did not yield a value" for every live test. monkeypatch
    already restores the original attribute at teardown, so nothing is lost.
    """
    if request.node.get_closest_marker("live"):
        return

    def guard(*args, **kwargs):
        raise RuntimeError(
            "Network access is blocked in offline tests. If this test genuinely needs "
            "a real model, mark it with @pytest.mark.live and run `pytest -m live`."
        )

    monkeypatch.setattr(socket, "socket", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


# ------------------------------------------------------------------------ fake model
@pytest.fixture
def fake_llm():
    """
    A deterministic chat model that never touches the network.

    LangChain ships fakes for exactly this purpose, but their import path has moved
    between versions, so each known location is tried in turn. If none is importable
    the test skips with a clear reason rather than falling back to a real model.
    """
    responses = ["This is a deterministic fake response for testing."] * 50

    candidates = [
        ("langchain_core.language_models.fake_chat_models", "FakeListChatModel"),
        ("langchain_core.language_models", "FakeListChatModel"),
        ("langchain_community.chat_models.fake", "FakeListChatModel"),
    ]
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            fake_class = getattr(module, class_name)
            return fake_class(responses=responses)
        except Exception:
            continue

    pytest.skip(
        "No fake chat model available. Install langchain-core to run tests that need "
        "a stand-in model."
    )


@pytest.fixture
def offline_llm():
    """
    A model object CrewAI accepts, that never makes a call.

    CrewAI validates the `llm` argument against its own `LLM` type, so a LangChain fake
    is not interchangeable here. Constructing `LLM` performs no request - the call only
    happens on kickoff, which offline tests never reach - and the obviously-fake key
    plus the socket guard mean an accidental request fails loudly rather than billing
    anyone.
    """
    crewai = pytest.importorskip("crewai", reason="crewai is not installed")
    return crewai.LLM(model="gpt-4o-mini", api_key="test-not-a-real-key")


# --------------------------------------------------------------------- live fixtures
@pytest.fixture
def agent_module():
    """
    Import the generated agent module, for live tests only.

    Set AGENT_MODULE to the module name if your generated file is not `agent_system`.
    """
    module_name = os.environ.get("AGENT_MODULE", "agent_system")
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(
            f"Could not import generated module {module_name!r} ({exc}). "
            "Save your generated code next to these tests, or set AGENT_MODULE."
        )


@pytest.fixture
def evaluator():
    """The output-quality evaluator, used by live quality tests."""
    try:
        from multi_agent_generator.evaluation import AgentEvaluator
    except ImportError:
        pytest.skip("multi-agent-generator is not installed; needed for quality scoring.")
    return AgentEvaluator()
