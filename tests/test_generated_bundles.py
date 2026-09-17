# tests/test_generated_bundles.py
"""
Self-tests for the generator.

These exist because the project's central promise - "the agents it creates can be
tested" - was never itself verified. Every generated framework file and every generated
test bundle is compiled here, and one bundle is actually executed by a child pytest
process. That last test is the important one: it proves a handed-over bundle runs on a
machine with nothing but pytest installed, which is precisely what used to fail.

Nothing here needs credentials, a network connection, or any agent framework installed.

    pip install pytest
    pytest
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

from multi_agent_generator.dependencies import (
    FRAMEWORK_REQUIREMENTS,
    check_dependencies,
    requirements_for,
    requirements_txt,
)
# Aliased on import: pytest tries to *collect* any module-level name starting with
# "Test", so importing TestGenerator/TestBundle/TestType unaliased fills the run with
# PytestCollectionWarning noise about classes it cannot instantiate.
from multi_agent_generator.evaluation.test_generator import (
    TestBundle as Bundle,
)
from multi_agent_generator.evaluation.test_generator import (
    TestGenerator as Generator,
)
from multi_agent_generator.evaluation.test_generator import (
    TestType as Kind,
)
from multi_agent_generator.frameworks import FRAMEWORKS, generate_code
from multi_agent_generator.frameworks._common import (
    collect_tools,
    sanitize_identifier,
    tool_class_name,
)
from multi_agent_generator.providers import (
    PROVIDERS,
    UnsupportedCombinationError,
    get_provider,
    llm_snippet,
    resolve_generator_model,
    resolve_runtime_model,
)

# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
#: A config deliberately chosen to be awkward: multi-word names, punctuation, a leading
#: digit, colliding tool names, a task pointing at a real agent and delegation enabled.
#: Friendly inputs would not have caught the identifier and duplicate-class bugs.
AWKWARD_CONFIG = {
    "agents": [
        {
            "name": "Research & Analysis",
            "role": "Senior Research Analyst",
            "goal": "Find and verify primary sources",
            "backstory": "Ten years in evidence synthesis.",
            "tools": ["web_search", "web-search", "search"],
            "allow_delegation": True,
            "llm": "gpt-4.1-mini",
        },
        {
            "name": "2nd Writer",
            "role": "Technical Writer",
            "goal": "Turn findings into prose",
            "backstory": "",
            "tools": ["summarizer"],
        },
    ],
    "tasks": [
        {
            "name": "Gather evidence",
            "description": "Collect primary sources on the query",
            "expected_output": "An annotated source list",
            "agent": "Research & Analysis",
        },
        {
            "name": "Draft report",
            "description": "Write the report from the sources",
            "expected_output": "A finished report",
            "agent": "2nd Writer",
        },
    ],
    "tools": [
        {
            "name": "web_search",
            "description": 'Search the web for "primary" sources',
            "parameters": {"query": "The search string"},
        },
        {"name": "summarizer", "description": "Condense long text"},
    ],
    "process": "sequential",
}

#: The provider/framework pairs that are expected to work. Agno reaches models only
#: through an OpenAI-compatible endpoint, so it supports a deliberately smaller set.
AGNO_PROVIDERS = ("openai", "huggingface", "huggingface-local", "ollama")


def _provider_framework_pairs():
    for framework in FRAMEWORKS:
        for provider in PROVIDERS:
            if framework == "agno" and provider not in AGNO_PROVIDERS:
                continue
            yield framework, provider


@pytest.fixture
def clean_env(monkeypatch):
    """Remove env vars that would otherwise leak into model resolution."""
    for var in ("DEFAULT_MODEL", "HF_MODEL_ID", "API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# --------------------------------------------------------------------------------------
# Generated agent code
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("framework,provider", list(_provider_framework_pairs()))
def test_generated_code_is_valid_python(framework, provider):
    """
    Every framework/provider pair must produce a file that parses.

    This is the check that would have caught agent names being interpolated straight
    into identifiers, and the f-string brace bugs in the CrewAI Flow generator.
    """
    code = generate_code(AWKWARD_CONFIG, framework, provider=provider)
    compile(code, f"{framework}_{provider}.py", "exec")


@pytest.mark.parametrize("framework,provider", list(_provider_framework_pairs()))
def test_generated_code_defines_a_runner(framework, provider):
    """The generated module exposes a callable entry point."""
    code = generate_code(AWKWARD_CONFIG, framework, provider=provider)
    tree = ast.parse(code)
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert functions & {"run_workflow", "run_agent"}, (
        f"{framework}/{provider} defines no run_workflow or run_agent: {sorted(functions)}"
    )


@pytest.mark.parametrize("framework,provider", list(_provider_framework_pairs()))
def test_header_banner_command_is_runnable(framework, provider):
    """
    The `--check-deps` command in the generated banner must use real CLI values.

    The banner used to interpolate the display label ("CrewAI Flow", "ReAct (classic)"),
    producing a command argparse rejects - advice that cannot be followed is worse than
    no advice.
    """
    code = generate_code(AWKWARD_CONFIG, framework, provider=provider)
    marker = "--framework "
    line = next(ln for ln in code.splitlines() if marker in ln)
    quoted_framework = line.split(marker, 1)[1].split()[0]
    assert quoted_framework in FRAMEWORKS
    assert quoted_framework == framework


@pytest.mark.parametrize("framework,provider", list(_provider_framework_pairs()))
def test_banner_model_matches_generated_code(framework, provider):
    """The model named in the banner is the one the code below it constructs."""
    code = generate_code(AWKWARD_CONFIG, framework, provider=provider)
    banner_model = next(
        ln.split("Model:", 1)[1].strip()
        for ln in code.splitlines()
        if ln.startswith("Model:")
    )
    body = code.split('"""', 2)[-1]
    assert banner_model in body, (
        f"{framework}/{provider} banner advertises {banner_model!r}, which does not "
        "appear in the generated code"
    )


def test_agno_rejects_unsupported_providers():
    """An unsupported combination raises rather than emitting broken code."""
    for provider in PROVIDERS:
        if provider in AGNO_PROVIDERS:
            continue
        with pytest.raises(UnsupportedCombinationError):
            generate_code(AWKWARD_CONFIG, "agno", provider=provider)


def test_config_suggested_model_does_not_leak_across_providers():
    """
    A config that names an OpenAI model must not reach a non-OpenAI provider.

    The analysis prompt asks the model for a name like `gpt-4.1-mini`, and the fallback
    configs hardcode it, so a permissive rule emitted ChatGroq(model="gpt-4.1-mini") -
    code that fails on its first call.
    """
    for provider in ("groq", "anthropic", "ollama", "watsonx"):
        code = generate_code(AWKWARD_CONFIG, "langgraph", provider=provider)
        assert "gpt-4.1-mini" not in code, f"OpenAI model name leaked into {provider}"


def test_hub_repo_id_is_honoured_for_huggingface():
    """A Hub-style id in the config is used for the Hugging Face providers."""
    config = dict(AWKWARD_CONFIG, model_id="mistralai/Mistral-7B-Instruct-v0.3")
    code = generate_code(config, "langgraph", provider="huggingface")
    assert "mistralai/Mistral-7B-Instruct-v0.3" in code


def test_tool_class_collisions_are_dropped():
    """
    Colliding tool names must not produce two classes with the same name.

    `web_search`, `web-search` and `search` all end in a class called something ending
    "Tool"; emitting duplicates means the second definition silently shadows the first.
    """
    tools = collect_tools(AWKWARD_CONFIG["agents"])
    class_names = [tool_class_name(t) for t in tools]
    assert len(class_names) == len(set(class_names)), class_names


def test_generated_code_is_deterministic():
    """Two runs on the same input produce identical bytes, so diffs stay meaningful."""
    for framework in FRAMEWORKS:
        first = generate_code(AWKWARD_CONFIG, framework, provider="openai")
        second = generate_code(AWKWARD_CONFIG, framework, provider="openai")
        assert first == second, f"{framework} output is not deterministic"


def test_awkward_names_become_valid_identifiers():
    for raw in ("Research & Analysis", "2nd Writer", "class", "", "!!!", "  "):
        assert sanitize_identifier(raw, "agent").isidentifier(), raw
    # Reserved words must be escaped, not merely sanitised.
    assert sanitize_identifier("class") == "class_"
    # A leading digit cannot start an identifier, so the prefix has to be prepended.
    assert sanitize_identifier("2nd Writer", "agent").startswith("agent_")


# --------------------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------------------
def test_generator_model_gets_litellm_route_prefix(clean_env):
    """
    LiteLLM routes on the `provider/` prefix, so an unprefixed id goes to OpenAI.

    A bare `Qwen/Qwen2.5-7B-Instruct` used to be sent to OpenAI and fail with an
    authentication error that named the wrong provider entirely.
    """
    resolved = resolve_generator_model("huggingface", "Qwen/Qwen2.5-7B-Instruct")
    assert resolved == "huggingface/Qwen/Qwen2.5-7B-Instruct"

    # Already-prefixed ids are left alone rather than double-prefixed.
    assert (
        resolve_generator_model("huggingface", "huggingface/Qwen/Qwen2.5-7B-Instruct")
        == "huggingface/Qwen/Qwen2.5-7B-Instruct"
    )


def test_default_model_env_var_respects_selected_provider(clean_env):
    """A DEFAULT_MODEL in .env must not silently redirect a chosen provider to OpenAI."""
    clean_env.setenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
    assert resolve_generator_model("groq") == "groq/llama-3.3-70b-versatile"


def test_local_provider_keeps_bare_hub_id(clean_env):
    """
    The transformers backend loads a bare Hub repo id, so it must never be prefixed.

    Prefixing here meant AutoTokenizer.from_pretrained() was handed a LiteLLM route
    string and failed with a confusing "repo not found".
    """
    resolved = resolve_generator_model("huggingface-local", "Qwen/Qwen2.5-7B-Instruct")
    assert resolved == "Qwen/Qwen2.5-7B-Instruct"


def test_runtime_model_strips_redundant_prefix():
    """`--model huggingface/X` must not become `huggingface/huggingface/X` in output."""
    assert (
        resolve_runtime_model("huggingface", "huggingface/Qwen/Qwen2.5-7B-Instruct")
        == "Qwen/Qwen2.5-7B-Instruct"
    )
    # A repo id whose first segment merely resembles a prefix is left intact.
    assert (
        resolve_runtime_model("watsonx", "meta-llama/llama-3-3-70b-instruct")
        == "meta-llama/llama-3-3-70b-instruct"
    )
    assert (
        resolve_runtime_model("huggingface-local", "Qwen/Qwen2.5-7B-Instruct")
        == "Qwen/Qwen2.5-7B-Instruct"
    )


def test_snippet_records_the_model_it_used():
    for provider in PROVIDERS:
        snippet = llm_snippet("langgraph", provider)
        assert snippet.model, f"{provider} snippet did not record its model"
        assert snippet.model in snippet.setup


def test_provider_aliases_resolve():
    assert get_provider("hf").name == "huggingface"
    assert get_provider("HF-Local").name == "huggingface-local"


def test_unknown_provider_raises_an_actionable_error():
    """
    An unknown provider must surface as an application error, not a bare ValueError.

    The distinction is load-bearing: the API's error handler turns an AppError into a clean
    400 carrying the remediation text, while a ValueError would escape as a 500 with an
    incident id and tell the user nothing about the typo they made.
    """
    from multi_agent_generator.errors import UnknownProviderError

    with pytest.raises(UnknownProviderError) as caught:
        get_provider("not-a-provider")

    error = caught.value
    assert "not-a-provider" in error.message
    # The remediation has to name the valid options, or it is not actionable.
    assert "openai" in (error.action or "")
    assert error.http_status == 400


# --------------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_framework_declares_its_package(framework):
    """
    The original bug in one assertion: generated CrewAI code needed `crewai`, and
    nothing anywhere said so.
    """
    assert FRAMEWORK_REQUIREMENTS.get(framework), f"{framework} declares no packages"


def test_requirements_include_test_runner_and_fake_model():
    """
    A generated suite needs pytest-timeout for its @pytest.mark.timeout decorators and
    langchain-core for the fake chat model - even a CrewAI suite, which would otherwise
    never pull langchain-core in.
    """
    packages = {r.package for r in requirements_for("crewai", "openai", include_tests=True)}
    assert {"crewai", "pytest", "pytest-timeout", "langchain-core"} <= packages


def test_requirements_txt_is_installable_syntax():
    text = requirements_txt("crewai", "huggingface", include_tests=True)
    specs = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert specs
    for spec in specs:
        assert " " not in spec, f"malformed requirement line: {spec!r}"


def test_dependency_report_is_honest_about_what_is_missing():
    report = check_dependencies("crewai", "openai")
    assert report.installed or report.missing
    assert report.ok == (not report.missing)
    rendered = report.render()
    assert "Dependency check" in rendered
    if report.missing:
        assert "pip install" in report.install_command()


# --------------------------------------------------------------------------------------
# Generated test bundles
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_bundle_files_are_valid_python(framework):
    bundle = Generator().generate_bundle(AWKWARD_CONFIG, framework, "openai")
    assert isinstance(bundle, Bundle)
    for filename, content in bundle.files.items():
        if filename.endswith(".py"):
            compile(content, filename, "exec")


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_bundle_is_complete(framework):
    """
    A bundle has to carry everything needed to run, not just the test file.

    Handing over a bare .py was the whole problem: the recipient had no requirements
    file, no marker registration and no offline fixtures.
    """
    bundle = Generator().generate_bundle(AWKWARD_CONFIG, framework, "openai")
    expected = {
        "conftest.py",
        "_helpers.py",
        "pytest.ini",
        "requirements-test.txt",
        "README.md",
    }
    assert expected <= set(bundle.files)
    assert any(n.startswith("test_") and n.endswith(".py") for n in bundle.files)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_bundle_requirements_name_the_framework_package(framework):
    bundle = Generator().generate_bundle(AWKWARD_CONFIG, framework, "openai")
    requirements = bundle.files["requirements-test.txt"]
    for req in FRAMEWORK_REQUIREMENTS[framework]:
        assert req.package in requirements


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_generated_test_asserts_something(framework):
    """
    No test may pass without checking anything.

    The previous generator emitted `result = None` / `assert True` with the intended
    checks left as English prose in comments, so the suite reported green while
    exercising nothing at all. This is the guard against that returning.
    """
    bundle = Generator().generate_bundle(AWKWARD_CONFIG, framework, "openai")
    test_file = next(n for n in bundle.files if n.startswith("test_"))
    tree = ast.parse(bundle.files[test_file])

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert functions, "bundle generated no test functions"

    for func in functions:
        has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(func))
        # `with pytest.raises(...)` is a real check even though it emits no `assert`.
        has_raises = any(
            isinstance(n, ast.Attribute) and n.attr == "raises" for n in ast.walk(func)
        )
        assert has_assert or has_raises, f"{func.name} contains no assertion"


def test_contract_tests_import_nothing_third_party():
    """
    Contract tests must run on a machine with nothing but pytest.

    That is what makes the suite worth shipping: whatever else is missing, the
    configuration itself still gets validated.
    """
    suite = Generator().generate_test_suite(
        AWKWARD_CONFIG, "crewai", [Kind.CONTRACT]
    )
    assert suite.test_cases
    for test in suite.test_cases:
        for line in test.body:
            stripped = line.strip()
            assert not stripped.startswith(("import ", "from ")), (
                f"{test.name} imports {stripped!r}; contract tests must be dependency-free"
            )


def test_live_tests_are_marked_and_deselected():
    """
    Tests that cost money are opt-in.

    Checked by reading the decorators off the parsed tree rather than by slicing the
    rendered text: a string search would keep passing if the mark drifted onto the
    wrong function.
    """
    generator = Generator()
    suite = generator.generate_test_suite(AWKWARD_CONFIG, "crewai")
    live_names = {t.name for t in suite.test_cases if t.live}
    assert live_names, "no live tests were generated"

    tree = ast.parse(generator.generate_pytest_file(suite))
    marked = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(dec, ast.Attribute) and dec.attr == "live"
            for dec in node.decorator_list
        )
    }
    assert marked == live_names, (
        f"live marks do not match the live tests: marked={sorted(marked)}, "
        f"expected={sorted(live_names)}"
    )

    assert '-m "not live"' in generator.generate_pytest_ini()


def test_offline_tests_never_set_a_plausible_real_key():
    """
    Credentials in the generated conftest must be obviously fake.

    The previous suite did `setdefault("OPENAI_API_KEY", "test-key")` and then made real
    calls, so it returned 401s. Here the fake key is paired with a socket block.
    """
    conftest = Generator().generate_conftest("crewai", "huggingface")
    assert "test-hf-token" in conftest
    assert "socket" in conftest
    assert "monkeypatch.setattr(socket" in conftest
    # sk- is the shape of a real OpenAI key; nothing here should resemble one.
    assert "sk-" not in conftest


def test_block_network_fixture_is_not_a_broken_generator():
    """
    A fixture that mixes an early `return` with a `yield` yields nothing, and pytest
    then errors with "fixture did not yield a value" for every live test. Assert the
    shape directly rather than trusting it.
    """
    conftest = Generator().generate_conftest("crewai", "openai")
    tree = ast.parse(conftest)
    fixture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "block_network"
    )
    yields = [n for n in ast.walk(fixture) if isinstance(n, (ast.Yield, ast.YieldFrom))]
    returns = [n for n in ast.walk(fixture) if isinstance(n, ast.Return)]
    assert not (yields and returns), (
        "block_network mixes yield with an early return; it must be one or the other"
    )


def test_unknown_test_type_is_rejected():
    with pytest.raises(ValueError):
        Generator().generate_test_suite(AWKWARD_CONFIG, "crewai", test_types=["nope"])


def test_bundle_save_writes_files(tmp_path):
    written = Generator().generate_bundle(AWKWARD_CONFIG, "crewai", "openai").save(
        str(tmp_path / "generated_tests")
    )
    assert written
    for path in written:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


def test_suite_save_round_trips(tmp_path):
    """TestSuite.save is the API the README advertises; it must actually exist."""
    suite = Generator().generate_test_suite(AWKWARD_CONFIG, "langgraph")
    written = suite.save(str(tmp_path / "suite"))
    assert any(p.endswith("conftest.py") for p in written)


# --------------------------------------------------------------------------------------
# The end-to-end proof: run a generated bundle in a child pytest
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("framework", ["crewai", "langgraph"])
def test_generated_bundle_actually_passes(tmp_path, framework):
    """
    Execute a generated bundle in a child pytest process and require a clean run.

    This is the test that answers the original complaint. It runs with no credentials,
    no network and (unless they happen to be installed) no framework packages: contract
    tests must pass, framework tests must *skip* rather than error, and live tests must
    be deselected. Exit code 0 is the whole assertion.
    """
    bundle_dir = tmp_path / framework
    Generator().generate_bundle(AWKWARD_CONFIG, framework, "openai").save(
        str(bundle_dir)
    )

    env = dict(os.environ)
    # Strip real credentials so a developer's own key cannot make this pass by accident.
    for var in ("OPENAI_API_KEY", "API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        env.pop(var, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--no-header"],
        cwd=str(bundle_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, textwrap.dedent(
        f"""
        Generated {framework} bundle did not pass.

        --- stdout ---
        {result.stdout}
        --- stderr ---
        {result.stderr}
        """
    )
    # A run in which everything skipped would also exit 0, which would hide a
    # regression, so require that the dependency-free tier really executed.
    assert "passed" in result.stdout


def test_generated_bundle_cannot_reach_the_network(tmp_path):
    """
    The socket guard must actually bite.

    A test that merely *intends* to be offline is worth little; this drops a probe into
    a generated bundle and requires that opening a socket fails.
    """
    bundle_dir = tmp_path / "guarded"
    Generator().generate_bundle(AWKWARD_CONFIG, "crewai", "openai").save(
        str(bundle_dir)
    )

    (bundle_dir / "test_zz_network_probe.py").write_text(
        textwrap.dedent(
            """
            import socket

            import pytest


            @pytest.mark.contract
            def test_socket_is_blocked():
                with pytest.raises(RuntimeError):
                    socket.socket()
            """
        ).lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "test_zz_network_probe.py",
        ],
        cwd=str(bundle_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
