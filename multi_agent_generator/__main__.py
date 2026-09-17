# mutli-agent-generator/__main__.py
"""
Command line interface for multi-agent-generator.

The default command runs the *same* pipeline the web API runs. That is the point of
:func:`multi_agent_generator.core.service.build_service`: analysis, framework selection,
generation, review, the improvement loop and the test run are defined once, and the CLI and the
API are two front doors onto them. A bug fixed for one is fixed for both, and neither can
quietly acquire behaviour the other lacks.

Two older paths are kept deliberately rather than deleted:

* ``--config PATH`` generates from a configuration file with no model call at all, which is what
  makes code and test generation usable with no credential and in CI.
* ``--legacy`` runs the original single-file generator. It is narrower than the pipeline (no
  review, no improvement, no test execution) and says so, but removing a working code path
  because a better one exists is how you break someone's script.

Exit codes are honest about verification. A run that produced files but whose tests never ran is
reported as such and, under ``--strict``, exits non-zero - because "it generated something" and
"it works" are different claims and the CLI must not conflate them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from .dependencies import check_dependencies, requirements_txt
from .errors import AppError, ConfigurationError
from .evaluation.evaluator import AgentEvaluator
from .evaluation.test_generator import TestGenerator
from .frameworks import FRAMEWORKS, generate_code
from .generator import AgentGenerator
from .orchestration.orchestrator import Orchestrator
from .orchestration.patterns import PatternType
from .providers import PROVIDERS, resolve_generator_model, resolve_runtime_model
from .settings import get_settings
from .tools.tool_generator import ToolGenerator
from .tools.tool_registry import get_tool_registry, ToolCategory


def _render_app_error(exc: AppError) -> str:
    """
    Format an application error for a terminal.

    Users get the message and the suggested action; the traceback stays out of the way.
    Section 21 of the brief is explicit about this: a wall of frames from inside a vendor
    SDK tells the user nothing they can act on. The full exception is still available with
    ``--debug``, and is what the API logs server-side.
    """
    lines = [f"Error: {exc.message}"]
    if exc.action:
        lines.append(f"  What to do: {exc.action}")
    if exc.detail:
        lines.append(f"  Details:    {exc.detail}")
    return "\n".join(lines)


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-agent AI code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate, review, improve and test a project - the full pipeline
  multi-agent-generator "Create a research assistant"

  # Force a framework instead of letting the selector choose
  multi-agent-generator "Create a research assistant" --framework langgraph

  # Fail the command unless review passed and the generated tests passed
  multi-agent-generator "Create a research assistant" --strict

  # No credential, no network: heuristic analysis and review only
  multi-agent-generator "Create a research assistant" --offline

  # Start the web app (API + frontend) on http://127.0.0.1:8000
  multi-agent-generator --serve

  # Use an open-source model hosted on Hugging Face
  multi-agent-generator "Create a research assistant" --provider huggingface

  # Run a small open-source model entirely on your own machine
  multi-agent-generator "Create a research assistant" --provider huggingface-local

  # See which providers are available and what each needs
  multi-agent-generator --list-providers

  # Check that everything a generated project needs is installed
  multi-agent-generator --check-deps --framework crewai --provider huggingface

  # The original single-file generator, with a test bundle
  multi-agent-generator "Create a research assistant" --legacy --framework crewai \\
      --output agent_system.py --generate-tests

  # Generate tests from a config you already have, with no LLM call
  multi-agent-generator --generate-tests --config config.json --framework crewai

  # Generate a custom tool
  multi-agent-generator --tool "Create a tool to fetch weather data from an API"

  # Evaluate agent output
  multi-agent-generator --evaluate --query "What is AI?" --response "AI is artificial intelligence..."

  # Suggest orchestration pattern
  multi-agent-generator --orchestrate "I need agents to debate and reach consensus"

  # List available tools
  multi-agent-generator --list-tools

  # List orchestration patterns
  multi-agent-generator --list-patterns
        """
    )
    parser.add_argument("prompt", nargs="?", help="Plain English description of what you need")
    parser.add_argument(
        "--framework",
        choices=FRAMEWORKS,
        default=None,
        help="Agent framework to use. Omit it and the pipeline picks one and explains why "
             "(the legacy path falls back to crewai)"
    )
    parser.add_argument(
        "--process",
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Process type for CrewAI (default: sequential)"
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default=None,
        help="LLM provider for the model that designs the agents. 'huggingface' uses hosted "
             "open-source models, 'huggingface-local' runs them on this machine "
             "(default: the DEFAULT_PROVIDER setting)"
    )
    parser.add_argument(
        "--model",
        help="Explicit model id, overriding the provider default "
             "(e.g. Qwen/Qwen2.5-7B-Instruct for Hugging Face)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: print to console)"
    )
    parser.add_argument(
        "--format",
        choices=["code", "json", "both"],
        default="code",
        help="Output format (default: code)"
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Load an agent configuration from a JSON file instead of analyzing a "
             "prompt. Lets you generate code or tests without any LLM call."
    )

    # Provider and dependency arguments
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List supported LLM providers, their default models and their credentials"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check whether the packages a generated project needs are installed, "
             "and print the pip command to fix any that are not"
    )
    parser.add_argument(
        "--emit-requirements",
        action="store_true",
        help="Print a requirements.txt for the chosen framework and provider"
    )

    # Test generation arguments
    parser.add_argument(
        "--generate-tests",
        action="store_true",
        help="Also generate a runnable test suite (test module, conftest.py, pytest.ini, "
             "requirements-test.txt and a README). Tests run offline by default."
    )
    parser.add_argument(
        "--test-dir",
        default="generated_tests",
        metavar="DIR",
        help="Directory to write the generated test bundle into (default: generated_tests)"
    )
    
    # Tool generation arguments
    parser.add_argument(
        "--tool",
        metavar="DESCRIPTION",
        help="Generate a custom tool from description"
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools in the registry"
    )
    parser.add_argument(
        "--tool-category",
        choices=[c.value for c in ToolCategory],
        help="Filter tools by category when using --list-tools"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate agent output quality"
    )
    parser.add_argument(
        "--query",
        help="Query/prompt for evaluation (used with --evaluate)"
    )
    parser.add_argument(
        "--response",
        help="Agent response to evaluate (used with --evaluate)"
    )
    parser.add_argument(
        "--expected",
        help="Expected output for accuracy comparison (optional, used with --evaluate)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum passing score threshold (default: 0.7)"
    )
    
    # Orchestration arguments
    parser.add_argument(
        "--orchestrate",
        metavar="DESCRIPTION",
        help="Get orchestration pattern suggestion for a task description"
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List all available orchestration patterns"
    )
    parser.add_argument(
        "--pattern",
        choices=[p.value for p in PatternType],
        help="Generate code for a specific orchestration pattern"
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of agents for orchestration (default: 3)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full traceback on error instead of a one-line message"
    )

    # ---------------------------------------------------------------- pipeline / server
    pipeline_group = parser.add_argument_group("full pipeline")
    pipeline_group.add_argument(
        "--out-dir",
        metavar="DIR",
        help="Directory to write the generated multi-file project into "
             "(default: generated/<slug>). Use --output for the single-file legacy path."
    )
    pipeline_group.add_argument(
        "--max-iterations",
        type=int,
        metavar="N",
        help="Cap on review/improve passes (default: the MAX_GENERATION_ITERATIONS setting)"
    )
    pipeline_group.add_argument(
        "--no-tests",
        action="store_true",
        help="Do not generate a test suite. Implies --no-run-tests, and means the project can "
             "never be reported as verified."
    )
    pipeline_group.add_argument(
        "--no-run-tests",
        action="store_true",
        help="Generate tests but do not execute them. The result is reported as unverified."
    )
    pipeline_group.add_argument(
        "--offline",
        action="store_true",
        help="Skip every model call: analysis and review fall back to the built-in heuristics. "
             "Needs no credential, and produces a shallower project."
    )
    pipeline_group.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not record this run in the local database"
    )
    pipeline_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless the project passed review and its tests ran and passed. "
             "Use this in CI."
    )
    pipeline_group.add_argument(
        "--legacy",
        action="store_true",
        help="Use the original single-file generator instead of the pipeline: no review, no "
             "improvement loop and no test execution."
    )

    server_group = parser.add_argument_group("web application")
    server_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the web application (API plus the built frontend, if present)"
    )
    server_group.add_argument("--host", help="Host to bind when serving (default: the HOST setting)")
    server_group.add_argument(
        "--port", type=int, help="Port to bind when serving (default: the PORT setting)"
    )


    args = parser.parse_args()

    # `.env` is loaded once, here, through the settings singleton - not as an import-time
    # side effect of some other module. This is the only place the CLI touches it.
    get_settings()

    # Every expected failure below raises an AppError, which we render as a short,
    # actionable message. Anything unexpected still gets a traceback, because that is a bug
    # in this tool rather than something the user can fix. `--debug` forces the traceback
    # in both cases.
    try:
        _dispatch(args, parser)
    except AppError as exc:
        if args.debug:
            raise
        print(_render_app_error(exc), file=sys.stderr)
        raise SystemExit(1)


def _dispatch(args, parser):
    """Run the command described by parsed ``args``."""

    # Informational commands that need neither a prompt nor credentials.
    if args.serve:
        handle_serve(args.host, args.port)
        return

    if args.list_providers:
        handle_list_providers()
        return

    if args.check_deps:
        report = check_dependencies(args.framework or "crewai", args.provider or "openai")
        print(report.render())
        # A non-zero exit makes this usable as a CI preflight gate.
        raise SystemExit(0 if report.ok else 1)

    if args.emit_requirements:
        text = requirements_txt(
            args.framework or "crewai", args.provider or "openai", include_tests=True
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
            print(f"Requirements written to {args.output}")
        else:
            print(text, end="")
        return

    # Handle tool listing
    if args.list_tools:
        handle_list_tools(args.tool_category)
        return

    # Handle tool generation
    if args.tool:
        handle_tool_generation(args.tool, args.output)
        return

    # Handle evaluation
    if args.evaluate:
        if not args.query or not args.response:
            parser.error("--evaluate requires both --query and --response")
        handle_evaluation(args.query, args.response, args.expected, args.threshold, args.output)
        return

    # Handle orchestration pattern listing
    if args.list_patterns:
        handle_list_patterns()
        return

    # Handle orchestration suggestion
    if args.orchestrate:
        handle_orchestration(
            args.orchestrate, args.pattern, args.num_agents,
            args.framework or "langgraph", args.output,
        )
        return

    # Handle pattern code generation
    if args.pattern:
        handle_orchestration(
            None, args.pattern, args.num_agents, args.framework or "langgraph", args.output
        )
        return

    # Default: agent code generation.
    #
    # Three routes, in order of preference. The pipeline is the product; --config is the
    # no-credential path that generates from an existing configuration; --legacy is the old
    # single-file generator, kept working for scripts that already call it.
    if not args.config and not args.legacy:
        if not args.prompt:
            parser.error(
                "a prompt is required for agent code generation "
                "(or pass --config PATH to use an existing configuration, "
                "or --serve to start the web application)"
            )
        handle_pipeline(args)
        return

    # The two legacy routes below need concrete names, where the pipeline is happy with None
    # and resolves its own defaults. Keeping the old defaults here means an existing command
    # line behaves exactly as it did.
    framework = args.framework or "crewai"
    provider = args.provider or get_settings().default_provider

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        print(f"Loaded configuration from {args.config}")
    else:
        if not args.prompt:
            parser.error(
                "a prompt is required for agent code generation "
                "(or pass --config PATH to use an existing configuration)"
            )
        generator = AgentGenerator(provider=args.provider, model=args.model)
        print(f"Analyzing prompt using {args.provider or 'the default provider'}...")
        config = generator.analyze_prompt(args.prompt, framework)

    # Add process type to config for CrewAI frameworks
    if framework in ["crewai", "crewai-flow"]:
        config["process"] = args.process
        print(f"Using {args.process} process for CrewAI...")

    # Generate code. Dispatch lives in frameworks/__init__.py so the provider and model
    # are threaded through in one place instead of a hand-written if/elif chain that has
    # to be edited every time a framework is added.
    runtime_model = resolve_runtime_model(provider, args.model)
    print(f"Generating {framework} code for {provider} ({runtime_model})...")
    code = generate_code(config, framework, provider=provider, model=args.model)

    # Prepare output
    if args.format == "code":
        output = code
    elif args.format == "json":
        output = json.dumps(config, indent=2)
    else:  # both
        output = f"// Configuration:\n{json.dumps(config, indent=2)}\n\n// Generated Code:\n{code}"

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output successfully written to {args.output}")
    else:
        print(output)

    if args.generate_tests:
        handle_generate_tests(config, framework, provider, args.test_dir)

    # Finish with a dependency check so nobody discovers a missing package the hard way,
    # the way `crewai` used to go missing from generated CrewAI projects.
    report = check_dependencies(framework, provider, include_tests=args.generate_tests)
    if not report.missing:
        return
    print("\nBefore running the generated code, install its dependencies:")
    for req in report.missing:
        print(f"  - {req.package:<26} {req.reason}")
    print(f"\n  {report.install_command()}")


def handle_serve(host=None, port=None):
    """
    Start the web application.

    The import is deferred because the API's dependencies (FastAPI, uvicorn) are an optional
    extra: `multi-agent-generator "..."` must keep working on an install that has no web
    server, so importing one at module scope would be wrong. A missing package is reported as
    an install command rather than an ImportError traceback.
    """
    try:
        from backend.app import run as run_server
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigurationError(
            "The web application's dependencies are not installed.",
            action="Install them with: pip install 'multi-agent-generator[api]'",
            detail=str(exc),
        ) from exc

    settings = get_settings()
    bind_host = host or settings.host
    bind_port = port or settings.port
    print(f"Starting the web application on http://{bind_host}:{bind_port}")
    print("  API docs:  /docs")
    print("  Frontend:  built files from frontend/dist, if present")
    print("Press Ctrl+C to stop.\n")
    run_server(host=bind_host, port=bind_port)


def handle_pipeline(args):
    """
    Run the full pipeline and write the result to disk.

    Progress is printed as it happens rather than after the fact. A generation run makes
    several model calls and can execute a test suite, so a silent terminal for a minute is
    indistinguishable from a hang - the same reason the API streams its events.
    """
    from .core.models import GeneratedProject, PipelineResult
    from .core.service import build_service

    include_tests = not args.no_tests
    run_tests = include_tests and not args.no_run_tests

    service = build_service(persist=not args.no_persist)

    print(f'Requirement: "{args.prompt}"')
    if args.framework:
        print(f"Framework:   {args.framework} (your choice - the selector will not override it)")
    else:
        print("Framework:   chosen by the selector")
    if args.offline:
        print("Model:       none - heuristic analysis and review only")
    print()

    result: PipelineResult = service.generate(
        args.prompt,
        framework=args.framework,
        provider=args.provider,
        model=args.model,
        include_tests=include_tests,
        run_tests=run_tests,
        use_model=not args.offline,
        max_iterations=args.max_iterations,
        on_event=_print_event,
        metadata={"source": "cli"},
    )

    print()
    project: GeneratedProject = result.project

    if project is None or not project.files:
        # Nothing was produced. The pipeline records why, and that reason is the whole
        # message - a generic "generation failed" would be useless here.
        if result.error:
            raise AppError(
                result.error.get("message") or "Generation produced no files.",
                action=result.error.get("action"),
                detail=result.error.get("detail"),
            )
        raise AppError(
            "Generation produced no files.",
            action="Run again with --debug, or check the log for the failing stage.",
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path("generated") / _slug(args.prompt)
    written = _write_project(project, out_dir)

    print(f"Wrote {len(written)} files to {out_dir}{os.sep}")
    for path in written:
        print(f"  {path}")

    _print_verdict(result, out_dir)

    if args.format in ("json", "both"):
        payload = json.dumps(result.as_dict(), indent=2, default=str)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"\nFull run record written to {args.output}")
        else:
            print("\n" + payload)

    # --strict is what makes this usable as a gate. Without it the command reports the
    # problem and still exits 0, because the files are real and worth keeping.
    if args.strict and not result.succeeded:
        raise SystemExit(1)


def _print_event(event):
    """Render one pipeline event as a single line."""
    marks = {"started": "…", "succeeded": "ok", "failed": "!!", "warning": " ~", "info": "  "}
    mark = marks.get(event.status, "  ")
    stage = getattr(event.stage, "value", str(event.stage))
    took = f" ({event.duration_s:.1f}s)" if event.duration_s else ""
    iteration = f" [pass {event.iteration}]" if event.iteration else ""
    print(f"  {mark} {stage:<20}{iteration} {event.message}{took}")


def _print_verdict(result, out_dir: Path):
    """
    Say plainly whether the project is verified, and if not, why not.

    This is the part of the CLI that must not flatter the result. A project whose tests never
    ran is printed as unverified with the reasons listed, because the alternative - a green
    "done!" over an untested project - is exactly the failure mode the brief forbids.
    """
    review = result.final_review
    tests = result.final_tests

    print()
    if review is not None:
        verdict = "passed" if review.passed else "did not pass"
        blockers = len(review.blockers)
        print(f"Review:  {review.score:.1f}/10 - {verdict}", end="")
        print(f", {blockers} blocker(s)" if blockers else "")
        for change in review.required_changes[:5]:
            print(f"           - {change}")

    if tests is not None:
        if tests.timed_out:
            state = "timed out"
        elif not tests.ran:
            state = f"did not run ({tests.unavailable_reason or tests.status})"
        elif tests.ok:
            state = "passed"
        else:
            state = "failed"
        print(
            f"Tests:   {state} - {tests.passed_count} passed, "
            f"{tests.failed_count} failed, {tests.skipped_count} skipped"
        )
        for name in tests.failed_tests[:5]:
            print(f"           - {name}")
        # Refused requirement lines, when there were any. Printed next to the verdict because
        # they are usually the reason an import failed, and a refusal that only exists in a JSON
        # field is one the CLI user never sees.
        for note in tests.notes[:5]:
            print(f"           ! {note}")

    if result.succeeded:
        print("\nVerified: review passed and the generated tests ran and passed.")
    else:
        print("\nNot verified. Reasons:")
        for reason in result.unmet_criteria() or ["the pipeline did not report a reason"]:
            print(f"  - {reason}")
        print("The files above are real and complete; they have simply not passed the gate.")

    entry = result.project.entrypoint if result.project else None
    if entry:
        print(f"\nRun it with:\n  cd {out_dir}\n  pip install -r requirements.txt\n  python {entry}")


def _write_project(project, out_dir: Path):
    """
    Write a generated project to ``out_dir``, refusing any path that escapes it.

    The path check is not paranoia about our own generator: file paths in a project come from
    model output, and "write whatever path the model said" is how a generated file lands in
    the user's home directory.
    """
    root = out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for file in project.files:
        target = (root / file.path).resolve()
        if root != target and root not in target.parents:
            raise AppError(
                f"Refusing to write outside the output directory: {file.path}",
                action="This is a bug in the generated file list; please report it.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
        written.append(file.path)
    return written


def _slug(text: str, limit: int = 40) -> str:
    """A short, filesystem-safe directory name derived from the requirement."""
    keep = [c.lower() if c.isalnum() else "-" for c in text.strip()[:limit]]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "project"


def handle_list_providers():
    """List supported providers, their default models and their credentials."""
    print("\nSupported LLM providers\n" + "=" * 72)
    for name, spec in PROVIDERS.items():
        creds = ", ".join(spec.credential_env) if spec.credential_env else "none"
        status = "configured" if (not spec.credential_env or spec.resolve_credential()) else "not configured"
        print(f"\n  {name}  ({spec.label}) - {status}")
        print(f"    {spec.description}")
        print(f"    Generator model: {resolve_generator_model(name)}")
        print(f"    Generated code:  {resolve_runtime_model(name)}")
        print(f"    Credentials:     {creds}")
        if spec.extras:
            extras = ",".join(spec.extras)
            print(f"    Install:         pip install 'multi-agent-generator[{extras}]'")
        else:
            print("    Install:         nothing extra needed")
        if spec.notes:
            print(f"    Note: {spec.notes}")
    print(
        "\nOpen-source options: 'huggingface' runs hosted open-weight models, "
        "'huggingface-local' runs them on this machine, 'ollama' uses a local Ollama server."
    )


def handle_generate_tests(config, framework: str, provider: str, test_dir: str):
    """
    Write a runnable test bundle for a generated agent system.

    A bare test file is not enough to hand over: it needs its dependencies recorded, its
    markers registered and fixtures that keep it offline. That is why this writes a
    directory rather than a single file.
    """
    print(f"\nGenerating test suite in {test_dir}/ ...")
    bundle = TestGenerator().generate_bundle(config, framework, provider)
    written = bundle.save(test_dir)
    for path in written:
        print(f"  wrote {path}")
    print(
        f"\nRun them with:\n"
        f"  pip install -r {os.path.join(test_dir, 'requirements-test.txt')}\n"
        f"  cd {test_dir} && pytest\n"
        "\nThe suite runs offline: no API key, no network. Tests that need a real model "
        "are marked 'live' and are skipped unless you run `pytest -m live`."
    )


def handle_list_tools(category_filter: str = None):
    """List all available tools in the registry."""
    registry = get_tool_registry()
    
    if category_filter:
        category = ToolCategory(category_filter)
        tools = registry.list_by_category(category)
        print(f"\n📦 Tools in category '{category_filter}':\n")
    else:
        tools = registry.list_all()
        print("\n📦 All Available Tools:\n")
    
    if not tools:
        print("  No tools found.")
        return
    
    # Group by category if not filtered
    if not category_filter:
        from collections import defaultdict
        by_category = defaultdict(list)
        for tool in tools:
            by_category[tool.category.value].append(tool)
        
        for cat, cat_tools in sorted(by_category.items()):
            print(f"  [{cat.upper()}]")
            for tool in cat_tools:
                print(f"    • {tool.name}: {tool.description}")
            print()
    else:
        for tool in tools:
            print(f"  • {tool.name}: {tool.description}")
            if tool.parameters:
                print(f"    Parameters: {list(tool.parameters.keys())}")


def handle_tool_generation(description: str, output_file: str = None):
    """Generate a custom tool from description."""
    print(f"🔧 Generating tool from description...")
    print(f"   \"{description}\"\n")
    
    generator = ToolGenerator()
    tool = generator.generate_from_description(description)
    
    output = f'''# Auto-generated tool: {tool.name}
# Category: {tool.category.value}
# Description: {tool.description}

{tool.code}

# Tool metadata
TOOL_INFO = {{
    "name": "{tool.name}",
    "description": "{tool.description}",
    "category": "{tool.category.value}",
    "parameters": {json.dumps(tool.parameters, indent=8)}
}}
'''
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
        print(f"✅ Tool generated and saved to {output_file}")
    else:
        print(output)
    
    print(f"\n📋 Tool Summary:")
    print(f"   Name: {tool.name}")
    print(f"   Category: {tool.category.value}")
    print(f"   Parameters: {list(tool.parameters.keys()) if tool.parameters else 'None'}")


def handle_evaluation(query: str, response: str, expected: str = None, threshold: float = 0.7, output_file: str = None):
    """Evaluate agent output quality."""
    print("📊 Evaluating agent output...\n")
    
    evaluator = AgentEvaluator(thresholds={"overall": threshold})
    
    if expected:
        result = evaluator.evaluate(query, response, ground_truth=expected)
    else:
        result = evaluator.evaluate(query, response)
    
    # Display results
    metrics = result.metrics
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    
    output_lines = [
        f"Evaluation Results: {status}",
        f"{'=' * 50}",
        f"Query: {query[:100]}{'...' if len(query) > 100 else ''}",
        f"Response: {response[:100]}{'...' if len(response) > 100 else ''}",
        "",
        "Metrics:",
        f"  • Relevance:        {metrics.relevance_score:.2f}",
        f"  • Completeness:     {metrics.completeness_score:.2f}",
        f"  • Coherence:        {metrics.coherence_score:.2f}",
        f"  • Accuracy:         {metrics.accuracy_score:.2f}",
        f"  • Task Completion:  {metrics.task_completion_rate:.2f}",
        f"  • Response Time:    {metrics.response_time_ms:.2f}ms",
        f"  • Token Count:      {metrics.token_count}",
        "",
        f"Overall Score: {metrics.overall_score():.3f} (threshold: {threshold})",
    ]
    
    if result.feedback:
        output_lines.append("\nFeedback:")
        for fb in result.feedback:
            output_lines.append(f"  • {fb}")
    
    if result.errors:
        output_lines.append("\nErrors:")
        for err in result.errors:
            output_lines.append(f"  ⚠️  {err}")
    
    output = "\n".join(output_lines)
    
    if output_file:
        # Write JSON format to file
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(output)
        print(f"\n📄 Full results saved to {output_file}")
    else:
        print(output)


def handle_list_patterns():
    """List all available orchestration patterns."""
    orchestrator = Orchestrator()
    patterns = orchestrator.list_available_patterns()
    
    print("\n🔄 Available Orchestration Patterns:\n")
    
    for pattern in patterns:
        print(f"  [{pattern['name'].upper()}]")
        print(f"    Description: {pattern['description']}")
        print(f"    Use Cases:")
        for use_case in pattern.get('use_cases', [])[:3]:
            print(f"      • {use_case}")
        print()


def handle_orchestration(description: str = None, pattern_name: str = None, num_agents: int = 3, framework: str = "langgraph", output_file: str = None):
    """Handle orchestration pattern suggestion and code generation."""
    from .orchestration.orchestrator import OrchestrationConfig
    from .orchestration.patterns import get_pattern
    
    orchestrator = Orchestrator()
    
    # Determine pattern
    if pattern_name:
        pattern_type = PatternType(pattern_name)
        print(f"🔄 Using orchestration pattern: {pattern_name}")
    elif description:
        pattern_type = orchestrator.suggest_pattern(description)
        print(f"🔄 Analyzing task description...")
        print(f"   \"{description}\"\n")
        print(f"📌 Recommended pattern: {pattern_type.value}")
    else:
        print("Error: Either --orchestrate or --pattern is required")
        return
    
    # Generate orchestration code
    print(f"\n🏗️  Generating {pattern_type.value} orchestration code for {framework}...\n")
    
    # Create configuration from description or pattern
    if description:
        config = orchestrator.create_config_from_description(description, num_agents, framework)
    else:
        # Generate agents for the pattern
        agents = orchestrator._generate_agents_for_pattern(pattern_type, "Generic task", num_agents)
        template = get_pattern(pattern_type).get_config_template()
        config = OrchestrationConfig(
            pattern=pattern_type,
            agents=agents,
            settings=template.get("settings", {}),
            framework=framework
        )
    
    code = orchestrator.generate_code(config)
    
    if output_file:
        with open(output_file, "w") as f:
            f.write(code)
        print(f"✅ Orchestration code saved to {output_file}")
    else:
        print(code)
    
    print(f"\n📋 Orchestration Summary:")
    print(f"   Pattern: {pattern_type.value}")
    print(f"   Framework: {framework}")
    print(f"   Agents: {[a['name'] for a in config.agents]}")


if __name__ == "__main__":
    main()
