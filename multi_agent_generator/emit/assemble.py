# multi_agent_generator/emit/assemble.py
"""
Turns a :class:`ProjectManifest` and a framework emitter into a real multi-file project.

This is where the two halves meet. The planner decided what files exist and what each one is
for; the emitter knows how to write this framework's tools, agents and workflow; ``shared.py``
writes everything that is the same in every project. The assembler places each fragment into
the file the plan gave it, works out the imports between them, and produces a
:class:`GeneratedProject`.

Cross-module imports are computed, not written by hand
-----------------------------------------------------
A fragment declares the third-party imports its own code needs, and nothing else. It does not
know whether ``get_llm`` ended up in ``app/services/llm_service.py`` or three lines above it
in the same file - only the assembler knows that, because only the assembler knows the tier.

So the assembler builds an index of every name the project defines and which module defines
it, then reads each assembled file to see which of those names it actually uses. The reading
is done with :mod:`tokenize` rather than a text search, so a name mentioned in a docstring or
a comment does not produce a phantom import. That matters more than it sounds: a phantom
import between two modules that already reference each other is a circular import, and it
fails at run time rather than at generation time.

The result is that a name is imported exactly when it is used, from the module that actually
defines it. There is no list of hand-written import lines anywhere in this package that could
drift out of step with the layout.
"""
from __future__ import annotations

import io
import keyword
import re
import tokenize
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.manifest import FileKind, ProjectManifest, Tier
from ..core.models import GeneratedProject
from . import shared
from .agno_emitter import AgnoEmitter
from .base import Emitter, Fragment, banner, join_blocks
from .crewai_emitter import CrewAIEmitter, CrewAIFlowEmitter
from .langgraph_emitter import LangGraphEmitter
from .react_emitter import ReActEmitter, ReActLCELEmitter

__all__ = ["EMITTERS", "emitter_for", "assemble_project", "AssemblyError"]


#: Framework key -> emitter class. The same keys the CLI and the API accept, so a framework
#: that can be selected is a framework that can be emitted; there is no way to offer one in
#: the UI that has no implementation behind it.
EMITTERS = {
    CrewAIEmitter.key: CrewAIEmitter,
    CrewAIFlowEmitter.key: CrewAIFlowEmitter,
    LangGraphEmitter.key: LangGraphEmitter,
    ReActEmitter.key: ReActEmitter,
    ReActLCELEmitter.key: ReActLCELEmitter,
    AgnoEmitter.key: AgnoEmitter,
}


class AssemblyError(RuntimeError):
    """
    Raised when the plan and the emitter cannot be reconciled.

    Its own type so the pipeline can tell a generation bug - a planned file with nothing to
    put in it - apart from a model producing poor content. The two need different responses:
    one is a defect to fix, the other is a retry.
    """


def emitter_for(manifest: ProjectManifest) -> Emitter:
    """
    Build the emitter for this manifest's framework.

    Raises rather than falling back to a default. Silently emitting CrewAI code for a project
    the user asked to be LangGraph would produce something that runs and is not what was
    asked for, which is the worst of both outcomes.
    """
    try:
        emitter_class = EMITTERS[manifest.framework]
    except KeyError:
        raise AssemblyError(
            f"No emitter for framework {manifest.framework!r}. "
            f"Available: {', '.join(sorted(EMITTERS))}."
        ) from None
    return emitter_class(manifest)


# ============================================================================ name lookup
#: Matches the names an import line binds, so a name the fragment already imported is not
#: imported a second time by the cross-module pass.
_IMPORT_BINDING = re.compile(
    r"^\s*(?:from\s+[\w.]+\s+import\s+(?P<names>.+)|import\s+(?P<modules>.+))$"
)


def _imported_names(lines: Iterable[str]) -> Set[str]:
    """
    The names a set of import lines already brings into scope.

    Used to subtract from the cross-module pass. Without it a fragment that imports a project
    symbol explicitly - the test files do, because they are clearer read that way - would get
    a second identical import line appended below.
    """
    bound: Set[str] = set()
    for line in lines:
        match = _IMPORT_BINDING.match(str(line))
        if not match:
            continue
        raw = match.group("names") or match.group("modules") or ""
        for piece in raw.replace("(", "").replace(")", "").split(","):
            piece = piece.strip()
            if not piece:
                continue
            # `x as y` binds y; `a.b.c` binds a.
            if " as " in piece:
                bound.add(piece.split(" as ")[-1].strip())
            else:
                bound.add(piece.split(".")[0].strip())
    return bound


def _names_used(source: str) -> Set[str]:
    """
    Every identifier the source actually uses, read from its tokens.

    Tokenising rather than searching the text is the point. ``record_call`` appearing in a
    docstring is not a use of ``record_call``, and importing it because of that would add an
    edge between two modules that do not depend on each other - which, if the dependency runs
    the other way too, is a circular import discovered on the user's first run.

    Attribute access is excluded: in ``response.output`` the name ``output`` belongs to the
    object, not to this module, so importing something called ``output`` would be wrong.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Malformed source is a real problem, but it is not this function's problem to report.
        # Returning nothing means no cross-imports are added, so the file stays exactly as the
        # emitter wrote it and the validator downstream reports the syntax error by name.
        return set()

    used: Set[str] = set()
    previous_was_dot = False
    for token in tokens:
        if token.type == tokenize.OP:
            previous_was_dot = token.string == "."
            continue
        if token.type == tokenize.NAME and not previous_was_dot:
            if token.string not in keyword.kwlist:
                used.add(token.string)
        if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT):
            previous_was_dot = False
    return used


def _import_lines(index: Dict[str, str], own_module: str, used: Set[str]) -> List[str]:
    """
    The ``from ... import ...`` lines this file needs for names defined elsewhere in the project.

    Grouped one line per source module and sorted, so two runs over the same plan produce
    byte-identical files - which is what makes a diff between two versions readable.
    """
    wanted: Dict[str, Set[str]] = {}
    for name in used:
        module = index.get(name)
        if not module or module == own_module:
            continue
        wanted.setdefault(module, set()).add(name)

    return [
        f"from {module} import {', '.join(sorted(names))}"
        for module, names in sorted(wanted.items())
    ]


def _sorted_imports(lines: Iterable[str]) -> List[str]:
    """
    Order import lines deterministically: plain ``import`` first, then ``from`` imports.

    Not a full isort - it does not need to be. It needs to be stable, so that regenerating a
    project produces the same bytes and a version diff shows only what really changed.
    """
    unique = list(dict.fromkeys(line.strip() for line in lines if line and line.strip()))
    plain = sorted(line for line in unique if line.startswith("import "))
    from_lines = sorted(line for line in unique if line.startswith("from "))
    other = [line for line in unique if line not in plain and line not in from_lines]
    return [*plain, *from_lines, *other]


def _render(header: str, imports: Sequence[str], preamble: str, body: str) -> str:
    """Assemble one Python file from its parts, with one blank-line convention throughout."""
    sections: List[str] = [header.rstrip()]
    if imports:
        sections.append("\n".join(imports))
    if preamble.strip():
        sections.append(preamble.strip())
    if body.strip():
        sections.append(body.strip())
    return "\n\n\n".join(s for s in sections if s.strip()) + "\n"


# =============================================================================== placement
class _Placement:
    """
    Which fragments go into which file, and what each file's module name is.

    Built once and then read, so the tier decision is made in one place instead of being
    re-derived by every function that needs to know whether this is the flat layout.
    """

    def __init__(self, manifest: ProjectManifest, emitter: Emitter) -> None:
        self.manifest = manifest
        self.emitter = emitter
        self.flat = manifest.tier == Tier.SIMPLE
        #: path -> the fragment that file's Python content is built from.
        self.fragments: Dict[str, Fragment] = {}
        #: tool name -> the names its module defined, for the registry's imports.
        self.tool_symbols: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------------------ build
    def build(self) -> None:
        """Produce every fragment and assign it to a planned path."""
        manifest = self.manifest
        emitter = self.emitter

        llm = emitter.llm_fragment()
        telemetry = shared.telemetry_module(manifest)
        settings = shared.settings_module(manifest)

        tool_fragments = {t.name: emitter.tool_fragment(t) for t in manifest.tools}
        self.tool_symbols = {
            name: list(fragment.provides) for name, fragment in tool_fragments.items()
        }
        registry = emitter.tool_registry_fragment(manifest.tools)

        agent_fragments = {a.name: emitter.agent_fragment(a) for a in manifest.agents}
        workflow_fragments = {
            w.name: emitter.workflow_fragment(w) for w in manifest.workflows
        }
        state = shared.state_module(manifest, emitter.state_fragment())

        if self.flat:
            self._place_flat(
                settings=settings,
                telemetry=telemetry,
                llm=llm,
                tool_fragments=tool_fragments,
                registry=registry,
                agent_fragments=agent_fragments,
                workflow_fragments=workflow_fragments,
            )
        else:
            self._place_package(
                settings=settings,
                telemetry=telemetry,
                llm=llm,
                state=state,
                tool_fragments=tool_fragments,
                registry=registry,
                agent_fragments=agent_fragments,
                workflow_fragments=workflow_fragments,
            )

        self.fragments["main.py"] = shared.entrypoint_module(manifest)
        self._place_tests()

    # ------------------------------------------------------------------------------- flat
    def _place_flat(
        self,
        *,
        settings: Fragment,
        telemetry: Fragment,
        llm: Fragment,
        tool_fragments: Dict[str, Fragment],
        registry: Fragment,
        agent_fragments: Dict[str, Fragment],
        workflow_fragments: Dict[str, Fragment],
    ) -> None:
        """
        Three files: ``config.py``, ``tools.py``, ``agent.py``.

        The fragments are merged rather than rewritten. That is the whole reason the emitters
        produce fragments instead of files - the same CrewAI agent code lands in its own
        module in the package layout and inside ``agent.py`` here, with no second code path.
        """
        self.fragments["config.py"] = settings

        if tool_fragments:
            merged = telemetry
            for fragment in tool_fragments.values():
                merged = merged.merge(fragment)
            self.fragments["tools.py"] = merged.merge(registry)
            agent_file = llm
        else:
            # Nothing to record, but the workflow still clears and reads the record, so the
            # helpers travel with the code that calls them.
            agent_file = telemetry.merge(llm).merge(registry)

        for fragment in agent_fragments.values():
            agent_file = agent_file.merge(fragment)
        for fragment in workflow_fragments.values():
            agent_file = agent_file.merge(fragment)
        self.fragments["agent.py"] = agent_file

    # ---------------------------------------------------------------------------- package
    def _place_package(
        self,
        *,
        settings: Fragment,
        telemetry: Fragment,
        llm: Fragment,
        state: Fragment,
        tool_fragments: Dict[str, Fragment],
        registry: Fragment,
        agent_fragments: Dict[str, Fragment],
        workflow_fragments: Dict[str, Fragment],
    ) -> None:
        """One module per responsibility, matching the paths the planner recorded."""
        manifest = self.manifest
        package = manifest.package or "app"

        for spec in manifest.files_of_kind(FileKind.CONFIG):
            self.fragments[spec.path] = settings

        for spec in manifest.files_of_kind(FileKind.SERVICE):
            if "telemetry" in spec.path:
                self.fragments[spec.path] = telemetry
            elif "llm" in spec.path:
                self.fragments[spec.path] = llm

        for spec in manifest.files_of_kind(FileKind.STATE):
            self.fragments[spec.path] = state

        for tool in manifest.tools:
            path = f"{package}/tools/{tool.name}.py"
            fragment = tool_fragments.get(tool.name)
            if fragment is None:
                raise AssemblyError(f"No emitted code for planned tool {tool.name!r}.")
            self.fragments[path] = fragment

        if manifest.tools:
            self.fragments[f"{package}/tools/__init__.py"] = shared.tool_registry_init(
                manifest, registry, self.tool_symbols
            )

        for agent in manifest.agents:
            path = f"{package}/agents/{agent.name}.py"
            fragment = agent_fragments.get(agent.name)
            if fragment is None:
                raise AssemblyError(f"No emitted code for planned agent {agent.name!r}.")
            self.fragments[path] = fragment
        self.fragments[f"{package}/agents/__init__.py"] = shared.agents_init(manifest)

        for workflow in manifest.workflows:
            path = f"{package}/workflows/{workflow.name}.py"
            fragment = workflow_fragments.get(workflow.name)
            if fragment is None:
                raise AssemblyError(
                    f"No emitted code for planned workflow {workflow.name!r}."
                )
            self.fragments[path] = fragment

        for spec in manifest.files_of_kind(FileKind.RUNTIME):
            self.fragments[spec.path] = shared.runner_module(manifest)

        for spec in manifest.files_of_kind(FileKind.MEMORY):
            self.fragments[spec.path] = shared.memory_module(manifest)

        # Package markers last, so any __init__ the planner asked for and nothing above
        # filled gets an honest empty one rather than being missing from the project.
        for spec in manifest.files_of_kind(FileKind.PACKAGE):
            self.fragments.setdefault(spec.path, shared.package_init(spec.provides))

    # ------------------------------------------------------------------------------ tests
    def _place_tests(self) -> None:
        """Attach the test files the planner asked for, by filename."""
        manifest = self.manifest
        builders = {
            "tests/conftest.py": shared.conftest_module,
            "tests/test_imports.py": shared.test_imports_module,
            "tests/test_config.py": shared.test_config_module,
            "tests/test_agents.py": shared.test_agents_module,
            "tests/test_tools.py": shared.test_tools_module,
            "tests/test_workflow.py": shared.test_workflow_module,
            "tests/test_agent.py": shared.test_simple_module,
        }
        for spec in manifest.files_of_kind(FileKind.TEST):
            build = builders.get(spec.path)
            if build is None:
                raise AssemblyError(
                    f"The plan asked for {spec.path}, but no test template writes it."
                )
            self.fragments[spec.path] = build(manifest)


# ============================================================================== assemble
def _symbol_index(
    manifest: ProjectManifest, placement: _Placement
) -> Tuple[Dict[str, str], List[str]]:
    """
    Map every name the project defines to the module that defines it.

    Returns the index and a list of ambiguities.

    Definitions win over re-exports. ``app/agents/__init__.py`` re-exports every builder, so
    without that rule a workflow importing ``build_researcher`` might be pointed at the
    package while the package is still importing the agent - a cycle. Pointing it at the
    defining module makes the import graph a tree in the direction the planner intended.

    When two modules define the same name - two agent modules each with an ``INSTRUCTIONS``
    constant, say - the name is left out of the index entirely and reported. Guessing which
    one was meant would produce an import that is silently wrong, and a wrong import that
    resolves is far harder to find than a missing one, which fails immediately by name.
    """
    definitions: Dict[str, Set[str]] = {}
    reexports: Dict[str, Set[str]] = {}

    for path, fragment in sorted(placement.fragments.items()):
        module = manifest.module_for(path)
        if not module or path.startswith("tests/"):
            # Test modules define helpers with ordinary names. Nothing may import from them:
            # a project that depends on its own test suite cannot be shipped without it.
            continue
        target = reexports if path.endswith("__init__.py") else definitions
        for name in fragment.provides:
            target.setdefault(name, set()).add(module)

    index: Dict[str, str] = {}
    ambiguous: List[str] = []
    for name, modules in sorted(definitions.items()):
        if len(modules) == 1:
            index[name] = next(iter(modules))
        else:
            ambiguous.append(
                f"{name} is defined in {', '.join(sorted(modules))}; it was left out of the "
                "automatic imports, so any module that needs it must import it explicitly."
            )
    for name, modules in sorted(reexports.items()):
        if name not in index and len(modules) == 1:
            index[name] = next(iter(modules))
    return index, ambiguous


def assemble_project(
    manifest: ProjectManifest,
    *,
    emitter: Optional[Emitter] = None,
) -> GeneratedProject:
    """
    Build the complete project described by ``manifest``.

    Every file in the plan comes out with content, and no file comes out that the plan did not
    ask for. That symmetry is deliberate: it means the file tree the user is shown, the
    dependency graph, and what is actually on disk are three views of one decision rather
    than three things that have to be kept in agreement.

    Raises :class:`AssemblyError` when a planned file has nothing to write into it, rather
    than emitting a project with a silently empty module in it.
    """
    emitter = emitter or emitter_for(manifest)
    placement = _Placement(manifest, emitter)
    placement.build()

    index, ambiguous = _symbol_index(manifest, placement)
    notes = list(manifest.notes)
    # Reported rather than dropped. A name the assembler declined to wire up is something the
    # reviewer and the import validator both need to know about.
    notes.extend(ambiguous)

    project = GeneratedProject(
        framework=manifest.framework,
        provider=manifest.provider,
        model=manifest.model,
        dependencies=list(manifest.dependencies),
        config=manifest.as_dict(),
        entrypoint=manifest.entrypoint or "main.py",
        notes=notes,
    )

    written: Set[str] = set()

    # ------------------------------------------------------------------- python modules
    for spec in manifest.files:
        if not spec.path.endswith(".py"):
            continue
        fragment = placement.fragments.get(spec.path)
        if fragment is None:
            raise AssemblyError(
                f"The plan includes {spec.path} but nothing was emitted for it. "
                "Either the planner added a file the emitters do not know about, or the "
                "emitter for this framework is incomplete."
            )

        own_module = manifest.module_for(spec.path) or ""
        header = banner(manifest, spec.purpose)
        core = join_blocks([fragment.preamble, fragment.body], "\n\n")
        declared = _sorted_imports(fragment.imports)
        used = _names_used(core) - _imported_names(declared)
        project_imports = _import_lines(index, own_module, used)

        imports = list(declared)
        if project_imports:
            imports = [*imports, "", "# --- this project", *sorted(project_imports)]

        project.add_file(
            spec.path,
            _render(header, imports, fragment.preamble, fragment.body),
            is_entrypoint=spec.is_entrypoint,
            description=spec.purpose,
        )
        written.add(spec.path)

    # ------------------------------------------------------------------ everything else
    entry_command = "python main.py" if manifest.entrypoint == "main.py" else f"python {manifest.entrypoint}"
    text_files = {
        "requirements.txt": lambda: shared.requirements_txt(manifest),
        ".env.example": lambda: shared.env_example(manifest),
        "pytest.ini": lambda: shared.pytest_ini(manifest),
        "README.md": lambda: shared.readme(manifest, entry_command),
        "Dockerfile": lambda: shared.dockerfile(manifest),
        ".dockerignore": shared.dockerignore,
    }
    for spec in manifest.files:
        if spec.path in written or spec.path.endswith(".py"):
            continue
        build = text_files.get(spec.path)
        if build is None:
            raise AssemblyError(
                f"The plan includes {spec.path} but no writer produces it."
            )
        project.add_file(spec.path, build(), description=spec.purpose)
        written.add(spec.path)

    project.run_instructions = _run_instructions(manifest, entry_command)
    return project


def _run_instructions(manifest: ProjectManifest, entry_command: str) -> str:
    """
    The exact commands to install, configure and run this project.

    Shown in the workspace and written into the export, and they name this project's real
    entry point and this provider's real environment variable - so they can be copied and
    pasted rather than adapted.
    """
    required = [v.name for v in manifest.env_vars if v.required]
    lines = [
        "pip install -r requirements.txt",
        "cp .env.example .env",
    ]
    if required:
        lines.append(f"# then set {', '.join(required)} in .env")
    lines.append(f"{entry_command} \"your question here\"")
    lines.append("pytest    # runs offline; needs no API key")
    return "\n".join(lines)
