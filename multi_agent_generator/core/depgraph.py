# multi_agent_generator/core/depgraph.py
"""
The file dependency graph, recovered from the code that was actually emitted.

Section 7 of the brief asks for one specific capability: *the system must know what breaks if a
particular file changes*. That question cannot be answered from the plan alone. The plan records
what the planner *intended* each file to depend on; what matters at run time is what the emitted
source really imports. Those two can differ, and when they do it is the emitted code that fails.

So this module reads the source. For every Python file in a :class:`GeneratedProject` it parses
the import statements, resolves each one against the project's own module layout, and builds a
directed graph of file-to-file edges. From that graph it can answer:

* what a file imports, directly and transitively;
* what imports a file - and therefore what breaks if it changes (:meth:`DependencyGraph.impact_of`);
* whether any import cannot be satisfied at all, either by a file in the project or by a
  declared requirement (section 28's import validation, the check that has to pass before a
  project is allowed to be called READY);
* whether a name imported from a sibling module is actually defined there;
* whether the imports form a cycle, and whether that cycle is the kind that fails on import;
* where the emitted imports and the planned ``depends_on`` disagree.

Why parse instead of trusting the assembler
-------------------------------------------
:mod:`multi_agent_generator.emit.assemble` computes cross-module imports itself, and it is
careful about it. But it is the thing being checked. A validator that asks the generator whether
the generator got it right is not a validator. Reading the finished text with :mod:`ast` is the
only way for this module to disagree with the code that produced it, which is the entire reason
it exists.

Parsing also means this works on a project that was *modified* - by the repair engine, by the
improvement stage, or by a user editing a file in the workspace - not only on one that was just
assembled. The graph is recomputed from whatever the files currently say.

Honesty rules applied here
--------------------------
Nothing in this module guesses in a way that produces a confident-sounding wrong answer. An
import whose module cannot be found inside the project *and* whose distribution is not in the
requirements is reported as unresolved rather than quietly assumed to be fine. A circular import
is reported as a blocker only when every edge in it is a module-level ``from ... import name``,
because that is the shape that actually raises on import; a cycle held together by an import
inside a function is reported at a lower severity with the reason stated. And a mismatch between
the plan and the emitted code is reported as information about the plan, not as a defect in the
project - the plan is a prediction, and a prediction that turned out to be unnecessary is not a
bug.

No network, no model call, no framework import. A graph can be built in a test from a project
assembled in memory.
"""
from __future__ import annotations

import ast
import posixpath
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..dependencies import (
    BASE_REQUIREMENTS,
    FRAMEWORK_REQUIREMENTS,
    GENERATED_CODE_REQUIREMENTS,
    PROVIDER_REQUIREMENTS,
    TEST_REQUIREMENTS,
    Requirement,
    distribution_name,
)
from .manifest import ProjectManifest
from .models import GeneratedProject, ReviewIssue, Severity

__all__ = [
    "ModuleRef",
    "ImportEdge",
    "FileNode",
    "GraphProblem",
    "DependencyGraph",
    "build_graph",
    "validate_imports",
    "import_roots_for",
]


# ============================================================================== vocabulary
#: Import roots the interpreter provides. ``sys.stdlib_module_names`` is exact for the running
#: interpreter, which is better than a hand-maintained list that would slowly go stale - and the
#: generated project runs on this same interpreter, so "exact for us" is the right definition.
_STDLIB_ROOTS = frozenset(sys.stdlib_module_names) | {"__future__"}

#: Distributions whose import name is not the distribution name with dashes turned into
#: underscores. Only consulted when :mod:`multi_agent_generator.dependencies` - which is where
#: this project's own requirements are written down - has nothing to say about the distribution,
#: so it exists for requirements a tool or a model added, not for the ones the planner emits.
_IMPORT_NAME_OVERRIDES: Dict[str, str] = {
    "attrs": "attr",
    "beautifulsoup4": "bs4",
    "faiss-cpu": "faiss",
    "google-generativeai": "google",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "protobuf": "google",
    "pycryptodome": "Crypto",
    "python-dateutil": "dateutil",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
}

#: Problem codes. Strings rather than an enum because they travel to the frontend as JSON and
#: are matched there; an enum would only add a translation step at the boundary.
CODE_SYNTAX_ERROR = "syntax_error"
CODE_UNRESOLVED_INTERNAL = "unresolved_internal_import"
CODE_UNDECLARED_DEPENDENCY = "undeclared_dependency"
CODE_MISSING_SYMBOL = "missing_symbol"
CODE_CIRCULAR_IMPORT = "circular_import"
CODE_AMBIGUOUS_MODULE = "ambiguous_module"
CODE_TEST_IMPORTED = "test_imported_by_source"
CODE_BAD_RELATIVE_IMPORT = "unresolvable_relative_import"
CODE_ORPHAN_FILE = "orphan_file"
CODE_UNPLANNED_EDGE = "unplanned_dependency"
CODE_UNREALISED_EDGE = "unrealised_dependency"


def _registry_import_names() -> Dict[str, str]:
    """
    Distribution name -> import root, read from this project's requirement registry.

    Built from :mod:`multi_agent_generator.dependencies` rather than from a second table,
    because that module already pairs every requirement with the module name used to detect it.
    Keeping one source means the import validator and ``--check-deps`` cannot disagree about
    what ``langchain-core`` is called once it is installed.
    """
    groups: List[Sequence[Requirement]] = [BASE_REQUIREMENTS, TEST_REQUIREMENTS]
    groups.extend(FRAMEWORK_REQUIREMENTS.values())
    groups.extend(PROVIDER_REQUIREMENTS.values())
    groups.extend(GENERATED_CODE_REQUIREMENTS.values())

    index: Dict[str, str] = {}
    for group in groups:
        for requirement in group:
            root = (requirement.import_name or "").split(".")[0].strip()
            package = requirement.package.strip().lower()
            if root and package:
                index.setdefault(package, root)
    return index


_REGISTRY_IMPORT_NAMES = _registry_import_names()


def _distribution_of(line: str) -> str:
    """
    The bare distribution name from one ``requirements.txt`` line, or ``""``.

    Delegates to :func:`multi_agent_generator.dependencies.distribution_name` so that the
    validator, the installer's screen and ``--check-deps`` all parse a requirement line the
    same way. This used to be a second implementation here, and a second implementation of
    "what is this line's package name" is exactly how a project comes to be reported as
    having an undeclared dependency that it has in fact declared.
    """
    return distribution_name(line)


def import_roots_for(requirements: Iterable[str]) -> Tuple[Set[str], List[str]]:
    """
    The top-level module names a set of pip requirements makes importable.

    Returns the roots and a list of the mappings that had to be *guessed* - a distribution the
    registry has never heard of, normalised by the usual dash-to-underscore rule. The guesses
    are returned rather than hidden because the rule is right for most of PyPI and wrong for
    some of it, and a wrong guess here would turn into a false "undeclared dependency" report.
    Saying which mappings were assumed lets a reader discount exactly those.
    """
    roots: Set[str] = set()
    guessed: List[str] = []
    for line in requirements:
        distribution = _distribution_of(line)
        if not distribution:
            continue
        key = distribution.lower()
        known = _REGISTRY_IMPORT_NAMES.get(key) or _IMPORT_NAME_OVERRIDES.get(key)
        if known:
            roots.add(known)
            continue
        fallback = key.replace("-", "_").replace(".", "_")
        roots.add(fallback)
        guessed.append(f"{distribution} was assumed to be imported as {fallback!r}")
    return roots, guessed


# ================================================================================= parsing
@dataclass(frozen=True)
class ModuleRef:
    """
    One import statement, as written.

    ``module`` is the absolute dotted name after any relative prefix has been resolved, or
    ``""`` when the relative prefix reached past the project root. ``top_level`` records
    whether the statement runs at import time: an import inside a function body is still a
    dependency, but it cannot participate in a circular-import failure, and conflating the two
    would mean reporting a harmless cycle as a blocker.
    """

    module: str
    names: Tuple[str, ...] = ()
    line: int = 0
    #: What the source said, including any leading dots. Kept for error messages.
    raw: str = ""
    level: int = 0
    star: bool = False
    top_level: bool = True
    #: True for ``importlib.import_module("app.agents.x")``. A real dependency - the file will
    #: not work if that module goes away - but it binds no name, so it can never be the step
    #: that makes a circular import fail.
    dynamic: bool = False

    @property
    def relative(self) -> bool:
        return self.level > 0


@dataclass
class _ParsedModule:
    """The three things this module needs from one file's source, plus any parse failure."""

    refs: List[ModuleRef] = field(default_factory=list)
    #: Names bound at module level, for checking that ``from x import y`` can succeed.
    symbols: Set[str] = field(default_factory=set)
    #: The subset of ``symbols`` this file actually creates, excluding imported bindings.
    #:
    #: Both questions are real and they are not the same one. ``from config import
    #: get_settings`` makes ``get_settings`` importable *from this module*, so it belongs in
    #: ``symbols``; it does not make this module the place ``get_settings`` lives, so it does
    #: not belong here. Asking the first question when you meant the second is how the
    #: validator came to report "configuration is loaded from one place (workflow.py)" for a
    #: project whose config module had not been generated at all.
    defines: Set[str] = field(default_factory=set)
    #: ``from x import *`` makes the symbol set unknowable, so name checks are skipped.
    star_import: bool = False
    error: Optional[str] = None
    error_line: Optional[int] = None


def _add_target(target: ast.expr, out: Set[str]) -> None:
    """Record the names one assignment target binds, unpacking tuples and stars."""
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _add_target(element, out)
    elif isinstance(target, ast.Starred):
        _add_target(target.value, out)
    # Attribute and subscript targets (``obj.x = 1``) bind nothing at module level.


def _walk_module_level(
    statements: Sequence[ast.stmt],
    *,
    symbols: Set[str],
    defines: Set[str],
    imports: Set[int],
    star: List[bool],
) -> None:
    """
    Collect module-level bindings and module-level import statements.

    Descends into ``if``/``try``/``with``/``for``/``while`` because a name bound inside one of
    those at module level is still a module-level name - which matters, since the guarded
    ``try: import x / except ImportError:`` shape is exactly how a generated project makes an
    optional dependency optional. Function and class bodies are not descended into: their
    names belong to their own scope, and treating them as module attributes would make the
    "does this module define that name" check useless.

    ``symbols`` collects every binding; ``defines`` collects every binding except the ones an
    import statement made. See :class:`_ParsedModule` for why the distinction matters.
    """
    for node in statements:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
            defines.add(node.name)
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                _add_target(target, symbols)
                _add_target(target, defines)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _add_target(node.target, symbols)
            _add_target(node.target, defines)
        elif isinstance(node, ast.Import):
            imports.add(id(node))
            for alias in node.names:
                symbols.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            imports.add(id(node))
            for alias in node.names:
                if alias.name == "*":
                    star[0] = True
                    continue
                symbols.add(alias.asname or alias.name)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _add_target(node.target, symbols)
            _add_target(node.target, defines)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _add_target(item.optional_vars, symbols)
                    _add_target(item.optional_vars, defines)

        for field_name in ("body", "orelse", "finalbody"):
            children = getattr(node, field_name, None)
            if isinstance(children, list):
                nested = [child for child in children if isinstance(child, ast.stmt)]
                if nested:
                    _walk_module_level(
                        nested, symbols=symbols, defines=defines, imports=imports, star=star
                    )
        for handler in getattr(node, "handlers", None) or []:
            body = [child for child in getattr(handler, "body", []) if isinstance(child, ast.stmt)]
            if body:
                _walk_module_level(
                    body, symbols=symbols, defines=defines, imports=imports, star=star
                )
        # ``match`` cases are ``match_case`` objects rather than statements, so the loop above
        # skips them. Missing a name bound in one would produce a false "does not define it"
        # blocker, which is the more expensive way to be wrong.
        for case in getattr(node, "cases", None) or []:
            body = [child for child in getattr(case, "body", []) if isinstance(child, ast.stmt)]
            if body:
                _walk_module_level(
                    body, symbols=symbols, defines=defines, imports=imports, star=star
                )


def _parse(source: str) -> _ParsedModule:
    """
    Read one file's imports and module-level names.

    A syntax error is returned rather than raised. A project with one unparseable file still
    has a meaningful graph for its other twelve, and the syntax error itself is reported as a
    problem - which is more useful than a traceback that hides the rest of the analysis.
    """
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        detail = (exc.msg or "invalid syntax").strip()
        return _ParsedModule(error=detail, error_line=exc.lineno)
    except (ValueError, RecursionError) as exc:  # null bytes, absurd nesting
        return _ParsedModule(error=str(exc) or "could not be parsed")

    symbols: Set[str] = set()
    defines: Set[str] = set()
    top_level_imports: Set[int] = set()
    star = [False]
    _walk_module_level(
        tree.body,
        symbols=symbols,
        defines=defines,
        imports=top_level_imports,
        star=star,
    )

    refs: List[ModuleRef] = []
    for node in ast.walk(tree):
        top_level = id(node) in top_level_imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(
                    ModuleRef(
                        module=alias.name,
                        names=(),
                        line=node.lineno,
                        raw=alias.name,
                        top_level=top_level,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            refs.append(
                ModuleRef(
                    module=node.module or "",
                    names=tuple(n for n in names if n != "*"),
                    line=node.lineno,
                    raw=("." * (node.level or 0)) + (node.module or ""),
                    level=node.level or 0,
                    star="*" in names,
                    top_level=top_level,
                )
            )
        else:
            dynamic = _dynamic_import(node)
            if dynamic:
                refs.append(dynamic)

    return _ParsedModule(refs=refs, symbols=symbols, defines=defines, star_import=star[0])


def _dynamic_import(node: ast.AST) -> Optional[ModuleRef]:
    """
    Recognise ``importlib.import_module("some.module")`` with a literal name.

    The generated ``tests/test_imports.py`` imports every module this way on purpose - it is how
    one test file can check a whole package without a hand-written import list. Ignoring those
    calls would leave the graph claiming that nothing depends on the agent modules, which is
    exactly the question section 7 wants answered correctly.

    Only a literal string is followed. ``import_module(name)`` where ``name`` is a variable is
    left alone rather than guessed at: an invented edge would be worse than a missing one,
    because it would show up in the impact analysis as a fact.
    """
    if not isinstance(node, ast.Call) or not node.args:
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        called = func.attr
    elif isinstance(func, ast.Name):
        called = func.id
    else:
        return None
    if called != "import_module":
        return None
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return None
    module = first.value.strip()
    if not module or module.startswith("."):
        # A relative dynamic import needs a package= argument to mean anything; skip rather
        # than resolve it against the wrong package.
        return None
    return ModuleRef(
        module=module,
        names=(),
        line=getattr(node, "lineno", 0),
        raw=module,
        top_level=False,
        dynamic=True,
    )


# =============================================================================== graph data
@dataclass(frozen=True)
class ImportEdge:
    """
    One file importing another.

    ``names`` is empty for ``import package.module`` and populated for
    ``from package.module import a, b``. The distinction decides how badly a cycle hurts: a
    plain ``import`` binds a module object and tolerates a partially initialised module, while
    ``from ... import name`` needs the name to exist already and raises if it does not.
    """

    source: str
    target: str
    module: str
    names: Tuple[str, ...] = ()
    line: int = 0
    top_level: bool = True
    star: bool = False
    dynamic: bool = False

    @property
    def binds_names(self) -> bool:
        """
        True when this import needs names to already exist in the target module.

        ``from x import y`` and ``from x import *`` both do; ``import x.y`` and a dynamic
        ``import_module("x.y")`` do not, because they bind a module object and are happy with
        one that is only half initialised. This is the whole difference between a circular
        import that raises and one that does not.
        """
        return bool(self.names) or self.star

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "module": self.module,
            "names": list(self.names),
            "line": self.line,
            "top_level": self.top_level,
            "star": self.star,
            "dynamic": self.dynamic,
            # Serialised rather than left to the reader to recompute: the workflow and files
            # views colour an edge by whether it is load-bearing at import time, and that rule
            # should be stated once, here.
            "binds_names": self.binds_names,
        }


@dataclass
class FileNode:
    """
    One Python file in the graph.

    ``purpose`` comes from the manifest when there is one. It is carried here so the code
    viewer (section 26) can show File / Purpose / Used By / Dependencies from a single object
    instead of joining three sources at the API boundary.
    """

    path: str
    module: Optional[str] = None
    purpose: str = ""
    kind: str = ""
    is_entrypoint: bool = False
    is_test: bool = False
    is_package_init: bool = False
    #: Project files this file imports, sorted.
    imports: List[str] = field(default_factory=list)
    #: Standard-library roots it imports.
    stdlib: List[str] = field(default_factory=list)
    #: Third-party roots it imports that a declared requirement provides.
    third_party: List[str] = field(default_factory=list)
    #: Roots that nothing in the project and nothing in requirements.txt provides.
    unresolved: List[str] = field(default_factory=list)
    #: Names this file binds at module level, including ones an import statement bound.
    #:
    #: This is the right list for answering "can ``from this_file import name`` succeed?".
    provides: List[str] = field(default_factory=list)
    #: The subset of :attr:`provides` this file creates itself, excluding imported bindings.
    #:
    #: This is the right list for answering "which file *is* the configuration module?". The two
    #: questions read the same and are not: ``from config import get_settings`` puts
    #: ``get_settings`` in ``provides`` because the name really is importable from here, but it
    #: does not make this file the place ``get_settings`` lives. The validator asked the first
    #: question when it meant the second and so reported "configuration is loaded from one place
    #: (workflow.py)" for a project that had no config module at all.
    defines: List[str] = field(default_factory=list)
    syntax_error: Optional[str] = None

    @property
    def external(self) -> List[str]:
        return sorted({*self.stdlib, *self.third_party, *self.unresolved})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "module": self.module,
            "purpose": self.purpose,
            "kind": self.kind,
            "is_entrypoint": self.is_entrypoint,
            "is_test": self.is_test,
            "is_package_init": self.is_package_init,
            "imports": list(self.imports),
            "stdlib": list(self.stdlib),
            "third_party": list(self.third_party),
            "unresolved": list(self.unresolved),
            "provides": list(self.provides),
            "defines": list(self.defines),
            "syntax_error": self.syntax_error,
        }


@dataclass(frozen=True)
class GraphProblem:
    """
    Something wrong with the import structure.

    Carries a plain-language ``message`` and a ``suggestion`` that names the fix, because both
    ends consume this: the repair engine reads the code and the path, and the user reads the
    sentence. Section 46's rule - a person should never have to read a traceback to find out
    what went wrong - starts with the message being written for a person in the first place.
    """

    code: str
    severity: Severity
    message: str
    path: Optional[str] = None
    line: Optional[int] = None
    suggestion: str = ""
    #: Extra machine-readable context, e.g. the cycle's file list.
    detail: Tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.severity.blocks_release

    def to_issue(self) -> ReviewIssue:
        """As a :class:`ReviewIssue`, so the reviewer and the pipeline need no adapter."""
        return ReviewIssue(
            category="imports",
            severity=self.severity,
            message=self.message,
            file=self.path,
            line=self.line,
            suggestion=self.suggestion,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "line": self.line,
            "suggestion": self.suggestion,
            "detail": list(self.detail),
        }


# ==================================================================================== graph
@dataclass
class DependencyGraph:
    """
    The import structure of one generated project.

    Built by :func:`build_graph` and then only read. Every number it reports is derived from
    the parsed source, so there is no code path that can claim a file has three dependents
    when it has none.
    """

    nodes: Dict[str, FileNode] = field(default_factory=dict)
    edges: List[ImportEdge] = field(default_factory=list)
    problems: List[GraphProblem] = field(default_factory=list)
    #: Assumptions made while building, e.g. a distribution whose import name was guessed.
    notes: List[str] = field(default_factory=list)
    entrypoint: Optional[str] = None
    #: Planned-vs-emitted differences, keyed by file path.
    drift: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    # ---------------------------------------------------------------------------- queries
    def paths(self) -> List[str]:
        return sorted(self.nodes)

    def imports(self, path: str) -> List[str]:
        """Files this file imports directly."""
        node = self.nodes.get(path)
        return list(node.imports) if node else []

    def dependents(self, path: str) -> List[str]:
        """Files that import this file directly."""
        return sorted({edge.source for edge in self.edges if edge.target == path})

    def impact_of(self, path: str) -> List[str]:
        """
        Every file that could break if this one changes.

        This is section 7's question, answered by walking the reverse edges to a fixed point:
        a file that imports a file that imports the changed one is affected too, because a
        name removed at the bottom fails at the top. The changed file itself is excluded - it
        is the cause, not part of the blast radius - and the result is sorted so two calls
        produce the same list.

        One relationship is deliberately left out. Importing ``app.workflow`` also executes
        ``app/__init__.py``, so in the strictest sense every importer of anything under ``app``
        depends on that file. Recording those edges would report every ordinary package as a
        circular import - ``app/__init__.py`` re-exports from ``app.workflow``, which would then
        import ``app`` back - and that cycle does not fail in practice, because Python is happy
        to hand a partially initialised parent package to its own submodule. A false blocker on
        every project is a far worse trade than an ``__init__`` file whose blast radius is
        understated, so the edges stop at the module that was named.
        """
        reverse: Dict[str, Set[str]] = {}
        for edge in self.edges:
            reverse.setdefault(edge.target, set()).add(edge.source)

        seen: Set[str] = set()
        frontier = [path]
        while frontier:
            current = frontier.pop()
            for dependent in reverse.get(current, ()):
                if dependent not in seen and dependent != path:
                    seen.add(dependent)
                    frontier.append(dependent)
        return sorted(seen)

    def dependencies_of(self, path: str) -> List[str]:
        """Every file this one depends on, directly or through another file."""
        seen: Set[str] = set()
        frontier = [path]
        while frontier:
            current = frontier.pop()
            for target in self.imports(current):
                if target not in seen and target != path:
                    seen.add(target)
                    frontier.append(target)
        return sorted(seen)

    def cycles(self) -> List[List[str]]:
        """
        Import cycles, each as a path that returns to its start.

        Found with a depth-first search over the internal edges. Cycles that are rotations of
        one another are reported once: ``a -> b -> a`` and ``b -> a -> b`` are the same
        problem, and reporting both would just make the same fix look like two.
        """
        adjacency = {path: self.imports(path) for path in self.nodes}
        state: Dict[str, int] = {}
        stack: List[str] = []
        found: List[List[str]] = []
        seen_sets: Set[frozenset] = set()

        def visit(node: str) -> None:
            state[node] = 1
            stack.append(node)
            for nxt in adjacency.get(node, ()):
                colour = state.get(nxt, 0)
                if colour == 0:
                    visit(nxt)
                elif colour == 1:
                    cycle = stack[stack.index(nxt):]
                    signature = frozenset(cycle)
                    if signature not in seen_sets:
                        seen_sets.add(signature)
                        found.append([*cycle, nxt])
            stack.pop()
            state[node] = 2

        for path in sorted(adjacency):
            if state.get(path, 0) == 0:
                visit(path)
        return found

    def unresolved_imports(self) -> Dict[str, List[str]]:
        """Path -> the import roots nothing satisfies. Empty when the project is importable."""
        return {
            path: list(node.unresolved)
            for path, node in sorted(self.nodes.items())
            if node.unresolved
        }

    # ---------------------------------------------------------------------------- verdicts
    @property
    def blocking_problems(self) -> List[GraphProblem]:
        return [p for p in self.problems if p.blocking]

    @property
    def ok(self) -> bool:
        """
        True when nothing found here should stop the project being called READY.

        Deliberately not "no problems at all". An orphan file and a plan/emit mismatch are
        worth showing and are not reasons to withhold a working agent from the user.
        """
        return not self.blocking_problems

    def to_issues(self) -> List[ReviewIssue]:
        """Every problem as a review issue, worst first."""
        order = {
            Severity.BLOCKER: 0,
            Severity.MAJOR: 1,
            Severity.MINOR: 2,
            Severity.INFO: 3,
        }
        return [
            problem.to_issue()
            for problem in sorted(
                self.problems, key=lambda p: (order.get(p.severity, 4), p.path or "", p.code)
            )
        ]

    # ----------------------------------------------------------------------------- summary
    def summary(self) -> Dict[str, Any]:
        """Counts for the Overview and Files tabs. Every one of them is a ``len()``."""
        return {
            "files": len(self.nodes),
            "edges": len(self.edges),
            "problems": len(self.problems),
            "blocking": len(self.blocking_problems),
            "cycles": len(self.cycles()),
            "unresolved": sum(len(n.unresolved) for n in self.nodes.values()),
            "third_party": len(
                {root for node in self.nodes.values() for root in node.third_party}
            ),
            "ok": self.ok,
        }

    def as_dict(self) -> Dict[str, Any]:
        """
        The whole graph as JSON, including the reverse edges the UI needs.

        ``used_by`` and ``breaks_if_changed`` are precomputed per file because that is what the
        code viewer shows above the source, and making the frontend re-derive them from the
        edge list would put the same traversal in two languages.
        """
        return {
            "entrypoint": self.entrypoint,
            "summary": self.summary(),
            "notes": list(self.notes),
            "files": [
                {
                    **node.as_dict(),
                    "used_by": self.dependents(path),
                    "breaks_if_changed": self.impact_of(path),
                }
                for path, node in sorted(self.nodes.items())
            ],
            "edges": [edge.as_dict() for edge in self.edges],
            "problems": [problem.as_dict() for problem in self.problems],
            "cycles": self.cycles(),
            "drift": {path: dict(entry) for path, entry in sorted(self.drift.items())},
        }


# ================================================================================== build
def _module_of(path: str) -> Optional[str]:
    """Dotted module name for a ``.py`` path, with ``__init__`` folded into its package."""
    if not path.endswith(".py"):
        return None
    parts = [part for part in path[: -len(".py")].split("/") if part]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _is_test_path(path: str) -> bool:
    return path.startswith("tests/") or posixpath.basename(path).startswith("test_")


def _resolve_relative(
    own_module: str, is_package: bool, level: int, module: str
) -> Optional[str]:
    """
    Turn a relative import into an absolute module name.

    The anchor is the importing file's *package*, and an ``__init__`` module **is** its package
    while a plain module merely sits inside one - which is why ``is_package`` has to be passed in
    rather than inferred from the dots. ``from . import x`` means ``app.tools.x`` both in
    ``app/tools/__init__.py`` and in ``app/tools/web_search.py``, and it is this distinction that
    makes the two agree.

    Returns ``None`` when the import reaches past the top of the project, which Python rejects
    outright with ``ImportError: attempted relative import beyond top-level package``. Two shapes
    do that, and both used to resolve to a plausible-looking top-level module here instead:
    ``from ..outside import thing`` in ``app/agents.py`` - one dot too many, since the anchor is
    already the outermost package - and any relative import in a module that has no package at
    all, such as a top-level ``main.py``.
    """
    base = [part for part in own_module.split(".") if part] if own_module else []
    package = base if is_package else base[:-1]
    if level > 1:
        climb = level - 1
        if climb >= len(package):
            return None
        package = package[: len(package) - climb]
    if not package:
        return None
    parts = [*package, *[p for p in module.split(".") if p]] if module else list(package)
    return ".".join(parts) if parts else None


def _package_prefixes(modules: Iterable[str]) -> Set[str]:
    """
    Every dotted prefix of every module: the directories the project's packages occupy.

    Needed because a directory without an ``__init__.py`` is still importable - Python 3 treats
    it as a namespace package. ``from app.tools import web_search`` succeeds even when
    ``app/tools/__init__.py`` does not exist, so reporting that import as "no such module"
    would block a project that runs. The planner does emit the markers, which makes this a
    guard against a false alarm rather than a common path, and a false blocker is the more
    expensive mistake of the two.
    """
    prefixes: Set[str] = set()
    for module in modules:
        parts = module.split(".")
        for size in range(1, len(parts)):
            prefixes.add(".".join(parts[:size]))
    return prefixes


def build_graph(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest] = None,
) -> DependencyGraph:
    """
    Build the dependency graph for a generated project.

    ``manifest`` is optional and only adds context: each file's stated purpose and kind, and
    the planned ``depends_on`` list to compare against. Without it the graph is still complete,
    because everything structural is read from the source - which is what lets this be called
    on a project loaded back from storage, or on one the repair engine has just rewritten.
    """
    # ``project.entrypoint`` is the recorded one; the fallback covers a project whose file was
    # flagged ``is_entrypoint`` without the field being set, so "which file runs" has one answer
    # here rather than two that can disagree.
    entrypoint = project.entrypoint or (
        project.entrypoint_file.path if project.entrypoint_file else None
    )
    graph = DependencyGraph(entrypoint=entrypoint)

    declared_roots, guessed = import_roots_for(project.dependencies)
    graph.notes.extend(guessed)

    # ------------------------------------------------------------------ index the modules
    module_index: Dict[str, str] = {}
    for generated in project.source_files():
        module = _module_of(generated.path)
        if not module:
            continue
        existing = module_index.get(module)
        if existing is not None:
            # ``app/tools.py`` and ``app/tools/__init__.py`` cannot both be imported as
            # ``app.tools``; one of them silently wins at run time depending on path order.
            # Reported rather than resolved, because guessing which one was meant would hide
            # a planning bug behind a working-looking project.
            graph.problems.append(
                GraphProblem(
                    code=CODE_AMBIGUOUS_MODULE,
                    severity=Severity.MAJOR,
                    message=(
                        f"Both {existing} and {generated.path} would be imported as "
                        f"{module!r}. Only one of them can win, and which one depends on the "
                        "order of the import path."
                    ),
                    path=generated.path,
                    suggestion=f"Remove or rename one of {existing} and {generated.path}.",
                )
            )
            continue
        module_index[module] = generated.path

    #: First segment of every module in the project. An import starting with one of these is
    #: meant to be internal, so failing to resolve it is a missing file rather than a missing
    #: package - a different problem with a different fix.
    project_roots = {module.split(".")[0] for module in module_index}
    package_prefixes = _package_prefixes(module_index)

    # ------------------------------------------------------------------- parse every file
    parsed: Dict[str, _ParsedModule] = {}
    for generated in project.source_files():
        parsed[generated.path] = _parse(generated.content)

    # -------------------------------------------------------------------- build the nodes
    for generated in project.source_files():
        path = generated.path
        result = parsed[path]
        spec = manifest.file(path) if manifest else None
        node = FileNode(
            path=path,
            module=_module_of(path),
            purpose=(spec.purpose if spec else generated.description) or "",
            kind=(spec.kind if spec else ""),
            is_entrypoint=generated.is_entrypoint or path == entrypoint,
            is_test=_is_test_path(path),
            is_package_init=posixpath.basename(path) == "__init__.py",
            provides=sorted(result.symbols),
            defines=sorted(result.defines),
            syntax_error=result.error,
        )
        graph.nodes[path] = node

        if result.error:
            graph.problems.append(
                GraphProblem(
                    code=CODE_SYNTAX_ERROR,
                    severity=Severity.BLOCKER,
                    message=f"{path} is not valid Python: {result.error}.",
                    path=path,
                    line=result.error_line,
                    suggestion=(
                        "This file cannot be imported at all, so nothing in the project will "
                        "run until it parses. Fix the syntax error on the reported line."
                    ),
                )
            )

    # -------------------------------------------------------------------- resolve imports
    for path, node in graph.nodes.items():
        result = parsed[path]
        own_module = node.module or ""
        internal: Set[str] = set()
        stdlib: Set[str] = set()
        third_party: Set[str] = set()
        unresolved: Set[str] = set()

        for ref in result.refs:
            module = ref.module
            if ref.relative:
                resolved = _resolve_relative(
                    own_module, node.is_package_init, ref.level, ref.module
                )
                if resolved is None:
                    graph.problems.append(
                        GraphProblem(
                            code=CODE_BAD_RELATIVE_IMPORT,
                            severity=Severity.BLOCKER,
                            message=(
                                f"{path} imports {ref.raw!r}, which points above the project "
                                "root. Python raises ImportError for this at start-up."
                            ),
                            path=path,
                            line=ref.line,
                            suggestion="Use an absolute import of a module inside the project.",
                        )
                    )
                    continue
                module = resolved

            if not module:
                continue

            target = module_index.get(module)

            # ``from package import submodule`` binds a module, not a symbol. Recording the
            # edge to the submodule as well is what makes the graph agree with what actually
            # gets imported, and it keeps the name check below from complaining that a package
            # does not define one of its own modules.
            submodule_names: Set[str] = set()
            for name in ref.names:
                sub_path = module_index.get(f"{module}.{name}")
                if sub_path:
                    submodule_names.add(name)
                    if sub_path != path:
                        internal.add(sub_path)
                        graph.edges.append(
                            ImportEdge(
                                source=path,
                                target=sub_path,
                                module=f"{module}.{name}",
                                names=(),
                                line=ref.line,
                                top_level=ref.top_level,
                                dynamic=ref.dynamic,
                            )
                        )

            if target is not None:
                if target != path:
                    internal.add(target)
                    graph.edges.append(
                        ImportEdge(
                            source=path,
                            target=target,
                            module=module,
                            names=tuple(n for n in ref.names if n not in submodule_names),
                            line=ref.line,
                            top_level=ref.top_level,
                            star=ref.star,
                            dynamic=ref.dynamic,
                        )
                    )
                _check_names(
                    graph,
                    source_path=path,
                    target_path=target,
                    module=module,
                    ref=ref,
                    skip=submodule_names,
                    parsed=parsed,
                )
                continue

            if module in package_prefixes:
                # A directory in the project with no ``__init__.py``: importable as a namespace
                # package, but it defines nothing, so any name asked of it that is not one of
                # its own modules cannot be found.
                for name in ref.names:
                    if name in submodule_names:
                        continue
                    graph.problems.append(
                        GraphProblem(
                            code=CODE_MISSING_SYMBOL,
                            severity=Severity.BLOCKER,
                            message=(
                                f"{path} imports {name!r} from {module!r}, but that directory "
                                "has no __init__.py and contains no module of that name."
                            ),
                            path=path,
                            line=ref.line,
                            suggestion=(
                                f"Add an __init__.py to {module.replace('.', '/')} that exports "
                                f"{name!r}, or import it from the module that defines it."
                            ),
                        )
                    )
                continue

            root = module.split(".")[0]

            if root in project_roots:
                # Looks internal and is not there. This is the failure the user's original
                # complaint was about: generation "succeeded" and the first import died.
                graph.problems.append(
                    GraphProblem(
                        code=CODE_UNRESOLVED_INTERNAL,
                        severity=Severity.BLOCKER,
                        message=(
                            f"{path} imports {module!r}, but this project contains no such "
                            "module."
                        ),
                        path=path,
                        line=ref.line,
                        suggestion=(
                            "Either the file was never generated or the import names the wrong "
                            "module. Available modules: "
                            + ", ".join(sorted(module_index)[:12])
                            + ("." if len(module_index) <= 12 else ", ...")
                        ),
                    )
                )
                unresolved.add(module)
                continue

            if root in _STDLIB_ROOTS:
                stdlib.add(root)
                continue

            if root in declared_roots:
                third_party.add(root)
                continue

            unresolved.add(root)
            graph.problems.append(
                GraphProblem(
                    code=CODE_UNDECLARED_DEPENDENCY,
                    severity=Severity.MAJOR,
                    message=(
                        f"{path} imports {root!r}, which is not part of the standard library "
                        "and is not listed in requirements.txt."
                    ),
                    path=path,
                    line=ref.line,
                    suggestion=(
                        f"Add the distribution that provides {root!r} to requirements.txt, or "
                        "remove the import. Installing the listed requirements is not enough "
                        "to make this file import as it stands."
                    ),
                )
            )

        node.imports = sorted(internal)
        node.stdlib = sorted(stdlib)
        node.third_party = sorted(third_party)
        node.unresolved = sorted(unresolved)

    # ------------------------------------------------------------------------ whole-graph
    _check_test_imports(graph)
    _check_cycles(graph)
    _check_orphans(graph)
    if manifest is not None:
        _compare_with_plan(graph, manifest)

    return graph


def _check_names(
    graph: DependencyGraph,
    *,
    source_path: str,
    target_path: str,
    module: str,
    ref: ModuleRef,
    skip: Set[str],
    parsed: Dict[str, _ParsedModule],
) -> None:
    """
    Verify that a ``from X import y`` can actually find ``y``.

    This is the check that catches the most damaging class of generation bug: an import that
    reads perfectly and resolves to a real file which does not define the name. It fails at
    start-up with ``ImportError: cannot import name``, long after the point where the system
    would otherwise have declared the project ready.

    Skipped when the target file could not be parsed - the syntax error is already reported and
    a second complaint about every name it fails to define adds noise, not information - and
    when the target uses ``import *``, which makes its module namespace genuinely unknowable
    without executing it.
    """
    target = parsed.get(target_path)
    if target is None or target.error or target.star_import or ref.star:
        return

    for name in ref.names:
        if name in skip or name in target.symbols:
            continue
        graph.problems.append(
            GraphProblem(
                code=CODE_MISSING_SYMBOL,
                severity=Severity.BLOCKER,
                message=(
                    f"{source_path} imports {name!r} from {module!r}, but {target_path} does "
                    f"not define it."
                ),
                path=source_path,
                line=ref.line,
                suggestion=(
                    f"Define {name!r} in {target_path}, or import it from the module that "
                    "does. Python raises ImportError for this before any agent code runs."
                ),
                detail=(target_path,),
            )
        )


def _check_test_imports(graph: DependencyGraph) -> None:
    """
    Source files must not import the test suite.

    A project whose agent code depends on its own tests cannot be shipped without them, and the
    export deliberately allows a user to drop the ``tests/`` directory. The assembler already
    keeps test symbols out of its automatic imports; this catches an explicit one written by a
    model or added by hand.
    """
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        target = graph.nodes.get(edge.target)
        if source is None or target is None:
            continue
        if target.is_test and not source.is_test:
            graph.problems.append(
                GraphProblem(
                    code=CODE_TEST_IMPORTED,
                    severity=Severity.MAJOR,
                    message=(
                        f"{edge.source} imports {edge.target}, which is part of the test "
                        "suite. The project would stop working if the tests were removed."
                    ),
                    path=edge.source,
                    line=edge.line,
                    suggestion=(
                        "Move the shared code into the application package and import it from "
                        "both places."
                    ),
                    detail=(edge.target,),
                )
            )


def _check_cycles(graph: DependencyGraph) -> None:
    """
    Report import cycles, with a severity that reflects whether they actually fail.

    A cycle made entirely of module-level ``from ... import name`` statements raises
    ``ImportError`` the first time the project is imported, so it is a blocker. A cycle that
    passes through a plain ``import package.module``, or through an import inside a function,
    does not necessarily fail - Python tolerates a partially initialised module as long as
    nothing reaches into it during the import. Calling the second kind a blocker would be
    overstating it; ignoring it would be understating it, because it is fragile and the next
    edit can turn it into the first kind.
    """
    if not graph.edges:
        return

    by_pair: Dict[Tuple[str, str], List[ImportEdge]] = {}
    for edge in graph.edges:
        by_pair.setdefault((edge.source, edge.target), []).append(edge)

    for cycle in graph.cycles():
        pairs = list(zip(cycle, cycle[1:]))
        edges = [edge for pair in pairs for edge in by_pair.get(pair, ())]
        fails_on_import = bool(edges) and all(
            edge.top_level and edge.binds_names for edge in edges
        )
        chain = " -> ".join(cycle)
        if fails_on_import:
            graph.problems.append(
                GraphProblem(
                    code=CODE_CIRCULAR_IMPORT,
                    severity=Severity.BLOCKER,
                    message=(
                        f"These files import each other in a loop: {chain}. Because every step "
                        "imports a name directly, Python raises ImportError on the first run."
                    ),
                    path=cycle[0],
                    suggestion=(
                        "Move the shared name into a module both sides can import, or import "
                        "the module rather than the name at the point of use."
                    ),
                    detail=tuple(cycle),
                )
            )
        else:
            graph.problems.append(
                GraphProblem(
                    code=CODE_CIRCULAR_IMPORT,
                    severity=Severity.MINOR,
                    message=(
                        f"These files import each other in a loop: {chain}. It does not fail "
                        "today, because at least one step imports a module rather than a name "
                        "or does so inside a function."
                    ),
                    path=cycle[0],
                    suggestion=(
                        "Worth breaking anyway: the next change to either file can turn this "
                        "into an ImportError at start-up."
                    ),
                    detail=tuple(cycle),
                )
            )


def _check_orphans(graph: DependencyGraph) -> None:
    """
    Files nothing imports and nothing runs.

    Section 70 asks that every generated file have a reason to exist, and a module no other
    module mentions has none - it is dead weight in the export and a distraction in the file
    tree. Reported as minor: it is untidy rather than broken.

    Entry points, tests, ``conftest.py`` and package markers are excluded, because for each of
    them "nothing imports it" is the normal state rather than a symptom.
    """
    imported: Set[str] = {edge.target for edge in graph.edges}
    for path, node in sorted(graph.nodes.items()):
        if path in imported or node.is_entrypoint or node.is_test or node.is_package_init:
            continue
        if posixpath.basename(path) == "conftest.py":
            continue
        graph.problems.append(
            GraphProblem(
                code=CODE_ORPHAN_FILE,
                severity=Severity.MINOR,
                message=(
                    f"Nothing in the project imports {path}, and it is not the entry point, so "
                    "none of its code can run."
                ),
                path=path,
                suggestion=(
                    "Either import it from the code that needs it, or remove it - an unused "
                    "file in an exported project is a question the user has to answer."
                ),
            )
        )


def _compare_with_plan(graph: DependencyGraph, manifest: ProjectManifest) -> None:
    """
    Record where the planned ``depends_on`` and the emitted imports disagree.

    Both directions are informational, and deliberately so.

    An edge the plan predicted and the code does not have usually means the emitter found it
    did not need that module - a prediction that did not come true is not a defect. An edge the
    code has and the plan did not predict means the plan's picture of the project is incomplete,
    which matters because the plan is what the architecture view is drawn from; the impact
    analysis in this module reads the emitted graph precisely so it stays right either way.

    Reporting these as INFO keeps them visible to whoever is improving the planner without ever
    being the reason a working project is withheld from a user.
    """
    for spec in manifest.files:
        if not spec.path.endswith(".py"):
            continue
        node = graph.nodes.get(spec.path)
        if node is None:
            continue
        planned = {p for p in spec.depends_on if p}
        actual = set(node.imports)
        missing = sorted(planned - actual)
        extra = sorted(actual - planned)
        if not missing and not extra:
            continue
        graph.drift[spec.path] = {"planned_only": missing, "emitted_only": extra}
        if missing:
            graph.problems.append(
                GraphProblem(
                    code=CODE_UNREALISED_EDGE,
                    severity=Severity.INFO,
                    message=(
                        f"The plan expected {spec.path} to import "
                        f"{', '.join(missing)}, and the generated code does not."
                    ),
                    path=spec.path,
                    suggestion=(
                        "Not a fault in the project - the emitter did not need it. Worth "
                        "checking only if the plan and the code should have matched."
                    ),
                    detail=tuple(missing),
                )
            )
        if extra:
            graph.problems.append(
                GraphProblem(
                    code=CODE_UNPLANNED_EDGE,
                    severity=Severity.INFO,
                    message=(
                        f"{spec.path} imports {', '.join(extra)}, which the plan did not "
                        "list as a dependency."
                    ),
                    path=spec.path,
                    suggestion=(
                        "The dependency graph shown to the user is read from the emitted code, "
                        "so it is still accurate; the plan's own list is what is incomplete."
                    ),
                    detail=tuple(extra),
                )
            )


# ============================================================================== validation
def validate_imports(
    project: GeneratedProject,
    manifest: Optional[ProjectManifest] = None,
) -> List[ReviewIssue]:
    """
    Every import problem in a project, as review issues.

    The convenience form of :func:`build_graph` for callers that only want the verdict -
    section 28's gate, which has to pass before a project may be described as READY. A caller
    that also wants to *show* the graph should use :func:`build_graph` and keep the object,
    rather than calling both and parsing everything twice.
    """
    return build_graph(project, manifest).to_issues()
