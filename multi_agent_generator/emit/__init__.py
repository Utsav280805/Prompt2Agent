# multi_agent_generator/emit/__init__.py
"""
Multi-file project emission.

The old generators each produced a single Python string, which is why a complex agent arrived
as one long ``agent.py`` no matter how many agents it contained. This package replaces that
with a plan and an assembler: :mod:`multi_agent_generator.core.architecture` decides what
files exist, an emitter writes this framework's tools, agents and workflow as fragments, and
:func:`assemble_project` places them and computes the imports between them.

The entry point is :func:`assemble_project`. Everything else here supports it.

Three properties hold for every project this package produces, and they are the reason it
exists:

* **The layout is proportional.** A one-agent, one-tool project gets three Python files. A
  five-agent project with branching gets a package with a module per responsibility. Neither
  gets the other's structure.
* **Imports are computed from what the code uses**, not written by hand, so a name is imported
  from the module that actually defines it and no file carries an import it does not need.
* **Every generated project exposes ``run_workflow(query) -> dict``**, which is what the
  Playground calls. That single contract is the difference between a Playground that runs the
  agent and one that prints its source code.
"""
from __future__ import annotations

from .agno_emitter import AgnoEmitter
from .assemble import EMITTERS, AssemblyError, assemble_project, emitter_for
from .base import Emitter, Fragment
from .crewai_emitter import CrewAIEmitter, CrewAIFlowEmitter
from .langgraph_emitter import LangGraphEmitter
from .react_emitter import ReActEmitter, ReActLCELEmitter

__all__ = [
    "assemble_project",
    "emitter_for",
    "AssemblyError",
    "EMITTERS",
    "Emitter",
    "Fragment",
    "AgnoEmitter",
    "CrewAIEmitter",
    "CrewAIFlowEmitter",
    "LangGraphEmitter",
    "ReActEmitter",
    "ReActLCELEmitter",
]
