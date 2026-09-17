# multi_agent_generator/core/requirements_spec.py
"""
Requirement traceability: REQ-001 onward, and whether the built system honours each one.

This module exists because of one sentence in section 20 - *"'Code runs' does NOT mean: 'The
agent satisfies the user's request.'"* Every other gate in :mod:`multi_agent_generator.core.validation`
asks whether the project is correct *as a program*. None of them looks back at what the person
actually typed. A project can parse, import, expose a runnable entry point, pass its own tests
and still be the wrong system, and nothing downstream would notice.

So the requirement is turned into a numbered list of checkable statements, each one traced to
the plan and the code that satisfies it. The numbering is what makes it traceable: REQ-003 means
the same thing in the analysis, the gate report, the API payload and the Testing tab, so a user
can ask "which part of what I asked for is missing?" and get an answer rather than a score.

**Extraction is deterministic and offline.** No model is consulted. That is not a shortcut - it
is the point. A model asked to list requirements and then asked whether they were met will agree
with itself, which produces a coverage report that always says yes. Every requirement here comes
from a field the analysis stage already committed to, so the list is reproducible and the
verdict is computed from the plan rather than negotiated with a model.

**What blocks and what does not.** A requirement that can be checked mechanically - this role
needs an agent, this tool needs an implementation, ordered work needs an ordered workflow - is
allowed to block release, because a missing agent is a fact rather than an opinion. A
requirement that can only be checked by reading prose blocks nothing and is reported as
unverified. Section 87 forbids inventing numbers, and a coverage percentage produced by keyword
overlap would be exactly that: an invented number wearing a gate's clothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .manifest import AgentSpec, ProjectManifest
from .models import GeneratedProject, RequirementAnalysis, Severity
from .validation import Check, CheckStatus

__all__ = [
    "RequirementItem",
    "RequirementTrace",
    "extract_requirements",
    "trace_requirements",
    "requirement_checks",
    "coverage_summary",
]


#: Kinds of requirement, in the order they are numbered.
#:
#: Order is fixed so that REQ-003 keeps meaning the same thing between two runs of the same
#: requirement. A list that renumbers itself is not traceable, and section 56 asks for
#: traceability rather than for a list.
KINDS: Tuple[str, ...] = ("goal", "role", "tool", "behaviour")

#: Words that carry no meaning when matching a goal against generated text.
_NOISE = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that the their then
    there these this to with will would should must can using use used need needs
    """.split()
)


@dataclass(frozen=True)
class RequirementItem:
    """
    One numbered thing the user asked for.

    ``mechanical`` is the field that decides whether this can block a release. It is set at
    extraction time rather than judged later, so the rule "only check what can actually be
    checked" is visible in the data instead of buried in a branch.
    """

    id: str
    text: str
    kind: str
    #: The analysis field this came from, for the Overview tab. Never a guess.
    source: str
    #: True when satisfaction is decidable from the plan rather than from reading prose.
    mechanical: bool = False
    #: Lower-cased significant words, used only by the prose fallback.
    keywords: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "mechanical": self.mechanical,
        }


@dataclass
class RequirementTrace:
    """
    One requirement, and what in the built system answers it.

    ``paths`` and ``evidence`` are deliberately separate from the verdict. A user who disagrees
    with a verdict needs to see what it was based on, and "REQ-002 is satisfied" with nothing
    behind it is indistinguishable from a hard-coded pass.
    """

    item: RequirementItem
    satisfied: Optional[bool]
    #: Plain-language statement of what was found. Written for the person who typed the request.
    finding: str = ""
    paths: Tuple[str, ...] = ()
    evidence: Tuple[str, ...] = ()

    @property
    def unverified(self) -> bool:
        """True when nothing could be established either way. Not the same as a failure."""
        return self.satisfied is None

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.item.as_dict(),
            "satisfied": self.satisfied,
            "finding": self.finding,
            "paths": list(self.paths),
            "evidence": list(self.evidence),
        }


# ----------------------------------------------------------------------------- extraction
def extract_requirements(analysis: RequirementAnalysis) -> List[RequirementItem]:
    """
    Turn an analysis into a numbered requirement list.

    Args:
        analysis: The structural reading of what the user asked for.

    Returns:
        ``RequirementItem`` objects numbered REQ-001 upward. Empty only when the analysis
        itself is empty, which is worth reporting rather than papering over: a requirement that
        yielded no goals, no roles and no tools is one the analysis stage failed to read.
    """
    items: List[RequirementItem] = []

    def add(text: str, kind: str, source: str, *, mechanical: bool) -> None:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return
        if any(existing.text.lower() == cleaned.lower() for existing in items):
            return
        items.append(
            RequirementItem(
                id=f"REQ-{len(items) + 1:03d}",
                text=cleaned,
                kind=kind,
                source=source,
                mechanical=mechanical,
                keywords=_keywords(cleaned),
            )
        )

    # Goals first: they are the closest thing to the user's own words, so they get the low
    # numbers a person will quote back. Not mechanical - a goal is prose.
    for goal in analysis.goals:
        add(goal, "goal", "analysis.goals", mechanical=False)

    for role in analysis.suggested_roles:
        add(
            f"The system needs an agent responsible for the {role} work.",
            "role",
            "analysis.suggested_roles",
            mechanical=True,
        )

    for tool in analysis.suggested_tools:
        add(
            f"The system needs a working {tool} tool.",
            "tool",
            "analysis.suggested_tools",
            mechanical=True,
        )

    # Behaviour flags. Each of these is decidable from the workflow plan, which is why they are
    # allowed to block: "the steps must happen in order" is either true of the graph or it is not.
    if analysis.needs_sequential_steps:
        add(
            "The work must happen as ordered steps rather than in one shot.",
            "behaviour",
            "analysis.needs_sequential_steps",
            mechanical=True,
        )
    if analysis.needs_branching:
        add(
            "What happens next must be able to depend on an earlier result.",
            "behaviour",
            "analysis.needs_branching",
            mechanical=True,
        )
    if analysis.needs_delegation:
        add(
            "One agent must be able to hand work to another.",
            "behaviour",
            "analysis.needs_delegation",
            mechanical=True,
        )
    if analysis.needs_state:
        add(
            "Information must carry across steps rather than being recomputed.",
            "behaviour",
            "analysis.needs_state",
            mechanical=True,
        )
    if analysis.needs_human_input:
        # Deliberately not mechanical. Whether a project genuinely pauses for a person is a
        # runtime property, and asserting it from the plan would be a gate that cannot fail.
        add(
            "The system must be able to pause for a person to answer.",
            "behaviour",
            "analysis.needs_human_input",
            mechanical=False,
        )

    return items


def _keywords(text: str) -> Tuple[str, ...]:
    """Significant lower-cased words, for the prose fallback only."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
    return tuple(sorted({w for w in words if w not in _NOISE}))


# -------------------------------------------------------------------------------- tracing
def trace_requirements(
    items: Sequence[RequirementItem],
    manifest: Optional[ProjectManifest],
    project: Optional[GeneratedProject] = None,
) -> List[RequirementTrace]:
    """
    Decide, for each requirement, what in the built system answers it.

    Args:
        items: The requirement list from :func:`extract_requirements`.
        manifest: The architecture plan. Without it every verdict is ``None`` - unverified -
            because a claim of coverage with nothing to check it against is worthless.
        project: The generated project, used for file paths and for the prose fallback.

    Returns:
        One :class:`RequirementTrace` per item, in the same order.
    """
    if manifest is None:
        return [
            RequirementTrace(
                item=item,
                satisfied=None,
                finding=(
                    "There is no architecture plan for this project, so this part of the "
                    "request could not be traced to anything."
                ),
            )
            for item in items
        ]

    traces: List[RequirementTrace] = []
    for item in items:
        if item.kind == "role":
            traces.append(_trace_role(item, manifest))
        elif item.kind == "tool":
            traces.append(_trace_tool(item, manifest))
        elif item.kind == "behaviour":
            traces.append(_trace_behaviour(item, manifest))
        else:
            traces.append(_trace_goal(item, manifest, project))
    return traces


def _trace_role(item: RequirementItem, manifest: ProjectManifest) -> RequirementTrace:
    """Match a suggested role to a planned agent, by name first and by wording second."""
    role = _subject(item.text, "responsible for the ", " work.")
    agent = manifest.agent(role) if role else None
    if agent is None and role:
        agent = next(
            (a for a in manifest.agents if _covers_name(role, _agent_haystack(a))),
            None,
        )
    if agent is None:
        return RequirementTrace(
            item=item,
            satisfied=False,
            finding=f"No agent covers the {role or 'requested'} work.",
            evidence=tuple(f"planned agents: {a.label}" for a in manifest.agents[:6]),
        )
    module = agent.module or manifest.entrypoint
    return RequirementTrace(
        item=item,
        satisfied=True,
        finding=f"The {agent.label} agent covers this.",
        paths=(module,) if module else (),
        evidence=(f"{agent.label} - {agent.role}",),
    )


def _trace_tool(item: RequirementItem, manifest: ProjectManifest) -> RequirementTrace:
    """Match a suggested tool to a planned tool, and say which agent holds it."""
    wanted = _subject(item.text, "needs a working ", " tool.")
    tool = manifest.tool(wanted) if wanted else None
    if tool is None and wanted:
        tool = next(
            (
                t
                for t in manifest.tools
                if _covers_name(
                    wanted,
                    (t.name.replace("_", " "), t.label, t.purpose or ""),
                )
            ),
            None,
        )
    if tool is None:
        return RequirementTrace(
            item=item,
            satisfied=False,
            finding=f"No tool implements {wanted or 'this capability'}.",
            evidence=tuple(f"planned tools: {t.name}" for t in manifest.tools[:6]),
        )
    holders = [a.label for a in manifest.agents_using(tool.name)]
    used_by = f", used by {', '.join(holders)}." if holders else ", though no agent uses it."
    paths = (tool.module,) if tool.module else ()
    evidence = (f"{tool.label} - {tool.purpose}",)
    if not tool.implemented:
        # A stub is a real answer to "is there a tool for this?" and not a real answer to
        # "does it work?", so the verdict withholds rather than passing. Reporting a
        # clearly-labelled placeholder as a satisfied requirement is the manufactured green
        # section 87 forbids; reporting it as a failure would block every project whose plan
        # includes a capability the user is expected to finish, which is most of them.
        return RequirementTrace(
            item=item,
            satisfied=None,
            finding=(
                f"The {tool.label} tool is planned{used_by} It is a labelled placeholder, so "
                "whether this requirement is really met depends on completing it."
            ),
            paths=paths,
            evidence=evidence + ("implementation: placeholder",),
        )
    return RequirementTrace(
        item=item,
        satisfied=True,
        finding=f"The {tool.label} tool implements this{used_by}",
        paths=paths,
        evidence=evidence,
    )


def _trace_behaviour(item: RequirementItem, manifest: ProjectManifest) -> RequirementTrace:
    """
    Check a behavioural requirement against the workflow plan.

    Each branch here answers a question about the *graph*, not about the source text. That is
    what makes these verdicts worth blocking on: "there are two or more ordered agent steps" is
    a property of the plan that cannot be argued with, whereas grepping the code for the word
    "sequential" would pass on a comment.
    """
    workflow = manifest.primary_workflow
    source = item.source

    if source == "analysis.needs_sequential_steps":
        ordered = workflow.ordered_agent_names() if workflow else []
        ok = len(ordered) >= 2
        return RequirementTrace(
            item=item,
            satisfied=ok,
            finding=(
                f"The workflow runs {len(ordered)} steps in order: {' then '.join(ordered)}."
                if ok
                else "The plan has no ordered sequence of steps, so the work happens in one shot."
            ),
            evidence=(f"{len(ordered)} ordered agent step(s)",),
        )

    if source == "analysis.needs_branching":
        ok = bool(workflow and workflow.has_branching)
        return RequirementTrace(
            item=item,
            satisfied=ok,
            finding=(
                "The workflow has a conditional edge, so a result can change what happens next."
                if ok
                else "The workflow is a straight line, so no result can change what happens next."
            ),
        )

    if source == "analysis.needs_delegation":
        delegators = [a.label for a in manifest.agents if a.allow_delegation]
        hierarchical = bool(workflow and workflow.kind == "hierarchical")
        ok = bool(delegators) or hierarchical
        return RequirementTrace(
            item=item,
            satisfied=ok,
            finding=(
                f"{', '.join(delegators)} can hand work to another agent."
                if delegators
                else (
                    "The workflow is hierarchical, so a manager can hand work to specialists."
                    if hierarchical
                    else "No agent in the plan is allowed to delegate."
                )
            ),
            evidence=tuple(f"{name} may delegate" for name in delegators[:4]),
        )

    if source == "analysis.needs_state":
        # State is real when something is threaded between steps. Architecture records that
        # on workflow nodes more often than on AgentSpec.inputs, so both are checked.
        threaded = [a.label for a in manifest.agents if a.inputs.strip()]
        ordered = workflow.ordered_agent_names() if workflow else []
        node_inputs = [
            n.id
            for n in (workflow.agent_nodes() if workflow else [])
            if (n.inputs or "").strip()
        ]
        ok = len(ordered) >= 2 or len(threaded) >= 2 or len(node_inputs) >= 2
        return RequirementTrace(
            item=item,
            satisfied=ok,
            finding=(
                "Each step declares what it receives from the one before it."
                if ok
                else "Nothing in the plan carries information from one step to the next."
            ),
            evidence=(f"{len(threaded)} agent(s) declare their inputs",),
        )

    return RequirementTrace(
        item=item,
        satisfied=None,
        finding="This is a runtime behaviour, so the plan alone cannot confirm it.",
    )


def _trace_goal(
    item: RequirementItem,
    manifest: ProjectManifest,
    project: Optional[GeneratedProject],
) -> RequirementTrace:
    """
    Relate a prose goal to the plan, without pretending the result is a verdict.

    ``satisfied`` stays ``None`` whatever the overlap turns out to be, and that is the whole
    design of this function. Keyword overlap between a sentence the user typed and text a
    generator wrote is a hint worth showing; treating it as a pass would manufacture the exact
    kind of unearned green the brief opens by rejecting, and treating it as a failure would
    block real projects on vocabulary.
    """
    haystack = " ".join(
        [
            manifest.description,
            " ".join(manifest.capabilities),
            " ".join(a.goal for a in manifest.agents),
            " ".join(a.role for a in manifest.agents),
            " ".join(" ".join(a.responsibilities) for a in manifest.agents),
            " ".join(f"{t.label} {t.purpose}" for t in manifest.tools),
        ]
    ).lower()

    hits = [word for word in item.keywords if word in haystack]
    where: Tuple[str, ...] = ()
    if project is not None and hits:
        where = tuple(
            f.path
            for f in project.source_files()
            if not f.path.startswith("tests/")
            and any(word in f.content.lower() for word in hits)
        )[:4]

    if hits:
        finding = (
            "The plan addresses this in the words it uses for its agents and tools "
            f"({', '.join(hits[:5])}), but only a person can confirm the goal is met."
        )
    else:
        finding = (
            "Nothing in the plan visibly addresses this goal. It may still be covered - this "
            "check compares wording, which is why it does not block."
        )

    return RequirementTrace(
        item=item,
        satisfied=None,
        finding=finding,
        paths=where,
        evidence=(f"{len(hits)} of {len(item.keywords)} significant word(s) matched",),
    )


# --------------------------------------------------------------------------------- checks
def requirement_checks(
    analysis: Optional[RequirementAnalysis],
    manifest: Optional[ProjectManifest] = None,
    project: Optional[GeneratedProject] = None,
) -> List[Check]:
    """
    The requirement traces as validation checks, ready for ``validate_project(extra_checks=...)``.

    One check per requirement plus a coverage check, all in the ``requirements`` category that
    :data:`multi_agent_generator.core.validation.CATEGORIES` already reserves. Reaching
    validation through ``extra_checks`` rather than being called from inside it is deliberate:
    these are the only gates that need the original request, and the validator should not have
    to know about the analysis stage to do its job.
    """
    if analysis is None:
        return []

    items = extract_requirements(analysis)
    if not items:
        return [
            Check(
                id="requirements.extracted",
                category="requirements",
                title="The request was broken into requirements",
                status=CheckStatus.FAILED,
                summary=(
                    "Nothing checkable could be read out of the request, so there is no way to "
                    "tell whether the generated system does what was asked."
                ),
                severity=Severity.MAJOR,
                detail="The analysis produced no goals, no roles, no tools and no behaviours.",
                fix="Describe the request in more detail, naming what the system should do.",
            )
        ]

    traces = trace_requirements(items, manifest, project)
    checks: List[Check] = [_check_for(trace) for trace in traces]
    checks.append(_coverage_check(traces))
    return checks


def _check_for(trace: RequirementTrace) -> Check:
    """One requirement, as a gate. The id carries the REQ number so the UI can link to it."""
    item = trace.item
    check_id = f"requirements.{item.id.lower().replace('-', '_')}"
    title = f"{item.id}: {item.text}"

    if trace.satisfied is True:
        status, severity, fix = CheckStatus.PASSED, Severity.INFO, ""
    elif trace.satisfied is False:
        status = CheckStatus.FAILED
        # Only a mechanically-decided miss blocks. A requirement judged by reading prose must
        # never be the reason a working project is withheld from its author.
        severity = Severity.MAJOR if item.mechanical else Severity.MINOR
        fix = "Use Add Capability to cover this, or refine the request and regenerate."
    else:
        status = CheckStatus.WARNING
        severity = Severity.INFO
        fix = "Read the generated code for this part of the request and confirm it yourself."

    return Check(
        id=check_id,
        category="requirements",
        title=title,
        status=status,
        summary=trace.finding,
        severity=severity,
        detail=f"Source: {item.source}. Kind: {item.kind}.",
        fix=fix,
        paths=trace.paths,
        evidence=trace.evidence,
        # An unverified prose requirement is explicitly not required: a check nobody can decide
        # must not block, or every run would stall on wording.
        required=item.mechanical,
    )


def _coverage_check(traces: Sequence[RequirementTrace]) -> Check:
    """
    How much of the request is accounted for, counted rather than estimated.

    The numbers are counts of traces, not a confidence figure. Section 41 forbids invented
    confidence scores and section 87 forbids hard-coded counts, so this reports exactly what the
    traces say and nothing more - including the unverified ones, which are listed as unverified
    rather than quietly folded into the pass column.
    """
    summary = coverage_summary(traces)
    missing = [t.item.id for t in traces if t.satisfied is False and t.item.mechanical]
    unverified = [t.item.id for t in traces if t.unverified]

    if missing:
        status, severity = CheckStatus.FAILED, Severity.MAJOR
        text = (
            f"{len(missing)} of {summary['total']} things you asked for are not in the built "
            f"system ({', '.join(missing)})."
        )
        fix = "Add the missing capability, or refine the request and generate again."
    elif unverified and summary["satisfied"] == 0:
        status, severity = CheckStatus.WARNING, Severity.INFO
        text = (
            f"None of the {summary['total']} requirements could be checked automatically, so "
            "read the generated code before relying on it."
        )
        fix = "Review the requirement list on the Overview tab."
    else:
        status, severity = CheckStatus.PASSED, Severity.INFO
        text = (
            f"{summary['satisfied']} of {summary['total']} requirements are traced to code"
            + (f"; {len(unverified)} need your own reading." if unverified else ".")
        )
        fix = ""

    return Check(
        id="requirements.coverage",
        category="requirements",
        title="Everything asked for is accounted for",
        status=status,
        summary=text,
        severity=severity,
        fix=fix,
        evidence=(
            f"{summary['satisfied']} satisfied",
            f"{summary['missing']} missing",
            f"{summary['unverified']} unverified",
        ),
    )


def coverage_summary(traces: Sequence[RequirementTrace]) -> Dict[str, int]:
    """Counts of what was traced. Counts only - no percentage, no score."""
    return {
        "total": len(traces),
        "satisfied": sum(1 for t in traces if t.satisfied is True),
        "missing": sum(1 for t in traces if t.satisfied is False),
        "unverified": sum(1 for t in traces if t.unverified),
    }


def _agent_haystack(agent: AgentSpec) -> Tuple[str, ...]:
    return (
        agent.label,
        agent.role,
        agent.name.replace("_", " "),
        agent.goal,
        " ".join(agent.responsibilities),
    )


def _covers_name(wanted: str, haystacks: Sequence[str]) -> bool:
    """
    Whether a planned name answers a suggested role or tool.

    ``researcher`` must match ``Research Specialist``, and ``web_search`` must match
    ``duckduckgo_search``. Exact substring is tried first; then tokens, including one being
    a prefix of the other so stems like research/researcher count.
    """
    needle = (wanted or "").strip().lower().replace("_", " ")
    if not needle:
        return False
    blobs = [h.lower().replace("_", " ") for h in haystacks if h]
    if any(needle in blob or blob in needle for blob in blobs if blob):
        return True
    ntoks = _name_tokens(needle)
    atoks = set()
    for blob in blobs:
        atoks.update(_name_tokens(blob))
    for left in ntoks:
        for right in atoks:
            if left == right or left.startswith(right) or right.startswith(left):
                return True
    return False


def _name_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()) if w not in _NOISE}


def _subject(text: str, after: str, before: str) -> str:
    """Pull the noun back out of a generated requirement sentence."""
    start = text.find(after)
    if start < 0:
        return ""
    rest = text[start + len(after) :]
    end = rest.find(before)
    return (rest[:end] if end >= 0 else rest).strip()
