// frontend/src/pages/Workspace.tsx
//
// Section 18: one project, in tabs.
//
// A generated project is several kinds of artefact at once - a plan, a file tree, a review, a
// test report, a thing you can run - and tabs are here because those are genuinely separate
// questions ("what did it decide?" vs "does it pass?") that share one URL. Everything is
// rendered from a single `GET /api/projects/{id}`; there is no per-tab fetch, so switching tabs
// is instant and cannot show two inconsistent snapshots of the same project.
//
// The header is deliberately blunt about verification. A project that generated fine but whose
// tests never ran is *not* shown as ready, and `unmet_criteria` is printed verbatim rather than
// summarised, because the whole point of the acceptance gate is that the user sees the reason
// the system refused to vouch for the code.
//
// The Playground tab is gated rather than hidden. Running an agent needs a credential and an
// installed framework; a disabled tab with the reason attached teaches that in one step, where
// a hidden tab would just look like a missing feature.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { downloadProjectUrl, getHealth, getProject, toApiError } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import { FileBrowser } from "../components/FileBrowser";
import { ReviewPanel, TestPanel } from "../components/Panels";
import { Pipeline } from "../components/Pipeline";
import { Playground } from "../components/Playground";
import type {
  ApiError,
  ArchitecturePlan,
  ArchitectureWorkflow,
  Health,
  ProjectRecord,
} from "../types";

type Tab = "overview" | "architecture" | "files" | "review" | "tests" | "playground";

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "architecture", label: "Architecture" },
  { id: "files", label: "Files" },
  { id: "review", label: "Review" },
  { id: "tests", label: "Tests" },
  { id: "playground", label: "Playground" },
];

export function WorkspacePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<ProjectRecord | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");

  const load = useCallback(() => {
    if (!projectId) return;
    setLoading(true);
    getProject(projectId)
      .then((record) => {
        setProject(record);
        setError(null);
      })
      .catch((err) => setError(toApiError(err)))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(load, [load]);
  useEffect(() => {
    // Only needed to explain a disabled Playground, so a failure here is not worth reporting.
    getHealth().then(setHealth, () => {});
  }, []);

  const result = project?.result ?? null;
  const generated = result?.project ?? null;
  const unmet = result?.unmet_criteria ?? [];

  // Why the Playground cannot run, in the order the user would hit the problems.
  const playgroundBlock = useMemo(() => {
    if (!generated) return "This project has no generated code to run.";
    if (health && !health.credential_configured) {
      return "No API key is configured, so the agent cannot reach a model. Add one in Settings.";
    }
    return null;
  }, [generated, health]);

  if (!projectId) return <div className="empty">No project id in the URL.</div>;

  if (loading && !project) {
    return (
      <div className="empty">
        <span className="spinner" /> Loading project…
      </div>
    );
  }

  if (!project) {
    return (
      <>
        <ErrorBanner error={error} onRetry={load} />
        <div className="empty">
          That project could not be loaded. <Link to="/projects">Back to projects</Link>.
        </div>
      </>
    );
  }

  const verified = result?.succeeded === true;

  return (
    <>
      <div className="page-head">
        <div className="card-title">
          <h1>{project.name}</h1>
          <div className="actions">
            <span className={`badge ${verified ? "ok" : project.status === "failed" ? "bad" : "warn"}`}>
              {verified ? "Verified" : project.status}
            </span>
            {generated && (
              <a href={downloadProjectUrl(project.id)}>
                <button className="primary">Download .zip</button>
              </a>
            )}
            <button className="ghost sm" onClick={load}>
              Refresh
            </button>
          </div>
        </div>
        <p className="muted">{project.requirement}</p>
      </div>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {result?.error && (
        <ErrorBanner error={result.error} />
      )}

      {!verified && unmet.length > 0 && (
        <div className="alert warn">
          <div className="alert-msg">This project is not marked verified.</div>
          <ul className="plain mt">
            {unmet.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="tabs" role="tablist">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            role="tab"
            aria-selected={tab === entry.id}
            className={`tab${tab === entry.id ? " active" : ""}`}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
            {entry.id === "files" && generated ? ` (${generated.file_count})` : ""}
            {entry.id === "review" && result?.review && !result.review.passed ? " ⚠" : ""}
            {entry.id === "tests" && result?.tests && !result.tests.ok ? " ⚠" : ""}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <>
          <div className="card">
            <div className="card-title">
              <h3>Summary</h3>
              <div className="actions">
                {generated && <span className="badge info">{generated.framework}</span>}
                {result && <span className="badge">{result.duration_s.toFixed(1)}s</span>}
              </div>
            </div>
            <dl className="kv">
              <dt>Framework</dt>
              <dd>{project.framework ?? "—"}</dd>
              <dt>Meta provider</dt>
              <dd>{project.provider ?? "—"}</dd>
              <dt>Model</dt>
              <dd className="mono">{project.model ?? "—"}</dd>
              <dt>Files</dt>
              <dd>
                {generated ? `${generated.file_count} files, ${generated.total_lines} lines` : "—"}
              </dd>
              <dt>Improvement passes</dt>
              <dd>{result ? result.iterations.length : "—"}</dd>
              <dt>Created</dt>
              <dd>{new Date(project.created_at).toLocaleString()}</dd>
            </dl>
          </div>

          {generated && (
            <div className="card">
              <div className="card-title">
                <h3>How to run it</h3>
              </div>
              <pre className="output">{generated.run_instructions}</pre>
              {generated.dependencies.length > 0 && (
                <>
                  <h4 className="faint">Dependencies</h4>
                  <div className="pills">
                    {generated.dependencies.map((dep) => (
                      <span className="pill mono" key={dep}>
                        {dep}
                      </span>
                    ))}
                  </div>
                </>
              )}
              {generated.notes.length > 0 && (
                <>
                  <h4 className="faint">Notes</h4>
                  <ul className="plain">
                    {generated.notes.map((note, i) => (
                      <li key={i}>{note}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          {result && result.events.length > 0 && (
            <div className="card">
              <div className="card-title">
                <h3>What happened</h3>
              </div>
              <Pipeline events={result.events} finished />
            </div>
          )}
        </>
      )}

      {tab === "architecture" && (
        <>
          {result?.manifest && <ArchitecturePlanView plan={result.manifest} />}

          {result?.framework_choice && (
            <div className="card">
              <div className="card-title">
                <h3>Framework choice</h3>
                <div className="actions">
                  <span className="badge">
                    {Math.round(result.framework_choice.confidence * 100)}% confident
                  </span>
                  {result.framework_choice.user_specified && (
                    <span className="badge info">You chose this</span>
                  )}
                </div>
              </div>
              <p className="muted">{result.framework_choice.reason}</p>
              {result.framework_choice.alternatives.length > 0 && (
                <>
                  <h4 className="faint">Also considered</h4>
                  <ul className="plain">
                    {result.framework_choice.alternatives.map((alt, i) => (
                      <li key={i}>
                        <span className="mono">{alt.framework}</span>
                        {alt.reason ? ` — ${alt.reason}` : ""}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          )}

          {result?.analysis ? (
            <div className="card">
              <div className="card-title">
                <h3>Requirement analysis</h3>
                <div className="actions">
                  <span className="badge">{result.analysis.complexity}</span>
                  {!result.analysis.from_model && (
                    <span
                      className="badge warn"
                      title="Produced by the offline heuristic, not a model"
                    >
                      heuristic
                    </span>
                  )}
                </div>
              </div>
              <p className="muted">{result.analysis.summary}</p>

              <div className="grid-2">
                <div>
                  <h4 className="faint">Goals</h4>
                  <ul className="plain">
                    {result.analysis.goals.map((goal, i) => (
                      <li key={i}>{goal}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="faint">Agents</h4>
                  <div className="pills mb">
                    {result.analysis.suggested_roles.map((role) => (
                      <span className="pill" key={role}>
                        {role}
                      </span>
                    ))}
                  </div>
                  <h4 className="faint">Tools</h4>
                  <div className="pills">
                    {result.analysis.suggested_tools.length === 0 ? (
                      <span className="faint">none</span>
                    ) : (
                      result.analysis.suggested_tools.map((tool) => (
                        <span className="pill mono" key={tool}>
                          {tool}
                        </span>
                      ))
                    )}
                  </div>
                </div>
              </div>

              <h4 className="faint">Shape of the workflow</h4>
              <div className="pills mb">
                {(
                  [
                    ["sequential steps", result.analysis.needs_sequential_steps],
                    ["branching", result.analysis.needs_branching],
                    ["delegation", result.analysis.needs_delegation],
                    ["shared state", result.analysis.needs_state],
                    ["human input", result.analysis.needs_human_input],
                  ] as [string, boolean][]
                )
                  .filter(([, on]) => on)
                  .map(([label]) => (
                    <span className="pill" key={label}>
                      {label}
                    </span>
                  ))}
              </div>

              {result.analysis.risks.length > 0 && (
                <>
                  <h4 className="faint">Risks it flagged</h4>
                  <ul className="plain">
                    {result.analysis.risks.map((risk, i) => (
                      <li key={i}>{risk}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          ) : (
            <div className="empty">No analysis was recorded for this project.</div>
          )}
        </>
      )}

      {tab === "files" &&
        (generated && generated.files.length > 0 ? (
          <FileBrowser files={generated.files} entrypoint={generated.entrypoint} />
        ) : (
          <div className="empty">This project has no generated files.</div>
        ))}

      {tab === "review" && (
        <div className="card">
          <div className="card-title">
            <h3>Code review</h3>
            {result && result.iterations.length > 0 && (
              <span className="badge">
                after {result.iterations.length} pass
                {result.iterations.length === 1 ? "" : "es"}
              </span>
            )}
          </div>
          <ReviewPanel review={result?.review ?? null} />
        </div>
      )}

      {tab === "tests" && (
        <div className="card">
          <div className="card-title">
            <h3>Generated test suite</h3>
          </div>
          <TestPanel tests={result?.tests ?? null} />
        </div>
      )}

      {tab === "playground" && (
        <Playground
          projectId={project.id}
          framework={generated?.framework ?? project.framework ?? "unknown"}
          provider={generated?.provider ?? project.provider ?? "unknown"}
          disabledReason={playgroundBlock}
        />
      )}
    </>
  );
}

function ArchitecturePlanView({ plan }: { plan: ArchitecturePlan }) {
  const workflow = plan.workflows[0];
  const summary = plan.summary;

  return (
    <>
      <div className="card">
        <div className="card-title">
          <h3>Build plan</h3>
          <div className="actions">
            <span className="badge info">{summary.tier}</span>
            <span className="badge">{summary.runtime}</span>
          </div>
        </div>
        <p className="muted">{plan.description || plan.requirement}</p>
        <div className="stats-grid">
          {[
            ["Agents", summary.agents],
            ["Tools", summary.tools],
            ["Workflow steps", summary.workflow_steps],
            ["Services", summary.services],
            ["Python files", summary.python_files],
            ["Dependencies", summary.dependencies],
          ].map(([label, value]) => (
            <div className="stat" key={label}>
              <strong>{value}</strong>
              <span>{label}</span>
            </div>
          ))}
        </div>
        <div className="architecture-meta">
          <span><b>Package</b> <span className="mono">{summary.package || "—"}</span></span>
          <span><b>Entrypoint</b> <span className="mono">{summary.entrypoint || "—"}</span></span>
          <span><b>Model</b> <span className="mono">{summary.model || "—"}</span></span>
        </div>
      </div>

      {workflow && <WorkflowPlan workflow={workflow} />}

      <div className="grid-2">
        <div className="card">
          <div className="card-title"><h3>Agents</h3><span className="badge">{plan.agents.length}</span></div>
          {plan.agents.map((agent) => (
            <div className="architecture-row" key={agent.name}>
              <div className="row-head"><strong>{agent.label}</strong><span className="badge info">{agent.role}</span></div>
              <p>{agent.goal}</p>
              <div className="faint">{agent.inputs || "Query"} → {agent.outputs || "Agent response"}</div>
              {agent.tools.length > 0 && <div className="pills mt">{agent.tools.map((tool) => <span className="pill mono" key={tool}>{tool}</span>)}</div>}
            </div>
          ))}
        </div>
        <div className="card">
          <div className="card-title"><h3>Tools & configuration</h3><span className="badge">{plan.tools.length}</span></div>
          {plan.tools.map((tool) => (
            <div className="architecture-row" key={tool.name}>
              <div className="row-head"><strong>{tool.label}</strong><span className={`badge ${tool.implemented ? "ok" : "bad"}`}>{tool.implemented ? "implemented" : "missing"}</span></div>
              <p>{tool.purpose}</p>
              <div className="faint mono">{tool.module || tool.class_name}</div>
            </div>
          ))}
          {plan.env_vars.length > 0 && <><h4 className="faint">Environment</h4><div className="pills">{plan.env_vars.map((env) => <span className="pill mono" key={env.name}>{env.name}{env.required ? " *" : ""}</span>)}</div></>}
        </div>
      </div>

      <div className="card">
        <div className="card-title"><h3>Planned files & dependencies</h3><span className="badge">{plan.files.length} files</span></div>
        <div className="architecture-files">
          {plan.files.map((file) => (
            <div className="architecture-file" key={file.path}>
              <span className="mono">{file.path}</span>
              <span className="badge">{file.kind}</span>
              <span className="faint">{file.purpose}</span>
            </div>
          ))}
        </div>
        <h4 className="faint">Install set</h4>
        <div className="pills">{plan.dependencies.map((dependency) => <span className="pill mono" key={dependency}>{dependency}</span>)}</div>
      </div>
    </>
  );
}

function WorkflowPlan({ workflow }: { workflow: ArchitectureWorkflow }) {
  return (
    <div className="card">
      <div className="card-title">
        <h3>Workflow execution</h3>
        <div className="actions"><span className="badge info">{workflow.kind}</span><span className="badge">{workflow.nodes.length} nodes</span></div>
      </div>
      <p className="muted">{workflow.description || "The planned path the generated runtime will execute."}</p>
      <div className="workflow-plan">
        {workflow.nodes.map((node, index) => (
          <div className="workflow-node-wrap" key={node.id}>
            <div className={`workflow-node ${node.kind}`}>
              <span className="faint mono">{node.kind}</span>
              <strong>{node.label}</strong>
              <span>{node.description}</span>
              {node.agent && <span className="mono">agent: {node.agent}</span>}
              {node.tool && <span className="mono">tool: {node.tool}</span>}
            </div>
            {index < workflow.nodes.length - 1 && <span className="workflow-arrow">→</span>}
          </div>
        ))}
      </div>
      {workflow.edges.length > 0 && <div className="edge-list">{workflow.edges.map((edge) => <span className="pill mono" key={`${edge.source}-${edge.target}`}>{edge.source} → {edge.target}{edge.label ? ` · ${edge.label}` : ""}</span>)}</div>}
      {workflow.state_fields.length > 0 && <div className="hint mt">Shared state: {workflow.state_fields.join(", ")}</div>}
    </div>
  );
}
