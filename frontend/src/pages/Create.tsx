// frontend/src/pages/Create.tsx
//
// Section 17: the project-creation form.
//
// Framework is "let the system choose" by default. That is the product's actual claim - it can
// pick CrewAI over LangGraph and say why - and defaulting to a named framework would quietly
// bypass the feature. Choosing one explicitly is honoured absolutely by the backend, so the
// picker is an override rather than a suggestion, and the copy says so.
//
// The framework and provider lists come from the API, never from a constant in this file. A
// hard-coded list is how a picker ends up offering something the backend cannot generate; these
// arrive with `installed` and `configured` flags, so an unavailable option is shown with the
// exact pip command that fixes it instead of being silently absent or silently broken.
//
// Submitting navigates to /runs/:runId rather than rendering the run inline. The run has a real
// URL from the moment it exists, so a refresh, a bookmark or a shared link all resume the same
// progress view - which is the entire reason the backend returns a run id up front.

import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createProject, getFrameworks, getProviders, toApiError } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import type { ApiError, FrameworkInfo, Health, ProviderInfo } from "../types";

const EXAMPLES = [
  "Create a research assistant that finds recent papers on a topic and writes a short literature review",
  "Build a team that reviews a pull request for security problems and writes the review comment",
  "Make a support triage crew that classifies an incoming ticket and drafts a reply",
];

/** Shown immediately so the picker is never empty while `/api/frameworks` loads. */
const FALLBACK_FRAMEWORKS: FrameworkInfo[] = [
  {
    name: "crewai",
    label: "CrewAI",
    description: "Role-playing agents that run ordered tasks as a crew.",
    strengths: ["roles", "sequential", "delegation"],
    best_for: ["roles", "sequential"],
    installed: true,
    install_command: null,
  },
  {
    name: "langgraph",
    label: "LangGraph",
    description: "An explicit state graph with conditional edges and loops.",
    strengths: ["branching", "state", "cycles"],
    best_for: ["branching", "state"],
    installed: true,
    install_command: null,
  },
  {
    name: "react",
    label: "LangChain ReAct agent",
    description: "One agent that reasons and calls tools in a loop.",
    strengths: ["tools", "simplicity"],
    best_for: ["tools"],
    installed: true,
    install_command: null,
  },
  {
    name: "react-lcel",
    label: "LangChain (LCEL chain)",
    description: "A prompt-to-model chain with no multi-agent coordination.",
    strengths: ["simplicity"],
    best_for: ["simplicity"],
    installed: true,
    install_command: null,
  },
  {
    name: "crewai-flow",
    label: "CrewAI Flow",
    description: "Event-driven CrewAI with listeners and branching.",
    strengths: ["roles", "branching", "events"],
    best_for: ["roles", "branching"],
    installed: true,
    install_command: null,
  },
  {
    name: "agno",
    label: "Agno",
    description: "Lightweight agents and teams that share memory.",
    strengths: ["roles", "collaboration"],
    best_for: ["roles"],
    installed: true,
    install_command: null,
  },
];

export function CreatePage({ health }: { health: Health | null }) {
  const navigate = useNavigate();

  const [requirement, setRequirement] = useState("");
  const [name, setName] = useState("");
  const [framework, setFramework] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [includeTests, setIncludeTests] = useState(true);
  const [runTests, setRunTests] = useState(true);
  const [useModel, setUseModel] = useState(true);
  const [maxIterations, setMaxIterations] = useState("");
  const [advanced, setAdvanced] = useState(false);

  const [frameworks, setFrameworks] = useState<FrameworkInfo[]>(FALLBACK_FRAMEWORKS);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    // A failure here is not fatal - the form still submits with defaults - so it is reported
    // without blocking, and the pickers just stay empty.
    getFrameworks().then(setFrameworks, (err) => setError(toApiError(err)));
    getProviders().then(setProviders, () => {});
  }, []);

  const chosenFramework = frameworks.find((f) => f.name === framework);
  const chosenProvider = providers.find((p) => p.name === provider);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!requirement.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createProject({
        requirement: requirement.trim(),
        name: name.trim() || null,
        framework: framework || null,
        provider: provider || null,
        model: model.trim() || null,
        include_tests: includeTests,
        // Running tests you did not generate is meaningless, so this is forced consistent
        // rather than left as two independent switches that can contradict each other.
        run_tests: includeTests && runTests,
        use_model: useModel,
        max_iterations: maxIterations ? Number(maxIterations) : null,
      });
      navigate(`/runs/${created.run_id}`, {
        state: { projectId: created.project_id, persisted: created.persisted },
      });
    } catch (err) {
      setError(toApiError(err));
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-head">
        <h1>Describe what you want the agents to do</h1>
        <p>
          Plain English is enough. The pipeline analyses the requirement, picks a framework and
          explains why, generates a real multi-file project, reviews it, improves it, and runs its
          tests before calling it done.
        </p>
      </div>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {health && !health.credential_configured && (
        <ErrorBanner
          variant="warn"
          error={{
            code: "missing_credential",
            message: "No API key is configured, so generation will fail at the first model call.",
            action: "Add one on the Settings page, or set it in your .env file.",
          }}
        />
      )}

      <form onSubmit={submit} className="grid-side">
        <div>
          <div className="card">
            <div className="field">
              <label htmlFor="req">Requirement</label>
              <textarea
                id="req"
                value={requirement}
                onChange={(e) => setRequirement(e.target.value)}
                placeholder="Create a research assistant that…"
                rows={5}
                maxLength={8000}
                disabled={busy}
                required
              />
              <div className="hint">
                {requirement.length}/8000 · be specific about the steps and the output you want
              </div>
            </div>

            <div className="field">
              <div className="hint mb">Or start from an example:</div>
              <div className="pills">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    className="pill"
                    style={{ cursor: "pointer", textAlign: "left" }}
                    onClick={() => setRequirement(example)}
                    disabled={busy}
                  >
                    {example.slice(0, 52)}…
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <label htmlFor="name">Project name (optional)</label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Derived from the requirement if left blank"
                maxLength={200}
                disabled={busy}
              />
            </div>

            <div className="actions">
              <button className="primary" type="submit" disabled={busy || !requirement.trim()}>
                {busy ? "Starting…" : "Generate project"}
              </button>
              {busy && <span className="spinner" />}
            </div>
          </div>
        </div>

        <div>
          <div className="card">
            <div className="card-title">
              <h3>Framework</h3>
            </div>

            <div className="field">
              <select
                value={framework}
                onChange={(e) => setFramework(e.target.value)}
                disabled={busy}
                aria-label="Framework"
              >
                <option value="">Let the system choose</option>
                {frameworks.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.label}
                    {f.installed ? "" : " (not installed)"}
                  </option>
                ))}
              </select>
              <div className="hint">
                {framework
                  ? "An explicit choice is always respected — the selector will not override it."
                  : "The selector returns a framework, a reason and a confidence score."}
              </div>
            </div>

            {chosenFramework && (
              <>
                <p className="muted" style={{ fontSize: "0.85rem" }}>
                  {chosenFramework.description}
                </p>
                {chosenFramework.strengths.length > 0 && (
                  <div className="pills mb">
                    {chosenFramework.strengths.slice(0, 5).map((s) => (
                      <span className="pill" key={s}>
                        {s}
                      </span>
                    ))}
                  </div>
                )}
                {!chosenFramework.installed && chosenFramework.install_command && (
                  <div className="alert warn">
                    <div className="alert-msg">This framework is not installed.</div>
                    <div className="alert-action mono">{chosenFramework.install_command}</div>
                    <div className="hint">
                      Generation still works — only running the agent needs the package.
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="card">
            <div className="card-title">
              <h3>Options</h3>
              <button
                type="button"
                className="ghost sm"
                onClick={() => setAdvanced((v) => !v)}
              >
                {advanced ? "Hide advanced" : "Advanced"}
              </button>
            </div>

            <label className="check">
              <input
                type="checkbox"
                checked={includeTests}
                onChange={(e) => setIncludeTests(e.target.checked)}
                disabled={busy}
              />
              <span>Generate a test suite</span>
            </label>

            <label className="check">
              <input
                type="checkbox"
                checked={runTests && includeTests}
                onChange={(e) => setRunTests(e.target.checked)}
                disabled={busy || !includeTests}
              />
              <span>
                Run the tests
                <div className="hint">
                  A project is only marked ready when its tests actually ran and passed.
                </div>
              </span>
            </label>

            {advanced && (
              <>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={useModel}
                    onChange={(e) => setUseModel(e.target.checked)}
                    disabled={busy}
                  />
                  <span>
                    Use a model to analyse and review
                    <div className="hint">
                      Off runs the offline heuristic only — instant and free, but shallower and
                      it needs no credential.
                    </div>
                  </span>
                </label>

                <div className="field mt">
                  <label htmlFor="prov">Provider</label>
                  <select
                    id="prov"
                    value={provider}
                    onChange={(e) => setProvider(e.target.value)}
                    disabled={busy}
                  >
                    <option value="">Server default</option>
                    {providers.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.label}
                        {p.configured ? "" : " (no key)"}
                      </option>
                    ))}
                  </select>
                  {chosenProvider && !chosenProvider.configured && (
                    <div className="hint" style={{ color: "var(--warn)" }}>
                      No credential resolved. Set {chosenProvider.credential_env.join(" or ")}.
                    </div>
                  )}
                </div>

                <div className="field">
                  <label htmlFor="model">Model</label>
                  <input
                    id="model"
                    type="text"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder={chosenProvider?.default_model ?? "Provider default"}
                    disabled={busy}
                  />
                  <div className="hint">
                    Plain id — no route prefix. Use <span className="mono">Qwen/Qwen2.5-7B-Instruct</span>,
                    not <span className="mono">huggingface/Qwen/…</span>.
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="iters">Maximum improvement passes</label>
                  <input
                    id="iters"
                    type="number"
                    min={1}
                    max={10}
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(e.target.value)}
                    placeholder="Server default (3)"
                    disabled={busy}
                  />
                </div>
              </>
            )}
          </div>
        </div>
      </form>
    </>
  );
}
