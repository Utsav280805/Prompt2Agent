// frontend/src/components/Playground.tsx
//
// Section 19: the Agent Playground. Ask the generated agent a real question and see what it
// actually says.
//
// This is the only place in the UI that triggers execution of generated code, and the copy
// reflects that honestly rather than hiding it - the user is told the code runs in a sandboxed
// workspace with a time limit, because "click here to execute code a language model wrote" is
// something a person is entitled to understand before they click.
//
// Submission is short-lived and the backend owns the long-running install/execute job. The API
// client polls that job until the sandbox returns a terminal result, so a browser refresh or a
// Vite proxy timeout cannot kill the work halfway through dependency installation.
//
// There is no "install dependencies" toggle. There used to be, and it was a control whose off
// state could only break the run: each run gets a fresh workspace, so nothing is carried over
// from a previous one, and without the framework on the import path the agent cannot start at
// all. The install is now cached by requirement content on the server, which is what made the
// toggle pointless as well as harmful - the first run for a given framework pays the download
// and the rest do not.

import { useEffect, useRef, useState } from "react";
import { getPlaygroundRun, startPlaygroundRun, toApiError } from "../api";
import { ErrorBanner } from "./ErrorBanner";
import type { AgentRunResult, ApiError } from "../types";

interface Props {
  projectId: string;
  framework: string;
  provider: string;
  /** Blocks the run and explains why, rather than letting the user hit a sandbox failure. */
  disabledReason?: string | null;
}

export function Playground({ projectId, framework, provider, disabledReason }: Props) {
  const storageKey = `magen:playground:${projectId}`;
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AgentRunResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const pollToken = useRef(0);

  const watchRun = async (runId: string) => {
    const token = ++pollToken.current;
    setBusy(true);
    try {
      while (token === pollToken.current) {
        const snapshot = await getPlaygroundRun(runId);
        if (snapshot.finished) {
          sessionStorage.removeItem(storageKey);
          if (snapshot.result) setResult(snapshot.result);
          else if (snapshot.error) setError(snapshot.error);
          setBusy(false);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 750));
      }
    } catch (err) {
      if (token === pollToken.current) setError(toApiError(err));
    } finally {
      if (token === pollToken.current) setBusy(false);
    }
  };

  useEffect(() => {
    const saved = sessionStorage.getItem(storageKey);
    if (!saved) return;
    try {
      const pending = JSON.parse(saved) as { runId?: string; query?: string };
      if (pending.runId) {
        setQuery(pending.query ?? "");
        setError(null);
        void watchRun(pending.runId);
      }
    } catch {
      sessionStorage.removeItem(storageKey);
    }
    return () => {
      pollToken.current += 1;
    };
  }, [projectId]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const started = await startPlaygroundRun(projectId, { query: query.trim() });
      sessionStorage.setItem(
        storageKey,
        JSON.stringify({ runId: started.run_id, query: query.trim() }),
      );
      await watchRun(started.run_id);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {disabledReason && (
        <ErrorBanner
          variant="warn"
          error={{
            code: "not_runnable",
            message: "This project cannot be run yet.",
            action: disabledReason,
          }}
        />
      )}

      <div className="card">
        <div className="card-title">
          <h3>Ask the agent</h3>
          <div className="actions">
            <span className="badge">{framework}</span>
            <span className="badge">{provider}</span>
          </div>
        </div>

        <form onSubmit={submit}>
          <div className="field">
            <label htmlFor="pg-query">Your question</label>
            <textarea
              id="pg-query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What are the three biggest risks in adopting solid-state batteries?"
              disabled={busy || !!disabledReason}
            />
            <div className="hint">
              The generated code runs in an isolated workspace with a wall-clock timeout. It
              receives only the one API key its own provider needs, and its dependencies are
              installed away from the server's own — never into it.
            </div>
          </div>

          <div className="actions mt">
            <button
              className="primary"
              type="submit"
              disabled={busy || !query.trim() || !!disabledReason}
            >
              {busy ? "Running…" : "Run agent"}
            </button>
            {busy && (
              <span className="muted">
                <span className="spinner" /> queued, installing, or executing — this can take a few minutes on the first run
              </span>
            )}
          </div>
        </form>
      </div>

      {result && <RunOutput result={result} />}
    </>
  );
}

function RunOutput({ result }: { result: AgentRunResult }) {
  const tone = result.ok ? "ok" : "bad";
  return (
    <div className="card">
      <div className="card-title">
        <h3>Result</h3>
        <div className="actions">
          <span className={`badge ${tone}`}>
            {result.timed_out ? "Timed out" : result.status}
          </span>
          <span className="badge">{result.duration_s.toFixed(1)}s</span>
          {result.exit_code != null && (
            <span className="badge">exit {result.exit_code}</span>
          )}
        </div>
      </div>

      {/* The extracted answer first when the harness could isolate it. Falling back to raw
          stdout rather than showing nothing matters: an agent that printed its answer without
          the expected marker still answered. */}
      {result.output ? (
        <pre className="output">{result.output}</pre>
      ) : result.stdout ? (
        <>
          <div className="hint mb">
            The agent's answer could not be isolated, so this is its full output.
          </div>
          <pre className="output">{result.stdout}</pre>
        </>
      ) : (
        <div className="empty">The agent produced no output.</div>
      )}

      {result.error && (
        <div className="mt">
          <ErrorBanner error={result.error} />
        </div>
      )}

      <RunTrace result={result} />

      {result.stderr && (
        <details className="mt">
          <summary className="faint" style={{ cursor: "pointer" }}>
            Standard error
          </summary>
          <pre className="output mt" style={{ color: "var(--bad)" }}>
            {result.stderr}
          </pre>
        </details>
      )}

      {result.output && result.stdout && result.stdout !== result.output && (
        <details className="mt">
          <summary className="faint" style={{ cursor: "pointer" }}>
            Full stdout
          </summary>
          <pre className="output mt">{result.stdout}</pre>
        </details>
      )}
    </div>
  );
}

/**
 * What the agent did, taken from the run's own record.
 *
 * Every value here comes from the framework's report of the execution that just happened. When
 * a project's run function returns only an answer, there is nothing to show, and this says so
 * rather than displaying a plausible-looking trace that never occurred.
 */
function RunTrace({ result }: { result: AgentRunResult }) {
  const steps = result.steps ?? [];
  const calls = result.tool_calls ?? [];

  // Nothing recorded and nothing to explain: stay out of the way rather than adding an empty
  // panel to every successful run.
  if (steps.length === 0 && calls.length === 0 && !result.entry) return null;

  return (
    <details className="mt" open={steps.length > 0 || calls.length > 0}>
      <summary className="faint" style={{ cursor: "pointer" }}>
        Execution trace
        {steps.length > 0 && ` — ${steps.length} step${steps.length === 1 ? "" : "s"}`}
        {calls.length > 0 && `, ${calls.length} tool call${calls.length === 1 ? "" : "s"}`}
      </summary>

      {result.entry && (
        <div className="hint mt">
          Called <span className="mono">{result.entry}()</span> in the generated project.
        </div>
      )}

      {steps.length > 0 ? (
        <ol className="plain mt">
          {steps.map((step, i) => (
            <li key={i}>
              <div>
                <span className="mono">{step.agent || step.action || `step ${i + 1}`}</span>
                {step.status && <span className="badge ml">{step.status}</span>}
                {typeof step.duration_s === "number" && (
                  <span className="badge ml">{step.duration_s.toFixed(1)}s</span>
                )}
              </div>
              {step.output && <pre className="output mt">{asText(step.output)}</pre>}
            </li>
          ))}
        </ol>
      ) : (
        <div className="hint mt">
          This project's run function returned an answer without a step-by-step record, so no
          steps are available for this run.
        </div>
      )}

      {calls.length > 0 && (
        <>
          <div className="faint mt">Tool calls</div>
          <ol className="plain">
            {calls.map((call, i) => (
              <li key={i}>
                <span className="mono">{call.name || `tool ${i + 1}`}</span>
                {call.args !== undefined && (
                  <pre className="output mt">{asText(call.args)}</pre>
                )}
                {call.result !== undefined && (
                  <pre className="output mt">{asText(call.result)}</pre>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </details>
  );
}

/** A step or tool value as display text. Frameworks put strings, dicts and lists in here. */
function asText(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
