// frontend/src/pages/Run.tsx
//
// Live progress for one generation run.
//
// The run has its own URL, so this page is reachable three ways that all have to work: straight
// from the create form, from a refresh mid-run, and from a link opened after the run finished.
// `useRun` collapses those into one state, and this page's only job is to render it and then get
// out of the way - once a run succeeds and produced a project, the workspace is where the user
// actually wants to be, so it offers that as the primary action rather than keeping them on a
// finished progress bar.
//
// A finished-but-unsuccessful run is treated as a first-class outcome, not an error. The pipeline
// only reports success when review passed and the generated tests ran and passed, so a run can
// legitimately end with working-looking code and `unmet_criteria` explaining why nobody should
// trust it yet. Those reasons are shown verbatim.

import { Link, useParams } from "react-router-dom";
import { ErrorBanner } from "../components/ErrorBanner";
import { Pipeline } from "../components/Pipeline";
import { useRun } from "../hooks/useRun";

export function RunPage() {
  const { runId } = useParams<{ runId: string }>();
  const run = useRun(runId ?? null);

  const result = run.result;
  const projectId = run.projectId;
  const unmet = result?.unmet_criteria ?? [];

  return (
    <>
      <div className="page-head">
        <div className="card-title">
          <h1>
            {!run.finished
              ? "Generating…"
              : result?.succeeded
                ? "Project ready"
                : run.error
                  ? "Run failed"
                  : "Run finished with problems"}
          </h1>
          <div className="actions">
            {!run.finished && !run.loading && <span className="spinner" />}
            {projectId && (
              <Link to={`/projects/${projectId}`}>
                <button className={result?.succeeded ? "primary" : ""}>Open workspace</button>
              </Link>
            )}
          </div>
        </div>
        {run.requirement && <p className="muted">{run.requirement}</p>}
      </div>

      {/* A stream error is reported without claiming the run failed - the server is still
          working, the browser just lost the narration. */}
      <ErrorBanner
        error={run.error}
        variant={run.finished ? "error" : "warn"}
        onRetry={run.reload}
      />

      {run.finished && !result?.succeeded && unmet.length > 0 && (
        <div className="alert warn">
          <div className="alert-msg">
            This project was not marked ready, and here is exactly why.
          </div>
          <ul className="plain mt">
            {unmet.map((reason, i) => (
              <li key={i}>{reason}</li>
            ))}
          </ul>
          <div className="hint mt">
            The code still exists and is downloadable from the workspace — it just has not passed
            the gate, so it is not presented as verified.
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-title">
          <h2>Pipeline</h2>
          <div className="actions">
            {result && <span className="badge">{result.duration_s.toFixed(1)}s total</span>}
            {result?.framework_choice && (
              <span className="badge info">{result.framework_choice.framework}</span>
            )}
            {result?.succeeded && <span className="badge ok">Ready</span>}
          </div>
        </div>

        {run.loading ? (
          <div className="empty">
            <span className="spinner" /> Loading run…
          </div>
        ) : (
          <Pipeline events={run.events} finished={run.finished} />
        )}
      </div>

      {result?.framework_choice && (
        <div className="card">
          <div className="card-title">
            <h3>Why this framework</h3>
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
    </>
  );
}
