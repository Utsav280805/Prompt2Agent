// frontend/src/components/Panels.tsx
//
// Read-only renderings of the review verdict and the test report.
//
// Both are written to make an unsuccessful outcome *unmissable*, which is the frontend half of
// the brief's "do not cheat the tests" rule. The backend already refuses to call a run
// successful unless review passed and the generated suite actually ran and passed; the UI's job
// is not to soften that. So a skipped suite renders as a warning that says "did not run" rather
// than as an absence, and a project with code but a failed gate shows its `unmet_criteria`
// verbatim instead of a green checkmark next to a download button.

import type { ReviewResult, TestReport } from "../types";

export function ReviewPanel({ review }: { review: ReviewResult | null }) {
  if (!review) {
    return (
      <div className="empty">
        This project was never reviewed, so its quality is unverified.
      </div>
    );
  }

  const tone = review.passed ? "ok" : "bad";

  return (
    <>
      <div className="card">
        <div className="card-title">
          <div>
            <div className="faint" style={{ fontSize: "0.78rem" }}>
              REVIEW SCORE
            </div>
            <div className="score" style={{ color: `var(--${tone})` }}>
              {review.score.toFixed(1)}
              <span className="faint" style={{ fontSize: "0.9rem", fontWeight: 400 }}>
                {" "}
                / 10
              </span>
            </div>
          </div>
          <div className="actions">
            <span className={`badge ${tone}`}>{review.passed ? "Passed" : "Did not pass"}</span>
            {review.blocker_count > 0 && (
              <span className="badge bad">{review.blocker_count} blocker</span>
            )}
            {!review.from_model && <span className="badge warn">Static checks only</span>}
          </div>
        </div>

        {review.summary && <p className="muted">{review.summary}</p>}

        {Object.keys(review.dimension_scores).length > 0 && (
          <dl className="kv mt">
            {Object.entries(review.dimension_scores).map(([name, score]) => (
              <div key={name} style={{ display: "contents" }}>
                <dt>{name.replace(/_/g, " ")}</dt>
                <dd className="mono">{score.toFixed(1)}</dd>
              </div>
            ))}
          </dl>
        )}
      </div>

      {review.required_changes.length > 0 && (
        <div className="card">
          <h3>Required changes</h3>
          <ul className="plain">
            {review.required_changes.map((change, i) => (
              <li key={i}>{change}</li>
            ))}
          </ul>
        </div>
      )}

      {review.issues.length > 0 && (
        <div className="card">
          <div className="card-title">
            <h3>Issues ({review.issues.length})</h3>
          </div>
          {review.issues.map((issue, i) => (
            <div key={i} className={`issue ${issue.severity}`}>
              <div className="issue-msg">{issue.message}</div>
              <div className="issue-meta">
                {issue.severity} · {issue.category}
                {issue.file && ` · ${issue.file}${issue.line ? `:${issue.line}` : ""}`}
              </div>
              {issue.suggestion && <div className="issue-fix">→ {issue.suggestion}</div>}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export function TestPanel({ tests }: { tests: TestReport | null }) {
  if (!tests) {
    return (
      <div className="empty">
        The generated test suite was never run, so nothing here is verified.
      </div>
    );
  }

  // Three outcomes, not two. "Ran and passed" is the only green one; "did not run" is its own
  // state and must not be shown as either success or a test failure, because the fix differs.
  const tone = tests.ok ? "ok" : tests.ran ? "bad" : "warn";
  const headline = tests.timed_out
    ? "Timed out"
    : !tests.ran
      ? "Did not run"
      : tests.ok
        ? "Passed"
        : "Failed";

  return (
    <>
      <div className="card">
        <div className="card-title">
          <h3>Generated test suite</h3>
          <div className="actions">
            <span className={`badge ${tone}`}>{headline}</span>
            <span className="badge">{tests.duration_s.toFixed(1)}s</span>
          </div>
        </div>

        <dl className="kv">
          <dt>passed</dt>
          <dd className="mono">{tests.counts.passed}</dd>
          <dt>failed</dt>
          <dd className="mono">{tests.counts.failed}</dd>
          <dt>skipped</dt>
          <dd className="mono">{tests.counts.skipped}</dd>
          <dt>errors</dt>
          <dd className="mono">{tests.counts.errors}</dd>
          <dt>exit code</dt>
          <dd className="mono">{tests.exit_code ?? "—"}</dd>
        </dl>

        {tests.unavailable_reason && (
          <div className="alert warn mt">
            <div className="alert-msg">The suite could not be run.</div>
            <div className="alert-action">{tests.unavailable_reason}</div>
          </div>
        )}

        {tests.notes && tests.notes.length > 0 && (
          <div className="alert warn mt">
            <div className="alert-msg">
              Some of this project's requirements were not installed.
            </div>
            <div className="alert-action">
              Generated projects may only install named packages from the configured index.
              These lines were refused, which is usually why an import failed:
            </div>
            <ul className="plain mono mt">
              {tests.notes.map((note, i) => (
                <li key={i}>{note}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {tests.failed_tests.length > 0 && (
        <div className="card">
          <h3>Failing tests</h3>
          <ul className="plain mono">
            {tests.failed_tests.map((name, i) => (
              <li key={i}>{name}</li>
            ))}
          </ul>
        </div>
      )}

      {(tests.stdout || tests.stderr) && (
        <div className="card">
          <h3>Output</h3>
          {tests.stdout && <pre className="output">{tests.stdout}</pre>}
          {tests.stderr && (
            <pre className="output mt" style={{ color: "var(--bad)" }}>
              {tests.stderr}
            </pre>
          )}
        </div>
      )}
    </>
  );
}
