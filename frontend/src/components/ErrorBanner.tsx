// frontend/src/components/ErrorBanner.tsx
//
// Section 21 of the brief: friendly errors, never tracebacks. This is the single component that
// renders a failure, and there is only one because the API returns exactly one error shape.
//
// The layout encodes what matters. `message` is the headline, `action` sits directly under it in
// full weight because it is the sentence that unblocks the user, and `detail` - which is where a
// stack trace or stderr ends up - is collapsed behind a disclosure. That ordering is the whole
// point: the diagnostic is available to someone who wants it without being the first thing a
// user reads.

import type { ApiError } from "../types";

interface Props {
  error: ApiError | null | undefined;
  /** Rendered as a warning rather than a failure - for degraded state, not a broken request. */
  variant?: "error" | "warn";
  onDismiss?: () => void;
  onRetry?: () => void;
}

export function ErrorBanner({ error, variant = "error", onDismiss, onRetry }: Props) {
  if (!error) return null;

  return (
    <div className={`alert${variant === "warn" ? " warn" : ""}`} role="alert">
      <div className="alert-head">
        <span className="alert-msg">{error.message}</span>
        <span className="actions">
          {onRetry && (
            <button className="ghost sm" onClick={onRetry}>
              Retry
            </button>
          )}
          {onDismiss && (
            <button className="ghost sm" onClick={onDismiss} aria-label="Dismiss">
              ✕
            </button>
          )}
        </span>
      </div>

      {error.action && <div className="alert-action">{error.action}</div>}

      {error.detail && (
        <details className="alert-detail">
          <summary>Technical detail</summary>
          <pre>{error.detail}</pre>
        </details>
      )}

      {/* The error code is shown quietly. It is useless to most users and essential in a bug
          report, which is exactly what a small monospace footnote is for. */}
      <div className="faint mono" style={{ fontSize: "0.72rem", marginTop: "0.4rem" }}>
        {error.code}
      </div>
    </div>
  );
}
