// frontend/src/components/Pipeline.tsx
//
// The visible pipeline (section 17): every stage the run goes through, its state, and the
// message it emitted.
//
// Stage state is *derived* from the event list rather than tracked in its own state variable.
// That is the design decision worth defending: events arrive out of a stream that can be
// replayed, reconnected or loaded from the database after the fact, and any parallel state
// machine would need to handle all three paths identically. Folding the event list into a
// status map on each render means a reconnecting client and a fresh page load cannot disagree
// about what happened - there is only one source of truth and it is the events themselves.
//
// Stages the pipeline chose not to run (improvement, when the first review passed) never appear
// as "pending forever": they are simply absent from the list once the run is finished, because
// showing an eternally-unstarted step reads as a hang.

import { STAGE_LABELS, STAGE_ORDER } from "../types";
import type { PipelineEvent, StageName } from "../types";

type StageState = "pending" | "running" | "done" | "failed" | "warn";

interface StageView {
  stage: StageName;
  state: StageState;
  message: string;
  duration: number | null;
  iteration: number | null;
}

function eventKey(event: PipelineEvent): string {
  return `${event.timestamp}|${event.stage}|${event.status}|${event.message}|${event.iteration ?? ""}`;
}

function uniqueEvents(events: PipelineEvent[]): PipelineEvent[] {
  const seen = new Set<string>();
  const out: PipelineEvent[] = [];
  for (const event of events) {
    const key = eventKey(event);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(event);
  }
  return out;
}

function fold(events: PipelineEvent[], finished: boolean): StageView[] {
  const seen = new Map<StageName, StageView>();

  for (const event of events) {
    const current = seen.get(event.stage);
    let state: StageState;
    if (event.status === "failed") state = "failed";
    else if (event.status === "succeeded") state = "done";
    else if (event.status === "started") state = "running";
    else if (event.status === "warning") state = current?.state === "running" ? "warn" : (current?.state ?? "warn");
    else state = current?.state ?? "running";

    const next: StageView = {
      stage: event.stage,
      state,
      message: event.message,
      duration: event.duration_s ?? current?.duration ?? null,
      iteration: event.iteration ?? current?.iteration ?? null,
    };
    // A failure is sticky: a later info event must not repaint a failed stage as running.
    if (current?.state === "failed" && next.state !== "failed") next.state = "failed";
    seen.set(event.stage, next);
  }

  const extras = [...seen.keys()].filter((stage) => !STAGE_ORDER.includes(stage));
  const order = [...STAGE_ORDER, ...extras];

  return order
    .filter((stage) => {
      if (seen.has(stage)) return true;
      return !finished;
    })
    .map(
      (stage) =>
        seen.get(stage) ?? {
          stage,
          state: "pending" as StageState,
          message: "",
          duration: null,
          iteration: null,
        },
    );
}

const ICON: Record<StageState, string> = {
  pending: "",
  running: "",
  done: "✓",
  failed: "✕",
  warn: "!",
};

interface Props {
  events: PipelineEvent[];
  finished: boolean;
  /** Show the raw event log under the stage list. */
  showLog?: boolean;
}

export function Pipeline({ events, finished, showLog = true }: Props) {
  const unique = uniqueEvents(events);
  const stages = fold(unique, finished);

  return (
    <div>
      <div className="pipeline">
        {stages.map((view) => (
          <div
            key={view.stage}
            className={`stage ${view.state}${view.state === "running" ? " active" : ""}`}
          >
            <span className="stage-icon" aria-hidden="true">
              {ICON[view.state]}
            </span>
            <span>
              <span className="stage-label">
                {STAGE_LABELS[view.stage] ?? view.stage.replace(/_/g, " ")}
                {view.iteration != null && view.iteration > 1 && (
                  <span className="faint"> · pass {view.iteration}</span>
                )}
              </span>
              {view.message && <div className="stage-msg">{view.message}</div>}
            </span>
            <span className="stage-time">
              {view.duration != null ? `${view.duration.toFixed(1)}s` : ""}
            </span>
          </div>
        ))}
      </div>

      {showLog && unique.length > 0 && (
        <div className="log">
          {unique.map((event, index) => (
            <div key={index} className={`log-line ${event.status}`}>
              <span className="t">{event.timestamp.slice(11, 19)}</span>
              <span className="t">[{event.stage}]</span>
              <span>{event.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
