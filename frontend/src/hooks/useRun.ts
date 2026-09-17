// frontend/src/hooks/useRun.ts
//
// Follow a generation run.
//
// The hook exists because "watch a run" is three different situations that must look identical
// to a component: a run that started in this browser tab, a run already in progress when the
// page loaded, and a run that finished before the page loaded. All three resolve to the same
// `{events, result, error, finished}`.
//
// The order of operations is what makes a refresh mid-run safe. It fetches the snapshot *first*
// and only then subscribes, and it de-duplicates by event index when the stream replays history
// it has already seen. Subscribing first would open a window where events arriving between the
// snapshot and the subscription are lost - the exact race that makes a progress panel silently
// skip a stage.

import { useCallback, useEffect, useState } from "react";
import { getRun, subscribeToRun, toApiError } from "../api";
import type { ApiError, PipelineEvent, PipelineResult, RunSnapshot } from "../types";

export interface RunState {
  events: PipelineEvent[];
  result: PipelineResult | null;
  error: ApiError | null;
  finished: boolean;
  /** True until the initial snapshot resolves, so the UI can avoid an empty-looking flash. */
  loading: boolean;
  projectId: string | null;
  /** Carried on the snapshot, so the page can show what was asked for before a result exists. */
  requirement: string;
}

const EMPTY: RunState = {
  events: [],
  result: null,
  error: null,
  finished: false,
  loading: true,
  projectId: null,
  requirement: "",
};

export function useRun(runId: string | null): RunState & { reload: () => void } {
  const [state, setState] = useState<RunState>(EMPTY);
  const [nonce, setNonce] = useState(0);

  const absorb = useCallback((snapshot: RunSnapshot) => {
    setState({
      events: snapshot.events ?? [],
      result: snapshot.result,
      error: snapshot.error,
      finished: snapshot.finished,
      loading: false,
      projectId: snapshot.project_id,
      requirement: snapshot.requirement ?? "",
    });
  }, []);

  useEffect(() => {
    if (!runId) {
      setState({ ...EMPTY, loading: false });
      return;
    }

    let cancelled = false;
    let subscription: { close: () => void } | null = null;
    setState((s) => ({ ...s, loading: true }));

    (async () => {
      try {
        const snapshot = await getRun(runId);
        if (cancelled) return;
        absorb(snapshot);

        if (snapshot.finished) return; // Nothing left to stream.

        // The stream replays history, so `seen` is how a reconnect avoids appending events the
        // snapshot already provided. Timestamp + stage + status + message is unique per event
        // in practice; omitting the message collapsed two analysis warnings into one key.
        const seen = new Set(
          (snapshot.events ?? []).map(
            (e) => `${e.timestamp}|${e.stage}|${e.status}|${e.message}|${e.iteration ?? ""}`,
          ),
        );

        subscription = subscribeToRun(runId, {
          onStage: (event) => {
            if (cancelled) return;
            const key = `${event.timestamp}|${event.stage}|${event.status}|${event.message}|${event.iteration ?? ""}`;
            if (seen.has(key)) return;
            seen.add(key);
            setState((s) => ({ ...s, events: [...s.events, event], loading: false }));
          },
          onDone: (snap) => {
            if (cancelled) return;
            // The terminal snapshot omits events (the client already has them), so keep ours.
            setState((s) => ({
              ...s,
              events: snap.events?.length ? snap.events : s.events,
              result: snap.result,
              error: snap.error,
              finished: true,
              loading: false,
              projectId: snap.project_id ?? s.projectId,
              requirement: snap.requirement || s.requirement,
            }));
          },
          onError: (err) => {
            if (cancelled) return;
            // A broken stream is not a failed run. The run is still going server-side, so this
            // is reported without marking the run finished.
            setState((s) => ({ ...s, error: s.error ?? err }));
          },
        });
      } catch (err) {
        if (cancelled) return;
        setState({ ...EMPTY, loading: false, error: toApiError(err) });
      }
    })();

    return () => {
      cancelled = true;
      subscription?.close();
    };
  }, [runId, nonce, absorb]);

  return { ...state, reload: () => setNonce((n) => n + 1) };
}
