// frontend/src/api.ts
//
// The only module that talks to the network.
//
// Two things are centralised here on purpose.
//
// **Error normalisation.** The backend guarantees one error shape, so this layer converts every
// non-2xx response into a thrown `ApiFailure` carrying that shape. Callers therefore never
// inspect `res.status` or guess at a body: they `catch (e)` and render `e.error.message` plus
// `e.error.action`. A network failure (server not running) is normalised into the same shape,
// because from the user's point of view "cannot reach the server" is just another error with an
// action attached.
//
// **The SSE subscription.** `subscribeToRun` wraps `EventSource` so components deal in
// callbacks rather than raw events. The important detail is the explicit close on the `done`
// event: `EventSource` reconnects automatically when a stream ends, which is normally the
// feature you want and here would mean silently re-running the replay forever after a finished
// run. Closing on `done` is what makes the stream terminate.

import type {
  AgentRunResult,
  ApiError,
  ConnectionCheck,
  CreateProjectResponse,
  FrameworkInfo,
  Health,
  LLMSettings,
  PipelineEvent,
  ProjectRecord,
  ProjectSummary,
  ProviderInfo,
  RequirementAnalysis,
  RunSnapshot,
} from "./types";

const BASE = "/api";

/** A failed request, carrying the backend's structured error. */
export class ApiFailure extends Error {
  readonly error: ApiError;
  readonly status: number;

  constructor(error: ApiError, status: number) {
    super(error.message);
    this.name = "ApiFailure";
    this.error = error;
    this.status = status;
  }
}

const OFFLINE: ApiError = {
  code: "network_error",
  message: "Could not reach the server.",
  action: "Check that the backend is running: uvicorn backend.app:app --reload",
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    // fetch only rejects for transport-level problems, which always mean the same thing here.
    throw new ApiFailure(OFFLINE, 0);
  }

  if (res.status === 204) return undefined as T;

  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }

  if (!res.ok) {
    // A well-formed error from our own handlers. Anything else (a proxy's HTML error page,
    // say) still has to become an ApiError, so fall back to the status line.
    const parsed = body as Partial<ApiError> | null;
    throw new ApiFailure(
      parsed && typeof parsed.message === "string"
        ? (parsed as ApiError)
        : {
            code: "http_error",
            message: `The server returned ${res.status}.`,
            action: "Try again, or check the server log.",
            detail: text.slice(0, 500) || null,
          },
      res.status,
    );
  }

  return body as T;
}

const json = (payload: unknown): RequestInit => ({
  method: "POST",
  body: JSON.stringify(payload),
});

// ------------------------------------------------------------------------------- meta
export const getHealth = () => request<Health>("/health");
export const getFrameworks = () => request<FrameworkInfo[]>("/frameworks");
export const getProviders = () => request<ProviderInfo[]>("/providers");

// --------------------------------------------------------------------------- analysis
export const analyze = (payload: {
  requirement: string;
  provider?: string | null;
  model?: string | null;
  use_model?: boolean;
}) => request<RequirementAnalysis>("/analyze", json(payload));

// --------------------------------------------------------------------------- projects
export interface CreateProjectPayload {
  requirement: string;
  name?: string | null;
  framework?: string | null;
  provider?: string | null;
  model?: string | null;
  runtime_provider?: string | null;
  runtime_model?: string | null;
  include_tests?: boolean;
  run_tests?: boolean;
  use_model?: boolean;
  max_iterations?: number | null;
}

export const createProject = (payload: CreateProjectPayload) =>
  request<CreateProjectResponse>("/projects", json(payload));

export const listProjects = (limit = 50, offset = 0) =>
  request<ProjectSummary[]>(`/projects?limit=${limit}&offset=${offset}`);

export const getProject = (id: string) => request<ProjectRecord>(`/projects/${id}`);

export const getProjectRuns = (id: string) =>
  request<Record<string, unknown>[]>(`/projects/${id}/runs`);

export const deleteProject = (id: string) =>
  request<void>(`/projects/${id}`, { method: "DELETE" });

/** Browser-native download. Not a fetch: letting the browser stream it avoids buffering the
 *  whole archive in memory, and the Content-Disposition filename is honoured for free. */
export const downloadProjectUrl = (id: string) => `${BASE}/projects/${id}/download`;

// -------------------------------------------------------------------------- playground
/** `install` is deliberately not exposed in the Playground UI - see Playground.tsx for why.
 *  It stays in the payload type for callers that know a project has no dependencies. */
export interface PlaygroundRunSnapshot {
  run_id: string;
  project_id: string;
  status: string;
  finished: boolean;
  duration_s: number;
  result?: AgentRunResult | null;
  error?: ApiError | null;
}

export const startPlaygroundRun = (
  id: string,
  payload: { query: string; install?: boolean; timeout?: number | null },
) => request<PlaygroundRunSnapshot>(`/projects/${id}/run`, json(payload));

export const getPlaygroundRun = (runId: string) =>
  request<PlaygroundRunSnapshot>(`/playground-runs/${runId}`);

// ---------------------------------------------------------------------------- settings
export const getSettings = () => request<LLMSettings>("/settings/llm");

export const saveSettings = (payload: {
  provider: string;
  model?: string | null;
  /** Omit to keep the stored key. Sending "" does not clear it - use deleteApiKey. */
  api_key?: string | null;
  base_url?: string | null;
  temperature?: number | null;
  max_tokens?: number | null;
  timeout?: number | null;
  top_p?: number | null;
  extra?: Record<string, unknown>;
}) => request<LLMSettings>("/settings/llm", { method: "PUT", body: JSON.stringify(payload) });

export const deleteApiKey = () =>
  request<void>("/settings/llm/key", { method: "DELETE" });

export const testConnection = (payload: {
  provider?: string | null;
  model?: string | null;
  api_key?: string | null;
  base_url?: string | null;
  use_saved?: boolean;
}) => request<ConnectionCheck>("/settings/llm/test", json(payload));

// -------------------------------------------------------------------------------- runs
export const getRun = (runId: string) => request<RunSnapshot>(`/runs/${runId}`);

export interface RunSubscription {
  close: () => void;
}

/**
 * Follow a run's progress.
 *
 * `onStage` fires per pipeline event, `onDone` once with the terminal snapshot. `onError`
 * fires only for a stream that broke before finishing - a *failed run* is not a stream
 * error, it arrives through `onDone` with `status: "error"`, which is the distinction that
 * lets the UI show a pipeline failure without also claiming the connection dropped.
 */
export function subscribeToRun(
  runId: string,
  handlers: {
    onStage?: (event: PipelineEvent) => void;
    onDone?: (snapshot: RunSnapshot) => void;
    onError?: (error: ApiError) => void;
  },
): RunSubscription {
  const source = new EventSource(`${BASE}/runs/${runId}/events`);
  let finished = false;

  source.addEventListener("stage", (raw) => {
    try {
      handlers.onStage?.(JSON.parse((raw as MessageEvent).data) as PipelineEvent);
    } catch {
      // A malformed frame is not worth tearing the stream down for; the terminal snapshot
      // carries the authoritative event list anyway.
    }
  });

  source.addEventListener("done", (raw) => {
    finished = true;
    try {
      handlers.onDone?.(JSON.parse((raw as MessageEvent).data) as RunSnapshot);
    } catch {
      handlers.onError?.({
        code: "stream_error",
        message: "The run finished but its result could not be read.",
        action: "Reload the page to fetch the saved result.",
      });
    }
    // Without this, EventSource treats the closed stream as a dropped connection and
    // reconnects, replaying the whole finished run on a loop.
    source.close();
  });

  source.onerror = () => {
    if (finished) return;
    // EventSource retries on its own while CONNECTING. Only a definitively closed stream is
    // worth reporting, otherwise a momentary blip would surface as an error banner.
    if (source.readyState === EventSource.CLOSED) {
      handlers.onError?.({
        code: "stream_closed",
        message: "Lost the connection to the run.",
        action: "Reload the page - the run is still going on the server.",
      });
    }
  };

  return {
    close: () => {
      finished = true;
      source.close();
    },
  };
}

/** Narrow an unknown catch value to the structured error the UI renders. */
export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiFailure) return err.error;
  if (err instanceof Error) {
    return { code: "client_error", message: err.message };
  }
  return { code: "client_error", message: "Something went wrong." };
}
