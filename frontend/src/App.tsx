// frontend/src/App.tsx
//
// The shell: navigation, routes, and the one global piece of state worth having - health.
//
// Health is fetched once here rather than per page because two of its fields change what the
// whole app can do. Without persistence, the workspace and settings pages cannot work at all,
// and without a credential, generation will fail at the first model call. Surfacing both in the
// top bar means a user learns that from a badge on arrival instead of from a failed run three
// clicks later.

import { useEffect, useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import { getHealth, toApiError } from "./api";
import { ErrorBanner } from "./components/ErrorBanner";
import { CreatePage } from "./pages/Create";
import { ProjectsPage } from "./pages/Projects";
import { RunPage } from "./pages/Run";
import { SettingsPage } from "./pages/Settings";
import { WorkspacePage } from "./pages/Workspace";
import type { ApiError, Health } from "./types";

export function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  useEffect(() => {
    getHealth().then(setHealth, (err) => setError(toApiError(err)));
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="dot" />
          Multi-Agent Generator
        </Link>

        <nav className="nav">
          <NavLink to="/" end>
            Create
          </NavLink>
          <NavLink to="/projects">Projects</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </nav>

        {health && (
          <div className="actions">
            {!health.credential_configured && (
              <Link to="/settings" className="badge warn" title="No API key resolved">
                No API key
              </Link>
            )}
            {!health.persistence && (
              <span className="badge bad" title={health.persistence_error ?? undefined}>
                No database
              </span>
            )}
            <span className="badge faint">v{health.version}</span>
          </div>
        )}
      </header>

      <main className="main">
        {/* Only a failed health check lands here. Everything else is reported by the page that
            caused it, next to the thing that failed. */}
        <ErrorBanner error={error} onRetry={() => window.location.reload()} />

        {health && !health.persistence && (
          <ErrorBanner
            variant="warn"
            error={{
              code: "storage_unavailable",
              message: "Running without a database — generated projects will not be saved.",
              action:
                "Check that DATA_DIR is writable and DATABASE_URL points at a SQLite file, then restart the server.",
              detail: health.persistence_error,
            }}
          />
        )}

        <Routes>
          <Route path="/" element={<CreatePage health={health} />} />
          <Route path="/runs/:runId" element={<RunPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId" element={<WorkspacePage />} />
          <Route path="/settings" element={<SettingsPage health={health} />} />
          <Route
            path="*"
            element={
              <div className="empty">
                That page does not exist. <Link to="/">Start a new project</Link>.
              </div>
            }
          />
        </Routes>
      </main>
    </div>
  );
}
