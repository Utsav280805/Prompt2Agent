// frontend/src/pages/Projects.tsx
//
// The workspace list.
//
// It reads the summary endpoint, which deliberately omits generated source - a project can be
// tens of kilobytes and this page only needs a title, a status and a timestamp. Pulling every
// byte the user has ever generated to render a list of rows is the kind of thing that works fine
// for the first three projects and then does not.
//
// Delete asks for confirmation because it cascades: the project's runs and their logs go with it.
// The confirmation names that consequence rather than asking "are you sure".

import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { deleteProject, listProjects, toApiError } from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import type { ApiError, ProjectStatus, ProjectSummary } from "../types";

const TONE: Partial<Record<ProjectStatus, string>> = {
  ready: "ok",
  failed: "bad",
  draft: "",
};

function relative(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

export function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const [removing, setRemoving] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    listProjects()
      .then((rows) => {
        setProjects(rows);
        setError(null);
      })
      .catch((err) => setError(toApiError(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const remove = async (project: ProjectSummary) => {
    const ok = window.confirm(
      `Delete "${project.name}"?\n\nIts generated files, run history and logs are removed too. This cannot be undone.`,
    );
    if (!ok) return;
    setRemoving(project.id);
    try {
      await deleteProject(project.id);
      setProjects((rows) => rows.filter((row) => row.id !== project.id));
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setRemoving(null);
    }
  };

  return (
    <>
      <div className="page-head">
        <div className="card-title">
          <h1>Projects</h1>
          <div className="actions">
            <button className="ghost sm" onClick={load} disabled={loading}>
              Refresh
            </button>
            <Link to="/">
              <button className="primary">New project</button>
            </Link>
          </div>
        </div>
      </div>

      <ErrorBanner error={error} onDismiss={() => setError(null)} onRetry={load} />

      {loading && projects.length === 0 ? (
        <div className="empty">
          <span className="spinner" /> Loading…
        </div>
      ) : projects.length === 0 ? (
        <div className="empty">
          Nothing here yet. <Link to="/">Describe a task</Link> and the generator will build it.
        </div>
      ) : (
        <div className="rows">
          {projects.map((project) => (
            <div className="row" key={project.id}>
              <div className="row-main">
                <div className="row-title">
                  <Link to={`/projects/${project.id}`}>{project.name}</Link>
                </div>
                <div className="row-sub">{project.requirement}</div>
              </div>

              <div className="actions nowrap">
                {project.framework && <span className="badge">{project.framework}</span>}
                <span className={`badge ${TONE[project.status] ?? "warn"}`}>
                  {project.status}
                </span>
                <span className="faint" style={{ fontSize: "0.78rem" }}>
                  {relative(project.updated_at)}
                </span>
                <button
                  className="danger sm"
                  onClick={() => remove(project)}
                  disabled={removing === project.id}
                >
                  {removing === project.id ? "…" : "Delete"}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </>
  );
}
