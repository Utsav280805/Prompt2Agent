// frontend/src/components/FileBrowser.tsx
//
// The Files tab: the generated project as a real tree with real contents.
//
// This is the component that makes the difference between "the tool printed some code" and "the
// tool produced a project" legible. It shows every file, marks the entrypoint, and reports line
// counts, because a user's first question about generated code is always "how much of it is
// there, and where do I start reading".
//
// No syntax-highlighting library. A highlighter is ~40kB of bundle to colour code the user is
// about to download and open in their own editor, and getting it wrong (mis-tokenising a
// docstring, say) actively misleads. A `<pre>` with the language named in the header is honest
// and costs nothing.

import { useEffect, useState } from "react";
import type { GeneratedFile } from "../types";

interface Props {
  files: GeneratedFile[];
  entrypoint?: string | null;
}

export function FileBrowser({ files }: Props) {
  const ordered = [...files].sort((a, b) => {
    // Entrypoint first, then tests last, then alphabetical. The entrypoint is where a reader
    // starts; tests are the least interesting thing to land on by default.
    if (a.is_entrypoint !== b.is_entrypoint) return a.is_entrypoint ? -1 : 1;
    const aTest = a.path.startsWith("tests/") || a.path.includes("/test_");
    const bTest = b.path.startsWith("tests/") || b.path.includes("/test_");
    if (aTest !== bTest) return aTest ? 1 : -1;
    return a.path.localeCompare(b.path);
  });

  const [selected, setSelected] = useState<string>(ordered[0]?.path ?? "");

  // A regenerated project can replace the file list entirely; without this the pane would keep
  // showing a path that no longer exists and render blank.
  useEffect(() => {
    if (!ordered.some((f) => f.path === selected)) {
      setSelected(ordered[0]?.path ?? "");
    }
  }, [files, ordered, selected]);

  const active = ordered.find((f) => f.path === selected);

  if (ordered.length === 0) {
    return <div className="empty">This project has no files.</div>;
  }

  const copy = () => {
    if (active) void navigator.clipboard?.writeText(active.content);
  };

  return (
    <div className="files">
      <div className="file-list">
        {ordered.map((file) => (
          <button
            key={file.path}
            className={`file-item${file.path === selected ? " active" : ""}`}
            onClick={() => setSelected(file.path)}
            title={file.description || file.path}
          >
            <span className="n">
              {file.is_entrypoint && <span style={{ color: "var(--accent)" }}>▸ </span>}
              {file.path}
            </span>
            <span className="l">{file.lines}</span>
          </button>
        ))}
      </div>

      {active && (
        <div className="code">
          <div className="code-head">
            <span>
              {active.path}
              <span className="faint"> · {active.language} · {active.lines} lines</span>
            </span>
            <button className="ghost sm" onClick={copy}>
              Copy
            </button>
          </div>
          {active.description && (
            <div
              className="muted"
              style={{
                padding: "0.5rem 0.7rem",
                borderBottom: "1px solid var(--border)",
                fontSize: "0.83rem",
              }}
            >
              {active.description}
            </div>
          )}
          <pre>
            <code>{active.content}</code>
          </pre>
        </div>
      )}
    </div>
  );
}
