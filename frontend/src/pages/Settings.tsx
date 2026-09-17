// frontend/src/pages/Settings.tsx
//
// Section 20: the LLM settings page.
//
// The credential rules here are not UI preferences, they are the security model, and this page
// is written to make the safe thing the default:
//
//   * The key input starts empty and is never populated from the server. The read model has no
//     `api_key` field at all, so there is nothing to populate it *with* - the page can show that
//     a key exists and its last four characters, and that is the most it can ever know.
//   * Saving with the field blank preserves the stored key. That is why the placeholder says so
//     explicitly: a user who edits the model name and saves must not silently wipe their key.
//   * Clearing a key is a separate, confirmed action hitting DELETE /api/settings/llm/key, so it
//     cannot happen as a side effect of saving something else.
//
// "Test connection" sends the typed key when there is one, so a key can be verified before it is
// stored, and falls back to the saved key otherwise. That ordering matters: verifying first is
// how a user avoids persisting a typo.
//
// When the server cannot encrypt at rest, saving a key is refused rather than downgraded to
// plaintext, and this page says so up front instead of letting the request fail.

import { useCallback, useEffect, useState } from "react";
import {
  deleteApiKey,
  getProviders,
  getSettings,
  saveSettings,
  testConnection,
  toApiError,
} from "../api";
import { ErrorBanner } from "../components/ErrorBanner";
import type { ApiError, ConnectionCheck, Health, LLMSettings, ProviderInfo } from "../types";

interface Form {
  provider: string;
  model: string;
  base_url: string;
  temperature: string;
  max_tokens: string;
  timeout: string;
}

const toForm = (s: LLMSettings): Form => ({
  provider: s.provider,
  model: s.model ?? "",
  base_url: s.base_url ?? "",
  temperature: s.temperature === null ? "" : String(s.temperature),
  max_tokens: s.max_tokens === null ? "" : String(s.max_tokens),
  timeout: s.timeout === null ? "" : String(s.timeout),
});

const num = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
};

export function SettingsPage({ health }: { health: Health | null }) {
  const [settings, setSettings] = useState<LLMSettings | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [form, setForm] = useState<Form | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);

  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<"save" | "test" | "clear" | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [saved, setSaved] = useState(false);
  const [check, setCheck] = useState<ConnectionCheck | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    getSettings()
      .then((s) => {
        setSettings(s);
        setForm(toForm(s));
        setError(null);
      })
      .catch((err) => setError(toApiError(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
    getProviders().then(setProviders, () => {});
  }, [load]);

  const set = <K extends keyof Form>(key: K, value: Form[K]) => {
    setForm((f) => (f ? { ...f, [key]: value } : f));
    setSaved(false);
  };

  const chosen = providers.find((p) => p.name === form?.provider);
  const canStore = settings?.can_store_credentials ?? false;

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form || busy) return;
    setBusy("save");
    setError(null);
    setCheck(null);
    try {
      const updated = await saveSettings({
        provider: form.provider,
        model: form.model.trim() || null,
        base_url: form.base_url.trim() || null,
        temperature: num(form.temperature),
        max_tokens: num(form.max_tokens),
        timeout: num(form.timeout),
        // Not editable here, but a PUT replaces the whole row - so they are echoed back rather
        // than dropped. Saving a temperature change must not quietly reset top_p.
        top_p: settings?.top_p ?? null,
        extra: settings?.extra ?? {},
        // Omitted entirely when blank. Sending null or "" would read as "clear it".
        ...(apiKey ? { api_key: apiKey } : {}),
      });
      setSettings(updated);
      setForm(toForm(updated));
      setApiKey("");
      setShowKey(false);
      setSaved(true);
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(null);
    }
  };

  const test = async () => {
    if (!form || busy) return;
    setBusy("test");
    setError(null);
    setCheck(null);
    try {
      setCheck(
        await testConnection({
          provider: form.provider,
          model: form.model.trim() || null,
          base_url: form.base_url.trim() || null,
          // Prefer the typed key so it can be validated before it is ever persisted.
          ...(apiKey ? { api_key: apiKey, use_saved: false } : { use_saved: true }),
        }),
      );
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(null);
    }
  };

  const clearKey = async () => {
    if (busy) return;
    const ok = window.confirm(
      "Remove the saved API key?\n\nGeneration will fail until another key is configured, either here or through an environment variable.",
    );
    if (!ok) return;
    setBusy("clear");
    setError(null);
    setCheck(null);
    try {
      await deleteApiKey();
      setApiKey("");
      load();
    } catch (err) {
      setError(toApiError(err));
    } finally {
      setBusy(null);
    }
  };

  if (loading && !form) {
    return (
      <div className="empty">
        <span className="spinner" /> Loading settings…
      </div>
    );
  }

  if (!form || !settings) {
    return (
      <>
        <ErrorBanner error={error} onRetry={load} />
        <div className="empty">Settings could not be loaded.</div>
      </>
    );
  }

  return (
    <>
      <div className="page-head">
        <h1>Model settings</h1>
        <p>
          This is the model that designs your agents — it analyses the requirement, picks the
          framework, writes the code and reviews it. Generated projects get their own runtime
          provider, chosen per project.
        </p>
      </div>

      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {!canStore && (
        <ErrorBanner
          variant="warn"
          error={{
            code: "encryption_unavailable",
            message: "Keys cannot be saved on this server, because they cannot be encrypted at rest.",
            action:
              "Install the cryptography package (pip install 'multi-agent-generator[secure]') and restart the server. Until then, set the provider's environment variable in .env instead.",
          }}
        />
      )}

      {saved && (
        <div className="alert ok">
          <div className="alert-msg">Saved.</div>
        </div>
      )}

      {check && (
        <div className={`alert ${check.ok ? "ok" : ""}`}>
          <div className="alert-msg">
            {check.ok ? "Connection works." : "Connection failed."} {check.message}
          </div>
          {check.latency_s != null && (
            <div className="hint">
              Responded in {check.latency_s.toFixed(2)}s
              {check.model ? ` using ${check.model}` : ""}.
            </div>
          )}
          {check.code && <div className="hint mono">{check.code}</div>}
        </div>
      )}

      <form onSubmit={save} className="grid-side">
        <div>
          <div className="card">
            <div className="card-title">
              <h3>Provider</h3>
              <div className="actions">
                {settings.credential_configured ? (
                  <span className="badge ok">
                    Key configured
                    {/* The hint already carries its own mask prefix from Secret.hint(). */}
                    {settings.credential_hint ? ` ${settings.credential_hint}` : ""}
                  </span>
                ) : (
                  <span className="badge warn">No key</span>
                )}
                {settings.credential_from_env && (
                  <span className="badge info" title="Resolved from the environment, not the database">
                    from .env
                  </span>
                )}
              </div>
            </div>

            <div className="field">
              <label htmlFor="provider">Provider</label>
              <select
                id="provider"
                value={form.provider}
                onChange={(e) => set("provider", e.target.value)}
                disabled={busy !== null}
              >
                {providers.length === 0 && <option value={form.provider}>{form.provider}</option>}
                {providers.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.label}
                    {p.installed ? "" : " (not installed)"}
                  </option>
                ))}
              </select>
              {chosen && <div className="hint">{chosen.description}</div>}
              {chosen && !chosen.installed && chosen.install_command && (
                <div className="alert warn mt">
                  <div className="alert-msg">This provider's package is not installed.</div>
                  <div className="alert-action mono">{chosen.install_command}</div>
                </div>
              )}
              {chosen?.notes && <div className="hint">{chosen.notes}</div>}
            </div>

            <div className="field">
              <label htmlFor="key">API key</label>
              <div className="field-row">
                <input
                  id="key"
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value);
                    setSaved(false);
                  }}
                  placeholder={
                    settings.credential_configured
                      ? "Leave blank to keep the saved key"
                      : chosen?.local
                        ? "Not required for a local provider"
                        : "Paste a key"
                  }
                  autoComplete="off"
                  spellCheck={false}
                  disabled={busy !== null || !canStore}
                />
                <button
                  type="button"
                  className="ghost sm"
                  onClick={() => setShowKey((v) => !v)}
                  disabled={!apiKey}
                >
                  {showKey ? "Hide" : "Show"}
                </button>
              </div>
              <div className="hint">
                Stored encrypted and never sent back to the browser — this page can only ever see
                the last four characters.
                {chosen && chosen.credential_env.length > 0 && (
                  <>
                    {" "}
                    You can also set{" "}
                    <span className="mono">{chosen.credential_env.join(" or ")}</span> in{" "}
                    <span className="mono">.env</span>.
                  </>
                )}
              </div>
              {settings.credential_configured && !settings.credential_from_env && (
                <div className="actions mt">
                  <button
                    type="button"
                    className="danger sm"
                    onClick={clearKey}
                    disabled={busy !== null}
                  >
                    {busy === "clear" ? "Removing…" : "Remove saved key"}
                  </button>
                </div>
              )}
              {settings.credential_from_env && (
                <div className="hint">
                  The active key comes from the environment, so it cannot be removed here — unset
                  the variable and restart the server.
                </div>
              )}
            </div>

            <div className="field">
              <label htmlFor="model">Model</label>
              <input
                id="model"
                type="text"
                value={form.model}
                onChange={(e) => set("model", e.target.value)}
                placeholder={chosen?.default_model ?? "Provider default"}
                disabled={busy !== null}
              />
              <div className="hint">
                Plain model id, no route prefix.
                {chosen?.default_model && (
                  <>
                    {" "}
                    Default: <span className="mono">{chosen.default_model}</span>
                  </>
                )}
              </div>
            </div>

            <div className="actions">
              <button className="primary" type="submit" disabled={busy !== null}>
                {busy === "save" ? "Saving…" : "Save settings"}
              </button>
              <button type="button" className="ghost" onClick={test} disabled={busy !== null}>
                {busy === "test" ? "Testing…" : "Test connection"}
              </button>
              {busy && <span className="spinner" />}
            </div>
          </div>
        </div>

        <div>
          <div className="card">
            <div className="card-title">
              <h3>Generation behaviour</h3>
            </div>

            <div className="field">
              <label htmlFor="temp">Temperature</label>
              <input
                id="temp"
                type="number"
                step="0.05"
                min={0}
                max={2}
                value={form.temperature}
                onChange={(e) => set("temperature", e.target.value)}
                placeholder="Provider default"
                disabled={busy !== null}
              />
              <div className="hint">Low values keep generated code predictable.</div>
            </div>

            <div className="field">
              <label htmlFor="max-tokens">Max output tokens</label>
              <input
                id="max-tokens"
                type="number"
                min={256}
                value={form.max_tokens}
                onChange={(e) => set("max_tokens", e.target.value)}
                placeholder="Provider default"
                disabled={busy !== null}
              />
              <div className="hint">
                A multi-file project needs room. Too low and generation is truncated mid-file.
              </div>
            </div>

            <div className="field">
              <label htmlFor="timeout">Request timeout (seconds)</label>
              <input
                id="timeout"
                type="number"
                min={5}
                value={form.timeout}
                onChange={(e) => set("timeout", e.target.value)}
                placeholder="Provider default"
                disabled={busy !== null}
              />
              <div className="hint">Every model call is bounded — nothing waits forever.</div>
            </div>

            <div className="field">
              <label htmlFor="base">Base URL</label>
              <input
                id="base"
                type="text"
                value={form.base_url}
                onChange={(e) => set("base_url", e.target.value)}
                placeholder={chosen?.local ? "http://localhost:11434" : "Provider default"}
                disabled={busy !== null}
              />
              <div className="hint">
                For a self-hosted or proxied endpoint. Leave blank for the provider's own API.
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              <h3>Server</h3>
            </div>
            <dl className="kv">
              <dt>settings source</dt>
              <dd className="mono">{settings.source}</dd>
              <dt>encryption</dt>
              <dd>{health?.encryption ? "available" : "unavailable"}</dd>
              <dt>persistence</dt>
              <dd>{health?.persistence ? "sqlite" : "none"}</dd>
              <dt>default provider</dt>
              <dd className="mono">{health?.default_provider ?? "—"}</dd>
              <dt>version</dt>
              <dd className="mono">{health?.version ?? "—"}</dd>
            </dl>
          </div>
        </div>
      </form>
    </>
  );
}
