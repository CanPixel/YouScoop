import { useEffect, useState } from "react";
import { openUrl } from "@tauri-apps/plugin-opener";
import { api, events } from "../api";
import type { WorkerConfig, WorkerStatus } from "../types";

export default function DesktopRuntime({ status, onChanged }: { status: WorkerStatus | null; onChanged: () => void }) {
  const [draft, setConfig] = useState<WorkerConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [progress, setProgress] = useState("");
  const [fraction, setFraction] = useState<number | null>(null);
  useEffect(() => {
    const subscription = events.onRuntimeProgress((event) => {
      if (event.detail) setProgress(event.detail);
      setFraction(event.received !== undefined && event.total ? event.received / event.total : null);
    });
    return () => { void subscription.then((unsubscribe) => unsubscribe()); };
  }, []);
  async function install(task: () => Promise<unknown>) {
    setSaving(true); setError(null); setProgress("Preparing desktop setup"); setFraction(null);
    try { await task(); setConfig(null); onChanged(); }
    catch (e) { setError(String(e)); }
    finally { setSaving(false); }
  }
  const config = draft ?? status?.config;
  if (!config) return null;
  return <details className="desktop-runtime">
    <summary>Desktop model · Gemma {status?.gemmaReady ? "available" : "needs setup"}</summary>
    <p>Gemma runs separately and releases its memory when finished. Setup preserves a pinned Python environment and the verified model in Application Support. Inference works offline after installation.</p>
    <div className="runtime-install-actions">
      <button className="btn btn-ghost btn-sm" disabled={saving || !status?.uvAvailable} onClick={() => void install(api.setupDesktopRuntime)}>{status?.runtimeReady ? "Check / repair runtime" : "Install desktop runtime"}</button>
      {!status?.uvAvailable && <button className="btn btn-ghost btn-sm" onClick={() => void openUrl("https://docs.astral.sh/uv/getting-started/installation/").catch((e) => setError(String(e)))}>Install uv first ↗</button>}
      <button className="btn btn-ghost btn-sm" disabled={saving || !status?.runtimeReady} onClick={() => void install(() => api.installDesktopModel("gemma", status?.cachedGemma ?? null))}>{status?.gemmaReady ? "Verify Gemma files" : status?.cachedGemma ? "Install Gemma from local cache" : "Download Gemma · 15.4 GB"}</button>
      {saving && <button className="btn btn-ghost btn-sm" onClick={() => void api.stopWorker().catch((e) => setError(String(e)))}>Cancel setup</button>}
    </div>
    {progress && <p role="status" className="inference-status">{progress}</p>}
    {saving && fraction !== null && <progress max={1} value={fraction} aria-label="Model file verification progress" />}
    <details><summary>Use a custom runtime or snapshot</summary>
    <form onSubmit={(event) => {
      event.preventDefault(); void install(() => api.configureWorker(config));
    }}>
      <label>Python executable<input value={config.python} disabled={saving} onChange={(e) => setConfig({ ...config, python: e.target.value })} /></label>
      <label>Gemma A4B MLX snapshot<input value={config.gemmaModel} disabled={saving} onChange={(e) => setConfig({ ...config, gemmaModel: e.target.value })} /></label>
      <button className="btn btn-ghost btn-sm" disabled={saving}>{saving ? "Saving…" : "Save runtime paths"}</button>
    </form>
    </details>
    {error && <p role="alert" className="archive-save-error">{error}</p>}
  </details>;
}
