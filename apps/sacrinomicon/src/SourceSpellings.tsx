import { useState } from "react";
import { api } from "./api";
import type { Assessment, SpellingCorrection } from "./types";

export default function SourceSpellings({ sourceId, assessment, onChanged }: { sourceId: string; assessment?: Assessment; onChanged: () => void }) {
  const [items, setItems] = useState<SpellingCorrection[]>(assessment?.sourceSpellings ?? []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  async function save(values: SpellingCorrection[]) {
    setBusy(true); setError(null); setSaved(false);
    try { await api.correctSourceSpellings(sourceId, values); setItems(values); setSaved(true); onChanged(); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  }
  return <details className="source-spellings">
    <summary>Spelling corrections for this source</summary>
    <p className="muted">Apply a spelling to this source’s Oracle readings. Matches use whole terms and exact letter case. Originals remain preserved. Use a specific phrase when the same word has several meanings.</p>
    <form onSubmit={(e) => { e.preventDefault(); void save(items); }}>
      {items.map((item, index) => <div className="spelling-row" key={index}>
        <label>Extracted term<input value={item.original} maxLength={100} required disabled={busy} onChange={(e) => { setSaved(false); setItems((list) => list.map((v, i) => i === index ? { ...v, original: e.target.value } : v)); }} /></label>
        <label>Correct spelling<input value={item.corrected} maxLength={100} required disabled={busy} onChange={(e) => { setSaved(false); setItems((list) => list.map((v, i) => i === index ? { ...v, corrected: e.target.value } : v)); }} /></label>
        <button type="button" disabled={busy} aria-label={`Remove correction ${index + 1}`} onClick={() => { setSaved(false); setItems((list) => list.filter((_, i) => i !== index)); }}>×</button>
      </div>)}
      <button type="button" disabled={busy || items.length >= 32} onClick={() => { setSaved(false); setItems((list) => [...list, { original: "", corrected: "" }]); }}>Add spelling correction</button>
      <button className="primary" disabled={busy}>{busy ? "Saving…" : "Save source spellings"}</button>
      {saved && <p role="status">Source spellings saved.</p>}
      {error && <p className="error" role="alert">{error}</p>}
    </form>
  </details>;
}
