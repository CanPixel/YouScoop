import { useEffect, useState } from "react";
import { api } from "./api";
import { timestamp } from "./EvidenceDrawer";
import type { Assessment, SpellingCorrection, SpellingSuggestion } from "./types";

export default function SourceSpellings({ sourceId, assessment, onChanged }: { sourceId: string; assessment?: Assessment; onChanged: () => void }) {
  const [items, setItems] = useState<SpellingCorrection[]>(assessment?.sourceSpellings ?? []);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [suggestions, setSuggestions] = useState<SpellingSuggestion[]>([]);
  useEffect(() => {
    let active = true;
    void api.spellingSuggestions(sourceId).then((list) => { if (active) setSuggestions(list); }).catch(() => { /* suggestions are optional */ });
    return () => { active = false; };
  }, [sourceId]);
  const open = suggestions.filter((s) => !items.some((i) => i.original === s.original));
  async function save(values: SpellingCorrection[]) {
    setBusy(true); setError(null); setSaved(false);
    try { await api.correctSourceSpellings(sourceId, values); setItems(values); setSaved(true); onChanged(); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  }
  return <details className="source-spellings">
    <summary>Spelling corrections for this source</summary>
    <p className="muted">Apply a spelling to this source’s Oracle readings. Matches use whole terms and exact letter case. Originals remain preserved. Use a specific phrase when the same word has several meanings.</p>
    {open.length > 0 && <section className="spelling-suggestions" aria-label="Suggested spellings">
      <h4>Suggested from on-screen text</h4>
      <p className="muted">These caption words sound like names the video shows on screen. Whisper re-transcribed a few of their moments to check. Nothing changes until you apply one.</p>
      {open.map((s) => <article key={`${s.original}-${s.suggested}`}>
        <p><strong>{s.original} → {s.suggested}</strong> · {s.evidence} · {s.captionUses} caption uses, shown on {s.screens} screens</p>
        {s.keptOriginalAt.length > 0 && <p className="muted">Whisper kept “{s.original}” at {s.keptOriginalAt.map(timestamp).join(", ")} even with the hint, so that moment may use the ordinary word. Matching is whole-word and case-sensitive.</p>}
        <details><summary>What Whisper heard</summary>{s.checks.map((c) => <p key={c.at} className="muted">{timestamp(c.at)} · without hint: “{c.withoutHint}” · with hint: “{c.withHint}”</p>)}</details>
        <button type="button" disabled={busy || items.length >= 32} onClick={() => void save([...items, { original: s.original, corrected: s.suggested }])}>Apply {s.original} → {s.suggested}</button>
      </article>)}
    </section>}
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
