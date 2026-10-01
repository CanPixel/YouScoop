import { useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { api } from "./api";
import SourceSpellings from "./SourceSpellings";
import type { Evidence, Library, SourceDetail } from "./types";

export const timestamp = (seconds: number) => {
  const s = Math.floor(seconds); return s >= 3600 ? `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};
export default function EvidenceDrawer({ evidence, library, onClose, onChanged }: { evidence: Evidence; library: Library; onClose: () => void; onChanged: () => void }) {
  const drawer = useRef<HTMLElement>(null);
  const closeRef = useRef(onClose);
  useEffect(() => { closeRef.current = onClose; }, [onClose]);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const panel = drawer.current;
    const controls = () => Array.from(panel?.querySelectorAll<HTMLElement>("button, input, select, textarea, summary, [tabindex='0']") ?? []).filter((el) => el.getClientRects().length && !(el as HTMLButtonElement).disabled);
    controls()[0]?.focus();
    const keys = (event: KeyboardEvent) => {
      if (event.key === "Escape") { event.preventDefault(); closeRef.current(); }
      if (event.key === "Tab") {
        const items = controls(); const first = items[0], last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    panel?.addEventListener("keydown", keys);
    return () => { panel?.removeEventListener("keydown", keys); previous?.focus(); };
  }, []);
  const assessment = library.assessments.find((a) => a.entityId === evidence.id);
  const [status, setStatus] = useState(assessment?.status ?? "unreviewed");
  const [note, setNote] = useState(assessment?.note ?? "");
  const [reading, setReading] = useState(assessment?.correctedText ?? evidence.text);
  const [oracleReading, setOracleReading] = useState<string | null>(null);
  const [concept, setConcept] = useState("");
  const [statement, setStatement] = useState("");
  const [kind, setKind] = useState("interpretation");
  const [detail, setDetail] = useState<SourceDetail | null>(null);
  const [media, setMedia] = useState<{ src: string; kind: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [relation, setRelation] = useState("related");
  const [explanation, setExplanation] = useState("");
  useEffect(() => {
    let active = true;
    void api.reading(evidence.id).then((text) => { if (active) setOracleReading(text); }).catch((e) => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [evidence.id, library.assessments]);
  useEffect(() => {
    let active = true;
    void api.source(evidence.sourceId).then(async (source) => {
      if (!active) return;
      setDetail(source);
      const original = source.assets.find((a) => a.id === evidence.provenance.originalAsset && a.kind === "keyframe");
      if (original) {
        const path = await api.assetPath(evidence.sourceId, original.id);
        if (active) setMedia({ src: convertFileSrc(path), kind: original.kind });
      }
    }).catch((e) => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [evidence]);
  async function change(task: Promise<unknown>) { setError(null); try { await task; onChanged(); return true; } catch (e) { setError(String(e)); return false; } }
  const concepts = library.concepts.filter((c) => c.evidenceIds.includes(evidence.id));
  return <aside ref={drawer} className="evidence-drawer" role="dialog" aria-modal="true" aria-label="Preserved evidence">
    <header><span className="eyebrow">Preserved passage</span><button aria-label="Close evidence" onClick={onClose}>×</button></header>
    <h2>{detail?.source.title ?? library.sources.find((s) => s.id === evidence.sourceId)?.title}</h2>
    <div className="passage-meta">{timestamp(evidence.start)} to {timestamp(evidence.end)} · {evidence.kind}</div>
    {media?.kind === "keyframe" && <img src={media.src} alt={`Preserved frame at ${timestamp(evidence.start)}`} />}
    <p className="eyebrow">Original extraction</p><blockquote>{evidence.text}</blockquote>
    {oracleReading !== null && oracleReading !== evidence.text && <><p className="eyebrow">Oracle reading · your corrections</p><blockquote>{oracleReading}</blockquote></>}
    <p className="provenance">{evidence.provenance.method.replaceAll("_", " ")}<br />{evidence.provenance.model ?? "Source captions"}<br />Recorded {new Date(evidence.provenance.generatedAt * 1000).toLocaleDateString()}</p>
    {detail?.source.uri && <button className="subtle" onClick={() => void openUrl(detail.source.uri).catch((e) => setError(String(e)))}>Open source ↗</button>}
    <form onSubmit={(e) => { e.preventDefault(); void change(api.assess(evidence.id, status, note)); }}>
      <h3>Your assessment</h3>
      <label>Standing<select value={status} onChange={(e) => setStatus(e.target.value)}>{["unreviewed", "supported", "contested", "uncertain", "belief", "rejected"].map((value) => <option key={value}>{value}</option>)}</select></label>
      <label>Notes and corrections<textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3} placeholder="What is established, uncertain, or corrected?" /></label>
      <button className="primary">Save assessment</button>
    </form>
    <details>
      <summary>Correct the extracted reading</summary>
      <p className="muted">The Oracle uses this reading for this passage. The original above and its provenance remain preserved.</p>
      <form onSubmit={(e) => { e.preventDefault(); void change(api.correctReading(evidence.id, reading)); }}>
        <label>Your corrected reading<textarea value={reading} onChange={(e) => setReading(e.target.value)} rows={6} required /></label>
        <button className="primary">Save corrected reading</button>
        {assessment?.correctedText && <button type="button" onClick={() => { void change(api.correctReading(evidence.id, null)).then((saved) => { if (saved) setReading(evidence.text); }); }}>Use original reading</button>}
      </form>
    </details>
    <SourceSpellings sourceId={evidence.sourceId} assessment={library.assessments.find((a) => a.entityId === evidence.sourceId)} onChanged={onChanged} />
    <form onSubmit={(e) => { e.preventDefault(); void change(api.addConcept(concept, evidence.id)).then((saved) => { if (saved) setConcept(""); }); }}>
      <h3>Connect a concept</h3><div className="tags">{concepts.map((c) => <span key={c.id}>{c.label}</span>)}</div>
      <label>Name<input value={concept} maxLength={200} onChange={(e) => setConcept(e.target.value)} placeholder="Egregores, symbolism, collective attention…" required /></label><button>Add concept</button>
    </form>
    <form onSubmit={(e) => { e.preventDefault(); void change(api.addClaim(evidence.id, statement, kind)).then((saved) => { if (saved) setStatement(""); }); }}>
      <h3>Preserve a claim or interpretation</h3>
      <label>Type<select value={kind} onChange={(e) => setKind(e.target.value)}>{["speaker_assertion", "observation", "interpretation", "metaphysical_hypothesis"].map((v) => <option key={v} value={v}>{v.replaceAll("_", " ")}</option>)}</select></label>
      <label>Statement<textarea value={statement} onChange={(e) => setStatement(e.target.value)} rows={2} required /></label><button>Preserve with this passage</button>
    </form>
    {library.concepts.length > 1 && <details><summary>Link two concepts through this evidence</summary><form onSubmit={(e) => { e.preventDefault(); void change(api.addRelationship(from, to, relation, evidence.id, explanation)); }}>
      <label>From<select required value={from} onChange={(e) => setFrom(e.target.value)}><option value="">Select concept</option>{library.concepts.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label>
      <label>To<select required value={to} onChange={(e) => setTo(e.target.value)}><option value="">Select concept</option>{library.concepts.filter((c) => c.id !== from).map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}</select></label>
      <label>Relationship<select value={relation} onChange={(e) => setRelation(e.target.value)}>{["mentions", "supports", "contradicts", "resembles", "influenced", "related"].map((r) => <option key={r}>{r}</option>)}</select></label>
      <label>Reason<textarea required value={explanation} onChange={(e) => setExplanation(e.target.value)} rows={2} /></label><button>Save relationship</button>
    </form></details>}
    {error && <p className="error" role="alert">{error}</p>}
  </aside>;
}
