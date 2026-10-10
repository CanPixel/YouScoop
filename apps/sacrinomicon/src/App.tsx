import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api } from "./api";
import ConceptMap from "./ConceptMap";
import EvidenceDrawer, { timestamp } from "./EvidenceDrawer";
import Oracle from "./Oracle";
import { SigilMark } from "./Sigil";
import type { Concept, Evidence, Library, ModelStatus, SearchHit, SourceDetail } from "./types";
import "@fontsource/cormorant-garamond/400.css";
import "@fontsource/cormorant-garamond/500.css";
import "@fontsource/cormorant-garamond/600.css";
import "@fontsource/cormorant-garamond/400-italic.css";
import "@fontsource/cinzel/400.css";
import "@fontsource/cinzel/500.css";
import "./style.css";
const empty: Library = { sources: [], evidenceCount: 0, claims: [], concepts: [], relationships: [], assessments: [], annotationConflicts: [], sourceConflicts: [] };
const SCREENS = {
  oracle: { label: "Oracle", glyph: "☉", eyebrow: "Consult the archive" },
  explore: { label: "Constellation", glyph: "✶", eyebrow: "Follow a thread" },
  library: { label: "Library", glyph: "❖", eyebrow: "Preserve and inspect" },
} as const;
export default function App() {
  const [screen, setScreen] = useState<"oracle" | "explore" | "library">("oracle");
  const [library, setLibrary] = useState<Library>(empty);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [source, setSource] = useState<SourceDetail | null>(null);
  const [selectedConcept, setSelectedConcept] = useState<Concept | null>(null);
  const [conceptEvidence, setConceptEvidence] = useState<Evidence[]>([]);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [exporting, setExporting] = useState(false);
  const [transferNotice, setTransferNotice] = useState<string | null>(null);
  const [importProgress, setImportProgress] = useState<number | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({});
  const refresh = useCallback(() => {
    void api.library().then(setLibrary).catch((e) => setError(String(e)));
    void api.modelStatus().then(setModels).catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    const visible = () => { if (document.visibilityState === "visible") refresh(); };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", visible);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [refresh]);
  useEffect(() => {
    refresh();
    const progress = listen<{ id: string; received: number; total: number; done: boolean; error?: string }>("model-progress", ({ payload }) => {
      setDownloadProgress((p) => ({ ...p, [payload.id]: payload.received / payload.total }));
      if (payload.done) refresh(); if (payload.error) setError(payload.error);
    });
    return () => { void progress.then((unlisten) => unlisten()); };
  }, [refresh]);
  useEffect(() => {
    if (!query.trim()) return;
    let active = true;
    const timeout = setTimeout(() => { void api.search(query).then((results) => { if (active) setHits(results); }).catch((e) => { if (active) setError(String(e)); }); }, 220);
    return () => { active = false; clearTimeout(timeout); };
  }, [query]);
  const download = (id: string) => {
    setDownloadProgress((p) => ({ ...p, [id]: 0 }));
    void api.download(id).then(refresh).catch((e) => { setError(String(e)); setDownloadProgress((p) => { const next = { ...p }; delete next[id]; return next; }); });
  };
  async function openSource(id: string) { try { setSource(await api.source(id)); setSourceId(id); setScreen("library"); } catch (e) { setError(String(e)); } }
  async function selectConcept(concept: Concept) { setSelectedConcept(concept); try { setConceptEvidence(await Promise.all(concept.evidenceIds.map((id) => api.evidence(id)))); } catch (e) { setError(String(e)); } }
  const navigate = (next: typeof screen) => { setScreen(next); setError(null); };
  const standing = (id: string) => library.assessments.find((a) => a.entityId === id)?.status ?? "unreviewed";
  return <div className="archive-shell">
    <aside className="archive-nav">
      <button className="brand" onClick={() => navigate("oracle")}><SigilMark size={38} /><span>SACRiNOMiCON<small>A private constellation</small></span></button>
      <nav>{(Object.keys(SCREENS) as (keyof typeof SCREENS)[]).map((name) => <button key={name} className={screen === name ? "active" : ""} aria-current={screen === name ? "page" : undefined} onClick={() => navigate(name)}><span className="nav-glyph">{SCREENS[name].glyph}</span>{name === "library" ? "Library / Inbox" : SCREENS[name].label}</button>)}</nav>
      <div className="nav-foot"><i />Sealed on this device<small>{library.sources.length} source{library.sources.length === 1 ? "" : "s"} · {library.evidenceCount} passages</small></div>
    </aside>
    <main className={`archive-main ${screen}`}>
      <header className="topbar"><span className="eyebrow">{SCREENS[screen].eyebrow}</span><button className="ghost" onClick={refresh}>Refresh</button></header>
      {error && <div className="error notice" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div>}
      <div className="screen-host" hidden={screen !== "oracle"}><Oracle library={library} models={models} active={screen === "oracle"} sourceId={sourceId} setSourceId={setSourceId} input={input} setInput={setInput} onEvidence={setEvidence} onError={setError} downloadProgress={downloadProgress} onDownload={download} /></div>
      {screen === "explore" && <><div className="screen-heading"><h1>Constellation</h1><p>Concepts, shared passages, and the relationships you choose to name.</p></div><div className="explore-layout"><ConceptMap library={library} selected={selectedConcept?.id ?? null} onSelect={(c) => void selectConcept(c)} /><aside className="concept-focus"><h2>{selectedConcept?.label ?? "Choose a concept"}</h2><div className="concept-list">{library.concepts.map((c) => <button key={c.id} className={c.id === selectedConcept?.id ? "active" : ""} onClick={() => void selectConcept(c)}>{c.label}<span>{c.evidenceIds.length}</span></button>)}</div>{selectedConcept && <><p className="eyebrow">Anchored passages</p>{conceptEvidence.map((e) => <button className="passage-card" key={e.id} onClick={() => setEvidence(e)}><span>{timestamp(e.start)} · {e.kind}</span><p>{e.text.slice(0, 220)}</p></button>)}<button onClick={() => { setInput(`Explore ${selectedConcept.label} and distinguish its meanings, interpretations, and evidence.`); navigate("oracle"); }}>Ask the Oracle about this</button></>}</aside></div></>}
      {screen === "library" && <><div className="screen-heading"><h1>The library</h1><p>Original sources survive alongside your interpretations.</p><label className="import-button" role="button" tabIndex={0} aria-disabled={importProgress !== null || exporting} onKeyDown={(e) => { if (["Enter", " "].includes(e.key) && importProgress === null && !exporting) { e.preventDefault(); e.currentTarget.querySelector<HTMLInputElement>("input")?.click(); } }}>{importProgress === null ? "＋ Import source or library" : `Importing ${Math.round(importProgress * 100)}%`}<input type="file" accept=".sacri,.sacri-library,.zip,application/zip" disabled={importProgress !== null || exporting} onChange={(e) => { const file = e.target.files?.[0]; if (!file) return; setImportProgress(0); setError(null); void api.import(file, setImportProgress).then(async (report) => { refresh(); setTransferNotice(`Imported ${report.sourceIds.length} source${report.sourceIds.length === 1 ? "" : "s"}.` + (report.annotationConflicts + report.sourceConflicts.length ? ` ${report.annotationConflicts + report.sourceConflicts.length} conflicts need review.` : "")); if (report.sourceIds.length === 1) await openSource(report.sourceIds[0]); }).catch((error) => setError(String(error))).finally(() => { setImportProgress(null); e.target.value = ""; }); }} /></label><button disabled={exporting || importProgress !== null || !library.sources.length} onClick={() => {
          setExporting(true); setError(null); setTransferNotice(null);
          void api.exportLibrary().then(async (result) => { setTransferNotice(result.saved ? "Library export includes originals, revision history, notes, and concept links." : "Export destination was cancelled."); if (result.path) await revealItemInDir(result.path); }).catch((error) => setError(String(error))).finally(() => setExporting(false));
        }}>{exporting ? "Preserving the library…" : "Export library"}</button></div>
        {transferNotice && <p className="transfer-notice" role="status">{transferNotice}</p>}
        {!!library.annotationConflicts.length && <section className="merge-review"><h2>Annotations to reconcile</h2><p>Both versions are preserved. Choose which notes, readings and spellings the Oracle should use.</p>{library.annotationConflicts.map((conflict) => {
          const local = library.assessments.find((a) => a.entityId === conflict.incoming.entityId);
          const label = library.sources.find((s) => s.id === conflict.incoming.entityId)?.title ?? library.claims.find((c) => c.id === conflict.incoming.entityId)?.statement ?? library.concepts.find((c) => c.id === conflict.incoming.entityId)?.label ?? "Preserved passage";
          return <article key={conflict.id}><h3>{label}</h3><div className="note-versions"><div><strong>This device · {local?.status}</strong><p>{local?.note}</p>{local?.correctedText && <><h4>Corrected reading</h4><p>{local.correctedText}</p></>}{!!local?.sourceSpellings?.length && <><h4>Source spellings</h4>{local.sourceSpellings.map((s) => <p key={s.original}>{s.original} → {s.corrected}</p>)}</>}</div><div><strong>Imported · {conflict.incoming.status}</strong><p>{conflict.incoming.note}</p>{conflict.incoming.correctedText && <><h4>Corrected reading</h4><p>{conflict.incoming.correctedText}</p></>}{!!conflict.incoming.sourceSpellings?.length && <><h4>Source spellings</h4>{conflict.incoming.sourceSpellings.map((s) => <p key={s.original}>{s.original} → {s.corrected}</p>)}</>}</div></div><button onClick={() => { void api.resolveAnnotation(conflict.id, false).then(refresh).catch((error) => setError(String(error))); }}>Keep this device’s annotation</button><button onClick={() => { void api.resolveAnnotation(conflict.id, true).then(refresh).catch((error) => setError(String(error))); }}>Use imported annotation</button></article>;
        })}</section>}
        {!!library.sourceConflicts.length && <section className="merge-review"><h2>Source revisions to review</h2><p>The current revision on this device was kept. Imported alternatives remain preserved.</p>{[...new Set(library.sourceConflicts.map((c) => c.sourceId))].map((id) => <button key={id} onClick={() => void openSource(id)}>{library.sources.find((s) => s.id === id)?.title ?? "Review source"}</button>)}</section>}
        <div className="library-search"><input aria-label="Search preserved passages" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search the preserved record…" /></div>
        {query.trim() ? <div className="search-results">{hits.map((hit) => <button className="passage-card" key={hit.evidence.id} onClick={() => setEvidence(hit.evidence)}><span>{hit.sourceTitle} · {timestamp(hit.evidence.start)}</span><p>{hit.evidence.text.slice(0, 350)}</p></button>)}{!hits.length && <p className="muted">No matching preserved passages.</p>}</div> : <div className="library-layout"><div className="source-list">{library.sources.map((s) => <button key={s.id} className={source?.source.id === s.id ? "active" : ""} onClick={() => void openSource(s.id)}><span className="standing">{standing(s.id)}</span><h2>{s.title}</h2><p>{s.publisher} · {timestamp(s.durationSeconds)}</p></button>)}{!library.sources.length && <div className="empty-library"><span>▤</span><p>Save a video from YouScoop or import its portable .sacri package.</p></div>}</div>{source && <section className="source-record"><h2>{source.source.title}</h2><div className="source-actions"><button onClick={() => { navigate("oracle"); setSourceId(source.source.id); }}>Converse with this source</button><button onClick={() => { void api.assess(source.source.id, "collected", library.assessments.find((a) => a.entityId === source.source.id)?.note ?? "Added to the collection").then(refresh).catch((e) => setError(String(e))); }}>Move from Inbox to collection</button></div><p className="muted">{source.assets.length} original artifacts · {source.evidence.length} indexed passages</p><details><summary>Preserved revisions ({source.revisions.length})</summary>{source.revisions.map((r) => <div className="revision-record" key={r.revision}><span>{new Date(r.exportedAt * 1000).toLocaleString()} · {r.evidenceCount} passages{r.active ? " · Current" : ""}</span>{!r.active && <button onClick={() => { void api.activateRevision(source.source.id, r.revision).then(async () => { refresh(); await openSource(source.source.id); }).catch((error) => setError(String(error))); }}>Use this revision</button>}</div>)}</details><details><summary>Preserved assets and checksums</summary>{source.assets.map((a) => <div className="asset-record" key={a.id}><strong>{a.name}</strong><span>{(a.bytes / 1e6).toFixed(2)} MB · {a.kind}</span><code>{a.sha256}</code></div>)}</details><details><summary>{source.claims.length} claims and interpretations</summary>{source.claims.map((c) => <article className="claim-record" key={c.id}><span>{c.kind.replaceAll("_", " ")} · {standing(c.id)}</span><p>{c.statement}</p><select aria-label={`Assessment for ${c.statement}`} value={standing(c.id)} onChange={(e) => { void api.assess(c.id, e.target.value, library.assessments.find((a) => a.entityId === c.id)?.note ?? "").then(refresh).catch((error) => setError(String(error))); }}>{["unreviewed", "supported", "contested", "uncertain", "belief", "rejected"].map((s) => <option key={s}>{s}</option>)}</select><button onClick={() => { void api.evidence(c.evidenceIds[0]).then(setEvidence).catch((e) => setError(String(e))); }}>Trace evidence</button></article>)}</details>{source.evidence.map((e) => <button key={e.id} className="passage-card" onClick={() => setEvidence(e)}><span>{timestamp(e.start)} · {e.kind} · {standing(e.id)}</span><p>{e.text}</p></button>)}</section>}</div>}
      </>}
    </main>
    {evidence && <><button className="drawer-backdrop" aria-label="Close evidence panel" onClick={() => setEvidence(null)} /><EvidenceDrawer key={evidence.id} evidence={evidence} library={library} onClose={() => setEvidence(null)} onChanged={() => { refresh(); if (source) void api.source(source.source.id).then(setSource); }} /></>}
  </div>;
}
