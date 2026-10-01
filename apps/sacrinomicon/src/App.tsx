import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { api } from "./api";
import ConceptMap from "./ConceptMap";
import EvidenceDrawer, { timestamp } from "./EvidenceDrawer";
import { VoiceCapture } from "./voice";
import type { Concept, Evidence, Library, Message, ModelStatus, SearchHit, SourceDetail } from "./types";
import "./style.css";
const empty: Library = { sources: [], evidenceCount: 0, claims: [], concepts: [], relationships: [], assessments: [], annotationConflicts: [], sourceConflicts: [] };
function loadMessages(): Message[] { try { return JSON.parse(localStorage.getItem("sacri-oracle-v1") ?? "[]"); } catch { return []; } }
export default function App() {
  const [screen, setScreen] = useState<"oracle" | "explore" | "library">("oracle");
  const [library, setLibrary] = useState<Library>(empty);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [messages, setMessages] = useState<Message[]>(loadMessages);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [oracleStatus, setOracleStatus] = useState("Responding");
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
  const [readAloud, setReadAloud] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("Listening");
  const [voiceLevel, setVoiceLevel] = useState(0);
  const capture = useRef<VoiceCapture | null>(null);
  const listening = useRef(false);
  const voiceEnabled = useRef(false);
  const speaking = useRef(false);
  const working = useRef(false);
  const askRef = useRef<(question: string) => Promise<void>>(async () => {});
  const scroll = useRef<HTMLDivElement>(null);
  const screenRef = useRef(screen); screenRef.current = screen;
  const draftId = useRef<string | null>(null);
  const ready = models.filter((m) => m.id !== "voice").length === 2 && models.filter((m) => m.id !== "voice").every((m) => m.downloaded);
  const voiceReady = models.find((m) => m.id === "voice")?.downloaded ?? false;
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
    const subscriptions = [listen<string>("oracle-status", ({ payload }) => { setOracleStatus(payload); if (voiceEnabled.current) setVoiceStatus(payload); }), listen<string>("oracle-token", ({ payload }) => setMessages((list) => list.map((m) => m.id === draftId.current ? { ...m, text: m.text + payload } : m))), listen<{ id: string; received: number; total: number; done: boolean; error?: string }>("model-progress", ({ payload }) => {
      setDownloadProgress((p) => ({ ...p, [payload.id]: payload.received / payload.total }));
      if (payload.done) refresh(); if (payload.error) setError(payload.error);
    })];
    return () => { for (const p of subscriptions) void p.then((unlisten) => unlisten()); void capture.current?.close(); void api.voiceMode(false); void api.stopSpeech(); };
  }, [refresh]);
  useEffect(() => { try { localStorage.setItem("sacri-oracle-v1", JSON.stringify(messages)); } catch { /* chat is optional; evidence remains in SQLite */ } scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" }); }, [messages]);
  useEffect(() => {
    if (!query.trim()) return;
    let active = true;
    const timeout = setTimeout(() => { void api.search(query).then((results) => { if (active) setHits(results); }).catch((e) => { if (active) setError(String(e)); }); }, 220);
    return () => { active = false; clearTimeout(timeout); };
  }, [query]);
  async function speak(text: string) {
    speaking.current = true; listening.current = false; if (voiceEnabled.current) setVoiceStatus("Speaking");
    try { await api.speak(text.replace(/\[E\d+\]/g, "")); }
    catch (e) { setError(String(e)); }
    finally { speaking.current = false; if (voiceEnabled.current) { listening.current = true; setVoiceStatus("Listening"); } }
  }
  async function leaveVoice() {
    voiceEnabled.current = false; listening.current = false; setVoiceMode(false); setVoiceLevel(0);
    const current = capture.current; capture.current = null; await current?.close();
    await api.stopSpeech(); await api.voiceMode(false);
  }
  async function enterVoice() {
    if (!voiceReady || !ready || busy) return;
    setError(null); setVoiceStatus("Opening microphone"); setVoiceMode(true); voiceEnabled.current = true;
    try {
      capture.current = await VoiceCapture.start(() => listening.current && !working.current && !speaking.current, (pcm) => {
        listening.current = false; working.current = true; setVoiceStatus("Transcribing");
        void api.transcribe(pcm).then(async (text) => {
          working.current = false;
          if (!voiceEnabled.current) return;
          if (text.trim() && !/^\[.*\]$/.test(text.trim())) await askRef.current(text);
          else { listening.current = true; setVoiceStatus("Listening"); }
        }).catch((e) => { working.current = false; if (voiceEnabled.current) { setError(String(e)); listening.current = true; setVoiceStatus("Listening"); } });
      }, setVoiceLevel);
      await api.voiceMode(true);
      if (!voiceEnabled.current) { await capture.current?.close(); capture.current = null; await api.voiceMode(false); return; }
      listening.current = true; setVoiceStatus("Listening");
    } catch (e) { setError(String(e)); await leaveVoice(); }
  }
  async function ask(question = input) {
    if (!question.trim() || working.current) return;
    working.current = true; listening.current = false;
    setInput(""); setBusy(true); setOracleStatus("Responding"); setError(null);
    if (voiceEnabled.current) setVoiceStatus("Responding");
    const id = crypto.randomUUID(); draftId.current = id;
    const history: [string, string][] = [];
    for (let i = 0; i < messages.length - 1; i++) if (messages[i].role === "user" && messages[i + 1].role === "assistant" && !messages[i + 1].error) history.push([messages[i].text, messages[i + 1].text]);
    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: question }, { id, role: "assistant", text: "", streaming: true }]);
    try {
      await api.stopSpeech();
      const result = await api.ask(question, sourceId, history.slice(-2));
      setMessages((m) => m.map((message) => message.id === id ? { ...message, text: result.answer, sources: result.sources, streaming: false } : message));
      if (readAloud || voiceEnabled.current) await speak(result.answer);
    } catch (e) { setMessages((m) => m.map((message) => message.id === id ? { ...message, text: String(e), error: true, streaming: false } : message)); }
    finally { setBusy(false); working.current = false; draftId.current = null; if (voiceEnabled.current) { listening.current = true; setVoiceStatus("Listening"); } if (screenRef.current !== "oracle") void api.unload().catch(console.error); }
  }
  askRef.current = ask;
  async function openSource(id: string) { try { setSource(await api.source(id)); setSourceId(id); setScreen("library"); } catch (e) { setError(String(e)); } }
  async function selectConcept(concept: Concept) { setSelectedConcept(concept); try { setConceptEvidence(await Promise.all(concept.evidenceIds.map((id) => api.evidence(id)))); } catch (e) { setError(String(e)); } }
  const navigate = (next: typeof screen) => { setScreen(next); setError(null); if (voiceEnabled.current) void leaveVoice().catch((e) => setError(String(e))); void api.stopSpeech(); if (next !== "oracle" && !busy) void api.unload().catch(console.error); };
  const standing = (id: string) => library.assessments.find((a) => a.entityId === id)?.status ?? "unreviewed";
  return <div className="archive-shell">
    <aside className="archive-nav"><button className="brand" onClick={() => navigate("oracle")}><span className="brand-glyph">⟡</span><span>SACRiNOMiCON<small>A private constellation</small></span></button><nav>{(["oracle", "explore", "library"] as const).map((name) => <button key={name} className={screen === name ? "active" : ""} aria-current={screen === name ? "page" : undefined} onClick={() => navigate(name)}><span>{name === "oracle" ? "◉" : name === "explore" ? "⌘" : "▤"}</span>{name === "library" ? "Library / Inbox" : name[0].toUpperCase() + name.slice(1)}</button>)}</nav><div className="nav-foot"><i />Local archive<small>{library.sources.length} sources · {library.evidenceCount} passages</small></div></aside>
    <main className={`archive-main ${screen}`}>
      <header className="topbar"><span className="eyebrow">{screen === "oracle" ? "Consult the archive" : screen === "explore" ? "Follow a thread" : "Preserve and inspect"}</span><button className="subtle" onClick={refresh}>Refresh</button></header>
      {error && <div className="error notice" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div>}
      {screen === "oracle" && <>
        <div className="oracle-scroll" ref={scroll}>
          {voiceMode && <div className="voice-presence"><div className="oracle-sigil" style={{ transform: `scale(${1 + voiceLevel * .25})` }} aria-hidden="true"><i /><i /><span>⟡</span></div><p role="status">{voiceStatus}</p><button onClick={() => { listening.current = false; void api.stop().then(() => api.stopSpeech()).finally(() => { listening.current = true; setVoiceStatus("Listening"); }); }}>Interrupt</button><button onClick={() => void leaveVoice().catch((e) => setError(String(e)))}>End voice</button></div>}
          {!messages.length && !voiceMode && <div className="oracle-welcome"><div className="oracle-sigil" aria-hidden="true"><i /><i /><span>⟡</span></div><h1>Ask. Listen. Trace.</h1><p>Converse with what you have preserved.<br />Every thread can lead back to its source.</p></div>}
          {messages.map((message) => <article key={message.id} className={`oracle-message ${message.role} ${message.error ? "error" : ""}`}><span className="eyebrow">{message.role === "user" ? "You" : "Oracle"}</span><p>{message.text.split(/(\[E\d+\])/g).map((part, i) => {
            const match = /^\[E(\d+)\]$/.exec(part); const hit = match && message.sources?.[Number(match[1]) - 1];
            return hit ? <button key={i} className="citation" title={`${hit.sourceTitle} · ${timestamp(hit.evidence.start)}`} onClick={() => setEvidence(hit.evidence)}>{part}</button> : part;
          })}{message.streaming && <span className="oracle-caret">·</span>}</p>
            {message.sources && <details className="answer-sources"><summary>{message.sources.length} preserved passages consulted</summary>{message.sources.map((hit, i) => <button key={hit.evidence.id} onClick={() => setEvidence(hit.evidence)}><span>E{i + 1} · {timestamp(hit.evidence.start)} · {hit.evidence.kind}</span>{hit.sourceTitle}</button>)}</details>}
            {message.role === "assistant" && !message.streaming && !message.error && <button className="subtle" onClick={() => void speak(message.text)}>Listen</button>}
          </article>)}
        </div>
        <div className="oracle-compose">
          <div className="oracle-options"><select aria-label="Source scope" value={sourceId ?? ""} onChange={(e) => setSourceId(e.target.value || null)}><option value="">All preserved sources</option>{library.sources.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}</select><label className="toggle"><input type="checkbox" checked={readAloud} onChange={(e) => { setReadAloud(e.target.checked); if (!e.target.checked) void api.stopSpeech(); }} />Read aloud</label><button className="subtle" disabled={busy || !ready} onClick={() => voiceMode ? void leaveVoice() : void enterVoice()}>{voiceMode ? "End voice" : "Voice"}</button><button className="subtle" disabled={busy} onClick={() => setMessages([])}>New conversation</button></div>
          {(!ready || !voiceReady) && <div className="model-setup"><p>{!ready ? "The Oracle needs two local text models. Your library is available while they download." : "Voice conversation uses an optional offline transcription model."}</p>{models.filter((m) => !m.downloaded).map((m) => <button key={m.id} disabled={downloadProgress[m.id] !== undefined && downloadProgress[m.id] < 1} onClick={() => { setDownloadProgress((p) => ({ ...p, [m.id]: 0 })); void api.download(m.id).then(refresh).catch((e) => { setError(String(e)); setDownloadProgress((p) => { const next = { ...p }; delete next[m.id]; return next; }); }); }}>↓ {m.label} {downloadProgress[m.id] !== undefined ? `${Math.round(downloadProgress[m.id] * 100)}%` : `${(m.bytes / 1e9).toFixed(2)} GB`}</button>)}</div>}
          <form onSubmit={(e) => { e.preventDefault(); void ask(); }}><textarea aria-label="Ask the Oracle" value={input} maxLength={2000} onChange={(e) => setInput(e.target.value)} rows={2} placeholder="What thread shall we follow?" disabled={busy || !ready} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(); } }} />{busy ? <button type="button" aria-label="Stop response" onClick={() => { void api.stop(); void api.stopSpeech(); }}>■</button> : <button className="primary" aria-label="Send question" disabled={!ready || !input.trim()}>↑</button>}</form><small role={busy ? "status" : undefined}>{busy ? oracleStatus : "Preservation carries provenance. Claims carry uncertainty."}</small>
        </div>
      </>}
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
