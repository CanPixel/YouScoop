import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "./api";
import Answer from "./Answer";
import Sigil, { SigilMark, type SigilState } from "./Sigil";
import { timestamp } from "./EvidenceDrawer";
import { VoiceCapture } from "./voice";
import { VoiceStage } from "./oracle-voice";
import { SpeechQueue, spokenText } from "./speech-queue";
import type { Evidence, Lens, Library, Message, ModelStatus } from "./types";

function stored<T>(key: string, fallback: T): T {
  try { const value = localStorage.getItem(key); return value === null ? fallback : JSON.parse(value) as T; } catch { return fallback; }
}
function remember(key: string, value: unknown) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* preferences are optional; evidence remains in SQLite */ }
}
// Strip the former direct-answer template labels while a reply streams.
const streamingText = (text: string) => text.replace(/^\s*Answer:\s*/, "").replace(/\n\s*Scope:\s*/, "\n\n");
const sentences = (text: string) => text.split(/(?<=[.!?…])\s+(?=\S)/).filter((s) => s.trim());
const STATUS: Record<string, string> = {
  "Waiting for local inference": "Gathering stillness", "Waking the Oracle": "Waking the Oracle", "Following the thread": "Following the thread",
  "Reading the archive": "Reading the archive", "Speaking": "Speaking", "Transcribing": "Hearing you", "Responding": "Contemplating",
};
const LENS_NOTE: Record<Lens, string> = {
  archive: "Answers only from preserved passages.",
  esoteric: "Passages first; lore, symbol and correspondence marked apart.",
};
type Phase = "opening" | "listening" | "hearing" | "thinking" | "speaking";
type Spoken = { text: string; audio?: AudioBuffer };

export interface OracleProps {
  library: Library; models: ModelStatus[]; active: boolean;
  sourceId: string | null; setSourceId: (id: string | null) => void;
  input: string; setInput: (text: string) => void;
  onEvidence: (evidence: Evidence) => void; onError: (message: string | null) => void;
  downloadProgress: Record<string, number>; onDownload: (id: string) => void;
}

export default function Oracle({ library, models, active, sourceId, setSourceId, input, setInput, onEvidence, onError, downloadProgress, onDownload }: OracleProps) {
  const [messages, setMessages] = useState<Message[]>(() => stored<Message[]>("sacri-oracle-v1", []).map((m) => m.streaming ? { ...m, streaming: false, text: m.text || "The reply was interrupted.", error: !m.text } : m));
  const [lens, setLens] = useState<Lens>(() => stored<Lens>("sacri-lens", "archive"));
  const [speakReplies, setSpeakReplies] = useState(() => stored("sacri-speak-replies", false));
  const [voiceId, setVoiceId] = useState(() => stored("sacri-voice", "lewis"));
  const [voices, setVoices] = useState<{ id: string; label: string }[]>([]);
  const [engine, setEngine] = useState<"neural" | "system" | "native">("system");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Responding");
  const [voiceMode, setVoiceMode] = useState(false);
  const [phase, setPhase] = useState<Phase>("opening");
  const [heard, setHeard] = useState("");
  const [caption, setCaption] = useState("");
  const [settings, setSettings] = useState(false);
  const [openSources, setOpenSources] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const scroll = useRef<HTMLDivElement>(null);
  const capture = useRef<VoiceCapture | null>(null);
  const micLevel = useRef(0);
  const listening = useRef(false), voiceEnabled = useRef(false), speaking = useRef(false), working = useRef(false);
  const draftId = useRef<string | null>(null), speakDraft = useRef<string | null>(null);
  const activeRef = useRef(active); activeRef.current = active;
  const settingsRef = useRef({ speakReplies, voiceId, engine, lens });
  settingsRef.current = { speakReplies, voiceId, engine, lens };
  const askRef = useRef<(question: string) => Promise<void>>(async () => {});
  const [stage] = useState(() => new VoiceStage());

  const ready = ["chat", "embed"].every((id) => models.find((m) => m.id === id)?.downloaded);
  const voiceReady = models.find((m) => m.id === "voice")?.downloaded ?? false;
  const missing = models.filter((m) => !m.downloaded);

  const beginListening = useCallback(() => {
    if (!voiceEnabled.current) return;
    listening.current = false; setPhase("listening");
    stage.chime();
    // Let the chime fade before the microphone may hear a question.
    window.setTimeout(() => { if (voiceEnabled.current && !working.current && !speaking.current) listening.current = true; }, 520);
  }, [stage]);

  const [speech] = useState(() => new SpeechQueue<Spoken>(
    async (item) => {
      setCaption(item.text);
      if (item.audio) await stage.play(item.audio); else await api.speak(item.text);
    },
    async () => { stage.stop(); await api.stopSpeech().catch(() => {}); },
    (on) => {
      speaking.current = on;
      if (on) { listening.current = false; if (voiceEnabled.current) setPhase("speaking"); }
      else { setCaption(""); if (voiceEnabled.current && !working.current) beginListening(); }
    },
    (e) => onError(String(e)),
    async (text) => settingsRef.current.engine === "native" ? { text } : { text, audio: await stage.decode(await api.synthesize(text, settingsRef.current.voiceId)) },
  ));

  useEffect(() => { remember("sacri-oracle-v1", messages); scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" }); }, [messages]);
  useEffect(() => remember("sacri-lens", lens), [lens]);
  useEffect(() => remember("sacri-speak-replies", speakReplies), [speakReplies]);
  useEffect(() => remember("sacri-voice", voiceId), [voiceId]);
  useEffect(() => {
    void api.speechEngine().then(setEngine).catch(() => {});
    void api.voices().then(setVoices).catch(() => {});
  }, [models]);

  useEffect(() => {
    const subscriptions = [
      listen<string>("oracle-status", ({ payload }) => setStatus(payload)),
      listen<string>("oracle-token", ({ payload }) => setMessages((list) => list.map((m) => m.id === draftId.current ? { ...m, text: m.text + payload } : m))),
      listen<string>("oracle-sentence", ({ payload }) => {
        if (speakDraft.current && speakDraft.current === draftId.current && (settingsRef.current.speakReplies || voiceEnabled.current)) speech.push(payload);
      }),
    ];
    return () => {
      for (const p of subscriptions) void p.then((unlisten) => unlisten());
      void capture.current?.close(); void api.voiceMode(false); void speech.cancel(); void stage.close();
    };
  }, [speech, stage]);

  async function cancelSpeech() { speakDraft.current = null; await speech.cancel(); }
  function speak(text: string) {
    void cancelSpeech().then(() => { for (const s of sentences(text)) speech.push(s); }).catch((e) => onError(String(e)));
  }

  const leaveVoice = useCallback(async () => {
    voiceEnabled.current = false; listening.current = false; setVoiceMode(false); micLevel.current = 0; setCaption(""); setHeard("");
    const current = capture.current; capture.current = null; await current?.close();
    speakDraft.current = null; await speech.cancel(); await api.voiceMode(false);
  }, [speech]);

  async function enterVoice() {
    if (!voiceReady || !ready || busy) { if (!voiceReady) setSettings(true); return; }
    onError(null); setPhase("opening"); setVoiceMode(true); voiceEnabled.current = true; setHeard(""); setCaption("");
    try {
      capture.current = await VoiceCapture.start(() => listening.current && !working.current && !speaking.current, (pcm) => {
        listening.current = false; working.current = true; setPhase("hearing");
        void api.transcribe(pcm).then(async (text) => {
          working.current = false;
          if (!voiceEnabled.current) return;
          const said = text.trim();
          if (said && !/^[[(].*[\])]$/.test(said)) { setHeard(said); await askRef.current(said); }
          else beginListening();
        }).catch((e) => { working.current = false; if (voiceEnabled.current) { onError(String(e)); beginListening(); } });
      }, (level) => { micLevel.current = level; });
      await api.voiceMode(true);
      if (!voiceEnabled.current) { await capture.current?.close(); capture.current = null; await api.voiceMode(false); return; }
      beginListening();
    } catch (e) { onError(String(e)); await leaveVoice(); }
  }

  function interrupt() {
    listening.current = false;
    void api.stop().then(() => cancelSpeech()).finally(() => { if (voiceEnabled.current) beginListening(); });
  }

  async function ask(question = input) {
    if (!question.trim() || working.current) return;
    working.current = true; listening.current = false;
    if (!voiceEnabled.current) setInput("");
    setBusy(true); setStatus("Responding"); onError(null); setOpenSources(null);
    if (voiceEnabled.current) setPhase("thinking");
    const id = crypto.randomUUID(); draftId.current = id;
    const spoken = voiceEnabled.current;
    const history: [string, string][] = [];
    for (let i = 0; i < messages.length - 1; i++) if (messages[i].role === "user" && messages[i + 1].role === "assistant" && !messages[i + 1].error && messages[i + 1].text) history.push([messages[i].text, messages[i + 1].text]);
    const asked = settingsRef.current.lens;
    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", text: question, at: Date.now(), spoken }, { id, role: "assistant", text: "", streaming: true, lens: asked, spoken, at: Date.now() }]);
    try {
      await cancelSpeech();
      speakDraft.current = id;
      const result = await api.ask(question, sourceId, history.slice(-3), asked, spoken);
      setMessages((m) => m.map((message) => message.id === id ? { ...message, text: result.answer, sources: result.sources, threads: result.threads, streaming: false } : message));
    } catch (e) {
      setMessages((m) => m.map((message) => message.id === id ? { ...message, text: String(e), error: true, streaming: false } : message));
    } finally {
      setBusy(false); working.current = false; draftId.current = null;
      if (voiceEnabled.current && !speech.speaking) beginListening();
      if (!activeRef.current) void api.unload().catch(console.error);
    }
  }
  askRef.current = ask;

  // Leaving the Oracle screen ends voice and frees the models when idle.
  useEffect(() => {
    if (active) return;
    setSettings(false);
    if (voiceEnabled.current) void leaveVoice().catch((e) => onError(String(e)));
    void cancelSpeech();
    if (!working.current) void api.unload().catch(console.error);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const keys = (event: KeyboardEvent) => {
      const typing = event.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
      if (event.key === "Escape") {
        if (settings) setSettings(false);
        else if (voiceEnabled.current) void leaveVoice();
        else if (working.current) { void api.stop(); void cancelSpeech(); }
        else if (speech.speaking) void cancelSpeech();
      }
      if (event.key === " " && voiceEnabled.current && !typing) { event.preventDefault(); interrupt(); }
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  });

  const sigilState: SigilState = !voiceMode ? (busy ? "thinking" : "idle") : phase === "listening" ? "listening" : phase === "speaking" ? "speaking" : "thinking";
  const level = useCallback(() => speaking.current ? stage.level() : listening.current ? micLevel.current : 0, [stage]);

  const openings = useMemo(() => {
    const concepts = [...library.concepts].sort((a, b) => b.evidenceIds.length - a.evidenceIds.length).map((c) => c.label);
    const list: string[] = [];
    if (concepts[0]) list.push(`What patterns run through ${concepts[0]}?`);
    if (concepts[1] && concepts[2]) list.push(`How does ${concepts[1]} connect to ${concepts[2]}?`);
    if (concepts[3]) list.push(`What is the hidden meaning of ${concepts[3]}?`);
    if (library.sources[0]) list.push(`What is “${library.sources[0].title}” really about?`);
    list.push("What do you hold?");
    return list.slice(0, 4);
  }, [library]);

  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const phaseWord = phase === "opening" ? "Opening the ear" : phase === "listening" ? "Listening" : phase === "hearing" ? "Hearing you" : phase === "speaking" ? "Speaking" : STATUS[status] ?? status;

  return <section className="oracle-screen" aria-label="Oracle">
    <div className="oracle-scroll" ref={scroll}>
      {!messages.length && <div className="oracle-welcome">
        <Sigil state={busy ? "thinking" : "idle"} size={220} />
        <p className="eyebrow">The Oracle of the archive</p>
        <h1>Speak, and the archive answers.</h1>
        <p className="welcome-sub">A private oracle of what you have preserved. Follow patterns, symbols and hidden threads; every answer can lead back to its source.</p>
        {ready && <div className="openings">{openings.map((q) => <button key={q} onClick={() => void ask(q)} disabled={busy}><span>✶</span>{q}</button>)}</div>}
      </div>}
      {messages.map((message, index) => message.role === "user"
        ? <article key={message.id} className="oracle-message user"><p>{message.spoken && <span className="spoken-mark" title="Spoken">◖</span>}{message.text}</p></article>
        : <article key={message.id} className={`oracle-message assistant ${message.error ? "error" : ""} ${message.lens === "esoteric" ? "esoteric" : ""}`}>
          <header><SigilMark size={18} /><span className="eyebrow">Oracle{message.lens === "esoteric" ? " · esoteric lens" : ""}</span>{message.at && <time>{new Date(message.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>}</header>
          {message.streaming && !message.text
            ? <div className="contemplating" role="status"><span>{STATUS[status] ?? status}</span><i /><i /><i /></div>
            : message.error ? <p className="error">{message.text}</p>
            : <Answer text={message.streaming ? streamingText(message.text) : message.text} sources={message.sources} onOpen={onEvidence} />}
          {message.streaming && message.text && <span className="oracle-caret" aria-hidden="true" />}
          {!message.streaming && !message.error && <footer>
            <div className="message-actions">
              <button className="ghost" onClick={() => speak(message.text)} aria-label="Hear this reply">◖ Hear</button>
              <button className="ghost" onClick={() => { void navigator.clipboard?.writeText(spokenText(message.text)).then(() => { setCopied(message.id); window.setTimeout(() => setCopied(null), 1400); }); }}>{copied === message.id ? "Copied" : "Copy"}</button>
              {!!message.sources?.length && <button className="ghost" aria-expanded={openSources === message.id} onClick={() => setOpenSources(openSources === message.id ? null : message.id)}>{message.sources.length} passages</button>}
            </div>
            {openSources === message.id && <ol className="answer-sources">{message.sources?.map((hit, i) => <li key={hit.evidence.id}><button onClick={() => onEvidence(hit.evidence)}><span className="n">{i + 1}</span><span><b>{timestamp(hit.evidence.start)} · {hit.evidence.kind === "document" ? "on-screen text" : hit.evidence.kind}</b>{hit.sourceTitle}<em>{(hit.reading ?? hit.evidence.text).slice(0, 140)}…</em></span></button></li>)}</ol>}
            {index === messages.length - 1 && !!message.threads?.length && <div className="threads"><span className="eyebrow">Threads to follow</span>{message.threads.map((t) => <button key={t} disabled={busy} onClick={() => void ask(`What does the archive hold about ${t}?`)}>{t}</button>)}</div>}
          </footer>}
        </article>)}
    </div>

    {voiceMode && <div className={`voice-presence phase-${phase}`} role="dialog" aria-label="Voice conversation">
      <div className="presence-veil" />
      <Sigil state={sigilState} level={level} size={300} />
      <p className="presence-phase" role="status" aria-live="polite">{phaseWord}</p>
      <div className="presence-captions" aria-live="polite">
        {heard && <p className="heard">“{heard}”</p>}
        {(caption || (phase === "thinking" && lastAssistant?.streaming)) && <p className="said">{caption ? spokenText(caption) : ""}</p>}
      </div>
      <div className="presence-controls">
        <button onClick={interrupt} disabled={phase === "listening" || phase === "opening"}>Interrupt <kbd>Space</kbd></button>
        <button onClick={() => void leaveVoice().catch((e) => onError(String(e)))}>End <kbd>Esc</kbd></button>
      </div>
      <p className="presence-note">{lens === "esoteric" ? "Esoteric lens: lore is spoken apart from the archive." : "Archive lens: only what was preserved."}</p>
    </div>}

    <div className="oracle-compose">
      {!ready && <div className="model-setup">
        <Sigil state="idle" size={54} />
        <div><strong>The Oracle sleeps until its two local models arrive.</strong><p>Your library stays open while they download. Nothing leaves this device.</p></div>
        <div className="setup-buttons">{models.filter((m) => !m.downloaded && (m.id === "chat" || m.id === "embed")).map((m) => <button key={m.id} disabled={downloadProgress[m.id] !== undefined && downloadProgress[m.id] < 1} onClick={() => onDownload(m.id)}>↓ {m.label} <small>{downloadProgress[m.id] !== undefined ? `${Math.round(downloadProgress[m.id] * 100)}%` : `${(m.bytes / 1e9).toFixed(2)} GB`}</small></button>)}</div>
      </div>}
      <div className="oracle-options">
        <div className="lens" role="radiogroup" aria-label="Lens">
          {(["archive", "esoteric"] as const).map((l) => <button key={l} role="radio" aria-checked={lens === l} className={lens === l ? "on" : ""} onClick={() => setLens(l)}>{l === "archive" ? "Archive" : "✶ Esoteric"}</button>)}
        </div>
        <select aria-label="Source scope" value={sourceId ?? ""} onChange={(e) => setSourceId(e.target.value || null)}><option value="">All preserved sources</option>{library.sources.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}</select>
        <span className="spacer" />
        <button className={`ghost toggle-chip ${speakReplies ? "on" : ""}`} aria-pressed={speakReplies} onClick={() => { setSpeakReplies(!speakReplies); if (speakReplies) void cancelSpeech(); }}>◖ Speak replies</button>
        <div className="settings-anchor">
          <button className="ghost" aria-label="Voice settings" aria-expanded={settings} onClick={() => setSettings(!settings)}>⚙</button>
          {settings && <div className="settings-pop" role="dialog" aria-label="Voice settings">
            <p className="eyebrow">The Oracle’s voice</p>
            {engine !== "native" && voices.map((v) => <label key={v.id} className="voice-choice"><input type="radio" name="voice" checked={voiceId === v.id} onChange={() => setVoiceId(v.id)} /><span>{v.label}</span><button className="ghost" onClick={(e) => { e.preventDefault(); setVoiceId(v.id); settingsRef.current.voiceId = v.id; speak("I am the voice of the archive. Ask, and I will follow the thread back to its source."); }}>Hear</button></label>)}
            {engine === "system" && <p className="muted">Until the neural voice is installed, the system voice speaks.</p>}
            {engine === "native" && <p className="muted">This device speaks with its own system voice.</p>}
            {missing.filter((m) => m.id === "speech" || m.id === "voice").map((m) => <button key={m.id} className="setup-line" disabled={downloadProgress[m.id] !== undefined && downloadProgress[m.id] < 1} onClick={() => onDownload(m.id)}>↓ {m.id === "speech" ? "Neural voice (Kokoro)" : "Voice conversation (Whisper)"} <small>{downloadProgress[m.id] !== undefined ? `${Math.round(downloadProgress[m.id] * 100)}%` : `${Math.round(m.bytes / 1e6)} MB`}</small></button>)}
          </div>}
        </div>
        <button className="ghost" disabled={busy || !messages.length} onClick={() => { void cancelSpeech(); setMessages([]); }}>New conversation</button>
      </div>
      <form onSubmit={(e) => { e.preventDefault(); void ask(); }} className={busy ? "busy" : ""}>
        <textarea aria-label="Ask the Oracle" value={input} maxLength={2000} onChange={(e) => setInput(e.target.value)} rows={1} placeholder={lens === "esoteric" ? "What pattern shall we trace?" : "What thread shall we follow?"} disabled={busy || !ready} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(); } }} />
        <button type="button" className="mic" aria-label={voiceReady ? "Speak with the Oracle" : "Set up voice conversation"} title={voiceReady ? "Speak with the Oracle" : "Voice conversation needs its transcription model"} disabled={busy || !ready} onClick={() => voiceMode ? void leaveVoice() : void enterVoice()}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" /></svg>
        </button>
        {busy ? <button type="button" className="send stop" aria-label="Stop response" onClick={() => { void api.stop(); void cancelSpeech(); }}>■</button>
          : <button className="send" aria-label="Send question" disabled={!ready || !input.trim()}><SigilMark size={20} /></button>}
      </form>
      <small className="compose-note" role={busy ? "status" : undefined}>{busy ? <>{STATUS[status] ?? status}<span className="dots"><i /><i /><i /></span></> : LENS_NOTE[lens]}</small>
    </div>
  </section>;
}
