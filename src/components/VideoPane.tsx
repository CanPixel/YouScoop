import { forwardRef, useImperativeHandle, useMemo, useRef, useState, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { openUrl, revealItemInDir } from "@tauri-apps/plugin-opener";
import { api } from "../api";
import { formatTs } from "../lib/format";
import UpdateDownloaderButton, { STALE_HINT } from "./UpdateDownloaderButton";
import DesktopRuntime from "./DesktopRuntime";
import { ANALYSES } from "../types";
import type { ChunkPublic, IngestProgress, SentNote, TranscriptCue, VideoMeta, VoiceSummary, WorkerStatus } from "../types";

export interface VideoPaneHandle {
  seekTo: (secs: number) => void;
}

interface Props {
  meta: VideoMeta;
  chunks: ChunkPublic[];
  transcript: TranscriptCue[];
  voices: VoiceSummary[];
  ingest?: IngestProgress;
  onCancelIngest: () => void;
  onUpdateAnalysis: () => Promise<void>;
  onRetry: () => void;
  worker: WorkerStatus | null;
  onWorkerChanged: () => void;
  busy: boolean;
}

const STAGES: { id: string; label: string }[] = [
  { id: "download", label: "Download" },
  { id: "captions", label: "Transcript" },
  { id: "frames", label: "Keyframes" },
  { id: "describe", label: "Vision pass" },
  { id: "index", label: "Semantic index" },
  { id: "voices", label: "Voices" },
  { id: "onscreen", label: "On-screen text" },
];

// Map backend stages onto the visual checklist.
const STAGE_ALIAS: Record<string, string> = {
  metadata: "download",
  download: "download",
  captions: "captions",
  transcribe: "captions",
  frames: "frames",
  describe: "describe",
  index: "index",
  voices: "voices",
  onscreen: "onscreen",
};

const SOURCE_LABEL: Record<ChunkPublic["source"], string> = {
  speech: "Speech",
  vision: "Frame model · visual",
  ocr: "On-screen text · OCR",
  frame: "Screen · from OCR",
};

const VideoPane = forwardRef<VideoPaneHandle, Props>(function VideoPane(
  { meta, chunks, transcript, voices, ingest, onCancelIngest, onUpdateAnalysis, onRetry, worker, onWorkerChanged, busy },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pendingSeekSource = useRef<ChunkPublic["source"] | undefined>(undefined);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [activeRow, setActiveRow] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<SentNote | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const currentId = useRef(meta.id);
  currentId.current = meta.id;
  // Voice separation often splits one narrator into several small clusters.
  // A voice is distinct with a tenth of the speaking time or three minutes;
  // with fewer than two distinct voices, naming them only adds noise.
  const distinctVoices = useMemo(() => {
    const total = voices.reduce((sum, v) => sum + v.seconds, 0);
    const distinct = voices.filter((v) => v.seconds >= 180 || v.seconds >= total * 0.1);
    return new Set(distinct.length >= 2 ? distinct.map((v) => v.voice) : []);
  }, [voices]);
  const entries = useMemo(() => {
    const speech: (ChunkPublic & { voice?: string | null })[] = transcript.length
      ? transcript.map((cue) => ({ ...cue, source: "speech" }))
      : chunks.filter((chunk) => chunk.source === "speech");
    const sorted: (ChunkPublic & { voice?: string | null; voiceChange?: boolean })[] = [...speech, ...chunks.filter((chunk) => chunk.source !== "speech")]
      .sort((a, b) => a.start - b.start);
    // Name the voice where speech changes hands, not on every caption line,
    // and only when the source really has more than one distinct voice.
    let previous: string | null | undefined;
    return sorted.map((entry) => {
      if (entry.source !== "speech" || !entry.voice || !distinctVoices.has(entry.voice)) return entry;
      const change = entry.voice !== previous;
      previous = entry.voice;
      return { ...entry, voiceChange: change };
    });
  }, [chunks, transcript, distinctVoices]);
  const missingAnalyses = meta.status === "ready" ? ANALYSES.filter((a) => !(meta.analysis ?? []).includes(a)) : [];
  const enriching = meta.status === "ready" && ["voices", "onscreen", "scenes", "describe", "spellings"].includes(ingest?.stage ?? "");
  const [enrichError, setEnrichError] = useState<string | null>(null);

  useEffect(() => {
    setActiveRow(null);
    setSaved(null);
    setSaveError(null);
    if (transcriptRef.current) transcriptRef.current.scrollTop = 0;
  }, [meta.id]);

  function showMoment(secs: number, source?: ChunkPublic["source"]) {
    let index = -1;
    for (let i = 0; i < entries.length; i++) {
      if (entries[i].start <= secs && (!source || entries[i].source === source)) index = i;
    }
    if (index < 0) index = entries.findIndex((entry) => !source || entry.source === source);
    setActiveRow(index < 0 ? null : index);
    const container = transcriptRef.current;
    const row = container?.querySelector<HTMLElement>(`[data-row="${index}"]`);
    if (container && row) {
      container.scrollTo({ top: row.offsetTop - container.offsetTop - container.clientHeight / 3, behavior: "smooth" });
    }
  }

  function seekTo(secs: number, source?: ChunkPublic["source"]) {
    pendingSeekSource.current = source;
    showMoment(secs, source);
    const el = videoRef.current;
    if (el) {
      el.currentTime = Math.max(0, secs);
      void el.play().catch(() => {});
    }
  }
  useImperativeHandle(ref, () => ({ seekTo }));

  const processing = meta.status === "processing" || meta.status === "pending";
  const currentStage = ingest ? STAGE_ALIAS[ingest.stage] ?? null : null;
  const currentIdx = STAGES.findIndex((s) => s.id === currentStage);

  const markers = useMemo(() => {
    if (!meta.durationSecs || meta.durationSecs <= 0) return [];
    return chunks.map((c, i) => ({
      key: i,
      left: Math.min(99.5, (c.start / meta.durationSecs) * 100),
      source: c.source,
      start: c.start,
      preview: c.text.length > 140 ? c.text.slice(0, 140) + "…" : c.text,
    }));
  }, [chunks, meta.durationSecs]);

  async function saveSource() {
    const id = meta.id;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await api.sendVideoToWizmuth(id);
      if (currentId.current === id) {
        setSaved(result);
      }
    } catch (error) {
      if (currentId.current === id) setSaveError(String(error));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="video-pane">
      <div className="player-shell">
        {meta.videoPath ? (
          <video
            ref={videoRef}
            className="player"
            src={convertFileSrc(meta.videoPath)}
            controls
            preload="metadata"
            onSeeked={(event) => {
              showMoment(event.currentTarget.currentTime, pendingSeekSource.current);
              pendingSeekSource.current = undefined;
            }}
            poster={meta.thumbPath ? convertFileSrc(meta.thumbPath) : undefined}
          />
        ) : (
          <div className="player player-placeholder">
            {processing ? <span className="spinner" /> : "▶"}
          </div>
        )}
      </div>

      {markers.length > 0 && (
        <div className="timeline">
          <div className="timeline-track">
            {markers.map((m) => (
              <button
                key={m.key}
                className={`timeline-marker ${m.source}`}
                style={{ left: `${m.left}%` }}
                title={`[${formatTs(m.start)}] ${m.preview}`}
                aria-label={`${m.source === "vision" ? "Visual description" : m.source === "ocr" ? "On-screen text" : "Speech"} at ${formatTs(m.start)}`}
                onClick={() => seekTo(m.start, m.source)}
              />
            ))}
          </div>
          <div className="timeline-legend">
            <span><i className="legend-dot speech" /> speech</span>
            <span><i className="legend-dot vision" /> visual</span>
            {chunks.some((c) => c.source === "ocr") && <span><i className="legend-dot ocr" /> on-screen text</span>}
            <span className="timeline-count">{chunks.length} indexed moments</span>
          </div>
        </div>
      )}

      <div className="video-meta">
        <h2 className="video-title">{meta.title}</h2>
        <div className="video-sub">
          {meta.channel}
          {meta.durationSecs > 0 && <> · {formatTs(meta.durationSecs)}</>}
          {meta.status === "ready" && (
            <>
              {" "}· {meta.hasCaptions ? "captions" : "whisper transcript"} · {meta.frameCount} frames
            </>
          )}
        </div>
        {meta.url && (
          <button
            className="video-source-link"
            title={meta.url}
            onClick={() => void openUrl(meta.url).catch(console.error)}
          >
            ▶ Watch on YouTube ↗
          </button>
        )}
        {meta.status === "ready" && (
          <button className="video-source-link archive-save" disabled={saving || busy} onClick={() => void saveSource()}>
            {saving ? "Sending…" : saved ? "✓ In WiZMUTH" : "◇ Send to WiZMUTH"}
          </button>
        )}
        {distinctVoices.size > 0 && <p className="voice-summary">Voices by speaking time: {voices.filter((v) => distinctVoices.has(v.voice)).map((v) => `${v.voice} ${formatTs(v.seconds)}`).join(" · ")}. Automatic clusters, not identities.</p>}
        {(missingAnalyses.length > 0 || enriching) && <div className="analysis-update" role="status">
          {enriching ? <>
            <span>{ingest?.detail}</span>
            <div className="progress-bar"><div className="progress-fill" style={{ width: `${(ingest?.progress ?? 0) * 100}%` }} /></div>
            <button className="btn btn-ghost btn-sm" onClick={onCancelIngest}>Cancel</button>
          </> : <>
            <span>{ingest?.stage === "ready" && ingest.detail.startsWith("Some analyses") ? ingest.detail : `Not yet analysed: ${missingAnalyses.map((a) => ({ voices: "who is speaking", "onscreen:3": "text shown on screen, with partial page readings merged", "scenes:4": "keyframes spread across the video, with complete scene descriptions (Qwen2.5-VL)", spellings: "caption misspellings of on-screen names" })[a]).join(", ")}. The video stays usable while this runs.`}</span>
            <button className="btn btn-ghost btn-sm" disabled={busy} onClick={() => { setEnrichError(null); void onUpdateAnalysis().catch((e) => setEnrichError(String(e))); }}>Update analysis</button>
          </>}
          {enrichError && <span className="archive-save-error" role="alert">{enrichError}</span>}
        </div>}
        {saved && (
          <div className="archive-receipt" role="status">
            <span>The transcript is a note in WiZMUTH/Inbox, with every paragraph linked to its moment in the video.</span>
            <button onClick={() => void revealItemInDir(saved.path).catch((error) => setSaveError(String(error)))}>Show note ↗</button>
          </div>
        )}
        {saveError && <p className="archive-save-error" role="alert">{saveError}</p>}
      </div>

      {entries.length > 0 && (
        <section className="transcript-panel" aria-label="Transcript and visual descriptions">
          <div className="transcript-header">
            <strong>Transcript & visuals</strong>
            <span>Original speech, on-screen text and automatic frame descriptions. Frame descriptions can be wrong.</span>
          </div>
          <div className="transcript-scroll" ref={transcriptRef}>
            {entries.map((entry, index) => (
              <button
                key={`${meta.id}-${index}`}
                data-row={index}
                className={`transcript-row ${entry.source}${activeRow === index ? " active" : ""}`}
                aria-current={activeRow === index ? "true" : undefined}
                onClick={() => seekTo(entry.start, entry.source)}
              >
                <span className="transcript-time">{formatTs(entry.start)}</span>
                <span className="transcript-content">
                  <span className="transcript-label">{SOURCE_LABEL[entry.source]}{entry.voiceChange && ` · ${entry.voice}`}</span>
                  <span className={entry.source === "ocr" ? "ocr-text" : undefined}>{entry.text}</span>
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      <DesktopRuntime status={worker} onChanged={onWorkerChanged} />

      {processing && (
        <div className="ingest-panel">
          <div className="ingest-head">
            <span className="ingest-title">
              <span className="spinner spinner-sm" /> Processing video
            </span>
            <button className="btn btn-ghost btn-sm" onClick={onCancelIngest}>
              Cancel
            </button>
          </div>
          <div className="ingest-stages">
            {STAGES.map((s, i) => {
              const state = currentIdx < 0 ? "pending" : i < currentIdx ? "done" : i === currentIdx ? "active" : "pending";
              return (
                <div key={s.id} className={`ingest-stage ${state}`}>
                  <span className="ingest-stage-icon">{state === "done" ? "✓" : i + 1}</span>
                  <span className="ingest-stage-label">{s.label}</span>
                  {state === "active" && ingest && ingest.progress > 0 && (
                    <span className="ingest-stage-pct">{Math.round(ingest.progress * 100)}%</span>
                  )}
                </div>
              );
            })}
          </div>
          {ingest && ingest.progress > 0 && (
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${ingest.progress * 100}%` }} />
            </div>
          )}
          {ingest?.detail && <div className="ingest-detail">{ingest.detail}</div>}
        </div>
      )}

      {meta.status === "error" && (
        <div className="ingest-panel ingest-error">
          <div className="ingest-fail-row">
            <span>
              <strong>Processing failed.</strong> {meta.error ?? "Unknown error."}
            </span>
            <span className="ingest-fail-actions">
              {meta.error?.includes(STALE_HINT) && (
                <UpdateDownloaderButton onUpdated={onRetry} />
              )}
              <button className="btn btn-ghost btn-sm" onClick={onRetry}>
                ↻ Retry
              </button>
            </span>
          </div>
        </div>
      )}
      {meta.status === "cancelled" && (
        <div className="ingest-panel ingest-muted">
          <div className="ingest-fail-row">
            <span>Processing was cancelled.</span>
            <button className="btn btn-ghost btn-sm" onClick={onRetry}>
              ↻ Retry
            </button>
          </div>
        </div>
      )}
    </section>
  );
});

export default VideoPane;
