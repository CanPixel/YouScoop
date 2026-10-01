import { Fragment, forwardRef, useImperativeHandle, useMemo, useRef, useState, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { openUrl, revealItemInDir } from "@tauri-apps/plugin-opener";
import { api } from "../api";
import { formatTs } from "../lib/format";
import UpdateDownloaderButton, { STALE_HINT } from "./UpdateDownloaderButton";
import DesktopRuntime from "./DesktopRuntime";
import type { ChunkPublic, IngestProgress, OmniMoment, SavedSource, TranscriptCue, VideoMeta, WorkerStatus } from "../types";

export interface VideoPaneHandle {
  seekTo: (secs: number) => void;
}

interface Props {
  meta: VideoMeta;
  chunks: ChunkPublic[];
  transcript: TranscriptCue[];
  moments: OmniMoment[];
  ingest?: IngestProgress;
  onCancelIngest: () => void;
  onRetry: () => void;
  worker: WorkerStatus | null;
  onWorkerChanged: () => void;
  busy: boolean;
  inspectionStatus: string;
  onInspect: (start: number, question: string) => Promise<void>;
  onInspectSource: () => Promise<{ changed: boolean; error?: string }>;
  onReviewMoment: (start: number, end: number, text: string) => Promise<void>;
  onRemoveReviewedMoment: (start: number, end: number) => Promise<void>;
}

const STAGES: { id: string; label: string }[] = [
  { id: "download", label: "Download" },
  { id: "captions", label: "Transcript" },
  { id: "frames", label: "Keyframes" },
  { id: "describe", label: "Vision pass" },
  { id: "index", label: "Semantic index" },
];

const SCENE_QUESTION = "In one short sentence, describe only what the camera shows. Ignore the spoken words. Do not quote or guess on-screen text, names, dates, numbers, or what a document proves.";
const JOINT_QUESTION = "What is said and what does the visible evidence actually show? Mark uncertain text and separate claims from observations.";

// Map backend stages onto the visual checklist.
const STAGE_ALIAS: Record<string, string> = {
  metadata: "download",
  download: "download",
  captions: "captions",
  transcribe: "captions",
  frames: "frames",
  describe: "describe",
  index: "index",
};

const VideoPane = forwardRef<VideoPaneHandle, Props>(function VideoPane(
  { meta, chunks, transcript, moments, ingest, onCancelIngest, onRetry, worker, onWorkerChanged, busy, inspectionStatus, onInspect, onInspectSource, onReviewMoment, onRemoveReviewedMoment },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pendingSeekSource = useRef<ChunkPublic["source"] | undefined>(undefined);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [activeRow, setActiveRow] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<SavedSource | null>(null);
  const [needsArchiveUpdate, setNeedsArchiveUpdate] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState<{ start: number; end: number } | null>(null);
  const [reviewText, setReviewText] = useState("");
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [reviewSaving, setReviewSaving] = useState(false);
  const [inspectQuestion, setInspectQuestion] = useState(SCENE_QUESTION);
  const latestMoments = useMemo(() => {
    const bySpan = new Map<string, OmniMoment>();
    for (const moment of moments) bySpan.set(`${moment.start}:${moment.end}`, moment);
    return [...bySpan.values()];
  }, [moments]);
  const currentId = useRef(meta.id);
  currentId.current = meta.id;
  const entries = useMemo(() => {
    const speech: ChunkPublic[] = transcript.length
      ? transcript.map((cue) => ({ ...cue, source: "speech" }))
      : chunks.filter((chunk) => chunk.source === "speech");
    const pending: ChunkPublic[] = latestMoments.map((m) => ({start:m.start,end:m.end,text:m.text,source:"audiovisual"}));
    return [...speech, ...chunks.filter((chunk) => chunk.source !== "speech" && chunk.source !== "audiovisual"), ...pending]
      .sort((a, b) => a.start - b.start);
  }, [chunks, transcript, latestMoments]);

  useEffect(() => {
    setActiveRow(null);
    setSaved(null);
    setNeedsArchiveUpdate(false);
    setSaveError(null);
    setReviewing(null);
    setReviewText("");
    setReviewError(null);
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
    return [...chunks.filter((c) => c.source !== "audiovisual"), ...latestMoments.map((m): ChunkPublic => ({start:m.start,end:m.end,text:m.text,source:"audiovisual"}))].map((c, i) => ({
      key: i,
      left: Math.min(99.5, (c.start / meta.durationSecs) * 100),
      source: c.source,
      start: c.start,
      preview: c.text.length > 140 ? c.text.slice(0, 140) + "…" : c.text,
    }));
  }, [chunks, latestMoments, meta.durationSecs]);

  async function saveSource() {
    const id = meta.id;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await api.saveToArchive(id);
      if (currentId.current === id) {
        setSaved(result);
        setNeedsArchiveUpdate(false);
      }
    } catch (error) {
      if (currentId.current === id) setSaveError(String(error));
    } finally {
      setSaving(false);
    }
  }

  function markArchiveChanged() {
    if (saved) setNeedsArchiveUpdate(true);
    setSaved(null);
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
                aria-label={`${m.source === "vision" ? "Visual description" : m.source === "audiovisual" ? "Raw Omni inspection" : m.source === "audiovisual_reviewed" ? "Reviewed observation" : "Speech"} at ${formatTs(m.start)}`}
                onClick={() => seekTo(m.start, m.source)}
              />
            ))}
          </div>
          <div className="timeline-legend">
            <span><i className="legend-dot speech" /> speech</span>
            <span><i className="legend-dot vision" /> visual</span>
            {latestMoments.length > 0 && <span><i className="legend-dot audiovisual" /> Omni raw</span>}
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
          <button className="video-source-link archive-save" disabled={saving || busy || inspecting || reviewSaving} onClick={() => void saveSource()}>
            {saving ? "Preserving source…" : saved ? "✓ Saved to SACRiNOMiCON" : needsArchiveUpdate ? "↻ Update SACRiNOMiCON" : "◇ Save to SACRiNOMiCON"}
          </button>
        )}
        {needsArchiveUpdate && <p className="archive-receipt" role="status">This source changed after the last save. Save again to update the archive’s current version. Earlier revisions remain preserved.</p>}
        {saved && (
          <div className="archive-receipt" role="status">
            <span>{saved.evidenceCount} passages and {saved.artifactCount} original artifacts preserved in your Inbox.</span>
            <button onClick={() => void revealItemInDir(saved.packagePath).catch((error) => setSaveError(String(error)))}>Show portable package ↗</button>
          </div>
        )}
        {saveError && <p className="archive-save-error" role="alert">{saveError}</p>}
      </div>

      {meta.status === "ready" && <section className="omni-inspection" aria-label="Audiovisual inspection">
        <div><strong>Inspect this moment with Omni</strong><span>Audio and video supplied · next 10 seconds · {latestMoments.length} raw inspected moments</span></div>
        <div className="inspection-presets" aria-label="Inspection question presets">
          <button className="btn btn-ghost btn-sm" disabled={busy || saving} onClick={() => setInspectQuestion(SCENE_QUESTION)}>Visible scene</button>
          <button className="btn btn-ghost btn-sm" disabled={busy || saving} onClick={() => setInspectQuestion(JOINT_QUESTION)}>Speech and visuals</button>
        </div>
        <textarea aria-label="Omni inspection question" value={inspectQuestion} maxLength={4000} rows={2} onChange={(e) => setInspectQuestion(e.target.value)} disabled={busy || saving} />
        <button className="btn btn-ghost btn-sm" disabled={busy || saving || !worker?.omniReady} onClick={() => {
          const start = videoRef.current?.currentTime ?? 0;
          setInspecting(true); setInspectError(null);
          void onInspect(start, inspectQuestion).then(markArchiveChanged).catch((e) => setInspectError(String(e))).finally(() => setInspecting(false));
        }}>{inspecting ? "Listening and watching…" : "Inspect playback position"}</button>
        <button className="btn btn-ghost btn-sm" disabled={busy || saving || !worker?.omniReady} onClick={() => {
          setInspecting(true); setInspectError(null);
          void onInspectSource().then(({changed,error}) => {
            if (changed) markArchiveChanged();
            if (error) setInspectError(error);
          }).catch((e) => setInspectError(String(e))).finally(() => setInspecting(false));
        }}>Experimental 12-moment sweep / resume</button>
        <p className="inference-status">Selective coverage: up to 12 ten-second moments. Omni stops after its first output line to avoid fabricated continuations, so a description can be incomplete. Raw model text stays outside search and the archive until you write a reviewed observation. The full speech transcript remains available.</p>
        {inspecting && <button className="btn btn-ghost btn-sm" onClick={() => void api.stopWorker()}>Cancel inspection</button>}
        {inspecting && inspectionStatus && <p className="inference-status" role="status">{inspectionStatus}</p>}
        {inspectError && <p role="alert" className="archive-save-error">{inspectError}</p>}
      </section>}

      {entries.length > 0 && (
        <section className="transcript-panel" aria-label="Transcript and visual descriptions">
          <div className="transcript-header">
            <strong>Transcript & visuals</strong>
            <span>Original speech, raw model output, and your reviewed observations. Raw model details can be invented.</span>
          </div>
          <div className="transcript-scroll" ref={transcriptRef}>
            {entries.map((entry, index) => {
              const rawMoment = entry.source === "audiovisual"
                ? latestMoments.find((moment) => moment.start === entry.start && moment.end === entry.end)
                : undefined;
              const alignedSpeech = rawMoment
                ? transcript.filter((cue) => cue.start < entry.end && cue.end > entry.start).map((cue) => cue.text).join(" ")
                : "";
              return <Fragment key={`${meta.id}-${index}`}>
              <button
                data-row={index}
                className={`transcript-row ${entry.source}${activeRow === index ? " active" : ""}`}
                aria-current={activeRow === index ? "true" : undefined}
                onClick={() => seekTo(entry.start, entry.source)}
              >
                <span className="transcript-time">{formatTs(entry.start)}</span>
                <span className="transcript-content">
                  <span className="transcript-label">{entry.source === "vision" ? "SmolVLM2 · visual" : entry.source === "audiovisual" ? rawMoment?.truncated ? "Omni · raw, incomplete" : "Omni · first output line, unverified" : entry.source === "audiovisual_reviewed" ? "Your reviewed observation" : "Speech"}</span>
                  <span>{entry.text}</span>
                  {rawMoment?.truncated && <span className="transcript-label">Output reached its length limit and may be incomplete.</span>}
                </span>
              </button>
              {entry.source === "audiovisual" && <div className="review-controls">
                <details className="review-context">
                  <summary>Compare with the speech transcript and inspection question</summary>
                  <p><strong>Asked Omni:</strong> {rawMoment?.question ?? "Question unavailable"}</p>
                  <p><strong>{meta.hasCaptions ? "Source captions" : "Automatic speech transcript"}:</strong> {alignedSpeech || "No aligned speech text for this moment."}</p>
                  <p>Captions and model output can both mishear or misread details. Replay the video before writing a reviewed observation.</p>
                </details>
                <button className="btn btn-ghost btn-sm" disabled={saving || busy} onClick={() => {
                  setReviewing({start:entry.start,end:entry.end});
                  setReviewText(chunks.find((c) => c.source === "audiovisual_reviewed" && c.start === entry.start && c.end === entry.end)?.text ?? "");
                  setReviewError(null);
                }}>Write reviewed observation</button>
                {reviewing?.start === entry.start && reviewing.end === entry.end && <div className="review-editor">
                  <p>Replay the moment and write only what you can confirm. Your observation is still not independent proof of any speaker claim.</p>
                  <textarea aria-label="Reviewed observation" maxLength={1200} rows={3} value={reviewText} onChange={(e) => setReviewText(e.target.value)} />
                  <button className="btn btn-ghost btn-sm" disabled={saving || reviewSaving || reviewText.trim().length < 8} onClick={() => {
                    setReviewSaving(true);setReviewError(null);
                    void onReviewMoment(entry.start,entry.end,reviewText).then(() => {setReviewing(null);markArchiveChanged();}).catch((e) => setReviewError(String(e))).finally(() => setReviewSaving(false));
                  }}>{reviewSaving ? "Saving…" : "Save reviewed observation"}</button>
                  <button className="btn btn-ghost btn-sm" onClick={() => setReviewing(null)}>Cancel</button>
                  {reviewError && <p role="alert">{reviewError}</p>}
                </div>}
              </div>}
              {entry.source === "audiovisual_reviewed" && <div className="review-controls">
                <button className="btn btn-ghost btn-sm" disabled={saving || busy} onClick={() => {
                  setReviewError(null);
                  void onRemoveReviewedMoment(entry.start,entry.end).then(markArchiveChanged).catch((e) => setReviewError(String(e)));
                }}>Remove from current YouScoop search</button>
                {reviewError && <p role="alert">{reviewError}</p>}
              </div>}
            </Fragment>;
            })}
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
