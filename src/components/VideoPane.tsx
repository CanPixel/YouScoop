import { forwardRef, useImperativeHandle, useMemo, useRef, useState, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { formatTs } from "../lib/format";
import UpdateDownloaderButton, { STALE_HINT } from "./UpdateDownloaderButton";
import type { ChunkPublic, IngestProgress, TranscriptCue, VideoMeta } from "../types";

export interface VideoPaneHandle {
  seekTo: (secs: number) => void;
}

interface Props {
  meta: VideoMeta;
  chunks: ChunkPublic[];
  transcript: TranscriptCue[];
  ingest?: IngestProgress;
  onCancelIngest: () => void;
  onRetry: () => void;
}

const STAGES: { id: string; label: string }[] = [
  { id: "download", label: "Download" },
  { id: "captions", label: "Transcript" },
  { id: "frames", label: "Keyframes" },
  { id: "describe", label: "Vision pass" },
  { id: "index", label: "Semantic index" },
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
};

const VideoPane = forwardRef<VideoPaneHandle, Props>(function VideoPane(
  { meta, chunks, transcript, ingest, onCancelIngest, onRetry },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const pendingSeekSource = useRef<ChunkPublic["source"] | undefined>(undefined);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [activeRow, setActiveRow] = useState<number | null>(null);
  const entries = useMemo(() => {
    const speech: ChunkPublic[] = transcript.length
      ? transcript.map((cue) => ({ ...cue, source: "speech" }))
      : chunks.filter((chunk) => chunk.source === "speech");
    return [...speech, ...chunks.filter((chunk) => chunk.source === "vision")]
      .sort((a, b) => a.start - b.start);
  }, [chunks, transcript]);

  useEffect(() => {
    setActiveRow(null);
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
                aria-label={`${m.source === "vision" ? "Visual description" : "Speech"} at ${formatTs(m.start)}`}
                onClick={() => seekTo(m.start, m.source)}
              />
            ))}
          </div>
          <div className="timeline-legend">
            <span><i className="legend-dot speech" /> speech</span>
            <span><i className="legend-dot vision" /> visual</span>
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
      </div>

      {entries.length > 0 && (
        <section className="transcript-panel" aria-label="Transcript and visual descriptions">
          <div className="transcript-header">
            <strong>Transcript & visuals</strong>
            <span>Speech and SmolVLM2 output</span>
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
                  <span className="transcript-label">{entry.source === "vision" ? "SmolVLM2 · visual" : "Speech"}</span>
                  <span>{entry.text}</span>
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

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
