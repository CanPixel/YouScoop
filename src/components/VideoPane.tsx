import { forwardRef, useImperativeHandle, useMemo, useRef } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import { formatTs } from "../lib/format";
import type { ChunkPublic, IngestProgress, VideoMeta } from "../types";

export interface VideoPaneHandle {
  seekTo: (secs: number) => void;
}

interface Props {
  meta: VideoMeta;
  chunks: ChunkPublic[];
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
  { meta, chunks, ingest, onCancelIngest, onRetry },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useImperativeHandle(ref, () => ({
    seekTo: (secs: number) => {
      const el = videoRef.current;
      if (!el) return;
      el.currentTime = Math.max(0, secs);
      void el.play().catch(() => {});
    },
  }));

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
                onClick={() => {
                  const el = videoRef.current;
                  if (el) {
                    el.currentTime = m.start;
                    void el.play().catch(() => {});
                  }
                }}
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
            <button className="btn btn-ghost btn-sm" onClick={onRetry}>
              ↻ Retry
            </button>
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
