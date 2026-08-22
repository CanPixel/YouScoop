import { useEffect, useRef, useState } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import logo from "../assets/logo.png";
import { formatDuration } from "../lib/format";
import UpdateDownloaderButton, { STALE_HINT } from "./UpdateDownloaderButton";
import type { IngestProgress, VideoMeta } from "../types";

interface Props {
  videos: VideoMeta[];
  selectedId: string | null;
  ingest: Record<string, IngestProgress>;
  addingUrl: boolean;
  addError: string | null;
  onSelect: (id: string) => void;
  onAdd: (url: string) => void;
  onRetry: (id: string) => void;
  onRemove: (id: string) => void;
}

export default function Sidebar({
  videos,
  selectedId,
  ingest,
  addingUrl,
  addError,
  onSelect,
  onAdd,
  onRetry,
  onRemove,
}: Props) {
  const [url, setUrl] = useState("");
  // Kept so a stale-downloader failure can re-run the same URL after updating;
  // the input itself is cleared on submit.
  const [lastUrl, setLastUrl] = useState("");

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed) return;
    onAdd(trimmed);
    setLastUrl(trimmed);
    setUrl("");
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <img src={logo} alt="" className="sidebar-logo" />
        <div>
          <div className="sidebar-title">SmolTalk</div>
          <div className="sidebar-tagline">Talk to your videos</div>
        </div>
      </div>

      <form className="add-form" onSubmit={submit}>
        <input
          type="text"
          className="add-input"
          placeholder="Paste a YouTube URL…"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          spellCheck={false}
        />
        <button type="submit" className="btn btn-primary add-btn" disabled={addingUrl || !url.trim()}>
          {addingUrl ? <span className="spinner spinner-sm" /> : "Add"}
        </button>
      </form>
      {addError && (
        <div className="add-error">
          <div>{addError}</div>
          {addError.includes(STALE_HINT) && lastUrl && (
            <UpdateDownloaderButton onUpdated={() => onAdd(lastUrl)} />
          )}
        </div>
      )}

      <div className="sidebar-section">Library</div>
      <nav className="video-list">
        {videos.length === 0 && (
          <div className="video-list-empty">
            Paste a YouTube link above to add your first video.
          </div>
        )}
        {videos.map((v) => (
          <VideoItem
            key={v.id}
            video={v}
            active={v.id === selectedId}
            ingest={ingest[v.id]}
            onSelect={() => onSelect(v.id)}
            onRetry={() => onRetry(v.id)}
            onRemove={() => onRemove(v.id)}
          />
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className="status-dot" />
        <span>Local engine · Metal</span>
      </div>
    </aside>
  );
}

const STAGE_LABEL: Record<string, string> = {
  metadata: "Fetching info",
  download: "Downloading",
  captions: "Captions",
  transcribe: "Transcribing",
  frames: "Extracting frames",
  describe: "Watching frames",
  index: "Indexing",
};

function VideoItem({
  video,
  active,
  ingest,
  onSelect,
  onRetry,
  onRemove,
}: {
  video: VideoMeta;
  active: boolean;
  ingest?: IngestProgress;
  onSelect: () => void;
  onRetry: () => void;
  onRemove: () => void;
}) {
  const processing = video.status === "processing" || video.status === "pending";
  const stageLabel = processing
    ? STAGE_LABEL[ingest?.stage ?? ""] ?? "Processing"
    : null;

  // Two-step delete: first click arms the button, second click removes.
  // (window.confirm doesn't work inside the Tauri webview.)
  const [confirming, setConfirming] = useState(false);
  const disarmTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(disarmTimer.current), []);

  const handleRemoveClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirming) {
      onRemove();
      return;
    }
    setConfirming(true);
    window.clearTimeout(disarmTimer.current);
    disarmTimer.current = window.setTimeout(() => setConfirming(false), 3000);
  };

  return (
    <div className={`video-item ${active ? "active" : ""}`} onClick={onSelect}>
      <div className="video-thumb">
        {video.thumbPath ? (
          <img src={convertFileSrc(video.thumbPath)} alt="" loading="lazy" />
        ) : (
          <div className="video-thumb-placeholder">▶</div>
        )}
        {video.durationSecs > 0 && (
          <span className="video-duration">{formatDuration(video.durationSecs)}</span>
        )}
      </div>
      <div className="video-item-info">
        <div className="video-item-title" title={video.title}>
          {video.title}
        </div>
        <div className="video-item-sub">
          {processing ? (
            <span className="pill pill-processing">
              <span className="spinner spinner-xs" /> {stageLabel}
            </span>
          ) : video.status === "ready" ? (
            <span className="video-channel">{video.channel}</span>
          ) : (
            <>
              <span className={`pill ${video.status === "error" ? "pill-error" : "pill-muted"}`}>
                {video.status === "error" ? "Failed" : "Cancelled"}
              </span>
              <button
                className="video-retry"
                title="Retry processing"
                onClick={(e) => {
                  e.stopPropagation();
                  onRetry();
                }}
              >
                ↻ Retry
              </button>
            </>
          )}
        </div>
      </div>
      <button
        className={`video-remove ${confirming ? "confirming" : ""}`}
        title={confirming ? "Click again to remove" : "Remove from library"}
        onClick={handleRemoveClick}
        onMouseLeave={() => setConfirming(false)}
      >
        {confirming ? "Remove?" : "×"}
      </button>
    </div>
  );
}
