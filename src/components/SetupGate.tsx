import { useMemo, useState } from "react";
import logo from "../assets/logo.png";
import { api } from "../api";
import { formatBytes } from "../lib/format";
import type { ModelProgress, ModelStatus, SetupStatus } from "../types";

interface Props {
  status: SetupStatus;
  progress: Record<string, ModelProgress>;
  engineError: string | null;
  onRefresh: () => void;
}

const REQUIRED = ["vlm", "mmproj", "embed"];

export default function SetupGate({ status, progress, engineError, onRefresh }: Props) {
  const [downloading, setDownloading] = useState<Set<string>>(new Set());
  const [errors, setErrors] = useState<Record<string, string>>({});

  const requiredReady = useMemo(
    () => status.models.filter((m) => REQUIRED.includes(m.id)).every((m) => m.downloaded),
    [status.models],
  );

  const startDownload = async (id: string) => {
    setErrors((e) => ({ ...e, [id]: "" }));
    setDownloading((d) => new Set(d).add(id));
    try {
      await api.downloadModel(id);
    } catch (err) {
      setErrors((e) => ({ ...e, [id]: String(err) }));
    } finally {
      setDownloading((d) => {
        const next = new Set(d);
        next.delete(id);
        return next;
      });
      onRefresh();
    }
  };

  const downloadAll = () => {
    for (const m of status.models) {
      if (!m.downloaded && !downloading.has(m.id)) void startDownload(m.id);
    }
  };

  const anythingToDownload = status.models.some((m) => !m.downloaded);

  return (
    <div className="setup-gate">
      <div className="setup-card">
        <img src={logo} alt="SmolTalk" className="setup-logo" />
        <h1 className="setup-title">Welcome to SmolTalk</h1>
        <p className="setup-sub">
          Talk to any YouTube video. SmolTalk needs a few
          open models on disk before it can watch videos for you. Everything runs on your
          Mac; nothing ever leaves it.
        </p>

        {!status.ffmpegOk && (
          <div className="setup-warning">
            <strong>ffmpeg not found.</strong> SmolTalk needs it to read video and audio. Install it
            with <code>brew install ffmpeg</code>, then restart the app.
          </div>
        )}

        <div className="setup-models">
          {status.models.map((m) => (
            <ModelRow
              key={m.id}
              model={m}
              required={REQUIRED.includes(m.id)}
              progress={progress[m.id]}
              downloading={downloading.has(m.id)}
              error={errors[m.id]}
              onDownload={() => startDownload(m.id)}
            />
          ))}
        </div>

        {engineError && <div className="setup-warning">{engineError}</div>}

        <div className="setup-actions">
          {anythingToDownload && (
            <button className="btn btn-primary" onClick={downloadAll} disabled={downloading.size > 0}>
              {downloading.size > 0 ? "Downloading…" : "Download all models"}
            </button>
          )}
          {requiredReady && (
            <div className="setup-loading-engine">
              {status.engineLoading ? (
                <>
                  <span className="spinner" /> Loading models into memory…
                </>
              ) : (
                <>Starting engine…</>
              )}
            </div>
          )}
        </div>

        <p className="setup-footnote">
          One-time download of ~1.8 GB from Hugging Face. Whisper is optional: it&apos;s only used
          when a video has no captions.
        </p>
      </div>
    </div>
  );
}

function ModelRow({
  model,
  required,
  progress,
  downloading,
  error,
  onDownload,
}: {
  model: ModelStatus;
  required: boolean;
  progress?: ModelProgress;
  downloading: boolean;
  error?: string;
  onDownload: () => void;
}) {
  const received = progress?.received ?? model.partialBytes;
  const pct = model.downloaded ? 100 : Math.min(100, (received / model.bytes) * 100);
  const active = downloading && !model.downloaded;

  return (
    <div className={`model-row ${model.downloaded ? "done" : ""}`}>
      <div className="model-row-main">
        <div className="model-row-head">
          <span className="model-name">{model.label}</span>
          {!required && <span className="model-tag">optional</span>}
          <span className="model-size">{formatBytes(model.bytes)}</span>
        </div>
        <div className="model-detail">{model.detail}</div>
        {(active || (!model.downloaded && received > 0)) && (
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
        )}
        {error && <div className="model-error">{error}</div>}
      </div>
      <div className="model-row-action">
        {model.downloaded ? (
          <span className="model-check" title="Downloaded">✓</span>
        ) : active ? (
          <span className="model-pct">{pct.toFixed(0)}%</span>
        ) : (
          <button className="btn btn-ghost" onClick={onDownload}>
            {received > 0 ? "Resume" : "Download"}
          </button>
        )}
      </div>
    </div>
  );
}
