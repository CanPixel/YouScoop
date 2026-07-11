//! Staged ingestion pipeline: metadata -> download -> captions/transcribe ->
//! frames -> describe -> index. Emits `ingest-progress` events throughout.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::media;
use crate::models;
use crate::state::AppState;
use crate::store::{self, Chunk, VideoMeta};
use crate::stt::WhisperEngine;
use crate::vtt;
use crate::ytdlp;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestProgress {
    pub video_id: String,
    pub stage: String,
    pub progress: f32,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<VideoMeta>,
}

fn emit(app: &AppHandle, p: IngestProgress) {
    let _ = app.emit("ingest-progress", &p);
}

fn emit_stage(app: &AppHandle, id: &str, stage: &str, progress: f32, detail: impl Into<String>) {
    emit(
        app,
        IngestProgress {
            video_id: id.to_string(),
            stage: stage.to_string(),
            progress,
            detail: detail.into(),
            meta: None,
        },
    );
}

struct Cancelled;

fn check_cancel(cancel: &AtomicBool) -> Result<(), Cancelled> {
    if cancel.load(Ordering::Relaxed) {
        Err(Cancelled)
    } else {
        Ok(())
    }
}

/// Entry point: run the whole pipeline for one URL. Never panics the app;
/// failures are written to the video's meta and emitted as an `error` stage.
pub async fn ingest(app: AppHandle, state: Arc<AppState>, url: String) {
    // Stage 0: metadata (before this we don't even know the video id).
    let _ = app.emit(
        "ingest-progress",
        &IngestProgress {
            video_id: String::new(),
            stage: "metadata".into(),
            progress: 0.0,
            detail: "Fetching video info…".into(),
            meta: None,
        },
    );

    let info = match ytdlp::fetch_info(&app, &url).await {
        Ok(i) => i,
        Err(e) => {
            emit(
                &app,
                IngestProgress {
                    video_id: String::new(),
                    stage: "error".into(),
                    progress: 0.0,
                    detail: format!("Could not read that URL: {e}"),
                    meta: None,
                },
            );
            return;
        }
    };

    let id = info.id.clone();

    // Already fully ingested? Just surface it.
    if let Ok(existing) = store::load_meta(&app, &id) {
        if existing.status == "ready" {
            emit(
                &app,
                IngestProgress {
                    video_id: id.clone(),
                    stage: "ready".into(),
                    progress: 1.0,
                    detail: "Already in your library".into(),
                    meta: Some(existing),
                },
            );
            return;
        }
        // Partial/failed leftover: wipe and start fresh.
        let _ = store::delete_video(&app, &id);
    }

    let cancel = Arc::new(AtomicBool::new(false));
    state.cancels.lock().unwrap().insert(id.clone(), cancel.clone());

    let mut meta = VideoMeta {
        id: id.clone(),
        url: url.clone(),
        title: info.title.clone(),
        channel: info.channel.clone(),
        duration_secs: info.duration_secs,
        added_at: store::now_secs(),
        status: "processing".into(),
        ..Default::default()
    };
    if store::save_meta(&app, &meta).is_err() {
        emit_stage(&app, &id, "error", 0.0, "Could not write to the library directory");
        return;
    }
    emit(
        &app,
        IngestProgress {
            video_id: id.clone(),
            stage: "metadata".into(),
            progress: 1.0,
            detail: info.title.clone(),
            meta: Some(meta.clone()),
        },
    );

    let result = run_stages(&app, &state, &mut meta, &cancel).await;
    state.cancels.lock().unwrap().remove(&id);

    match result {
        Ok(()) => {
            meta.status = "ready".into();
            meta.error = None;
            let _ = store::save_meta(&app, &meta);
            emit(
                &app,
                IngestProgress {
                    video_id: id.clone(),
                    stage: "ready".into(),
                    progress: 1.0,
                    detail: "Ready — ask away!".into(),
                    meta: Some(meta),
                },
            );
        }
        Err(e) => {
            let cancelled = cancel.load(Ordering::Relaxed);
            meta.status = if cancelled { "cancelled".into() } else { "error".into() };
            meta.error = Some(e.to_string());
            let _ = store::save_meta(&app, &meta);
            emit(
                &app,
                IngestProgress {
                    video_id: id.clone(),
                    stage: if cancelled { "cancelled".into() } else { "error".into() },
                    progress: 0.0,
                    detail: if cancelled { "Cancelled".into() } else { e.to_string() },
                    meta: Some(meta),
                },
            );
        }
    }
}

async fn run_stages(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &mut VideoMeta,
    cancel: &Arc<AtomicBool>,
) -> Result<()> {
    let id = meta.id.clone();
    let dir = store::video_dir(app, &id)?;

    // ---- Stage: download video ----
    let video_path = dir.join("video.mp4");
    emit_stage(app, &id, "download", 0.0, "Downloading video…");
    {
        let app2 = app.clone();
        let id2 = id.clone();
        ytdlp::download_video(app, &meta.url, &video_path, move |p| {
            emit_stage(&app2, &id2, "download", p, "Downloading video…");
        })
        .await?;
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }
    meta.video_path = Some(video_path.to_string_lossy().into_owned());
    store::save_meta(app, meta)?;
    emit_stage(app, &id, "download", 1.0, "Video downloaded");

    // ---- Stage: thumbnail (best effort) ----
    let thumb_path = dir.join("thumb.jpg");
    let thumb_at = (meta.duration_secs * 0.05).clamp(1.0, 30.0);
    if media::extract_thumb(&video_path, &thumb_path, thumb_at).await.is_ok() {
        meta.thumb_path = Some(thumb_path.to_string_lossy().into_owned());
        store::save_meta(app, meta)?;
    }

    // ---- Stage: captions ----
    emit_stage(app, &id, "captions", 0.2, "Looking for captions…");
    let cues = match ytdlp::fetch_captions(app, &meta.url, &dir).await {
        Ok(Some(vtt_text)) => {
            let cues = vtt::parse_vtt(&vtt_text);
            if cues.is_empty() { None } else { Some(cues) }
        }
        _ => None,
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // ---- Stage: transcribe (only when captions are missing) ----
    let cues = match cues {
        Some(c) => {
            meta.has_captions = true;
            store::save_meta(app, meta)?;
            emit_stage(app, &id, "captions", 1.0, format!("Found captions ({} lines)", c.len()));
            c
        }
        None => {
            emit_stage(app, &id, "transcribe", 0.0, "No captions — transcribing locally with Whisper…");
            let samples = media::extract_audio_f32(&video_path).await?;
            if cancel.load(Ordering::Relaxed) {
                return Err(anyhow!("cancelled"));
            }

            // Lazily load Whisper.
            {
                let mut guard = state.whisper.lock().unwrap();
                if guard.is_none() {
                    let spec = models::spec("whisper").unwrap();
                    let path = models::model_path(app, spec)?;
                    if !path.exists() {
                        return Err(anyhow!(
                            "This video has no captions and the Whisper model is not downloaded. \
                             Download it from Settings and try again."
                        ));
                    }
                    *guard = Some(WhisperEngine::load(&path.to_string_lossy())?);
                }
            }

            let whisper = state.whisper.clone();
            let app2 = app.clone();
            let id2 = id.clone();
            let cues = tauri::async_runtime::spawn_blocking(move || {
                let guard = whisper.lock().unwrap();
                let engine = guard.as_ref().expect("whisper loaded above");
                engine.transcribe(&samples, move |p| {
                    emit_stage(&app2, &id2, "transcribe", p, "Transcribing audio…");
                })
            })
            .await??;
            emit_stage(app, &id, "transcribe", 1.0, format!("Transcribed {} segments", cues.len()));
            cues
        }
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // ---- Stage: extract frames ----
    emit_stage(app, &id, "frames", 0.0, "Extracting keyframes…");
    let duration = if meta.duration_secs > 0.0 { meta.duration_secs } else { 600.0 };
    let interval = (duration / 48.0).clamp(4.0, 30.0);
    let frames_dir = dir.join("frames");
    let frames = media::extract_frames(&video_path, &frames_dir, interval).await?;
    if frames.is_empty() {
        return Err(anyhow!("no frames could be extracted from the video"));
    }
    meta.frame_count = frames.len();
    store::save_meta(app, meta)?;
    emit_stage(app, &id, "frames", 1.0, format!("{} keyframes extracted", frames.len()));

    // ---- Stage: describe frames with SmolVLM2 ----
    let engine = state.require_engine()?;
    let n_frames = frames.len();
    emit_stage(app, &id, "describe", 0.0, format!("Watching {n_frames} frames with SmolVLM2…"));

    let app2 = app.clone();
    let id2 = id.clone();
    let cancel2 = cancel.clone();
    let interval_copy = interval;
    let vision_chunks: Vec<Chunk> = tauri::async_runtime::spawn_blocking(move || {
        let guard = engine.lock().unwrap();
        let engine = guard.as_ref().ok_or_else(|| anyhow!("engine not loaded"))?;
        let mut out = Vec::with_capacity(n_frames);
        for (i, (ts, path)) in frames.iter().enumerate() {
            check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
            let desc = match engine.describe_frame(&path.to_string_lossy(), &cancel2) {
                Ok(d) if !d.is_empty() => d,
                Ok(_) => continue,
                Err(e) => {
                    // One bad frame shouldn't sink the whole video.
                    eprintln!("frame {i} description failed: {e}");
                    continue;
                }
            };
            emit_stage(
                &app2,
                &id2,
                "describe",
                (i + 1) as f32 / n_frames as f32,
                format!("[{}] {}", crate::rag::format_ts(*ts), desc),
            );
            out.push(Chunk {
                start: *ts,
                end: ts + interval_copy,
                text: desc,
                source: "vision".into(),
                vector: Vec::new(),
            });
        }
        Ok::<_, anyhow::Error>(out)
    })
    .await??;

    // ---- Stage: build the semantic index ----
    emit_stage(app, &id, "index", 0.1, "Embedding transcript and descriptions…");
    let mut chunks = vtt::chunk_cues(&cues, 40.0, 600);
    chunks.extend(vision_chunks);
    if chunks.is_empty() {
        return Err(anyhow!("nothing to index — the video produced no transcript or descriptions"));
    }

    let engine = state.require_engine()?;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let cancel3 = cancel.clone();
    let vectors = tauri::async_runtime::spawn_blocking(move || {
        check_cancel(&cancel3).map_err(|_| anyhow!("cancelled"))?;
        let guard = engine.lock().unwrap();
        let engine = guard.as_ref().ok_or_else(|| anyhow!("engine not loaded"))?;
        engine.embed_texts(&texts, false)
    })
    .await??;
    for (chunk, vec) in chunks.iter_mut().zip(vectors) {
        chunk.vector = vec;
    }

    store::save_index(app, &id, &chunks)?;
    meta.chunk_count = chunks.len();
    store::save_meta(app, meta)?;
    emit_stage(app, &id, "index", 1.0, format!("Indexed {} segments", chunks.len()));

    Ok(())
}
