//! Staged ingestion pipeline: metadata -> download/import -> captions or
//! transcribe -> frames -> describe -> index -> summarize. Emits
//! `ingest-progress` events throughout.
//!
//! Every stage persists its artifact (video.mp4, cues.json, frames.json,
//! descriptions.json, index.json, meta.summary) and is skipped when the
//! artifact already exists, so a retry resumes where the last attempt
//! stopped instead of re-downloading and re-transcribing everything.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};

use crate::media;
use crate::models;
use crate::onscreen;
use crate::spelling;
use crate::voices;
use crate::rag;
use crate::state::AppState;
use crate::store::{self, Chunk, FrameRef, VideoMeta};
use crate::stt::WhisperEngine;
use crate::vtt::{self, Cue};
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

/// An error surfaced before a video id exists (bad URL, over-long video, ...):
/// the frontend shows it next to the add form.
pub fn emit_global_error(app: &AppHandle, detail: String) {
    emit(
        app,
        IngestProgress {
            video_id: String::new(),
            stage: "error".into(),
            progress: 0.0,
            detail,
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

/// Save meta unless the video was removed mid-ingest; treat removal like a
/// cancellation so the pipeline stops writing (and stops recreating the
/// directory `delete_video` just removed).
fn persist(app: &AppHandle, meta: &VideoMeta) -> Result<()> {
    match store::save_meta_if_present(app, meta) {
        Ok(true) => Ok(()),
        Ok(false) => Err(anyhow!("cancelled")),
        Err(e) => Err(e),
    }
}

// Longer videos take ages to download, describe, and index locally.
const MAX_DURATION_SECS: f64 = 120.0 * 60.0;

/// Video file extensions accepted for local ingestion.
pub const LOCAL_VIDEO_EXTS: [&str; 6] = ["mp4", "mov", "m4v", "webm", "mkv", "avi"];

/// Entry point for a URL: fetch metadata, then run (or resume) the stages.
/// Never panics the app; failures are written to the video's meta and emitted
/// as an `error` stage.
pub async fn ingest(app: AppHandle, state: Arc<AppState>, url: String) {
    // Stage 0: metadata (before this we don't even know the video id).
    emit_stage(&app, "", "metadata", 0.0, "Fetching video info…");

    let info = match ytdlp::fetch_info(&app, &url).await {
        Ok(i) => i,
        Err(e) => {
            emit_global_error(&app, format!("Could not read that URL: {e}"));
            return;
        }
    };

    if info.duration_secs > MAX_DURATION_SECS {
        emit_global_error(
            &app,
            format!(
                "\"{}\" is {} long. YouScoop accepts videos up to 2:00:00.",
                info.title,
                rag::format_ts(info.duration_secs),
            ),
        );
        return;
    }

    // Reuse the existing meta (and its finished artifacts) when this video
    // was added before; refresh the fields that may have changed upstream.
    let meta = match store::load_meta(&app, &info.id) {
        Ok(mut existing) => {
            existing.url = url;
            existing.title = info.title;
            existing.channel = info.channel;
            existing.duration_secs = info.duration_secs;
            existing
        }
        Err(_) => VideoMeta {
            id: info.id.clone(),
            url,
            title: info.title,
            channel: info.channel,
            duration_secs: info.duration_secs,
            added_at: store::now_secs(),
            ..Default::default()
        },
    };

    run(app, state, meta, None).await;
}

/// Entry point for a local video file: import it into the library, skip the
/// download/captions stages, and transcribe with Whisper.
pub async fn ingest_local(app: AppHandle, state: Arc<AppState>, path: PathBuf) {
    emit_stage(&app, "", "metadata", 0.0, "Reading file…");

    let file_name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled video".into());
    let ext_ok = path
        .extension()
        .map(|e| LOCAL_VIDEO_EXTS.contains(&e.to_string_lossy().to_lowercase().as_str()))
        .unwrap_or(false);
    if !ext_ok || !path.is_file() {
        emit_global_error(&app, format!("\"{file_name}\" doesn't look like a video file."));
        return;
    }

    let duration = match media::probe_duration(&path).await {
        Ok(d) => d,
        Err(e) => {
            emit_global_error(&app, format!("Could not read \"{file_name}\": {e}"));
            return;
        }
    };
    if duration > MAX_DURATION_SECS {
        emit_global_error(
            &app,
            format!(
                "\"{}\" is {} long. YouScoop accepts videos up to 2:00:00.",
                file_name,
                rag::format_ts(duration),
            ),
        );
        return;
    }

    // Stable id from the source path + size, so re-adding the same file
    // resumes its artifacts instead of duplicating them.
    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
    let id = format!("local-{:016x}", fnv1a(format!("{}:{size}", path.display()).as_bytes()));

    let meta = match store::load_meta(&app, &id) {
        Ok(existing) => existing,
        Err(_) => VideoMeta {
            id: id.clone(),
            url: String::new(),
            title: file_name,
            channel: "Local file".into(),
            duration_secs: duration,
            added_at: store::now_secs(),
            ..Default::default()
        },
    };

    run(app, state, meta, Some(path)).await;
}

/// Resume a failed/cancelled ingest from its persisted artifacts.
pub async fn resume(app: AppHandle, state: Arc<AppState>, meta: VideoMeta) {
    run(app, state, meta, None).await;
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Shared driver: guard against double ingestion, run the stages, finalize.
async fn run(app: AppHandle, state: Arc<AppState>, mut meta: VideoMeta, local_src: Option<PathBuf>) {
    let id = meta.id.clone();

    if meta.status == "ready" {
        emit(
            &app,
            IngestProgress {
                video_id: id,
                stage: "ready".into(),
                progress: 1.0,
                detail: "Already in your library".into(),
                meta: Some(meta),
            },
        );
        return;
    }

    // One pipeline per video: a second add of the same URL must not race the
    // first one over the same directory.
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut cancels = state.cancels.lock().unwrap();
        if cancels.contains_key(&id) {
            emit_global_error(&app, format!("\"{}\" is already being processed.", meta.title));
            return;
        }
        cancels.insert(id.clone(), cancel.clone());
    }

    meta.status = "processing".into();
    meta.error = None;
    if store::save_meta(&app, &meta).is_err() {
        state.cancels.lock().unwrap().remove(&id);
        emit_stage(&app, &id, "error", 0.0, "Could not write to the library directory");
        return;
    }
    emit(
        &app,
        IngestProgress {
            video_id: id.clone(),
            stage: "metadata".into(),
            progress: 1.0,
            detail: meta.title.clone(),
            meta: Some(meta.clone()),
        },
    );

    let result = run_stages(&app, &state, &mut meta, &cancel, local_src).await;
    state.cancels.lock().unwrap().remove(&id);

    // With no ingest running, the vision models are dead weight (~1.7 GB):
    // drop them until the next describe stage needs them.
    if state.cancels.lock().unwrap().is_empty() {
        if let Ok(mut guard) = state.engine.lock() {
            if let Some(engine) = guard.as_mut() {
                engine.unload_vision();
            }
        }
    }

    match result {
        Ok(()) => {
            meta.status = "ready".into();
            meta.error = None;
            if store::save_meta_if_present(&app, &meta).unwrap_or(false) {
                emit(
                    &app,
                    IngestProgress {
                        video_id: id.clone(),
                        stage: "ready".into(),
                        progress: 1.0,
                        detail: "Ready. Ask away!".into(),
                        meta: Some(meta),
                    },
                );
            }
        }
        Err(e) => {
            let cancelled = cancel.load(Ordering::Relaxed);
            meta.status = if cancelled { "cancelled".into() } else { "error".into() };
            meta.error = Some(e.to_string());
            // If the video was removed mid-ingest, don't write state back
            // (that would resurrect the directory) and don't emit a meta the
            // UI would re-add to the library.
            if store::save_meta_if_present(&app, &meta).unwrap_or(false) {
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
}

async fn run_stages(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &mut VideoMeta,
    cancel: &Arc<AtomicBool>,
    local_src: Option<PathBuf>,
) -> Result<()> {
    let id = meta.id.clone();
    let dir = store::video_dir(app, &id)?;

    // ---- Stage: download / import ----
    let video_path = dir.join("video.mp4");
    let have_video = video_path.metadata().map(|m| m.len() > 0).unwrap_or(false);
    if have_video {
        emit_stage(app, &id, "download", 1.0, "Reusing the downloaded video");
    } else if let Some(src) = local_src {
        emit_stage(app, &id, "download", 0.1, "Importing file…");
        let dest = video_path.clone();
        tauri::async_runtime::spawn_blocking(move || {
            // Hard link when source and library share a volume (instant, no
            // extra disk); fall back to a real copy.
            if std::fs::hard_link(&src, &dest).is_err() {
                std::fs::copy(&src, &dest)?;
            }
            Ok::<_, std::io::Error>(())
        })
        .await??;
        emit_stage(app, &id, "download", 1.0, "File imported");
    } else if !meta.url.is_empty() {
        emit_stage(app, &id, "download", 0.0, "Downloading video…");
        let app2 = app.clone();
        let id2 = id.clone();
        ytdlp::download_video(app, &meta.url, &video_path, cancel, move |p| {
            emit_stage(&app2, &id2, "download", p, "Downloading video…");
        })
        .await?;
        emit_stage(app, &id, "download", 1.0, "Video downloaded");
    } else {
        return Err(anyhow!(
            "The original file for this video is no longer available — remove it and add the file again."
        ));
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }
    meta.video_path = Some(video_path.to_string_lossy().into_owned());
    persist(app, meta)?;

    // ---- Stage: thumbnail (best effort) ----
    let thumb_path = dir.join("thumb.jpg");
    if !thumb_path.exists() {
        let thumb_at = (meta.duration_secs * 0.05).clamp(1.0, 30.0);
        if media::extract_thumb(&video_path, &thumb_path, thumb_at).await.is_ok() {
            meta.thumb_path = Some(thumb_path.to_string_lossy().into_owned());
            persist(app, meta)?;
        }
    } else if meta.thumb_path.is_none() {
        meta.thumb_path = Some(thumb_path.to_string_lossy().into_owned());
        persist(app, meta)?;
    }

    // ---- Stage: transcript (restore / captions / whisper) ----
    let cues: Vec<Cue> = if let Ok(cues) = store::load_cues(app, &id) {
        emit_stage(app, &id, "captions", 1.0, format!("Transcript restored ({} lines)", cues.len()));
        cues
    } else {
        let cues = obtain_cues(app, state, meta, cancel, &video_path).await?;
        store::save_cues(app, &id, &cues)?;
        cues
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // ---- Stage: extract frames ----
    let frames_dir = dir.join("frames");
    let frames: Vec<(f64, PathBuf)> = match load_frames(app, &id, &frames_dir) {
        Some(frames) => {
            emit_stage(app, &id, "frames", 1.0, format!("{} keyframes restored", frames.len()));
            frames
        }
        None => {
            emit_stage(app, &id, "frames", 0.0, "Detecting scene changes…");
            let frames = extract_keyframes(app, &id, &video_path, &frames_dir, meta.duration_secs).await?;
            emit_stage(app, &id, "frames", 1.0, format!("{} keyframes extracted", frames.len()));
            frames
        }
    };
    meta.frame_count = frames.len();
    persist(app, meta)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // ---- Stage: describe frames with SmolVLM2 ----
    let vision_chunks = describe_frames(app, state, meta, cancel, &frames).await?;

    // The vision stack is no longer needed for this video; free its ~1.7 GB
    // unless another ingest is running concurrently.
    if state.cancels.lock().unwrap().len() <= 1 {
        if let Ok(mut guard) = state.engine.lock() {
            if let Some(engine) = guard.as_mut() {
                engine.unload_vision();
            }
        }
    }

    // ---- Stage: build the semantic index ----
    let chunks: Vec<Chunk> = match store::load_index(app, &id) {
        Ok(existing) if !existing.is_empty() => {
            emit_stage(app, &id, "index", 1.0, format!("Index restored ({} segments)", existing.len()));
            existing
        }
        _ => build_index(app, state, meta, cancel, &cues, vision_chunks).await?,
    };
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // ---- Stage: whole-video summary ----
    if meta.summary.is_none() {
        match summarize(app, state, meta, cancel, &chunks).await {
            Ok(Some(summary)) => {
                meta.summary = Some(summary);
                persist(app, meta)?;
                emit_stage(app, &id, "summarize", 1.0, "Summary ready");
            }
            Ok(None) => {}
            Err(e) => {
                if cancel.load(Ordering::Relaxed) {
                    return Err(anyhow!("cancelled"));
                }
                // A missing summary shouldn't sink an otherwise indexed video.
                eprintln!("summary generation failed for {id}: {e}");
            }
        }
    }

    enrich(app, state, meta, cancel).await
}

/// Captions when the video has a URL and provides them, Whisper otherwise.
async fn obtain_cues(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &mut VideoMeta,
    cancel: &Arc<AtomicBool>,
    video_path: &std::path::Path,
) -> Result<Vec<Cue>> {
    let id = meta.id.clone();

    if !meta.url.is_empty() {
        emit_stage(app, &id, "captions", 0.2, "Looking for captions…");
        let dir = store::video_dir(app, &id)?;
        let cues = match ytdlp::fetch_captions(app, &meta.url, &dir).await {
            Ok(Some(vtt_text)) => {
                let cues = vtt::parse_vtt(&vtt_text);
                if cues.is_empty() {
                    None
                } else {
                    Some(cues)
                }
            }
            _ => None,
        };
        if cancel.load(Ordering::Relaxed) {
            return Err(anyhow!("cancelled"));
        }
        if let Some(cues) = cues {
            meta.has_captions = true;
            persist(app, meta)?;
            emit_stage(app, &id, "captions", 1.0, format!("Found captions ({} lines)", cues.len()));
            return Ok(cues);
        }
    }

    // ---- Transcribe locally ----
    emit_stage(app, &id, "transcribe", 0.0, "No captions: transcribing locally with Whisper…");

    // Resolve a Whisper model, downloading the registry one on demand.
    // Progress mapping for this stage: download 0–25%, load ~28%, transcribe 30–100%.
    let spec = models::spec("whisper").expect("whisper model in registry");
    let mut whisper_path = models::model_path(app, spec)?;
    if !models::is_downloaded(app, spec) {
        let app3 = app.clone();
        let id3 = id.clone();
        let label = spec.label;
        let dl = models::download_with_progress(app, "whisper", move |received, total| {
            let frac = if total > 0 { received as f32 / total as f32 } else { 0.0 };
            emit_stage(
                &app3,
                &id3,
                "transcribe",
                frac * 0.25,
                format!(
                    "One-time download of {label} ({:.0} of {:.0} MB)…",
                    received as f64 / 1e6,
                    total as f64 / 1e6
                ),
            );
        })
        .await;
        if let Err(e) = dl {
            match models::any_whisper_fallback(app) {
                Some(alt) => {
                    let name =
                        alt.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    emit_stage(
                        app,
                        &id,
                        "transcribe",
                        0.26,
                        format!("Download failed ({e}); falling back to {name}"),
                    );
                    whisper_path = alt;
                }
                None => {
                    return Err(anyhow!(
                        "This video has no captions, and downloading the Whisper model failed: {e}. \
                         Check your connection and press Retry."
                    ));
                }
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    let samples = media::extract_audio_f32(video_path).await?;
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
    }

    // Load (or reload, if the model file changed) and transcribe in one
    // blocking task - the load alone can take a few seconds.
    let whisper = state.whisper.clone();
    let wanted_path = whisper_path.to_string_lossy().into_owned();
    let app_load = app.clone();
    let id_load = id.clone();
    let app2 = app.clone();
    let id2 = id.clone();
    let cancel2 = cancel.clone();
    let cues = tauri::async_runtime::spawn_blocking(move || {
        let mut guard = whisper.lock().unwrap();
        let stale = guard.as_ref().map(|w| w.model_path != wanted_path).unwrap_or(true);
        if stale {
            emit_stage(&app_load, &id_load, "transcribe", 0.28, "Loading Whisper model…");
            *guard = Some(WhisperEngine::load(&wanted_path)?);
        }
        let engine = guard.as_ref().expect("whisper loaded above");
        engine.transcribe(&samples, cancel2, move |p| {
            emit_stage(&app2, &id2, "transcribe", 0.3 + p * 0.7, "Transcribing audio…");
        })
    })
    .await??;
    emit_stage(app, &id, "transcribe", 1.0, format!("Transcribed {} segments", cues.len()));
    Ok(cues)
}

/// Restore the frame list from the manifest, or None when it's missing or
/// any file has gone away (then the caller re-extracts).
/// Extract keyframes at scene changes and save their manifest.
async fn extract_keyframes(app: &AppHandle, id: &str, video: &std::path::Path, frames_dir: &std::path::Path, duration_secs: f64) -> Result<Vec<(f64, PathBuf)>> {
    let frames = media::extract_frames(video, frames_dir, duration_secs).await?;
    if frames.is_empty() {
        return Err(anyhow!("no frames could be extracted from the video"));
    }
    let manifest: Vec<FrameRef> = frames
        .iter()
        .map(|(ts, p)| FrameRef {
            ts: *ts,
            file: p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        })
        .collect();
    store::save_frame_manifest(app, id, &manifest)?;
    Ok(frames)
}

fn load_frames(app: &AppHandle, id: &str, frames_dir: &std::path::Path) -> Option<Vec<(f64, PathBuf)>> {
    let manifest = store::load_frame_manifest(app, id).ok()?;
    if manifest.is_empty() {
        return None;
    }
    let frames: Vec<(f64, PathBuf)> = manifest
        .into_iter()
        .map(|f| (f.ts, frames_dir.join(&f.file)))
        .collect();
    frames.iter().all(|(_, p)| p.exists()).then_some(frames)
}

/// Describe frames with the VLM, resuming from previously saved descriptions.
/// The engine mutex is taken per frame, not for the whole batch, so a chat
/// question asked mid-ingest gets a slot within one frame (~seconds) instead
/// of hanging until the whole video is described.
async fn describe_frames(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &VideoMeta,
    cancel: &Arc<AtomicBool>,
    frames: &[(f64, PathBuf)],
) -> Result<Vec<Chunk>> {
    let id = meta.id.clone();
    let engine = state.require_engine()?;
    let n_frames = frames.len();

    let mut existing: Vec<Chunk> = store::load_descriptions(app, &id).unwrap_or_default();
    let done: HashSet<i64> = existing.iter().map(|c| (c.start * 1000.0) as i64).collect();

    // End of each frame's validity: the next frame's timestamp.
    let todo: Vec<(f64, f64, PathBuf)> = frames
        .iter()
        .enumerate()
        .filter(|(_, (ts, _))| !done.contains(&((ts * 1000.0) as i64)))
        .map(|(i, (ts, path))| {
            let end = frames.get(i + 1).map(|(t, _)| *t).unwrap_or(ts + 15.0);
            (*ts, end, path.clone())
        })
        .collect();

    if todo.is_empty() {
        emit_stage(app, &id, "describe", 1.0, format!("{} frame descriptions restored", existing.len()));
        existing.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
        return Ok(existing);
    }

    // Prefer Qwen2.5-VL, downloading it on first use; SmolVLM2 remains the
    // fallback when it cannot be fetched.
    let mut frame_model = (models::spec("vlm").unwrap(), models::spec("mmproj").unwrap());
    let preferred = (models::spec("scene-vlm").unwrap(), models::spec("scene-mmproj").unwrap());
    let mut available = models::is_downloaded(app, preferred.0) && models::is_downloaded(app, preferred.1);
    if !available {
        emit_stage(app, &id, "describe", 0.0, format!("Downloading {} for frame descriptions…", preferred.0.label));
        available = async {
            models::download(app, preferred.0.id).await?;
            models::download(app, preferred.1.id).await
        }
        .await
        .is_ok();
        check_cancel(cancel).map_err(|_| anyhow!("cancelled"))?;
    }
    if available {
        frame_model = preferred;
    }
    let label = {
        let mut guard = engine.lock().unwrap();
        let engine = guard.as_mut().ok_or_else(|| anyhow!("engine not loaded"))?;
        engine.use_frame_model(&models::model_path(app, frame_model.0)?.to_string_lossy(), &models::model_path(app, frame_model.1)?.to_string_lossy());
        engine.frame_model_label()
    };
    let version = if available { SCENE_PROMPT_VERSION.to_string() } else { "2".to_string() };
    emit_stage(
        app,
        &id,
        "describe",
        done.len() as f32 / n_frames as f32,
        format!("Watching {} frames with {label}…", todo.len()),
    );

    let app2 = app.clone();
    let id2 = id.clone();
    let cancel2 = cancel.clone();
    let n_done = done.len();
    let mut chunks = tauri::async_runtime::spawn_blocking(move || {
        let mut out = existing;
        for (i, (ts, end, path)) in todo.iter().enumerate() {
            check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
            // Lock per frame: release between frames so chat can interleave.
            let desc = {
                let mut guard = engine.lock().unwrap();
                let engine = guard.as_mut().ok_or_else(|| anyhow!("engine not loaded"))?;
                engine.describe_frame(&path.to_string_lossy(), &cancel2)
            };
            let desc = match desc {
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
                (n_done + i + 1) as f32 / n_frames as f32,
                format!("[{}] {}", rag::format_ts(*ts), desc),
            );
            out.push(Chunk {
                start: *ts,
                end: *end,
                text: desc,
                source: "vision".into(),
                vector: Vec::new(),
                cues: Vec::new(),
            });
            // Persist incrementally so a cancel/crash keeps completed VLM work.
            if store::meta_exists(&app2, &id2) {
                let _ = store::save_descriptions(&app2, &id2, &out);
                if let Ok(dir) = store::video_dir_path(&app2, &id2) {
                    let _ = std::fs::write(dir.join("descriptions.version"), format!("{version}\n{label}"));
                }
            }
        }
        Ok::<_, anyhow::Error>(out)
    })
    .await??;

    chunks.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    Ok(chunks)
}

/// Embed chunk texts in batches, locking the engine per batch so chat stays
/// responsive.
async fn embed_chunks(
    app: &AppHandle,
    state: &Arc<AppState>,
    id: &str,
    cancel: &Arc<AtomicBool>,
    chunks: &mut [Chunk],
    stage: &'static str,
    detail: &'static str,
) -> Result<()> {
    let engine = state.require_engine()?;
    const BATCH: usize = 32;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let n_texts = texts.len();
    let app2 = app.clone();
    let id2 = id.to_string();
    let cancel2 = cancel.clone();
    let vectors: Vec<Vec<f32>> = tauri::async_runtime::spawn_blocking(move || {
        let mut vectors = Vec::with_capacity(n_texts);
        for batch in texts.chunks(BATCH) {
            check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
            let guard = engine.lock().unwrap();
            let engine = guard.as_ref().ok_or_else(|| anyhow!("engine not loaded"))?;
            vectors.extend(engine.embed_texts(batch, false)?);
            drop(guard);
            emit_stage(&app2, &id2, stage, (vectors.len() as f32 / n_texts as f32).min(0.95), detail);
        }
        Ok::<_, anyhow::Error>(vectors)
    })
    .await??;
    for (chunk, vec) in chunks.iter_mut().zip(vectors) {
        chunk.vector = vec;
    }
    Ok(())
}

/// Analyses added after the original pipeline. A ready video missing any of
/// them can be updated in place with [`enrich`].
/// A version suffix re-runs an analysis whose method improved.
pub const ANALYSES: [&str; 4] = ["voices", "onscreen:3", "scenes:4", "spellings"];
/// Frame descriptions written with the prompt that leaves on-screen text to OCR.
/// Version 3: Qwen2.5-VL with the instruction that asks for people, clothing,
/// setting and objects and leaves wording to OCR.
/// Version 4: keyframes spread across time (no bursts) and descriptions that
/// end on a whole sentence. Older descriptions also get new keyframes.
const SCENE_PROMPT_VERSION: &str = "4";
fn scene_descriptions_current(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join("descriptions.version")).is_ok_and(|v| v.lines().next().map(str::trim) == Some(SCENE_PROMPT_VERSION))
}
pub fn needs_enrichment(meta: &VideoMeta) -> bool {
    ANALYSES.iter().any(|a| !meta.analysis.iter().any(|done| done == a))
}

/// Replace one source kind in the saved index. Reads the index fresh so a
/// reviewed observation written during a long analysis is not lost.
fn merge_index(app: &AppHandle, meta: &mut VideoMeta, sources: &[&str], replacement: Vec<Chunk>) -> Result<()> {
    if !store::meta_exists(app, &meta.id) {
        return Err(anyhow!("cancelled"));
    }
    let mut index = store::load_index(app, &meta.id)?;
    index.retain(|c| !sources.contains(&c.source.as_str()));
    index.extend(replacement);
    index.sort_by(|a, b| a.start.total_cmp(&b.start));
    store::save_index(app, &meta.id, &index)?;
    meta.chunk_count = index.len();
    Ok(())
}

/// Voice separation and on-screen text for a video whose transcript, frames
/// and index exist. Each analysis is recorded in `meta.analysis` when it
/// completes; a failure leaves the video usable and the analysis missing, so
/// it can be retried. Returns an error only when cancelled.
pub async fn enrich(app: &AppHandle, state: &Arc<AppState>, meta: &mut VideoMeta, cancel: &Arc<AtomicBool>) -> Result<()> {
    let id = meta.id.clone();
    let dir = store::video_dir(app, &id)?;
    let video = meta.video_path.clone().map(PathBuf::from).filter(|p| p.is_file())
        .ok_or_else(|| anyhow!("The original video file is missing"))?;
    let cancelled = |e: &anyhow::Error| cancel.load(Ordering::Relaxed) || e.to_string() == "cancelled";

    if !meta.analysis.iter().any(|a| a == "voices") {
        let result = async {
            let cues = store::load_cues(app, &id)?;
            let mut paths = Vec::new();
            for model in ["segmentation", "voiceprint"] {
                let spec = models::spec(model).unwrap();
                if !models::is_downloaded(app, spec) {
                    emit_stage(app, &id, "voices", 0.0, format!("Downloading {}…", spec.label));
                    models::download(app, model).await?;
                }
                paths.push(models::model_path(app, spec)?);
            }
            emit_stage(app, &id, "voices", 0.0, "Separating voices…");
            let record = voices::diarize(&paths[0], &paths[1], &video, &dir, cancel, |p| {
                emit_stage(app, &id, "voices", p * 0.9, "Separating voices…");
            })
            .await?;
            let assigned = voices::assign(&cues, &record.turns);
            let mut speech = vtt::chunk_cues_by_voice(&cues, &assigned, 40.0, 600);
            embed_chunks(app, state, &id, cancel, &mut speech, "voices", "Indexing speech by voice…").await?;
            merge_index(app, meta, &["speech"], speech)?;
            let found = record.voices.iter().map(|v| format!("{} {}", v.voice, rag::format_ts(v.seconds))).collect::<Vec<_>>().join(", ");
            Ok::<_, anyhow::Error>(found)
        }
        .await;
        match result {
            Ok(found) => {
                meta.analysis.push("voices".into());
                persist(app, meta)?;
                emit_stage(app, &id, "voices", 1.0, format!("Voices separated: {found}"));
            }
            Err(e) if cancelled(&e) => return Err(anyhow!("cancelled")),
            Err(e) => emit_stage(app, &id, "voices", 0.0, format!("Voice separation unavailable: {e}")),
        }
    }

    if !meta.analysis.iter().any(|a| a == "onscreen:3") {
        let result = async {
            let bin = app.path().app_data_dir()?.join("bin");
            emit_stage(app, &id, "onscreen", 0.0, "Reading on-screen text…");
            let record = onscreen::read(&bin, &video, &dir, meta.duration_secs, cancel, |p, detail| {
                emit_stage(app, &id, "onscreen", p * 0.9, detail);
            })
            .await?;
            // Each passage starts with its kind ("[Email]") when the layout shows it.
            let mut chunks: Vec<Chunk> = record
                .screens
                .iter()
                .map(|s| Chunk { start: s.start, end: s.end, text: onscreen::passage_text(s), source: "ocr".into(), vector: Vec::new(), cues: Vec::new() })
                .collect();
            embed_chunks(app, state, &id, cancel, &mut chunks, "onscreen", "Indexing on-screen text…").await?;
            let count = chunks.len();
            merge_index(app, meta, &["ocr"], chunks)?;
            Ok::<_, anyhow::Error>(count)
        }
        .await;
        match result {
            Ok(count) => {
                meta.analysis.retain(|a| !a.starts_with("onscreen"));
                meta.analysis.push("onscreen:3".into());
                persist(app, meta)?;
                emit_stage(app, &id, "onscreen", 1.0, format!("{count} screens of on-screen text indexed"));
            }
            Err(e) if cancelled(&e) => return Err(anyhow!("cancelled")),
            Err(e) => emit_stage(app, &id, "onscreen", 0.0, format!("On-screen text unavailable: {e}")),
        }
    }

    // Spelling suggestions: caption words that sound like names shown on
    // screen, kept only when Whisper writes the on-screen spelling.
    if !meta.analysis.iter().any(|a| a == "spellings") && meta.analysis.iter().any(|a| a == "onscreen:3") {
        let result = async {
            let cues = store::load_cues(app, &id)?;
            let record = onscreen::load(&dir).ok_or_else(|| anyhow!("On-screen text is missing"))?;
            let screens: Vec<(f64, f64, String)> = record.screens.iter().map(|s| (s.start, s.end, s.text.clone())).collect();
            let candidates = spelling::candidates(&cues, &screens);
            emit_stage(app, &id, "spellings", 0.1, format!("Checking {} possible caption misspellings by listening…", candidates.len()));
            let spec = models::spec("whisper").unwrap();
            if !candidates.is_empty() && !models::is_downloaded(app, spec) {
                emit_stage(app, &id, "spellings", 0.1, format!("Downloading {}…", spec.label));
                models::download(app, "whisper").await?;
            }
            let whisper_path = models::model_path(app, spec)?.to_string_lossy().into_owned();
            let ffmpeg = ytdlp::find_ffmpeg().ok_or_else(|| anyhow!("ffmpeg not found"))?;
            let whisper = state.whisper.clone();
            let cancel2 = cancel.clone();
            let video2 = video.clone();
            let counts: HashMap<String, usize> = cues.iter().flat_map(|c| c.text.split(|ch: char| !ch.is_alphabetic()).map(str::to_lowercase).collect::<Vec<_>>()).fold(HashMap::new(), |mut m, w| { *m.entry(w).or_default() += 1; m });
            let record = tauri::async_runtime::spawn_blocking(move || {
                if candidates.is_empty() {
                    return Ok(spelling::SpellingRecord { method: spelling::METHOD.into(), ..Default::default() });
                }
                let mut guard = whisper.lock().unwrap();
                if guard.as_ref().map(|w| w.model_path != whisper_path).unwrap_or(true) {
                    *guard = Some(WhisperEngine::load(&whisper_path)?);
                }
                let engine = guard.as_ref().expect("whisper loaded above");
                let clip = |start: f64, length: f64| -> Result<Vec<f32>> {
                    let output = std::process::Command::new(&ffmpeg)
                        .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{start:.3}"), "-t", &format!("{length:.3}"), "-i"])
                        .arg(&video2)
                        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
                        .output()?;
                    if !output.status.success() {
                        return Err(anyhow!("ffmpeg could not cut an audio clip"));
                    }
                    Ok(output.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
                };
                spelling::confirm(engine, &candidates, cancel2, clip, |w| counts.get(w).copied().unwrap_or(0))
            })
            .await??;
            let path = dir.join("spellings.json");
            let temp = path.with_extension("json.part");
            std::fs::write(&temp, serde_json::to_vec_pretty(&record)?)?;
            std::fs::rename(temp, path)?;
            Ok::<_, anyhow::Error>(record)
        }
        .await;
        match result {
            Ok(record) => {
                meta.analysis.push("spellings".into());
                persist(app, meta)?;
                let found = record.suggestions.iter().map(|s| format!("{} → {}", s.original, s.suggested)).collect::<Vec<_>>().join(", ");
                emit_stage(app, &id, "spellings", 1.0, if found.is_empty() { "No caption misspellings confirmed".to_string() } else { format!("Suggested spellings for the archive: {found}") });
            }
            Err(e) if cancelled(&e) => return Err(anyhow!("cancelled")),
            Err(e) => emit_stage(app, &id, "spellings", 0.0, format!("Spelling suggestions unavailable: {e}")),
        }
    }

    // Scene descriptions: re-describe with the prompt that leaves wording to
    // OCR when needed, then give keyframes that are screens of text a factual
    // label from OCR of that exact frame instead of the model's guess.
    if !meta.analysis.iter().any(|a| a == "scenes:4") && meta.analysis.iter().any(|a| a == "onscreen:3") {
        let result = async {
            if !scene_descriptions_current(&dir) {
                emit_stage(app, &id, "scenes", 0.0, "Choosing keyframes across the video…");
                let frames = extract_keyframes(app, &id, &video, &dir.join("frames"), meta.duration_secs).await?;
                meta.frame_count = frames.len();
                persist(app, meta)?;
                let _ = std::fs::remove_file(dir.join("descriptions.json"));
                describe_frames(app, state, meta, cancel, &frames).await?;
            }
            let descriptions = store::load_descriptions(app, &id)?;
            emit_stage(app, &id, "scenes", 0.5, "Checking which keyframes are screens of text…");
            let bin = app.path().app_data_dir()?.join("bin");
            let times: Vec<f64> = descriptions.iter().map(|d| d.start).collect();
            let keyframes = onscreen::read_keyframes(&bin, &video, &dir, &times).await?;
            let record = onscreen::load(&dir).ok_or_else(|| anyhow!("On-screen text is missing"))?;
            let overlays = onscreen::overlays(&record.samples, record.interval_secs);
            let key = |t: f64| (t * 1000.0).round() as i64;
            let mut chunks: Vec<Chunk> = descriptions
                .into_iter()
                .map(|mut d| {
                    if let Some(label) = keyframes.iter().find(|k| key(k.ts) == key(d.start)).and_then(|k| onscreen::screen_frame_label(k, &overlays)) {
                        d.text = label;
                        d.source = "frame".into();
                    }
                    d
                })
                .collect();
            // Reuse embeddings for unchanged passages; embed only new text.
            let index = store::load_index(app, &id)?;
            let mut fresh: Vec<Chunk> = chunks.iter().filter(|c| !index.iter().any(|i| i.start == c.start && i.text == c.text)).cloned().collect();
            embed_chunks(app, state, &id, cancel, &mut fresh, "scenes", "Indexing scene descriptions…").await?;
            for c in &mut chunks {
                c.vector = index.iter().chain(&fresh).find(|i| i.start == c.start && i.text == c.text).map(|i| i.vector.clone()).unwrap_or_default();
            }
            let screens = chunks.iter().filter(|c| c.source == "frame").count();
            merge_index(app, meta, &["vision", "frame"], chunks)?;
            Ok::<_, anyhow::Error>(screens)
        }
        .await;
        if state.cancels.lock().unwrap().len() <= 1 {
            if let Ok(mut guard) = state.engine.lock() {
                if let Some(engine) = guard.as_mut() {
                    engine.unload_vision();
                }
            }
        }
        match result {
            Ok(screens) => {
                meta.analysis.retain(|a| !a.starts_with("scenes"));
                meta.analysis.push("scenes:4".into());
                persist(app, meta)?;
                emit_stage(app, &id, "scenes", 1.0, format!("Scene descriptions updated; {screens} keyframes are screens of text, labeled from OCR"));
            }
            Err(e) if cancelled(&e) => return Err(anyhow!("cancelled")),
            Err(e) => emit_stage(app, &id, "scenes", 0.0, format!("Scene descriptions unavailable: {e}")),
        }
    }
    Ok(())
}

/// Run [`enrich`] on a ready video without taking it out of use. Progress is
/// reported as ingest events; a final "ready" event refreshes its passages.
pub async fn update_analysis(app: AppHandle, state: Arc<AppState>, mut meta: VideoMeta) {
    let id = meta.id.clone();
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut cancels = state.cancels.lock().unwrap();
        if cancels.contains_key(&id) {
            emit_global_error(&app, format!("\"{}\" is already being processed.", meta.title));
            return;
        }
        cancels.insert(id.clone(), cancel.clone());
    }
    let result = enrich(&app, &state, &mut meta, &cancel).await;
    state.cancels.lock().unwrap().remove(&id);
    if let Ok(fresh) = store::load_meta(&app, &id) {
        let detail = match result {
            Ok(()) if needs_enrichment(&fresh) => "Some analyses are unavailable. See the messages above.",
            Ok(()) => "Voices, on-screen text and scene descriptions are up to date.",
            Err(_) => "Analysis update cancelled. Completed work is kept.",
        };
        emit(&app, IngestProgress { video_id: id, stage: "ready".into(), progress: 1.0, detail: detail.into(), meta: Some(fresh) });
    }
}

/// Chunk the transcript, embed everything, save the index.
async fn build_index(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &mut VideoMeta,
    cancel: &Arc<AtomicBool>,
    cues: &[Cue],
    vision_chunks: Vec<Chunk>,
) -> Result<Vec<Chunk>> {
    let id = meta.id.clone();
    emit_stage(app, &id, "index", 0.05, "Embedding transcript and descriptions…");

    let mut chunks = vtt::chunk_cues(cues, 40.0, 600);
    chunks.extend(vision_chunks);
    if chunks.is_empty() {
        return Err(anyhow!("nothing to index: the video produced no transcript or descriptions"));
    }

    embed_chunks(app, state, &id, cancel, &mut chunks, "index", "Embedding transcript and descriptions…").await?;

    if !store::meta_exists(app, &id) {
        return Err(anyhow!("cancelled"));
    }
    store::save_index(app, &id, &chunks)?;
    meta.chunk_count = chunks.len();
    persist(app, meta)?;
    emit_stage(app, &id, "index", 1.0, format!("Indexed {} segments", chunks.len()));
    Ok(chunks)
}

/// Generate the whole-video overview + chapters with the chat model.
/// Short transcripts get one pass; long ones are map-reduced. Returns None
/// when there's nothing to summarize.
async fn summarize(
    app: &AppHandle,
    state: &Arc<AppState>,
    meta: &VideoMeta,
    cancel: &Arc<AtomicBool>,
    chunks: &[Chunk],
) -> Result<Option<String>> {
    let id = meta.id.clone();
    let lines = rag::summary_digest(chunks);
    if lines.is_empty() {
        return Ok(None);
    }

    emit_stage(app, &id, "summarize", 0.05, "Summarizing the whole video…");
    let engine = state.require_engine()?;
    let meta2 = meta.clone();
    let cancel2 = cancel.clone();
    let app2 = app.clone();
    let id2 = id.clone();

    let summary = tauri::async_runtime::spawn_blocking(move || {
        let chat = |prompt: &str, max_tokens: usize| -> Result<String> {
            let guard = engine.lock().unwrap();
            let engine = guard.as_ref().ok_or_else(|| anyhow!("engine not loaded"))?;
            engine.chat(prompt, max_tokens, &cancel2, &mut |_| {})
        };

        let digest = lines.join("\n");
        // Single pass when the digest fits the chat context comfortably;
        // map-reduce otherwise.
        const SINGLE_PASS_MAX: usize = 12_000;
        let material = if digest.len() <= SINGLE_PASS_MAX {
            digest
        } else {
            let parts = rag::split_digest(&lines, 10_000, 6);
            let n_parts = parts.len();
            let mut bullets = String::new();
            for (i, part) in parts.iter().enumerate() {
                check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
                let out = chat(&rag::build_map_prompt(&meta2, part), 256)?;
                bullets.push_str(&out);
                bullets.push('\n');
                emit_stage(
                    &app2,
                    &id2,
                    "summarize",
                    0.1 + 0.7 * ((i + 1) as f32 / n_parts as f32),
                    "Summarizing the whole video…",
                );
            }
            bullets
        };

        check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
        let summary = chat(&rag::build_final_summary_prompt(&meta2, &material), 400)?;
        check_cancel(&cancel2).map_err(|_| anyhow!("cancelled"))?;
        Ok::<_, anyhow::Error>(summary)
    })
    .await??;

    let summary = summary.trim().to_string();
    Ok((!summary.is_empty()).then_some(summary))
}
