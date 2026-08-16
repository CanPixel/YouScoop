//! On-disk library: one directory per video with meta.json, index.json, media files.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager};

#[derive(Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VideoMeta {
    pub id: String,
    pub url: String,
    pub title: String,
    pub channel: String,
    pub duration_secs: f64,
    pub added_at: u64,
    /// pending | processing | ready | error | cancelled
    pub status: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub video_path: Option<String>,
    #[serde(default)]
    pub thumb_path: Option<String>,
    #[serde(default)]
    pub has_captions: bool,
    #[serde(default)]
    pub frame_count: usize,
    #[serde(default)]
    pub chunk_count: usize,
    /// Auto-generated whole-video summary + chapter outline. Grounds global
    /// questions ("what is this video about?") that retrieval alone can't answer.
    #[serde(default)]
    pub summary: Option<String>,
}

impl VideoMeta {
    /// Local files are ingested with an empty URL.
    pub fn is_local(&self) -> bool {
        self.url.is_empty()
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Chunk {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// "speech" | "vision"
    pub source: String,
    #[serde(default)]
    pub vector: Vec<f32>,
    /// Fine-grained timestamps of the cues this chunk was merged from, so the
    /// chat prompt can cite the actual sentence instead of the chunk start.
    /// Empty on vision chunks and on indexes built before this field existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cues: Vec<SubCue>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SubCue {
    pub start: f64,
    pub text: String,
}

/// Chunk as exposed to the frontend (no embedding vector).
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkPublic {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub source: String,
}

impl From<&Chunk> for ChunkPublic {
    fn from(c: &Chunk) -> Self {
        Self { start: c.start, end: c.end, text: c.text.clone(), source: c.source.clone() }
    }
}

pub fn videos_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_data_dir()?.join("videos");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn check_id(id: &str) -> Result<()> {
    // Defend against path traversal via a hostile "id".
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(anyhow!("invalid video id"));
    }
    Ok(())
}

pub fn video_dir(app: &AppHandle, id: &str) -> Result<PathBuf> {
    check_id(id)?;
    let dir = videos_dir(app)?.join(id);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// The video's directory path without creating it. Lets callers test for
/// existence without resurrecting a directory that `delete_video` removed.
pub fn video_dir_path(app: &AppHandle, id: &str) -> Result<PathBuf> {
    check_id(id)?;
    Ok(videos_dir(app)?.join(id))
}

/// Whether the video still exists in the library. False after `delete_video`,
/// so an in-flight pipeline knows not to write its state back.
pub fn meta_exists(app: &AppHandle, id: &str) -> bool {
    video_dir_path(app, id).map(|d| d.join("meta.json").exists()).unwrap_or(false)
}

/// Save meta only if the video hasn't been removed meanwhile. Used by the
/// pipeline so a cancelled/failed ingest of a deleted video doesn't recreate
/// its directory.
pub fn save_meta_if_present(app: &AppHandle, meta: &VideoMeta) -> Result<bool> {
    if !meta_exists(app, &meta.id) {
        return Ok(false);
    }
    save_meta(app, meta)?;
    Ok(true)
}

pub fn save_meta(app: &AppHandle, meta: &VideoMeta) -> Result<()> {
    let path = video_dir(app, &meta.id)?.join("meta.json");
    std::fs::write(path, serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}

pub fn load_meta(app: &AppHandle, id: &str) -> Result<VideoMeta> {
    let path = video_dir_path(app, id)?.join("meta.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

// ---- Per-stage pipeline artifacts ----
//
// Each stage persists its output so a retry (or an app restart) resumes where
// it left off instead of re-downloading and re-transcribing everything.

/// Timestamped reference to one extracted keyframe on disk.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameRef {
    pub ts: f64,
    pub file: String,
}

pub fn save_cues(app: &AppHandle, id: &str, cues: &[crate::vtt::Cue]) -> Result<()> {
    let path = video_dir(app, id)?.join("cues.json");
    std::fs::write(path, serde_json::to_vec(cues)?)?;
    Ok(())
}

pub fn load_cues(app: &AppHandle, id: &str) -> Result<Vec<crate::vtt::Cue>> {
    let path = video_dir_path(app, id)?.join("cues.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn save_frame_manifest(app: &AppHandle, id: &str, frames: &[FrameRef]) -> Result<()> {
    let path = video_dir(app, id)?.join("frames.json");
    std::fs::write(path, serde_json::to_vec(frames)?)?;
    Ok(())
}

pub fn load_frame_manifest(app: &AppHandle, id: &str) -> Result<Vec<FrameRef>> {
    let path = video_dir_path(app, id)?.join("frames.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// Vision chunks (frame descriptions) saved incrementally during the describe
/// stage, so a cancelled or crashed ingest doesn't lose completed VLM work.
pub fn save_descriptions(app: &AppHandle, id: &str, chunks: &[Chunk]) -> Result<()> {
    let path = video_dir(app, id)?.join("descriptions.json");
    std::fs::write(path, serde_json::to_vec(chunks)?)?;
    Ok(())
}

pub fn load_descriptions(app: &AppHandle, id: &str) -> Result<Vec<Chunk>> {
    let path = video_dir_path(app, id)?.join("descriptions.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn save_index(app: &AppHandle, id: &str, chunks: &[Chunk]) -> Result<()> {
    let path = video_dir(app, id)?.join("index.json");
    std::fs::write(path, serde_json::to_vec(chunks)?)?;
    Ok(())
}

pub fn load_index(app: &AppHandle, id: &str) -> Result<Vec<Chunk>> {
    let path = video_dir_path(app, id)?.join("index.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn list_videos(app: &AppHandle) -> Result<Vec<VideoMeta>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(videos_dir(app)?)? {
        let entry = entry?;
        let meta_path = entry.path().join("meta.json");
        if meta_path.exists() {
            if let Ok(bytes) = std::fs::read(&meta_path) {
                if let Ok(meta) = serde_json::from_slice::<VideoMeta>(&bytes) {
                    out.push(meta);
                }
            }
        }
    }
    out.sort_by_key(|m| std::cmp::Reverse(m.added_at));
    Ok(out)
}

/// Called once at startup: videos left "processing" by a previous session are
/// orphaned, and directories without a meta.json (e.g. an in-flight artifact
/// write racing a removal) are dead weight.
pub fn cleanup_orphans(app: &AppHandle) {
    if let Ok(videos) = list_videos(app) {
        for mut meta in videos {
            if meta.status == "processing" || meta.status == "pending" {
                meta.status = "error".into();
                meta.error = Some("Processing was interrupted.".into());
                let _ = save_meta(app, &meta);
            }
        }
    }
    if let Ok(entries) = videos_dir(app).and_then(|d| Ok(std::fs::read_dir(d)?)) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && !path.join("meta.json").exists() {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }
}

pub fn delete_video(app: &AppHandle, id: &str) -> Result<()> {
    let dir = video_dir_path(app, id)?;
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    Ok(())
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
