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

pub fn video_dir(app: &AppHandle, id: &str) -> Result<PathBuf> {
    // Defend against path traversal via a hostile "id".
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(anyhow!("invalid video id"));
    }
    let dir = videos_dir(app)?.join(id);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn save_meta(app: &AppHandle, meta: &VideoMeta) -> Result<()> {
    let path = video_dir(app, &meta.id)?.join("meta.json");
    std::fs::write(path, serde_json::to_vec_pretty(meta)?)?;
    Ok(())
}

pub fn load_meta(app: &AppHandle, id: &str) -> Result<VideoMeta> {
    let path = video_dir(app, id)?.join("meta.json");
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn save_index(app: &AppHandle, id: &str, chunks: &[Chunk]) -> Result<()> {
    let path = video_dir(app, id)?.join("index.json");
    std::fs::write(path, serde_json::to_vec(chunks)?)?;
    Ok(())
}

pub fn load_index(app: &AppHandle, id: &str) -> Result<Vec<Chunk>> {
    let path = video_dir(app, id)?.join("index.json");
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
    out.sort_by(|a, b| b.added_at.cmp(&a.added_at));
    Ok(out)
}

/// Called once at startup: videos left "processing" by a previous session are orphaned.
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
}

pub fn delete_video(app: &AppHandle, id: &str) -> Result<()> {
    let dir = video_dir(app, id)?;
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
