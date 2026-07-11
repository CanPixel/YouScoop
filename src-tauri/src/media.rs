//! ffmpeg helpers: keyframe extraction and audio decoding for Whisper.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tokio::process::Command;

use crate::ytdlp::find_ffmpeg;

fn ffmpeg_cmd() -> Result<Command> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        anyhow!("ffmpeg not found — install it with `brew install ffmpeg` and restart SmolTalk")
    })?;
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-hide_banner").arg("-loglevel").arg("error").arg("-y");
    Ok(cmd)
}

/// Extract frames every `interval` seconds, scaled to 512px wide JPEGs.
/// Returns (timestamp_secs, path) pairs.
pub async fn extract_frames(
    video: &Path,
    frames_dir: &Path,
    interval: f64,
) -> Result<Vec<(f64, PathBuf)>> {
    std::fs::create_dir_all(frames_dir)?;
    let pattern = frames_dir.join("f_%05d.jpg");
    let status = ffmpeg_cmd()?
        .arg("-i")
        .arg(video)
        .arg("-vf")
        .arg(format!("fps=1/{interval:.3},scale=512:-2"))
        .arg("-q:v")
        .arg("4")
        .arg(&pattern)
        .status()
        .await?;
    if !status.success() {
        return Err(anyhow!("ffmpeg frame extraction failed"));
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(frames_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
        .collect();
    files.sort();
    Ok(files
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as f64 * interval, p))
        .collect())
}

/// Extract a single thumbnail at `at_secs`, 640px wide.
pub async fn extract_thumb(video: &Path, dest: &Path, at_secs: f64) -> Result<()> {
    let status = ffmpeg_cmd()?
        .arg("-ss")
        .arg(format!("{at_secs:.2}"))
        .arg("-i")
        .arg(video)
        .arg("-frames:v")
        .arg("1")
        .arg("-vf")
        .arg("scale=640:-2")
        .arg("-q:v")
        .arg("4")
        .arg(dest)
        .status()
        .await?;
    if !status.success() {
        return Err(anyhow!("ffmpeg thumbnail extraction failed"));
    }
    Ok(())
}

/// Decode the audio track to 16 kHz mono f32 PCM (what Whisper expects).
pub async fn extract_audio_f32(video: &Path) -> Result<Vec<f32>> {
    let output = ffmpeg_cmd()?
        .arg("-i")
        .arg(video)
        .arg("-vn")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg("-f")
        .arg("f32le")
        .arg("pipe:1")
        .output()
        .await?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let bytes = output.stdout;
    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(samples)
}
