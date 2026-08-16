//! ffmpeg helpers: keyframe extraction and audio decoding for Whisper.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tokio::process::Command;

use crate::ytdlp::find_ffmpeg;

fn ffmpeg_cmd() -> Result<Command> {
    let ffmpeg = find_ffmpeg().ok_or_else(|| {
        anyhow!("ffmpeg not found: install it with `brew install ffmpeg` and restart SmolTalk")
    })?;
    let mut cmd = Command::new(ffmpeg);
    cmd.arg("-hide_banner").arg("-loglevel").arg("error").arg("-y");
    Ok(cmd)
}

/// Hard cap on frames per video: each one costs a slow VLM pass.
const MAX_FRAMES: usize = 60;
/// Scene-change score above which a frame is kept (0..1, ffmpeg `scene`).
const SCENE_THRESHOLD: f64 = 0.24;

/// Extract keyframes at scene changes, scaled to 512px wide JPEGs.
/// Returns (timestamp_secs, path) pairs sorted by timestamp.
///
/// A fixed-interval grid wastes VLM passes describing 48 near-identical
/// talking-head frames and misses everything between samples in fast-cut
/// videos. Instead, select frames where ffmpeg's scene score spikes, plus a
/// gap-fill frame when nothing changed for a while (so static videos still
/// get sparse coverage). Falls back to fixed-interval sampling if the scene
/// pass produces nothing usable.
pub async fn extract_frames(
    video: &Path,
    frames_dir: &Path,
    duration_secs: f64,
) -> Result<Vec<(f64, PathBuf)>> {
    let duration = if duration_secs > 0.0 { duration_secs } else { 600.0 };
    match extract_frames_scene(video, frames_dir, duration).await {
        Ok(frames) if !frames.is_empty() => Ok(frames),
        _ => {
            let interval = (duration / 48.0).clamp(4.0, 30.0);
            extract_frames_interval(video, frames_dir, interval).await
        }
    }
}

async fn extract_frames_scene(
    video: &Path,
    frames_dir: &Path,
    duration_secs: f64,
) -> Result<Vec<(f64, PathBuf)>> {
    std::fs::create_dir_all(frames_dir)?;
    clear_jpgs(frames_dir)?;
    // Static stretches still get one frame per `gap` seconds.
    let gap = (duration_secs / 40.0).clamp(15.0, 90.0);
    let pattern = frames_dir.join("f_%05d.jpg");
    let output = ffmpeg_cmd()?
        // showinfo logs at info level; override ffmpeg_cmd's -loglevel error
        // (the last -loglevel wins).
        .arg("-loglevel")
        .arg("info")
        .arg("-i")
        .arg(video)
        .arg("-vf")
        .arg(format!(
            "select='gt(scene,{SCENE_THRESHOLD})+isnan(prev_selected_t)+gte(t-prev_selected_t,{gap:.2})',\
             showinfo,scale=512:-2"
        ))
        .arg("-fps_mode")
        .arg("vfr")
        .arg("-q:v")
        .arg("4")
        .arg(&pattern)
        .output()
        .await?;
    if !output.status.success() {
        return Err(anyhow!("ffmpeg scene extraction failed"));
    }

    // showinfo prints one line per selected frame, in output order:
    // "[Parsed_showinfo_1 @ 0x...] n:   0 ... pts_time:12.345 ..."
    let stderr = String::from_utf8_lossy(&output.stderr);
    let timestamps: Vec<f64> = stderr
        .lines()
        .filter(|l| l.contains("Parsed_showinfo"))
        .filter_map(|l| {
            let rest = l.split("pts_time:").nth(1)?;
            rest.split_whitespace().next()?.parse::<f64>().ok()
        })
        .collect();

    let mut files: Vec<PathBuf> = std::fs::read_dir(frames_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "jpg").unwrap_or(false))
        .collect();
    files.sort();
    if files.len() != timestamps.len() {
        // Can't trust the pairing; let the caller fall back.
        return Err(anyhow!(
            "frame/timestamp mismatch ({} files, {} timestamps)",
            files.len(),
            timestamps.len()
        ));
    }

    let mut frames: Vec<(f64, PathBuf)> = timestamps.into_iter().zip(files).collect();
    if frames.len() > MAX_FRAMES {
        frames = thin_frames(frames, MAX_FRAMES);
    }
    Ok(frames)
}

/// Keep `keep` frames spread evenly across the list; delete the rest from disk.
fn thin_frames(frames: Vec<(f64, PathBuf)>, keep: usize) -> Vec<(f64, PathBuf)> {
    let n = frames.len();
    let mut kept = Vec::with_capacity(keep);
    let mut next_pick = 0usize;
    for (i, frame) in frames.into_iter().enumerate() {
        // Even spread: pick indices round(k * (n-1) / (keep-1)).
        if kept.len() < keep && i == next_pick {
            kept.push(frame);
            next_pick = if kept.len() < keep {
                (kept.len() * (n - 1) + (keep - 1) / 2) / (keep - 1)
            } else {
                usize::MAX
            };
        } else {
            let _ = std::fs::remove_file(&frame.1);
        }
    }
    kept
}

fn clear_jpgs(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir)?.flatten() {
        let p = entry.path();
        if p.extension().map(|e| e == "jpg").unwrap_or(false) {
            let _ = std::fs::remove_file(&p);
        }
    }
    Ok(())
}

/// Fixed-interval fallback sampling.
async fn extract_frames_interval(
    video: &Path,
    frames_dir: &Path,
    interval: f64,
) -> Result<Vec<(f64, PathBuf)>> {
    std::fs::create_dir_all(frames_dir)?;
    clear_jpgs(frames_dir)?;
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

/// Video duration in seconds via ffprobe (sibling of ffmpeg), for local files.
pub async fn probe_duration(video: &Path) -> Result<f64> {
    let ffmpeg = find_ffmpeg()
        .ok_or_else(|| anyhow!("ffmpeg not found: install it with `brew install ffmpeg`"))?;
    let ffprobe = ffmpeg.with_file_name("ffprobe");
    if !ffprobe.exists() {
        return Err(anyhow!("ffprobe not found next to ffmpeg"));
    }
    let output = Command::new(ffprobe)
        .args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"])
        .arg(video)
        .output()
        .await?;
    if !output.status.success() {
        return Err(anyhow!("could not read the video file (is it a valid video?)"));
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|_| anyhow!("could not determine video duration"))
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
