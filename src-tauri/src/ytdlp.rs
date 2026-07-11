//! yt-dlp sidecar wrapper: metadata, video download (with progress), captions.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde::Deserialize;
use tauri::AppHandle;
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub id: String,
    pub title: String,
    pub channel: String,
    pub duration_secs: f64,
}

#[derive(Deserialize)]
struct RawInfo {
    id: String,
    title: Option<String>,
    channel: Option<String>,
    uploader: Option<String>,
    duration: Option<f64>,
}

/// Locate ffmpeg: GUI apps don't inherit the shell PATH on macOS.
pub fn find_ffmpeg() -> Option<PathBuf> {
    let candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ];
    for c in candidates {
        if Path::new(c).exists() {
            return Some(PathBuf::from(c));
        }
    }
    // Fall back to PATH lookup (works when launched from a terminal).
    which("ffmpeg")
}

fn which(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(bin))
            .find(|p| p.exists())
    })
}

fn base_args(url: &str) -> Vec<String> {
    vec![
        "--no-playlist".into(),
        "--no-warnings".into(),
        "--force-ipv4".into(),
        url.into(),
    ]
}

pub async fn fetch_info(app: &AppHandle, url: &str) -> Result<VideoInfo> {
    let mut args = vec!["-J".to_string(), "--flat-playlist".to_string()];
    args.extend(base_args(url));
    let output = app
        .shell()
        .sidecar("yt-dlp")
        .map_err(|e| anyhow!("yt-dlp sidecar not found: {e}"))?
        .args(args)
        .output()
        .await
        .map_err(|e| anyhow!("failed to run yt-dlp: {e}"))?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(short_ytdlp_error(&err)));
    }
    let raw: RawInfo = serde_json::from_slice(&output.stdout)
        .map_err(|e| anyhow!("could not parse video metadata: {e}"))?;
    Ok(VideoInfo {
        id: raw.id,
        title: raw.title.unwrap_or_else(|| "Untitled video".into()),
        channel: raw.channel.or(raw.uploader).unwrap_or_else(|| "Unknown channel".into()),
        duration_secs: raw.duration.unwrap_or(0.0),
    })
}

fn short_ytdlp_error(stderr: &str) -> String {
    // yt-dlp errors are long; surface the last ERROR: line.
    stderr
        .lines()
        .filter(|l| l.contains("ERROR"))
        .last()
        .map(|l| l.trim().trim_start_matches("ERROR:").trim().to_string())
        .unwrap_or_else(|| "yt-dlp failed".into())
}

/// Download the video as mp4 (<=720p) to `dest`. Calls `on_progress(0..=1)`.
pub async fn download_video(
    app: &AppHandle,
    url: &str,
    dest: &Path,
    mut on_progress: impl FnMut(f32),
) -> Result<()> {
    let mut args: Vec<String> = vec![
        "-f".into(),
        "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[ext=mp4]/b".into(),
        "--merge-output-format".into(),
        "mp4".into(),
        "--newline".into(),
        "-o".into(),
        dest.to_string_lossy().into_owned(),
    ];
    if let Some(ffmpeg) = find_ffmpeg() {
        args.push("--ffmpeg-location".into());
        args.push(ffmpeg.to_string_lossy().into_owned());
    }
    args.extend(base_args(url));

    let (mut rx, _child) = app
        .shell()
        .sidecar("yt-dlp")
        .map_err(|e| anyhow!("yt-dlp sidecar not found: {e}"))?
        .args(args)
        .spawn()
        .map_err(|e| anyhow!("failed to spawn yt-dlp: {e}"))?;

    let mut stderr_tail = String::new();
    let mut code: Option<i32> = None;
    while let Some(event) = rx.recv().await {
        match event {
            CommandEvent::Stdout(line) => {
                let line = String::from_utf8_lossy(&line);
                if let Some(pct) = parse_download_percent(&line) {
                    on_progress(pct);
                }
            }
            CommandEvent::Stderr(line) => {
                let line = String::from_utf8_lossy(&line);
                if line.contains("ERROR") {
                    stderr_tail = line.trim().to_string();
                }
            }
            CommandEvent::Terminated(payload) => {
                code = payload.code;
            }
            _ => {}
        }
    }
    if code != Some(0) {
        return Err(anyhow!(if stderr_tail.is_empty() {
            "video download failed".to_string()
        } else {
            short_ytdlp_error(&stderr_tail)
        }));
    }
    if !dest.exists() {
        return Err(anyhow!("video file missing after download"));
    }
    Ok(())
}

fn parse_download_percent(line: &str) -> Option<f32> {
    // "[download]  42.7% of  123.45MiB at ..."
    let line = line.trim();
    if !line.starts_with("[download]") {
        return None;
    }
    let pct = line.split_whitespace().nth(1)?;
    let pct = pct.strip_suffix('%')?;
    pct.parse::<f32>().ok().map(|p| (p / 100.0).clamp(0.0, 1.0))
}

/// Try to fetch English captions (manual first, then auto). Returns the VTT text if found.
pub async fn fetch_captions(app: &AppHandle, url: &str, work_dir: &Path) -> Result<Option<String>> {
    let out_tpl = work_dir.join("subs.%(ext)s");
    let mut args: Vec<String> = vec![
        "--skip-download".into(),
        "--write-subs".into(),
        "--write-auto-subs".into(),
        "--sub-langs".into(),
        "en.*,en,en-orig".into(),
        "--sub-format".into(),
        "vtt/best".into(),
        "--convert-subs".into(),
        "vtt".into(),
        "-o".into(),
        out_tpl.to_string_lossy().into_owned(),
    ];
    if let Some(ffmpeg) = find_ffmpeg() {
        args.push("--ffmpeg-location".into());
        args.push(ffmpeg.to_string_lossy().into_owned());
    }
    args.extend(base_args(url));

    let output = app
        .shell()
        .sidecar("yt-dlp")
        .map_err(|e| anyhow!("yt-dlp sidecar not found: {e}"))?
        .args(args)
        .output()
        .await
        .map_err(|e| anyhow!("failed to run yt-dlp: {e}"))?;

    if !output.status.success() {
        // Captions are best-effort; treat failure as "no captions".
        return Ok(None);
    }

    // Find any subs.*.vtt file produced.
    let mut vtt_path: Option<PathBuf> = None;
    for entry in std::fs::read_dir(work_dir)?.flatten() {
        let p = entry.path();
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.starts_with("subs.") && name.ends_with(".vtt") {
            vtt_path = Some(p);
            break;
        }
    }
    match vtt_path {
        Some(p) => {
            let content = std::fs::read_to_string(&p)?;
            let _ = std::fs::remove_file(&p);
            Ok(Some(content))
        }
        None => Ok(None),
    }
}
