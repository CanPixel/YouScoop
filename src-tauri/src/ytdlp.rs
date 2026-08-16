//! yt-dlp wrapper: metadata, video download (with progress), captions, and
//! self-update. Uses an updated binary from the app data dir when present,
//! falling back to the bundled sidecar.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::Deserialize;
use tauri::{AppHandle, Manager};
use tauri_plugin_shell::process::{Command, CommandEvent};
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

/// Where a self-updated yt-dlp binary lives. The bundled sidecar sits inside
/// the (signed, read-only) app bundle, so updates go to the app data dir.
fn updated_ytdlp_path(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_data_dir()?.join("bin");
    std::fs::create_dir_all(&dir)?;
    Ok(dir.join("yt-dlp"))
}

/// The yt-dlp command to run: the self-updated binary when present, else the
/// bundled sidecar.
fn ytdlp(app: &AppHandle) -> Result<Command> {
    if let Ok(updated) = updated_ytdlp_path(app) {
        if updated.exists() {
            return Ok(app.shell().command(updated));
        }
    }
    app.shell().sidecar("yt-dlp").map_err(|e| anyhow!("yt-dlp sidecar not found: {e}"))
}

const YTDLP_RELEASE_URL: &str =
    "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos";

/// Download the latest yt-dlp release to the app data dir and verify it runs.
/// Returns the new version string. YouTube regularly breaks old yt-dlp
/// releases, and the bundled sidecar can't be replaced in place.
pub async fn self_update(app: &AppHandle) -> Result<String> {
    let dest = updated_ytdlp_path(app)?;
    let part = dest.with_extension("part");

    let resp = reqwest::get(YTDLP_RELEASE_URL).await.context("could not reach GitHub")?;
    if !resp.status().is_success() {
        return Err(anyhow!("downloader update failed with HTTP {}", resp.status()));
    }
    let mut file = tokio::fs::File::create(&part).await?;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk.context("stream error")?).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut file).await?;
    drop(file);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&part, std::fs::Permissions::from_mode(0o755))?;
    }

    // Validate before swapping it in: a broken download must not shadow the
    // working sidecar.
    let output = app
        .shell()
        .command(&part)
        .args(["--version"])
        .output()
        .await
        .map_err(|e| anyhow!("updated yt-dlp does not run: {e}"))?;
    if !output.status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(anyhow!("updated yt-dlp does not run"));
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    std::fs::rename(&part, &dest)?;
    Ok(version)
}

pub async fn fetch_info(app: &AppHandle, url: &str) -> Result<VideoInfo> {
    let mut args = vec!["-J".to_string(), "--flat-playlist".to_string()];
    args.extend(base_args(url));
    let output = ytdlp(app)?
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
    let msg = stderr
        .lines()
        .filter(|l| l.contains("ERROR"))
        .next_back()
        .map(|l| l.trim().trim_start_matches("ERROR:").trim().to_string())
        .unwrap_or_else(|| "yt-dlp failed".into());
    // Extraction failures usually mean YouTube changed something and the
    // shipped yt-dlp is stale. Point at the in-app fix.
    let stale_signals =
        ["Unable to extract", "Sign in to confirm", "HTTP Error 403", "nsig extraction failed"];
    if stale_signals.iter().any(|s| msg.contains(s)) {
        format!("{msg} — this usually means YouTube changed something; try \"Update downloader\" below.")
    } else {
        msg
    }
}

/// Download the video as mp4 (<=720p) to `dest`. Calls `on_progress(0..=1)`.
/// Kills the yt-dlp process as soon as `cancel` flips, instead of letting the
/// full download finish first.
pub async fn download_video(
    app: &AppHandle,
    url: &str,
    dest: &Path,
    cancel: &Arc<AtomicBool>,
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

    let (mut rx, child) = ytdlp(app)?
        .args(args)
        .spawn()
        .map_err(|e| anyhow!("failed to spawn yt-dlp: {e}"))?;
    let mut child = Some(child);

    let mut stderr_tail = String::new();
    let mut code: Option<i32> = None;
    let mut poll = tokio::time::interval(std::time::Duration::from_millis(250));
    loop {
        tokio::select! {
            event = rx.recv() => {
                let Some(event) = event else { break };
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
            _ = poll.tick() => {
                if cancel.load(Ordering::Relaxed) {
                    if let Some(child) = child.take() {
                        let _ = child.kill();
                    }
                    return Err(anyhow!("cancelled"));
                }
            }
        }
    }
    if cancel.load(Ordering::Relaxed) {
        return Err(anyhow!("cancelled"));
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

    let output = ytdlp(app)?
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
