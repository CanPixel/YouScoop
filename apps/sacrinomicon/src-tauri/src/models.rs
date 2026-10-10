//! Model registry + downloader (Hugging Face -> app data dir) with resume and progress events.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;

pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub detail: &'static str,
    pub url: &'static str,
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

pub const MODELS: [ModelSpec; 4] = [
    ModelSpec {
        id: "voice", label: "Whisper small Q5_1", detail: "Optional offline voice transcription, loaded only in voice mode",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/3e251d613ae82c33b1dcca9f0406c697c3f9d083/ggml-small-q5_1.bin",
        file: "ggml-small-q5_1.bin", bytes: 190_085_487, sha256: "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
    },
    ModelSpec {
        id: "chat",
        label: "Qwen3 4B Instruct",
        detail: "Chat model: answers questions grounded in preserved evidence (Q4_K_M)",
        url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        bytes: 2_497_281_120,
        sha256: "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
    },
    ModelSpec {
        id: "embed",
        label: "BGE-small embeddings",
        detail: "Sentence embeddings: powering local archive retrieval",
        url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/d32f8c040ea3b516330eeb75b72bcc2d3a780ab7/bge-small-en-v1.5-q8_0.gguf",
        file: "bge-small-en-v1.5-q8_0.gguf",
        bytes: 36_806_944,
        sha256: "ec38e8da142596baa913124ae50550de284b6916bf59577ef2f0cb9660c2f514",
    },
    ModelSpec {
        id: "speech",
        label: "Kokoro voice",
        detail: "The Oracle's spoken voice: offline neural speech, unpacked on first use (macOS)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2",
        file: "kokoro-en-v0_19.tar.bz2",
        bytes: 319_625_534,
        sha256: "912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7",
    },
];

pub fn spec(id: &str) -> Option<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id)
}

pub fn models_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_data_dir()?.join("models");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn model_path(app: &AppHandle, spec: &ModelSpec) -> Result<PathBuf> {
    Ok(models_dir(app)?.join(spec.file))
}

pub fn is_downloaded(app: &AppHandle, spec: &ModelSpec) -> bool {
    model_path(app, spec)
        .map(|p| {
            p.exists()
                && std::fs::metadata(&p)
                    .map(|m| m.len() == spec.bytes)
                    .unwrap_or(false)
        })
        .unwrap_or(false)
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelProgress {
    pub id: String,
    pub received: u64,
    pub total: u64,
    pub done: bool,
    pub error: Option<String>,
}

fn emit_progress(app: &AppHandle, p: &ModelProgress) {
    let _ = app.emit("model-progress", p);
}

/// Best available Whisper model already on disk that is NOT the registry one
/// e.g. an older `ggml-base` left over from a previous app version. Used as an
/// offline fallback when the current registry model can't be downloaded.
/// Models currently being written to their `.part` file. Two concurrent
/// downloads of the same model (e.g. SetupGate's "download all" racing the
/// pipeline's on-demand Whisper fetch) would interleave appends and corrupt it.
fn in_flight() -> &'static Mutex<HashSet<String>> {
    static IN_FLIGHT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    IN_FLIGHT.get_or_init(|| Mutex::new(HashSet::new()))
}

struct FlightGuard(String);

impl FlightGuard {
    /// Claim the download slot for `id`; None if another download holds it.
    fn try_claim(id: &str) -> Option<Self> {
        in_flight()
            .lock()
            .unwrap()
            .insert(id.to_string())
            .then(|| Self(id.to_string()))
    }
}

impl Drop for FlightGuard {
    fn drop(&mut self) {
        in_flight().lock().unwrap().remove(&self.0);
    }
}

/// Download a model with HTTP range resume. Emits `model-progress` events.
pub async fn download(app: &AppHandle, id: &str) -> Result<()> {
    download_with_progress(app, id, |_, _| {}).await
}

/// Like [`download`], but also reports `(received, total)` to a callback so
/// callers (e.g. the ingest pipeline) can surface progress in their own UI.
/// If the same model is already downloading, waits for that download instead
/// of starting a second one.
pub async fn download_with_progress(
    app: &AppHandle,
    id: &str,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<()> {
    let spec = spec(id).ok_or_else(|| anyhow!("unknown model id: {id}"))?;
    let guard = loop {
        match FlightGuard::try_claim(id) {
            Some(g) => break g,
            None => {
                // Another task is downloading this model: report its progress
                // from the .part file until it finishes, then re-check.
                let part = model_path(app, spec)?.with_extension("part");
                on_progress(part.metadata().map(|m| m.len()).unwrap_or(0), spec.bytes);
                tokio::time::sleep(std::time::Duration::from_millis(400)).await;
                if is_downloaded(app, spec) {
                    on_progress(spec.bytes, spec.bytes);
                    return Ok(());
                }
            }
        }
    };
    let result = do_download(app, spec, id, &mut on_progress).await;
    drop(guard);
    result
}

async fn do_download(
    app: &AppHandle,
    spec: &ModelSpec,
    id: &str,
    on_progress: &mut impl FnMut(u64, u64),
) -> Result<()> {
    let final_path = model_path(app, spec)?;
    if final_path.exists() && std::fs::metadata(&final_path)?.len() == spec.bytes {
        verify_file(&final_path, spec)?;
        emit_progress(
            app,
            &ModelProgress {
                id: id.into(),
                received: spec.bytes,
                total: spec.bytes,
                done: true,
                error: None,
            },
        );
        return Ok(());
    }

    let part_path = final_path.with_extension("part");
    #[cfg(target_os = "macos")]
    if let Some(parent) = app.path().app_data_dir()?.parent() {
        let cached = parent.join("com.canpixel.youscoop").join("models").join(spec.file);
        if cached.is_file() && verify_file(&cached,spec).is_ok() {
            let _=std::fs::remove_file(&part_path);
            if std::fs::hard_link(&cached,&part_path).is_err() { std::fs::copy(&cached,&part_path)?; }
            verify_file(&part_path,spec)?;
            std::fs::rename(&part_path,&final_path)?;
            on_progress(spec.bytes,spec.bytes);
            emit_progress(app,&ModelProgress{id:id.into(),received:spec.bytes,total:spec.bytes,done:true,error:None});
            return Ok(());
        }
    }
    let mut offset: u64 = part_path.metadata().map(|m| m.len()).unwrap_or(0);
    if offset >= spec.bytes {
        // Corrupt partial larger than target: start over.
        let _ = std::fs::remove_file(&part_path);
        offset = 0;
    }

    let client = reqwest::Client::new();
    let mut req = client.get(spec.url);
    if offset > 0 {
        req = req.header("Range", format!("bytes={offset}-"));
    }
    let resp = req.send().await.context("request failed")?;
    if !resp.status().is_success() {
        return Err(anyhow!("download failed with HTTP {}", resp.status()));
    }
    // If the server ignored our Range header, start from scratch.
    if offset > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        offset = 0;
    }

    let total = spec.bytes;
    if offset == 0 {
        // Truncate any stale partial file when starting from scratch.
        let _ = std::fs::remove_file(&part_path);
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part_path)
        .await?;

    let mut received = offset;
    let mut last_emit = std::time::Instant::now();
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("stream error")?;
        file.write_all(&chunk).await?;
        received += chunk.len() as u64;
        if last_emit.elapsed().as_millis() > 150 {
            emit_progress(
                app,
                &ModelProgress {
                    id: id.into(),
                    received,
                    total,
                    done: false,
                    error: None,
                },
            );
            on_progress(received, total);
            last_emit = std::time::Instant::now();
        }
    }
    file.flush().await?;
    drop(file);

    let got = std::fs::metadata(&part_path)?.len();
    if got != total {
        emit_progress(
            app,
            &ModelProgress {
                id: id.into(),
                received: got,
                total,
                done: false,
                error: Some(format!(
                    "incomplete download ({got} of {total} bytes): retry to resume"
                )),
            },
        );
        return Err(anyhow!("incomplete download"));
    }
    verify_file(&part_path, spec)?;
    std::fs::rename(&part_path, &final_path)?;
    emit_progress(
        app,
        &ModelProgress {
            id: id.into(),
            received: total,
            total,
            done: true,
            error: None,
        },
    );
    on_progress(total, total);
    Ok(())
}

fn verify_file(path: &std::path::Path,spec:&ModelSpec)->Result<()> {
    use sha2::{Digest,Sha256};
    use std::io::Read;
    if path.metadata()?.len()!=spec.bytes { anyhow::bail!("Incomplete model file"); }
    let mut input=std::fs::File::open(path)?;let mut digest=Sha256::new();let mut buf=vec![0u8;1024*1024];
    loop { let n=input.read(&mut buf)?;if n==0 {break;}digest.update(&buf[..n]); }
    if format!("{:x}",digest.finalize())!=spec.sha256 { anyhow::bail!("Model checksum mismatch; remove the incomplete download and retry"); }
    Ok(())
}
