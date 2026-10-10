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
    /// Checked after download when present.
    pub sha256: Option<&'static str>,
}

pub const MODELS: [ModelSpec; 9] = [
    ModelSpec {
        id: "vlm",
        label: "SmolVLM2 2.2B Instruct",
        detail: "Vision-language model: describes video frames (Q4_K_M)",
        url: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        file: "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        bytes: 1_112_602_656,
        sha256: None,
    },
    ModelSpec {
        id: "mmproj",
        label: "SmolVLM2 vision projector",
        detail: "Multimodal projector: connects the vision encoder to the LLM (Q8_0)",
        url: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        file: "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        bytes: 592_523_200,
        sha256: None,
    },
    ModelSpec {
        id: "chat",
        label: "Qwen3 4B Instruct",
        detail: "Chat model: answers questions grounded in the video index (Q4_K_M)",
        url: "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        file: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        bytes: 2_497_281_120,
        sha256: None,
    },
    ModelSpec {
        id: "embed",
        label: "BGE-small embeddings",
        detail: "Sentence embeddings: powering semantic search over the video",
        url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        file: "bge-small-en-v1.5-q8_0.gguf",
        bytes: 36_806_944,
        sha256: None,
    },
    ModelSpec {
        id: "whisper",
        label: "Whisper large-v3-turbo",
        detail: "Local transcription fallback when a video has no captions (Q5_0)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        file: "ggml-large-v3-turbo-q5_0.bin",
        bytes: 574_041_195,
        sha256: None,
    },
    ModelSpec {
        id: "segmentation",
        label: "Pyannote segmentation 3.0",
        detail: "Finds where each voice speaks, for separating speakers (MIT, CNRS)",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx",
        file: "pyannote-segmentation-3.0.onnx",
        bytes: 5_992_913,
        sha256: Some("220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"),
    },
    ModelSpec {
        id: "voiceprint",
        label: "WeSpeaker ResNet34-LM",
        detail: "Speaker embeddings that group passages by voice (CC BY 4.0, WeSpeaker, VoxCeleb)",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
        file: "wespeaker_en_voxceleb_resnet34_LM.onnx",
        bytes: 26_530_550,
        sha256: Some("e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"),
    },
    // Frame descriptions: on the test source this invented about 3 details in
    // 20 frames against 11 for SmolVLM2 (experiments/model-trial/results/frame-models.json).
    ModelSpec {
        id: "scene-vlm",
        label: "Qwen2.5-VL 3B Instruct",
        detail: "Describes video frames more accurately than SmolVLM2 (Q4_K_M, Apache-2.0)",
        url: "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        file: "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
        bytes: 1_929_901_056,
        sha256: Some("d02fe9b69ad8cadbbd228e387667af66612c44bed29ffc8eb1e7caf9ac486c12"),
    },
    ModelSpec {
        id: "scene-mmproj",
        label: "Qwen2.5-VL 3B vision projector",
        detail: "Image encoder for Qwen2.5-VL (Q8_0)",
        url: "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        file: "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        bytes: 844_757_728,
        sha256: Some("980c9b2f78c04e6cff93d277ada09e768394f112d75db3b4e9dea8a69f9fb904"),
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
        .map(|p| p.exists() && std::fs::metadata(&p).map(|m| m.len() == spec.bytes).unwrap_or(false))
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
pub fn any_whisper_fallback(app: &AppHandle) -> Option<PathBuf> {
    let current = spec("whisper")?.file;
    let dir = models_dir(app).ok()?;
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.starts_with("ggml-") && name.ends_with(".bin") && name != current {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if size > 1_000_000 && best.as_ref().map(|(s, _)| size > *s).unwrap_or(true) {
                best = Some((size, path));
            }
        }
    }
    best.map(|(_, p)| p)
}

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
        in_flight().lock().unwrap().insert(id.to_string()).then(|| Self(id.to_string()))
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
        emit_progress(
            app,
            &ModelProgress { id: id.into(), received: spec.bytes, total: spec.bytes, done: true, error: None },
        );
        return Ok(());
    }

    let part_path = final_path.with_extension("part");
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
                &ModelProgress { id: id.into(), received, total, done: false, error: None },
            );
            on_progress(received, total);
            last_emit = std::time::Instant::now();
        }
    }
    file.flush().await?;
    drop(file);

    let got = std::fs::metadata(&part_path)?.len();
    let digest_ok = match spec.sha256 {
        Some(expected) if got == total => crate::digest::file_digest(&part_path)?.0 == expected,
        _ => true,
    };
    if !digest_ok {
        let _ = std::fs::remove_file(&part_path);
        let error = "downloaded model failed its checksum: retry to download it again".to_string();
        emit_progress(app, &ModelProgress { id: id.into(), received: 0, total, done: false, error: Some(error.clone()) });
        return Err(anyhow!(error));
    }
    if got != total {
        emit_progress(
            app,
            &ModelProgress {
                id: id.into(),
                received: got,
                total,
                done: false,
                error: Some(format!("incomplete download ({got} of {total} bytes): retry to resume")),
            },
        );
        return Err(anyhow!("incomplete download"));
    }
    std::fs::rename(&part_path, &final_path)?;
    emit_progress(
        app,
        &ModelProgress { id: id.into(), received: total, total, done: true, error: None },
    );
    on_progress(total, total);
    Ok(())
}
