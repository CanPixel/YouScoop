//! Model registry + downloader (Hugging Face -> app data dir) with resume and progress events.

use std::path::PathBuf;

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
}

pub const MODELS: [ModelSpec; 4] = [
    ModelSpec {
        id: "vlm",
        label: "SmolVLM2 2.2B Instruct",
        detail: "Vision-language model — describes frames and answers questions (Q4_K_M)",
        url: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        file: "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        bytes: 1_112_602_656,
    },
    ModelSpec {
        id: "mmproj",
        label: "SmolVLM2 vision projector",
        detail: "Multimodal projector — connects the vision encoder to the LLM (Q8_0)",
        url: "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        file: "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        bytes: 592_523_200,
    },
    ModelSpec {
        id: "embed",
        label: "BGE-small embeddings",
        detail: "Sentence embeddings powering semantic search over the video",
        url: "https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
        file: "bge-small-en-v1.5-q8_0.gguf",
        bytes: 36_806_944,
    },
    ModelSpec {
        id: "whisper",
        label: "Whisper base",
        detail: "Local transcription fallback when a video has no captions (Q5_1)",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        file: "ggml-base-q5_1.bin",
        bytes: 59_707_625,
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

/// Download a model with HTTP range resume. Emits `model-progress` events.
pub async fn download(app: &AppHandle, id: &str) -> Result<()> {
    let spec = spec(id).ok_or_else(|| anyhow!("unknown model id: {id}"))?;
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
        // Corrupt partial larger than target — start over.
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
                error: Some(format!("incomplete download ({got} of {total} bytes) — retry to resume")),
            },
        );
        return Err(anyhow!("incomplete download"));
    }
    std::fs::rename(&part_path, &final_path)?;
    emit_progress(
        app,
        &ModelProgress { id: id.into(), received: total, total, done: true, error: None },
    );
    Ok(())
}
