use crate::{state::AppState, store, CmdResult};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerConfig {
    pub python: String,
    pub gemma_model: String,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerStatus {
    pub config: WorkerConfig,
    pub runtime_ready: bool,
    pub uv_available: bool,
    pub cached_gemma: Option<String>,
    pub gemma_ready: bool,
}

fn config_path(app: &AppHandle) -> Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("desktop-runtime.json"))
}
pub(crate) fn config(app: &AppHandle) -> Result<WorkerConfig> {
    let path = config_path(app)?;
    if path.exists() {
        return Ok(serde_json::from_slice(&std::fs::read(path)?)?);
    }
    crate::runtime_setup::defaults(app)
}
pub(crate) fn save_config(app: &AppHandle, cfg: &WorkerConfig) -> Result<()> {
    let path = config_path(app)?;
    std::fs::create_dir_all(
        path.parent()
            .context("Application support directory unavailable")?,
    )?;
    let temporary = path.with_extension("json.part");
    std::fs::write(&temporary, serde_json::to_vec_pretty(cfg)?)?;
    std::fs::File::open(&temporary)?.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
/// Cheap structural readiness; managed installation also verifies full hashes.
pub(crate) fn valid_snapshot(path: &Path, id: &str) -> bool {
    use std::io::Read;
    let validate = || -> Result<()> {
        let cfg: Value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        let catalog = crate::runtime_setup::catalog();
        let expected = catalog["models"][id]["modelType"].as_str().context("Unknown desktop model")?;
        if cfg["model_type"] != expected {
            bail!("Wrong model architecture");
        }
        for file in ["tokenizer.json", "tokenizer_config.json"] {
            if !path.join(file).is_file() {
                bail!("Missing tokenizer");
            }
        }
        let index: Value =
            serde_json::from_slice(&std::fs::read(path.join("model.safetensors.index.json"))?)?;
        let mut files: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for (tensor, file) in index["weight_map"]
            .as_object()
            .context("Weight index missing")?
            .iter()
        {
            let name = file.as_str().context("Invalid shard name")?;
            if Path::new(name).file_name().and_then(|v| v.to_str()) != Some(name)
                || !name.ends_with(".safetensors")
            {
                bail!("Unsafe shard name");
            }
            files.entry(name).or_default().push(tensor);
        }
        if files.is_empty() || files.len() > 32 {
            bail!("Invalid shard count");
        }
        let mut total = 0u64;
        for (file, expected_tensors) in files {
            let mut input = std::fs::File::open(path.join(file))?;
            let size = input.metadata()?.len();
            total += size;
            let mut prefix = [0u8; 8];
            input.read_exact(&mut prefix)?;
            let header = u64::from_le_bytes(prefix);
            if header > 8 * 1024 * 1024 || size < header + 8 {
                bail!("Invalid tensor header");
            }
            let mut bytes = vec![0; header as usize];
            input.read_exact(&mut bytes)?;
            let parsed: Value = serde_json::from_slice(&bytes)?;
            if expected_tensors
                .iter()
                .any(|name| parsed.get(*name).is_none())
            {
                bail!("Indexed tensor missing from shard");
            }
            let mut end = 0;
            for (name, tensor) in parsed.as_object().context("Invalid tensor header")? {
                if name == "__metadata__" {
                    continue;
                }
                let offset = tensor["data_offsets"][1]
                    .as_u64()
                    .context("Invalid tensor offset")?;
                end = end.max(offset);
            }
            if end != size - header - 8 {
                bail!("Incomplete tensor shard");
            }
        }
        if total > 18 * 1024u64.pow(3) {
            bail!("This snapshot exceeds the desktop model budget");
        }
        Ok(())
    };
    validate().is_ok()
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    fn fixture(path: &Path) {
        std::fs::write(
            path.join("config.json"),
            br#"{"model_type":"gemma4"}"#,
        )
        .unwrap();
        std::fs::write(path.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(path.join("tokenizer_config.json"), b"{}").unwrap();
        std::fs::write(
            path.join("model.safetensors.index.json"),
            br#"{"weight_map":{"tensor":"model.safetensors"}}"#,
        )
        .unwrap();
        let header = br#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(path.join("model.safetensors"), bytes).unwrap();
    }
    #[test]
    fn readiness_requires_indexed_tensors_and_complete_shards() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        assert!(valid_snapshot(dir.path(), "gemma"));
        assert!(!valid_snapshot(dir.path(), "unknown"));
        let file = dir.path().join("model.safetensors");
        let mut bytes = std::fs::read(&file).unwrap();
        bytes.pop();
        std::fs::write(file, bytes).unwrap();
        assert!(!valid_snapshot(dir.path(), "gemma"));
        fixture(dir.path());
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"missing":"model.safetensors"}}"#,
        )
        .unwrap();
        assert!(!valid_snapshot(dir.path(), "gemma"));
    }
    #[test]
    fn readiness_rejects_missing_tokenizers_and_paths_outside_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());
        std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();
        assert!(!valid_snapshot(dir.path(), "gemma"));
        fixture(dir.path());
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"tensor":"../model.safetensors"}}"#,
        )
        .unwrap();
        assert!(!valid_snapshot(dir.path(), "gemma"));
    }
}
fn cached_model(cfg: &WorkerConfig) -> Option<String> {
    let id = "gemma";
    let configured = &cfg.gemma_model;
    if !configured.is_empty() && valid_snapshot(Path::new(configured), id) {
        return Some(configured.clone());
    }
    let catalog = crate::runtime_setup::catalog();
    let spec = &catalog["models"][id];
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
    let repo = spec["repository"].as_str()?.replace('/', "--");
    let path = root
        .join("out/model-trial/hf/hub")
        .join(format!("models--{repo}"))
        .join("snapshots")
        .join(spec["revision"].as_str()?);
    valid_snapshot(&path, id).then(|| path.to_string_lossy().into_owned())
}
#[tauri::command]
pub fn worker_status(app: AppHandle) -> CmdResult<WorkerStatus> {
    let config = config(&app).map_err(crate::err_str)?;
    let ready = crate::runtime_setup::is_ready(&app, &config);
    Ok(WorkerStatus {
        runtime_ready: ready,
        uv_available: crate::runtime_setup::uv_path().is_some(),
        cached_gemma: cached_model(&config),
        gemma_ready: ready && valid_snapshot(Path::new(&config.gemma_model), "gemma"),
        config,
    })
}
#[tauri::command]
pub async fn configure_worker(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    config: WorkerConfig,
) -> CmdResult<()> {
    if !Path::new(&config.python).is_file() {
        return Err("Select the Python executable in the installed runtime".into());
    }
    crate::runtime_setup::configure(&app, state.inner(), &config)
        .await
        .map_err(crate::err_str)
}
#[tauri::command]
pub fn stop_worker(state: State<'_, Arc<AppState>>) {
    state.worker_cancel.store(true, Ordering::Relaxed);
}

pub(crate) async fn run(
    app: &AppHandle,
    state: &Arc<AppState>,
    mut request: Value,
    mut event: impl FnMut(&Value),
) -> Result<Vec<Value>> {
    // Serialize dependency repair with the entire worker lifetime, including
    // child ownership if this app exits abruptly. Acquire before reading config.
    let setup_lease = crate::runtime_setup::setup_lease(app, state).await?;
    let cfg = config(app)?;
    let _ = app.emit(
        "worker-progress",
        json!({"kind":"status","detail":"Waiting for current local inference to finish"}),
    );
    let coordinator = crate::memory::coordinator(app)?;
    let cancel = state.worker_cancel.clone();
    let intent =
        tauri::async_runtime::spawn_blocking(move || coordinator.request_heavy(&cancel)).await??;
    let permit = tokio::select! {
        permit=state.inference_gate.clone().write_owned()=>permit,
        _=async {while !state.worker_cancel.load(Ordering::Relaxed){tokio::time::sleep(std::time::Duration::from_millis(40)).await;}}=>bail!("Cancelled"),
    };
    crate::memory::unload(state)?;
    let cancel = state.worker_cancel.clone();
    let _exclusive =
        tauri::async_runtime::spawn_blocking(move || intent.reserve(&cancel)).await??;
    // All normal inference has finished before the write permit is granted.
    let script = app.path().app_data_dir()?.join("runtime-worker.py");
    std::fs::write(&script, include_str!("../../runtime/worker.py"))?;
    request["ownerPid"] = json!(std::process::id());
    let mut command = Command::new(&cfg.python);
    command
        .arg("-u")
        .arg(script)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    setup_lease.inherit_into(command.as_std_mut())?;
    _exclusive.inherit_into(command.as_std_mut())?;
    let mut child = command
        .spawn()
        .context("Desktop runtime could not start. Check its Python path")?;
    let mut stdin = child.stdin.take().context("Worker stdin unavailable")?;
    stdin
        .write_all(serde_json::to_string(&request)?.as_bytes())
        .await?;
    stdin.write_all(b"\n").await?;
    drop(stdin);
    let mut lines =
        BufReader::new(child.stdout.take().context("Worker output unavailable")?).lines();
    let mut stderr = BufReader::new(
        child
            .stderr
            .take()
            .context("Worker diagnostics unavailable")?,
    )
    .lines();
    let diagnostics = tokio::spawn(async move {
        let mut tail = String::new();
        while let Ok(Some(line)) = stderr.next_line().await {
            tail.push_str(&line);
            tail.push('\n');
            if tail.len() > 12000 {
                let split = tail
                    .char_indices()
                    .find(|(i, _)| *i >= tail.len() - 12000)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                tail.drain(..split);
            }
        }
        tail
    });
    let mut events = Vec::new();
    let mut poll = tokio::time::interval(std::time::Duration::from_millis(100));
    let mut error = None;
    loop {
        tokio::select! {
            line=lines.next_line()=>match line {
                Ok(Some(line))=> {
                    match serde_json::from_str::<Value>(&line) {
                        Ok(value)=> {
                            event(&value);
                            let _=app.emit("worker-progress",&value);
                            if value["kind"]=="error" { error=Some(value["message"].as_str().unwrap_or("Model failed").to_string()); }
                            if value["kind"]=="result" { events.push(value); }
                        },
                        Err(_)=> { error=Some("Desktop runtime returned an invalid event".into());child.kill().await?;break; }
                    }
                },
                Ok(None)=>break,
                Err(e)=> {error=Some(e.to_string());child.kill().await?;break;}
            },
            _=poll.tick()=>if state.worker_cancel.load(Ordering::Relaxed) {
                child.kill().await?;error=Some("Cancelled".into());break;
            }
        }
    }
    let status = child.wait().await?;
    let tail = diagnostics.await.unwrap_or_default();
    drop(permit);
    // The worker process has exited, releasing all of its model memory.
    // The next normal request reloads the small models lazily.
    if let Some(error) = error {
        bail!("{error}");
    }
    if !status.success() {
        bail!("Desktop model process failed: {}", tail.trim());
    }
    if !events.iter().any(|e| e["kind"] == "result") {
        bail!("Desktop model returned no result");
    }
    Ok(events)
}

#[tauri::command]
pub async fn ask_gemma(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    video_id: String,
    question: String,
    history: Vec<crate::HistoryTurnDto>,
) -> CmdResult<crate::ChatAnswer> {
    ask_source(&app,state.inner(),&video_id,question,history).await
}
pub(crate) async fn ask_source(app:&AppHandle,state:&Arc<AppState>,video_id:&str,question:String,history:Vec<crate::HistoryTurnDto>)->CmdResult<crate::ChatAnswer> {
    let _job = crate::memory::WorkerJob::begin(state).map_err(crate::err_str)?;
    if question.trim().is_empty() || question.len() > 8000 {
        return Err("Enter a question below 8,000 characters".into());
    }
    let meta = store::load_meta(&app, &video_id).map_err(crate::err_str)?;
    if meta.status != "ready" {
        return Err("Finish processing first".into());
    }
    let chunks = store::load_index(&app, &video_id).map_err(crate::err_str)?;
    let cfg = config(&app).map_err(crate::err_str)?;
    let context = crate::investigation::context(&chunks, &question);
    let sources = context.passages.iter().map(|p| crate::SourceRef {
        start: p.start, end: (p.start + 12.0).min(meta.duration_secs),
        source: p.label.into(), text: p.text.clone(),
    }).collect();
    let turns: Vec<(String, String)> = history.into_iter().map(|t| (t.question, t.answer)).collect();
    let messages = crate::investigation::messages(&meta.title, &meta.channel, &context, &question, &turns);
    let vid = video_id.to_string();
    let handle = app.clone();
    let events = run(
        &app,
        state,
        json!({"task":"gemma","model":cfg.gemma_model,"messages":messages}),
        move |e| {
            if e["kind"] == "token" {
                let _ = handle.emit(
                    "chat-token",
                    crate::ChatToken {
                        video_id: vid.clone(),
                        token: e["text"].as_str().unwrap_or_default().into(),
                    },
                );
            }
        },
    )
    .await
    .map_err(crate::err_str)?;
    let raw = events
        .iter()
        .find(|e| e["kind"] == "result")
        .and_then(|e| e["answer"].as_str())
        .unwrap_or_default();
    let mut answer = crate::investigation::resolve(raw, &context.passages);
    if !context.spoken_complete { answer.push_str("\n\nCoverage: this answer used selected passages from the source, not its entire transcript."); }
    Ok(crate::ChatAnswer { answer, sources })
}
