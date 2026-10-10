//! Persistent, pinned desktop installation. No inference weights load in setup.
use crate::{
    desktop_worker::{self, WorkerConfig, WorkerStatus},
    state::AppState,
    CmdResult,
};
use anyhow::{anyhow, bail, Context, Result};
use inference_coordinator::{Coordinator, Exclusive};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{atomic::Ordering, Arc},
};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

const REQUIREMENTS: &str = include_str!("../../runtime/requirements.lock.txt");
const INSTALLER: &str = include_str!("../../runtime/setup.py");
const CATALOG: &str = include_str!("../../runtime/models.json");
const PYTHON_VERSION: &str = "3.13.15";

pub fn catalog() -> Value {
    serde_json::from_str(CATALOG).expect("Bundled model catalog")
}
pub fn root(app: &AppHandle) -> Result<PathBuf> {
    Ok(app.path().app_data_dir()?.join("desktop-runtime"))
}
fn environment(app: &AppHandle) -> Result<PathBuf> {
    let digest = knowledge_core::stable_id("runtime", REQUIREMENTS);
    Ok(root(app)?.join(format!("py31315-{}", &digest[8..24])))
}
pub fn defaults(app: &AppHandle) -> Result<WorkerConfig> {
    let root = root(app)?;
    let catalog = catalog();
    let model = |id: &str| {
        let path = root.join("models").join(format!(
            "{id}-{}",
            catalog["models"][id]["revision"].as_str().unwrap()
        ));
        if desktop_worker::valid_snapshot(&path, id) {
            path.to_string_lossy().into_owned()
        } else {
            String::new()
        }
    };
    Ok(WorkerConfig {
        python: environment(app)?
            .join("bin/python")
            .to_string_lossy()
            .into(),
        gemma_model: model("gemma"),
    })
}
pub fn uv_path() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).map(|p| p.join("uv")).collect())
        .unwrap_or_default();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".local/bin/uv"));
    }
    candidates.extend([
        PathBuf::from("/opt/homebrew/bin/uv"),
        PathBuf::from("/usr/local/bin/uv"),
    ]);
    candidates.into_iter().find(|p| p.is_file())
}
pub fn is_ready(app: &AppHandle, cfg: &WorkerConfig) -> bool {
    if !Path::new(&cfg.python).is_file() {
        return false;
    }
    root(app)
        .ok()
        .and_then(|root| std::fs::read(root.join("runtime-ready.json")).ok())
        .and_then(|data| serde_json::from_slice::<Value>(&data).ok())
        .is_some_and(|receipt| {
            receipt["configuredPython"] == cfg.python
                && receipt["requirementsId"] == knowledge_core::stable_id("runtime", REQUIREMENTS)
        })
}
fn status(app: &AppHandle, detail: &str) {
    let _ = app.emit(
        "runtime-progress",
        json!({"kind":"status", "detail":detail}),
    );
}
fn atomic_json(path: &Path, data: &Value) -> Result<()> {
    std::fs::create_dir_all(
        path.parent()
            .context("Installation directory unavailable")?,
    )?;
    let temporary = path.with_extension("json.part");
    std::fs::write(&temporary, serde_json::to_vec_pretty(data)?)?;
    std::fs::File::open(&temporary)?.sync_all()?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
pub(crate) async fn setup_lease(app: &AppHandle, state: &Arc<AppState>) -> Result<Exclusive> {
    let coordinator = Coordinator::new(root(app)?.join(".setup-lock"))?;
    let cancel = state.worker_cancel.clone();
    tauri::async_runtime::spawn_blocking(move || {
        coordinator.request_heavy(&cancel)?.reserve(&cancel)
    })
    .await?
    .map_err(Into::into)
}
/// Run one owned child, draining diagnostics, preserving setup leases on crash.
async fn execute(
    app: &AppHandle,
    state: &Arc<AppState>,
    mut command: Command,
    request: Option<Value>,
    lease: &Exclusive,
) -> Result<Option<Value>> {
    command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    lease.inherit_into(command.as_std_mut())?;
    let mut child = command
        .spawn()
        .context("Desktop setup process could not start")?;
    if let Some(mut request) = request.clone() {
        request["ownerPid"] = json!(std::process::id());
        let mut input = child.stdin.take().context("Setup input unavailable")?;
        input
            .write_all(serde_json::to_string(&request)?.as_bytes())
            .await?;
        input.write_all(b"\n").await?;
    } else {
        drop(child.stdin.take());
    }
    let mut lines =
        BufReader::new(child.stdout.take().context("Setup output unavailable")?).lines();
    let mut errors = BufReader::new(
        child
            .stderr
            .take()
            .context("Setup diagnostics unavailable")?,
    )
    .lines();
    let diagnostics = tokio::spawn(async move {
        let mut tail = String::new();
        while let Ok(Some(line)) = errors.next_line().await {
            tail.push_str(&line);
            tail.push('\n');
            if tail.len() > 12000 {
                let cut = tail
                    .char_indices()
                    .find(|(i, _)| *i >= tail.len() - 12000)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                tail.drain(..cut);
            }
        }
        tail
    });
    let mut result = None;
    let mut failure = None;
    let mut timer = tokio::time::interval(std::time::Duration::from_millis(100));
    loop {
        tokio::select! {
            line = lines.next_line() => match line {
                Ok(Some(line)) if request.is_some() => match serde_json::from_str::<Value>(&line) {
                    Ok(value) => {
                        if value["kind"] == "result" { result = Some(value.clone()); }
                        if value["kind"] == "error" { failure = Some(value["message"].as_str().unwrap_or("Setup failed").to_owned()); }
                        let _ = app.emit("runtime-progress", value);
                    },
                    Err(_) => { failure = Some("Invalid desktop setup event".into()); child.kill().await?; break; }
                },
                Ok(Some(_)) => {},
                Ok(None) => break,
                Err(error) => { failure = Some(error.to_string()); child.kill().await?; break; }
            },
            _ = timer.tick() => if state.worker_cancel.load(Ordering::Relaxed) {
                child.kill().await?; failure = Some("Cancelled".into()); break;
            }
        }
    }
    let code = child.wait().await?;
    let tail = diagnostics.await.unwrap_or_default();
    if let Some(failure) = failure {
        bail!("{failure}");
    }
    if !code.success() {
        bail!("Desktop setup failed: {}", tail.trim());
    }
    if request.is_some() && result.is_none() {
        bail!("Desktop setup returned no result");
    }
    Ok(result)
}
fn script(app: &AppHandle) -> Result<PathBuf> {
    let path = root(app)?.join("setup.py");
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, INSTALLER)?;
    Ok(path)
}
async fn probe(
    app: &AppHandle,
    state: &Arc<AppState>,
    python: &str,
    lease: &Exclusive,
) -> Result<Value> {
    let mut command = Command::new(python);
    command.arg("-u").arg(script(app)?);
    let result = execute(app, state, command, Some(json!({"task":"probe"})), lease)
        .await?
        .context("Runtime probe missing")?;
    Ok(result["probe"].clone())
}
fn remember_probe(app: &AppHandle, python: &str, probe: Value) -> Result<()> {
    atomic_json(
        &root(app)?.join("runtime-ready.json"),
        &json!({"configuredPython":python, "requirementsId":knowledge_core::stable_id("runtime", REQUIREMENTS), "probe":probe, "verifiedAt":knowledge_core::now()}),
    )
}
pub async fn configure(app: &AppHandle, state: &Arc<AppState>, cfg: &WorkerConfig) -> Result<()> {
    let _job = crate::memory::WorkerJob::begin(state)?;
    if !cfg.gemma_model.is_empty() && !desktop_worker::valid_snapshot(Path::new(&cfg.gemma_model), "gemma") {
        bail!("Select a complete compatible gemma snapshot directory");
    }
    let lease = setup_lease(app, state).await?;
    status(app, "Checking the configured Python runtime");
    let receipt = probe(app, state, &cfg.python, &lease).await?;
    remember_probe(app, &cfg.python, receipt)?;
    desktop_worker::save_config(app, cfg)
}
#[tauri::command]
pub async fn setup_desktop_runtime(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> CmdResult<WorkerStatus> {
    install_runtime(&app, state.inner())
        .await
        .map_err(crate::err_str)?;
    desktop_worker::worker_status(app)
}
async fn install_runtime(app: &AppHandle, state: &Arc<AppState>) -> Result<()> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        bail!("Desktop Gemma setup currently supports Apple Silicon Macs");
    }
    let _job = crate::memory::WorkerJob::begin(state)?;
    let os = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()?;
    let version = String::from_utf8_lossy(&os.stdout);
    if version
        .trim()
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
        < 26
    {
        bail!("The tested MLX runtime requires macOS 26 or newer");
    }
    let lease = setup_lease(app, state).await?;
    let uv = uv_path().context("Install uv from astral.sh/uv, then retry desktop runtime setup")?;
    let root = root(app)?;
    let env = environment(app)?;
    std::fs::create_dir_all(&root)?;
    let requirements = root.join("requirements.lock.txt");
    std::fs::write(&requirements, REQUIREMENTS)?;
    let python = env.join("bin/python");
    let uv_command = || {
        let mut command = Command::new(&uv);
        command
            .env("UV_PYTHON_INSTALL_DIR", root.join("python"))
            .env("UV_CACHE_DIR", root.join("cache"))
            .env("UV_NO_CONFIG", "1");
        command
    };
    if !python.is_file() {
        status(app, "Installing Python 3.13.15 in Application Support");
        let mut command = uv_command();
        command
            .args([
                "venv",
                "--no-project",
                "--clear",
                "--managed-python",
                "--python",
                PYTHON_VERSION,
            ])
            .arg(&env);
        execute(app, state, command, None, &lease).await?;
    }
    status(app, "Installing and checking pinned inference dependencies");
    let mut command = uv_command();
    command
        .args([
            "pip",
            "sync",
            "--require-hashes",
            "--reinstall",
            "--no-build",
            "--strict",
            "--link-mode",
            "clone",
            "--python",
        ])
        .arg(&python)
        .arg(&requirements);
    execute(app, state, command, None, &lease).await?;
    let receipt = probe(app, state, &python.to_string_lossy(), &lease).await?;
    remember_probe(app, &python.to_string_lossy(), receipt)?;
    let mut cfg = desktop_worker::config(app)?;
    cfg.python = python.to_string_lossy().into();
    desktop_worker::save_config(app, &cfg)?;
    status(app, "Persistent desktop runtime is ready");
    Ok(())
}

/// Actual native setup functions, with no windows or GUI interaction.
#[cfg(feature = "diagnostics")]
pub fn verify_desktop_setup(task: &str, source: Option<String>) -> Result<Value> {
    let mut context = crate::app_context();
    context.config_mut().app.windows.clear();
    let app = tauri::Builder::default().build(context)?;
    let state = Arc::new(AppState::new());
    tauri::async_runtime::block_on(async {
        match task {
            "runtime" => install_runtime(app.handle(), &state).await?,
            "gemma" => install_model(app.handle(), &state, task, source).await?,
            "status" => {}
            "gemma-source" => {
                let id = source.context("Gemma diagnostic requires a video ID")?;
                let answer=desktop_worker::ask_source(app.handle(),&state,&id,"Reconstruct the main argument from the supplied passages. Distinguish source assertions, quoted responses and interpretation. Cite supporting passage IDs.".into(),vec![]).await.map_err(anyhow::Error::msg)?;
                return Ok(serde_json::to_value(answer)?);
            }
            "worker" => {
                let path = source.context("Worker diagnostic requires a request JSON file")?;
                let mut request: Value = serde_json::from_slice(&std::fs::read(path)?)?;
                let cfg = desktop_worker::config(app.handle())?;
                request["model"] = match request["task"].as_str() {
                    Some("gemma") => json!(cfg.gemma_model),
                    _ => bail!("Choose a Gemma worker diagnostic"),
                };
                let _job = crate::memory::WorkerJob::begin(&state)?;
                let events = desktop_worker::run(app.handle(), &state, request, |_| {}).await?;
                return Ok(json!({"events": events}));
            }
            _ => bail!("Choose runtime, gemma, or status"),
        }
        Ok(serde_json::to_value(
            desktop_worker::worker_status(app.handle().clone()).map_err(anyhow::Error::msg)?,
        )?)
    })
}
#[tauri::command]
pub async fn install_desktop_model(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    model: String,
    source: Option<String>,
) -> CmdResult<WorkerStatus> {
    install_model(&app, state.inner(), &model, source)
        .await
        .map_err(crate::err_str)?;
    desktop_worker::worker_status(app)
}
async fn install_model(
    app: &AppHandle,
    state: &Arc<AppState>,
    model: &str,
    source: Option<String>,
) -> Result<()> {
    if model != "gemma" {
        bail!("Unknown desktop model");
    }
    let _job = crate::memory::WorkerJob::begin(state)?;
    let cfg = desktop_worker::config(app)?;
    if !is_ready(app, &cfg) {
        bail!("Install or configure the desktop Python runtime first");
    }
    let lease = setup_lease(app, state).await?;
    let mut command = Command::new(&cfg.python);
    command
        .arg("-u")
        .arg(script(app)?)
        .env("HF_HUB_OFFLINE", "0")
        .env("TRANSFORMERS_OFFLINE", "0");
    let result = execute(app,state,command,Some(json!({"task":"model","root":root(app)?,"catalog":catalog(),"model":model,"source":source})),&lease).await?.context("Model installation result missing")?;
    let path = result["modelPath"]
        .as_str()
        .context("Installed model path missing")?;
    if !desktop_worker::valid_snapshot(Path::new(path), model) {
        return Err(anyhow!("Installed model snapshot is incomplete"));
    }
    let mut cfg = desktop_worker::config(app)?;
    cfg.gemma_model = path.into();
    desktop_worker::save_config(app, &cfg)?;
    status(app, &format!("{model} installed and verified"));
    Ok(())
}
