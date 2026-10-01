use crate::state::AppState;
use anyhow::{anyhow, Context, Result};
use inference_coordinator::{Admission, Coordinator};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tauri::{AppHandle, Emitter, Manager};

pub struct NormalScope {
    pub admission: Admission,
    _local: tokio::sync::OwnedRwLockReadGuard<()>,
}
pub fn coordinator(app: &AppHandle) -> Result<Coordinator> {
    let dir = app.path().app_data_dir()?;
    Coordinator::new(
        dir.parent()
            .context("Application support directory is unavailable")?
            .join(".sacrinomicon-inference"),
    )
    .map_err(Into::into)
}
pub async fn normal(
    app: &AppHandle,
    state: &Arc<AppState>,
    cancel: Arc<AtomicBool>,
) -> Result<NormalScope> {
    let _ = app.emit("inference-status", "Waiting for local inference");
    let coordinator = coordinator(app)?;
    let admission =
        tauri::async_runtime::spawn_blocking(move || coordinator.admit(&cancel)).await??;
    let local = state.inference_gate.clone().read_owned().await;
    let _ = app.emit("inference-status", "Preparing local inference");
    Ok(NormalScope {
        admission,
        _local: local,
    })
}
pub fn resident(app: &AppHandle, state: &Arc<AppState>, admission: &Admission) -> Result<()> {
    let mut lease = state
        .memory_resident
        .lock()
        .map_err(|_| anyhow!("Memory state is unavailable"))?;
    if lease.is_none() {
        *lease = Some(coordinator(app)?.resident(admission)?);
    }
    Ok(())
}
pub fn unload(state: &Arc<AppState>) -> Result<()> {
    *state
        .engine
        .lock()
        .map_err(|_| anyhow!("Engine state unavailable"))? = None;
    *state
        .whisper
        .lock()
        .map_err(|_| anyhow!("Speech state unavailable"))? = None;
    *state
        .memory_resident
        .lock()
        .map_err(|_| anyhow!("Memory state unavailable"))? = None;
    Ok(())
}
pub fn watch(app: AppHandle, state: Arc<AppState>) -> Result<()> {
    let coordinator = coordinator(&app)?;
    tauri::async_runtime::spawn(async move {
        let mut timer = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            timer.tick().await;
            if coordinator.heavy_requested().unwrap_or(false) {
                if let Ok(_idle) = state.inference_gate.clone().try_write_owned() {
                    let _ = unload(&state);
                }
            }
        }
    });
    Ok(())
}
pub struct WorkerJob(Arc<AppState>);
impl WorkerJob {
    pub fn begin(state: &Arc<AppState>) -> Result<Self> {
        if state.worker_busy.swap(true, Ordering::SeqCst) {
            return Err(anyhow!("A desktop investigation is already running"));
        }
        state.worker_cancel.store(false, Ordering::Relaxed);
        Ok(Self(state.clone()))
    }
}
impl Drop for WorkerJob {
    fn drop(&mut self) {
        self.0.worker_busy.store(false, Ordering::SeqCst);
    }
}
