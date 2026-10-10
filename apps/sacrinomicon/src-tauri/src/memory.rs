use crate::{err, AppState, CmdResult};
use inference_coordinator::{Admission, Coordinator};
use std::sync::Arc;
use tauri::{AppHandle, Manager};
pub fn coordinator(app: &AppHandle) -> CmdResult<Coordinator> {
    let dir = app.path().app_data_dir().map_err(err)?;
    // Android's app sandbox has no writable shared parent. Its archive has
    // no heavyweight video worker, so coordinate within its own private root.
    #[cfg(target_os = "android")]
    let locks = dir.join(".inference");
    #[cfg(not(target_os = "android"))]
    let locks = dir.parent()
        .ok_or("Application support directory is unavailable")?
        .join(".sacrinomicon-inference");
    Coordinator::new(locks)
    .map_err(err)
}
pub fn resident(
    state: &Arc<AppState>,
    coordinator: &Coordinator,
    admission: &Admission,
) -> CmdResult<()> {
    let mut lease = state.memory_resident.lock().map_err(err)?;
    if lease.is_none() {
        *lease = Some(coordinator.resident(admission).map_err(err)?);
    }
    Ok(())
}
pub fn release_if_empty(state: &Arc<AppState>) -> CmdResult<()> {
    let engine = state.engine.lock().map_err(err)?;
    let speech = state.speech.lock().map_err(err)?;
    let voice = state.voice.lock().map_err(err)?;
    if engine.is_none() && speech.is_none() && voice.is_none() {
        *state.memory_resident.lock().map_err(err)? = None;
    }
    Ok(())
}
pub fn yield_if_idle(state: &Arc<AppState>) -> bool {
    let Ok(mut engine) = state.engine.try_lock() else {
        return false;
    };
    let Ok(mut speech) = state.speech.try_lock() else {
        return false;
    };
    let Ok(mut voice) = state.voice.try_lock() else {
        return false;
    };
    let Ok(mut lease) = state.memory_resident.lock() else {
        return false;
    };
    *engine = None;
    *speech = None;
    *voice = None;
    *lease = None;
    true
}
pub fn watch(app: AppHandle, state: Arc<AppState>) -> CmdResult<()> {
    let coordinator = coordinator(&app)?;
    tauri::async_runtime::spawn(async move {
        let mut timer = tokio::time::interval(std::time::Duration::from_millis(200));
        loop {
            timer.tick().await;
            if !coordinator.heavy_requested().unwrap_or(false) {
                continue;
            }
            // A request waiting for admission may already mark itself busy.
            // Only the actual inference mutexes determine whether models are idle.
            yield_if_idle(&state);
        }
    });
    Ok(())
}
