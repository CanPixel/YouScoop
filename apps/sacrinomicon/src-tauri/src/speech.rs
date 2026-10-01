use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use tauri::{Manager, State};
pub struct Speech {
    cancel: Mutex<Option<Arc<AtomicBool>>>,
    #[cfg(target_os = "android")]
    handle: tauri::plugin::PluginHandle<tauri::Wry>,
}
pub fn init() -> tauri::plugin::TauriPlugin<tauri::Wry> {
    tauri::plugin::Builder::new("sacri-voice")
        .setup(|app, api| {
            #[cfg(target_os = "android")]
            let handle = api.register_android_plugin("com.canpixel.sacrinomicon", "VoicePlugin")?;
            #[cfg(not(target_os = "android"))]
            let _ = api;
            app.manage(Arc::new(Speech {
                cancel: Mutex::new(None),
                #[cfg(target_os = "android")]
                handle,
            }));
            Ok(())
        })
        .build()
}
pub async fn save_export(app: &tauri::AppHandle, path: &std::path::Path) -> Result<bool, String> {
    #[cfg(target_os = "android")]
    {
        let state = app.state::<Arc<Speech>>().inner().clone();
        let path = path.to_string_lossy().to_string();
        let result = tauri::async_runtime::spawn_blocking(move || {
            state
                .handle
                .run_mobile_plugin::<serde_json::Value>(
                    "saveExport",
                    serde_json::json!({"path":path}),
                )
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())??;
        return Ok(result["saved"].as_bool().unwrap_or(false));
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (app, path);
        Ok(true)
    }
}
#[tauri::command]
pub async fn speak_text(state: State<'_, Arc<Speech>>, text: String) -> Result<(), String> {
    if text.trim().is_empty() || text.len() > 16000 {
        return Err("Speech text must be below 16,000 characters".into());
    }
    let flag = Arc::new(AtomicBool::new(false));
    {
        let mut current = state.cancel.lock().map_err(|e| e.to_string())?;
        if let Some(previous) = current.replace(flag.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
    }
    #[cfg(target_os = "android")]
    {
        let state = state.inner().clone();
        return tauri::async_runtime::spawn_blocking(move || {
            state
                .handle
                .run_mobile_plugin::<serde_json::Value>("speak", serde_json::json!({"text":text}))
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())?;
    }
    #[cfg(target_os = "macos")]
    {
        use tokio::io::AsyncWriteExt;
        let mut child = tokio::process::Command::new("/usr/bin/say")
            .stdin(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| e.to_string())?;
        if let Some(mut input) = child.stdin.take() {
            input
                .write_all(text.as_bytes())
                .await
                .map_err(|e| e.to_string())?;
        }
        loop {
            if flag.load(Ordering::Relaxed) {
                child.kill().await.map_err(|e| e.to_string())?;
                let _ = child.wait().await;
                return Ok(());
            }
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                return if status.success() {
                    Ok(())
                } else {
                    Err("The local system voice failed".into())
                };
            }
            tokio::time::sleep(std::time::Duration::from_millis(70)).await;
        }
    }
    #[cfg(not(any(target_os = "android", target_os = "macos")))]
    Err("Native offline speech is currently supported on macOS and Android".into())
}
#[tauri::command]
pub fn stop_speech(state: State<'_, Arc<Speech>>) -> Result<(), String> {
    if let Some(flag) = state.cancel.lock().map_err(|e| e.to_string())?.as_ref() {
        flag.store(true, Ordering::Relaxed);
    }
    #[cfg(target_os = "android")]
    {
        state
            .handle
            .run_mobile_plugin::<serde_json::Value>("stop", serde_json::json!({}))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
