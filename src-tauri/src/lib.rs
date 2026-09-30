pub mod llm;
mod media;
mod models;
mod pipeline;
pub mod rag;
mod state;
pub mod store;
pub mod stt;
mod vtt;
mod ytdlp;

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use state::AppState;
use store::{ChunkPublic, VideoMeta};

type CmdResult<T> = Result<T, String>;

fn err_str(e: impl std::fmt::Display) -> String {
    e.to_string()
}

// ---------- Setup / models ----------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    id: String,
    label: String,
    detail: String,
    bytes: u64,
    downloaded: bool,
    partial_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupStatus {
    models: Vec<ModelStatus>,
    ffmpeg_ok: bool,
    engine_loaded: bool,
    engine_loading: bool,
}

#[tauri::command]
fn setup_status(app: AppHandle, state: State<'_, Arc<AppState>>) -> CmdResult<SetupStatus> {
    let mut list = Vec::new();
    for spec in &models::MODELS {
        let path = models::model_path(&app, spec).map_err(err_str)?;
        let part = path.with_extension("part");
        list.push(ModelStatus {
            id: spec.id.into(),
            label: spec.label.into(),
            detail: spec.detail.into(),
            bytes: spec.bytes,
            downloaded: models::is_downloaded(&app, spec),
            partial_bytes: part.metadata().map(|m| m.len()).unwrap_or(0),
        });
    }
    Ok(SetupStatus {
        models: list,
        ffmpeg_ok: ytdlp::find_ffmpeg().is_some(),
        engine_loaded: state.engine_loaded(),
        engine_loading: state.engine_loading.load(Ordering::Relaxed),
    })
}

#[tauri::command]
async fn download_model(app: AppHandle, id: String) -> CmdResult<()> {
    models::download(&app, &id).await.map_err(err_str)
}

#[tauri::command]
async fn update_ytdlp(app: AppHandle) -> CmdResult<String> {
    ytdlp::self_update(&app).await.map_err(err_str)
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineStatus {
    loaded: bool,
    loading: bool,
    error: Option<String>,
}

/// Make sure the engine is loaded, starting or waiting for a load as needed.
/// Callers that need inference (adding videos, asking questions) funnel
/// through here, so the UI can open before the multi-GB models are resident.
async fn ensure_engine(app: &AppHandle, state: &Arc<AppState>) -> Result<(), String> {
    loop {
        if state.engine_loaded() {
            return Ok(());
        }
        if state.engine_loading.swap(true, Ordering::SeqCst) {
            // Another load is in flight; wait for it to settle, then re-check.
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            continue;
        }
        // We own the load now.
        return do_load_engine(app, state).await;
    }
}

/// Perform the actual engine load. Caller must own `engine_loading` (true).
async fn do_load_engine(app: &AppHandle, state: &Arc<AppState>) -> Result<(), String> {
    let _ = app.emit("engine-status", &EngineStatus { loaded: false, loading: true, error: None });

    let vlm = models::model_path(app, models::spec("vlm").unwrap()).map_err(err_str)?;
    let mmproj = models::model_path(app, models::spec("mmproj").unwrap()).map_err(err_str)?;
    let chat = models::model_path(app, models::spec("chat").unwrap()).map_err(err_str)?;
    let embed = models::model_path(app, models::spec("embed").unwrap()).map_err(err_str)?;
    for (p, name) in [
        (&vlm, "SmolVLM2"),
        (&mmproj, "vision projector"),
        (&chat, "chat"),
        (&embed, "embeddings"),
    ] {
        if !p.exists() {
            state.engine_loading.store(false, Ordering::SeqCst);
            let msg = format!("{name} model is missing: download it first.");
            let _ = app.emit(
                "engine-status",
                &EngineStatus { loaded: false, loading: false, error: Some(msg.clone()) },
            );
            return Err(msg);
        }
    }

    let result = tauri::async_runtime::spawn_blocking(move || {
        llm::Engine::load(
            &vlm.to_string_lossy(),
            &mmproj.to_string_lossy(),
            &chat.to_string_lossy(),
            &embed.to_string_lossy(),
        )
    })
    .await
    .map_err(err_str);

    state.engine_loading.store(false, Ordering::SeqCst);
    match result.and_then(|r| r.map_err(err_str)) {
        Ok(engine) => {
            *state.engine.lock().unwrap() = Some(engine);
            let _ = app.emit("engine-status", &EngineStatus { loaded: true, loading: false, error: None });
            Ok(())
        }
        Err(msg) => {
            let _ = app.emit(
                "engine-status",
                &EngineStatus { loaded: false, loading: false, error: Some(msg.clone()) },
            );
            Err(msg)
        }
    }
}

#[tauri::command]
async fn load_engine(app: AppHandle, state: State<'_, Arc<AppState>>) -> CmdResult<()> {
    let state = state.inner().clone();
    ensure_engine(&app, &state).await
}

// ---------- Library ----------

#[tauri::command]
fn library(app: AppHandle) -> CmdResult<Vec<VideoMeta>> {
    store::list_videos(&app).map_err(err_str)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VideoDetail {
    meta: VideoMeta,
    chunks: Vec<ChunkPublic>,
    transcript: Vec<vtt::Cue>,
}

#[tauri::command]
fn video_detail(app: AppHandle, id: String) -> CmdResult<VideoDetail> {
    let meta = store::load_meta(&app, &id).map_err(err_str)?;
    let chunks = store::load_index(&app, &id)
        .map(|cs| cs.iter().map(ChunkPublic::from).collect())
        .unwrap_or_default();
    let transcript = store::load_cues(&app, &id).unwrap_or_default();
    Ok(VideoDetail { meta, chunks, transcript })
}

/// Wait for the engine, then run the given ingest future; engine-load
/// failures surface as add errors in the UI.
fn spawn_ingest<F, Fut>(app: AppHandle, state: Arc<AppState>, ingest: F)
where
    F: FnOnce(AppHandle, Arc<AppState>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    tauri::async_runtime::spawn(async move {
        if let Err(e) = ensure_engine(&app, &state).await {
            pipeline::emit_global_error(&app, e);
            return;
        }
        ingest(app, state).await;
    });
}

#[tauri::command]
fn add_video(app: AppHandle, state: State<'_, Arc<AppState>>, url: String) -> CmdResult<()> {
    let url = url.trim().to_string();
    if url.is_empty() {
        return Err("Paste a YouTube URL first.".into());
    }
    spawn_ingest(app, state.inner().clone(), move |app, state| pipeline::ingest(app, state, url));
    Ok(())
}

#[tauri::command]
fn add_local_video(app: AppHandle, state: State<'_, Arc<AppState>>, path: String) -> CmdResult<()> {
    let path = std::path::PathBuf::from(path);
    if !path.is_file() {
        return Err("That file doesn't exist.".into());
    }
    spawn_ingest(app, state.inner().clone(), move |app, state| {
        pipeline::ingest_local(app, state, path)
    });
    Ok(())
}

#[tauri::command]
fn retry_video(app: AppHandle, state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    let meta = store::load_meta(&app, &id).map_err(err_str)?;
    match meta.status.as_str() {
        "processing" | "pending" => return Err("This video is already being processed.".into()),
        "ready" => return Err("This video is already in your library.".into()),
        _ => {}
    }
    spawn_ingest(app, state.inner().clone(), move |app, state| {
        pipeline::resume(app, state, meta)
    });
    Ok(())
}

#[tauri::command]
fn cancel_ingest(state: State<'_, Arc<AppState>>, id: String) {
    if let Some(flag) = state.cancels.lock().unwrap().get(&id) {
        flag.store(true, Ordering::Relaxed);
    }
}

#[tauri::command]
fn remove_video(app: AppHandle, state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    // If it's mid-ingest, cancel first; the pipeline notices the missing
    // meta.json and stops writing state back.
    if let Some(flag) = state.cancels.lock().unwrap().get(&id) {
        flag.store(true, Ordering::Relaxed);
    }
    store::delete_video(&app, &id).map_err(err_str)
}

// ---------- Chat ----------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct HistoryTurnDto {
    question: String,
    answer: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatToken {
    video_id: String,
    token: String,
}

/// One retrieved excerpt behind an answer, for the "sources" UI.
#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SourceRef {
    start: f64,
    end: f64,
    source: String,
    text: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatAnswer {
    answer: String,
    sources: Vec<SourceRef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frontend reads `answer` off this payload. Nothing in the type system
    /// spans the IPC boundary, so a rename here silently hands the UI an object
    /// where it expects a string — which blanks the window rather than erroring.
    #[test]
    fn chat_answer_keeps_its_wire_shape() {
        let payload = ChatAnswer {
            answer: "hello".into(),
            sources: vec![SourceRef {
                start: 1.0,
                end: 2.0,
                source: "speech".into(),
                text: "hi".into(),
            }],
        };
        let json = serde_json::to_value(&payload).unwrap();
        assert_eq!(json["answer"], "hello");
        assert!(json["sources"].is_array());
        assert_eq!(json["sources"][0]["start"], 1.0);
        assert_eq!(json["sources"][0]["source"], "speech");
    }
}

#[tauri::command]
async fn ask_video(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    video_id: String,
    question: String,
    history: Vec<HistoryTurnDto>,
) -> CmdResult<ChatAnswer> {
    let question = question.trim().to_string();
    if question.is_empty() {
        return Err("Ask something first.".into());
    }
    let meta = store::load_meta(&app, &video_id).map_err(err_str)?;
    if meta.status != "ready" {
        return Err("This video hasn't finished processing yet.".into());
    }
    let chunks = store::load_index(&app, &video_id).map_err(err_str)?;
    let state = state.inner().clone();
    ensure_engine(&app, &state).await?;
    let engine = state.require_engine().map_err(err_str)?;

    state.chat_cancel.store(false, Ordering::Relaxed);
    let cancel = state.chat_cancel.clone();
    let history: Vec<rag::HistoryTurn> = history
        .into_iter()
        .map(|h| rag::HistoryTurn { question: h.question, answer: h.answer })
        .collect();

    let app2 = app.clone();
    let vid = video_id.clone();
    let q = question.clone();
    let answer = tauri::async_runtime::spawn_blocking(move || {
        let guard = engine.lock().unwrap();
        let engine = guard.as_ref().ok_or_else(|| anyhow::anyhow!("engine not loaded"))?;

        // Follow-ups like "what did he say about it?" embed terribly; let the
        // chat model rewrite them into a standalone retrieval query first.
        // The user's original wording still goes into the answer prompt.
        let retrieval_q = if history.is_empty() {
            q.clone()
        } else {
            let prompt = rag::build_rewrite_prompt(&history, &q);
            match engine.chat(&prompt, 64, &cancel, &mut |_| {}) {
                Ok(out) => rag::clean_rewrite(&out, &q).unwrap_or_else(|| q.clone()),
                Err(_) => q.clone(),
            }
        };

        let query_vec = engine
            .embed_texts(std::slice::from_ref(&retrieval_q), true)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("query embedding failed"))?;
        let excerpts = rag::retrieve(&chunks, &query_vec, &retrieval_q, 6);
        let prompt = rag::build_chat_prompt(&meta, &excerpts, &history, &q);
        let sources: Vec<SourceRef> = excerpts
            .iter()
            .map(|c| SourceRef {
                start: c.start,
                end: c.end,
                source: c.source.clone(),
                text: c.text.clone(),
            })
            .collect();

        let mut on_token = |piece: &str| {
            let _ = app2.emit(
                "chat-token",
                &ChatToken { video_id: vid.clone(), token: piece.to_string() },
            );
        };
        let answer = engine.chat(&prompt, 512, &cancel, &mut on_token)?;
        // The model streams its own citations, which it routinely points at the
        // wrong excerpt line. Re-point them against the lines it was actually
        // shown before returning; the frontend replaces the streamed text with
        // this corrected answer when the request resolves.
        let answer = rag::snap_citations(&answer, &rag::excerpt_sublines(&excerpts));
        Ok::<_, anyhow::Error>(ChatAnswer { answer, sources })
    })
    .await
    .map_err(err_str)?
    .map_err(err_str)?;

    Ok(answer)
}

#[tauri::command]
fn stop_chat(state: State<'_, Arc<AppState>>) {
    state.chat_cancel.store(true, Ordering::Relaxed);
}

// ---------- Entry ----------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(Arc::new(AppState::new()))
        .setup(|app| {
            store::cleanup_orphans(app.handle());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            setup_status,
            download_model,
            update_ytdlp,
            load_engine,
            library,
            video_detail,
            add_video,
            add_local_video,
            retry_video,
            cancel_ingest,
            remove_video,
            ask_video,
            stop_chat,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // Drop the models before process exit so ggml's Metal cleanup
                // doesn't assert on still-resident buffers.
                let state = tauri::Manager::state::<Arc<AppState>>(app_handle).inner().clone();
                let engine = state.engine.lock().ok().and_then(|mut g| g.take());
                drop(engine);
                let whisper = state.whisper.lock().ok().and_then(|mut g| g.take());
                drop(whisper);
            }
        });
}
