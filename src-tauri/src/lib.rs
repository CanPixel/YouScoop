pub mod investigation;
pub mod llm;
mod digest;
mod media;
mod onscreen;
mod spelling;
pub mod voices;
mod models;
mod pipeline;
pub mod rag;
mod state;
pub mod store;
pub mod stt;
mod vtt;
mod wizmuth;
mod ytdlp;
mod desktop_worker;
mod runtime_setup;
#[cfg(feature = "diagnostics")]
pub use runtime_setup::verify_desktop_setup;

/// Run voice separation and on-screen text on one library video without a
/// window. Used for native verification.
#[cfg(feature = "diagnostics")]
pub fn verify_enrichment(video_id: &str) -> anyhow::Result<serde_json::Value> {
    let mut context = app_context();
    context.config_mut().app.windows.clear();
    let app = tauri::Builder::default().build(context)?;
    let handle = app.handle().clone();
    let state = Arc::new(AppState::new());
    tauri::async_runtime::block_on(async {
        let scope = memory::normal(&handle, &state, Arc::new(std::sync::atomic::AtomicBool::new(false))).await?;
        ensure_engine(&handle, &state, &scope.admission).await.map_err(anyhow::Error::msg)?;
        let mut meta = store::load_meta(&handle, video_id)?;
        let started = std::time::Instant::now();
        pipeline::enrich(&handle, &state, &mut meta, &Arc::new(std::sync::atomic::AtomicBool::new(false))).await?;
        let elapsed = started.elapsed().as_secs_f64();
        let index = store::load_index(&handle, video_id)?;
        let mut sources = std::collections::BTreeMap::<String, usize>::new();
        for chunk in &index { *sources.entry(chunk.source.clone()).or_default() += 1; }
        let dir = store::video_dir_path(&handle, video_id)?;
        Ok(serde_json::json!({
            "analysis": meta.analysis, "elapsedSeconds": elapsed, "indexBySource": sources,
            "voices": voices::load(&dir).map(|v| v.voices),
            "screens": onscreen::load(&dir).map(|r| r.screens.len()),
            "spellings": spelling::load(&dir),
        }))
    })
}
mod memory;

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

#[tauri::command]
async fn send_video_to_wizmuth(app: AppHandle, id: String, today: String) -> CmdResult<wizmuth::SentNote> {
    tauri::async_runtime::spawn_blocking(move || wizmuth::send_video(&app, &id, &today))
        .await.map_err(err_str)?.map_err(err_str)
}
#[tauri::command]
async fn send_answer_to_wizmuth(app: AppHandle, id: String, question: String, answer: String, today: String) -> CmdResult<wizmuth::SentNote> {
    tauri::async_runtime::spawn_blocking(move || wizmuth::send_answer(&app, &id, &question, &answer, &today))
        .await.map_err(err_str)?.map_err(err_str)
}

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
async fn ensure_engine(app: &AppHandle, state: &Arc<AppState>, admission:&inference_coordinator::Admission) -> Result<(), String> {
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
        struct Loading<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Loading<'_>{fn drop(&mut self){self.0.store(false,Ordering::SeqCst);}}
        let _loading=Loading(&state.engine_loading);
        memory::resident(app,state,admission).map_err(err_str)?;
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
    let scope=memory::normal(&app,&state,Arc::new(std::sync::atomic::AtomicBool::new(false))).await.map_err(err_str)?;
    ensure_engine(&app, &state,&scope.admission).await
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
    transcript: Vec<TranscriptCue>,
    /// Speaking time per voice, when voice separation ran.
    voices: Vec<voices::VoiceSummary>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptCue {
    start: f64,
    end: f64,
    text: String,
    voice: Option<String>,
}

#[tauri::command]
fn video_detail(app: AppHandle, id: String) -> CmdResult<VideoDetail> {
    let meta = store::load_meta(&app, &id).map_err(err_str)?;
    let chunks = store::load_index(&app, &id)
        .map(|cs| cs.iter().map(ChunkPublic::from).collect())
        .unwrap_or_default();
    let cues = store::load_cues(&app, &id).unwrap_or_default();
    let record = store::video_dir_path(&app, &id).ok().and_then(|dir| voices::load(&dir));
    let assigned = record.as_ref().map(|r| voices::assign(&cues, &r.turns)).unwrap_or_default();
    let transcript = cues
        .into_iter()
        .enumerate()
        .map(|(i, c)| TranscriptCue { start: c.start, end: c.end, text: c.text, voice: assigned.get(i).cloned().flatten() })
        .collect();
    Ok(VideoDetail { meta, chunks, transcript, voices: record.map(|r| r.voices).unwrap_or_default() })
}

/// Wait for the engine, then run the given ingest future; engine-load
/// failures surface as add errors in the UI.
fn spawn_ingest<F, Fut>(app: AppHandle, state: Arc<AppState>, ingest: F)
where
    F: FnOnce(AppHandle, Arc<AppState>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send,
{
    tauri::async_runtime::spawn(async move {
        let scope=match memory::normal(&app,&state,Arc::new(std::sync::atomic::AtomicBool::new(false))).await {
            Ok(scope)=>scope,Err(e)=>{pipeline::emit_global_error(&app,e.to_string());return;}
        };
        if let Err(e) = ensure_engine(&app, &state,&scope.admission).await {
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

/// Add voice separation and on-screen text to a ready video, keeping it usable.
#[tauri::command]
fn update_analysis(app: AppHandle, state: State<'_, Arc<AppState>>, id: String) -> CmdResult<()> {
    let meta = store::load_meta(&app, &id).map_err(err_str)?;
    if meta.status != "ready" {
        return Err("Finish processing the video first.".into());
    }
    if !pipeline::needs_enrichment(&meta) {
        return Err("This video's analysis is already up to date.".into());
    }
    spawn_ingest(app, state.inner().clone(), move |app, state| pipeline::update_analysis(app, state, meta));
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
    state.chat_cancel.store(false, Ordering::Relaxed);
    let scope=memory::normal(&app,&state,state.chat_cancel.clone()).await.map_err(err_str)?;
    ensure_engine(&app, &state,&scope.admission).await?;
    let engine = state.require_engine().map_err(err_str)?;

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
        // "Which emails are shown?" needs every email screen, not the six most similar passages.
        let listed = rag::listing(&chunks, &q, 16);
        let excerpts: Vec<&store::Chunk> = match &listed {
            Some((found, _)) => found.iter().collect(),
            None => rag::retrieve(&chunks, &query_vec, &retrieval_q, 6),
        };
        let asked = match &listed {
            Some((_, note)) => format!("{q}\n\n{note}"),
            None => q.clone(),
        };
        let prompt = rag::build_chat_prompt(&meta, &excerpts, &history, &asked);
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
        // A complete list needs more room than a direct answer.
        let answer = engine.chat(&prompt, if listed.is_some() { 1100 } else { 512 }, &cancel, &mut on_token)?;
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
            memory::watch(app.handle().clone(),tauri::Manager::state::<Arc<AppState>>(app).inner().clone())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            desktop_worker::worker_status,
            desktop_worker::configure_worker,
            runtime_setup::setup_desktop_runtime,
            runtime_setup::install_desktop_model,
            desktop_worker::ask_gemma,
            desktop_worker::stop_worker,
            send_video_to_wizmuth,
            send_answer_to_wizmuth,
            setup_status,
            download_model,
            update_ytdlp,
            load_engine,
            library,
            video_detail,
            add_video,
            add_local_video,
            retry_video,
            update_analysis,
            cancel_ingest,
            remove_video,
            ask_video,
            stop_chat,
        ])
        .build(app_context())
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

fn app_context() -> tauri::Context<tauri::Wry> {
    tauri::generate_context!()
}
