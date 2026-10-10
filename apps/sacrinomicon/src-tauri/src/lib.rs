mod memory;
mod models;
mod oracle;
mod oracle_voice;
mod speech;
pub use oracle::*;
#[cfg(target_os = "macos")]
pub use oracle_voice::OracleVoice;
pub use oracle_voice::{speakable, ORACLE_VOICES};
#[cfg(target_os = "macos")]
type VoiceEngine = oracle_voice::OracleVoice;
#[cfg(not(target_os = "macos"))]
type VoiceEngine = ();
use anyhow::Context;
use knowledge_core::{
    Archive, ArchiveSummary, Assessment, Claim, Concept, Evidence, RelationKind, Relationship,
    SearchHit, SourceDetail,
};
use local_inference::TextEngine;
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};
use tauri::{AppHandle, Emitter, Manager, State};
type CmdResult<T> = Result<T, String>;
fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}
struct AppState {
    engine: Mutex<Option<TextEngine>>,
    speech: Mutex<Option<whisper_rs::WhisperContext>>,
    voice: Mutex<Option<VoiceEngine>>,
    voice_enabled: AtomicBool,
    cancel: AtomicBool,
    busy: AtomicBool,
    memory_resident: Mutex<Option<inference_coordinator::Resident>>,
}
impl AppState {
    fn new() -> Self {
        Self {
            engine: Mutex::new(None),
            speech: Mutex::new(None),
            voice: Mutex::new(None),
            voice_enabled: AtomicBool::new(false),
            cancel: AtomicBool::new(false),
            busy: AtomicBool::new(false),
            memory_resident: Mutex::new(None),
        }
    }
}

#[cfg(feature = "diagnostics")]
pub fn verify_memory_handoff(
    directory: &std::path::Path,
    chat: &str,
    embed: &str,
    marker: &std::path::Path,
) -> anyhow::Result<()> {
    let coordinator = inference_coordinator::Coordinator::new(directory)?;
    let state = Arc::new(AppState::new());
    let admission = coordinator.admit(&state.cancel)?;
    memory::resident(&state, &coordinator, &admission).map_err(anyhow::Error::msg)?;
    *state.engine.lock().unwrap() = Some(TextEngine::load(chat, embed)?);
    drop(admission);
    std::fs::write(marker, "loaded")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !coordinator.heavy_requested()? {
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "No heavy worker requested memory"
        );
        std::thread::sleep(std::time::Duration::from_millis(40));
    }
    anyhow::ensure!(
        memory::yield_if_idle(&state),
        "Resident models were not idle"
    );
    anyhow::ensure!(
        state.engine.lock().unwrap().is_none() && state.memory_resident.lock().unwrap().is_none(),
        "Models or reservation remained resident"
    );
    std::fs::write(marker, "released")?;
    let _ = std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}
struct Busy(Arc<AppState>);
impl Drop for Busy {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::SeqCst);
    }
}
fn claim(state: &Arc<AppState>) -> CmdResult<Busy> {
    if state.busy.swap(true, Ordering::SeqCst) {
        return Err("The Oracle is already working".into());
    }
    state.cancel.store(false, Ordering::Relaxed);
    Ok(Busy(state.clone()))
}
fn root(app: &AppHandle) -> CmdResult<PathBuf> {
    Ok(app.path().app_data_dir().map_err(err)?.join("library"))
}
fn archive(app: &AppHandle) -> CmdResult<Archive> {
    Archive::open(root(app)?).map_err(err)
}
#[tauri::command]
fn library(app: AppHandle) -> CmdResult<ArchiveSummary> {
    archive(&app)?.summary().map_err(err)
}
#[tauri::command]
fn source_detail(app: AppHandle, id: String) -> CmdResult<SourceDetail> {
    archive(&app)?.source_detail(&id).map_err(err)
}
#[tauri::command]
fn evidence_detail(app: AppHandle, id: String) -> CmdResult<Evidence> {
    archive(&app)?.evidence(&id).map_err(err)
}
#[tauri::command]
fn asset_path(app: AppHandle, source_id: String, asset_id: String) -> CmdResult<String> {
    Ok(archive(&app)?
        .asset_path(&source_id, &asset_id)
        .map_err(err)?
        .to_string_lossy()
        .into())
}
#[tauri::command]
fn assess(
    app: AppHandle,
    entity_id: String,
    status: String,
    note: String,
) -> CmdResult<Assessment> {
    archive(&app)?
        .assess(&entity_id, &status, &note)
        .map_err(err)
}
#[tauri::command]
fn evidence_reading(app: AppHandle, id: String) -> CmdResult<String> {
    archive(&app)?.reading(&id).map_err(err)
}
#[tauri::command]
fn correct_reading(
    app: AppHandle,
    evidence_id: String,
    corrected: Option<String>,
) -> CmdResult<Assessment> {
    archive(&app)?
        .correct_reading(&evidence_id, corrected.as_deref())
        .map_err(err)
}
#[tauri::command]
fn correct_source_spellings(
    app: AppHandle,
    source_id: String,
    spellings: Vec<knowledge_core::SpellingCorrection>,
) -> CmdResult<Assessment> {
    archive(&app)?
        .correct_source_spellings(&source_id, &spellings)
        .map_err(err)
}
/// Spelling suggestions YouScoop preserved with a source (from on-screen text,
/// checked by re-listening). Proposals only; the user applies them.
#[tauri::command]
fn spelling_suggestions(app: AppHandle, source_id: String) -> CmdResult<serde_json::Value> {
    let archive = archive(&app)?;
    let detail = archive.source_detail(&source_id).map_err(err)?;
    let Some(asset) = detail.assets.iter().find(|a| a.kind == "spelling_suggestions") else {
        return Ok(serde_json::json!([]));
    };
    if asset.bytes > 4 * 1024 * 1024 {
        return Err("Spelling suggestions file is too large".into());
    }
    let path = archive.asset_path(&source_id, &asset.id).map_err(err)?;
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
    Ok(record.get("suggestions").cloned().unwrap_or_else(|| serde_json::json!([])))
}
#[tauri::command]
fn resolve_annotation(app: AppHandle, id: String, use_incoming: bool) -> CmdResult<()> {
    archive(&app)?
        .resolve_annotation(&id, use_incoming)
        .map_err(err)
}
#[tauri::command]
fn activate_revision(app: AppHandle, source_id: String, revision: String) -> CmdResult<()> {
    archive(&app)?
        .activate_revision(&source_id, &revision)
        .map_err(err)
}
#[tauri::command]
fn add_concept(app: AppHandle, label: String, evidence_id: String) -> CmdResult<Concept> {
    archive(&app)?
        .add_concept(&label, &evidence_id)
        .map_err(err)
}
#[tauri::command]
fn add_claim(
    app: AppHandle,
    evidence_id: String,
    statement: String,
    kind: String,
) -> CmdResult<Claim> {
    archive(&app)?
        .add_claim(&evidence_id, &statement, &kind)
        .map_err(err)
}
#[tauri::command]
fn add_relationship(
    app: AppHandle,
    from: String,
    to: String,
    kind: RelationKind,
    evidence_id: String,
    explanation: String,
) -> CmdResult<Relationship> {
    archive(&app)?
        .add_relationship(&from, &to, kind, &evidence_id, &explanation)
        .map_err(err)
}
#[tauri::command]
fn search(app: AppHandle, query: String, source_id: Option<String>) -> CmdResult<Vec<SearchHit>> {
    archive(&app)?
        .search(&query, None, source_id.as_deref(), 20)
        .map_err(err)
}

// Bounded upload chunks avoid JSON-encoding a complete video package in memory.
#[tauri::command]
fn begin_import(app: AppHandle) -> CmdResult<String> {
    let dir = app.path().app_data_dir().map_err(err)?.join("imports");
    std::fs::create_dir_all(&dir).map_err(err)?;
    let id = knowledge_core::stable_id(
        "import",
        &format!("{}:{:?}", knowledge_core::now(), std::time::Instant::now()),
    )
    .replace(':', "-");
    std::fs::File::create(dir.join(&id)).map_err(err)?;
    Ok(id)
}
fn import_path(app: &AppHandle, id: &str) -> CmdResult<PathBuf> {
    if !id.starts_with("import-")
        || id.len() != 71
        || !id[7..].bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("Invalid import session".into());
    }
    Ok(app
        .path()
        .app_data_dir()
        .map_err(err)?
        .join("imports")
        .join(id))
}
#[tauri::command]
fn append_import(app: AppHandle, id: String, offset: u64, bytes: Vec<u8>) -> CmdResult<()> {
    use std::io::Write;
    if bytes.len() > 128 * 1024 || offset + bytes.len() as u64 > 8 * 1024_u64.pow(3) {
        return Err("Package exceeds import limits".into());
    }
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(import_path(&app, &id)?)
        .map_err(err)?;
    if file.metadata().map_err(err)?.len() != offset {
        return Err("Import chunk arrived out of sequence".into());
    }
    file.write_all(&bytes).map_err(err)
}
#[tauri::command]
async fn finish_import(app: AppHandle, id: String) -> CmdResult<knowledge_core::ImportReport> {
    tauri::async_runtime::spawn_blocking(move || {
        let path = import_path(&app, &id)?;
        let result = archive(&app)?.import_portable(&path).map_err(err);
        let _ = std::fs::remove_file(path);
        result
    })
    .await
    .map_err(err)?
}
#[tauri::command]
fn cancel_import(app: AppHandle, id: String) -> CmdResult<()> {
    std::fs::remove_file(import_path(&app, &id)?).map_err(err)
}

#[derive(Serialize)]
struct ExportResult {
    saved: bool,
    path: Option<String>,
}
#[tauri::command]
async fn export_library(app: AppHandle) -> CmdResult<ExportResult> {
    let handle = app.clone();
    let path = tauri::async_runtime::spawn_blocking(move || {
        let dir = if cfg!(target_os = "android") {
            handle.path().app_cache_dir()
        } else {
            handle.path().app_data_dir()
        }
        .map_err(err)?
        .join("exports");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(err)?
            .as_millis();
        let path = dir.join(format!("SACRiNOMiCON-{stamp}.sacri-library"));
        archive(&handle)?.export_library(&path).map_err(err)?;
        Ok::<_, String>(path)
    })
    .await
    .map_err(err)??;
    let result = speech::save_export(&app, &path).await;
    #[cfg(target_os = "android")]
    let _ = std::fs::remove_file(&path);
    let saved = result?;
    Ok(ExportResult {
        saved,
        path: if cfg!(target_os = "android") {
            None
        } else {
            Some(path.to_string_lossy().into())
        },
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    id: String,
    label: String,
    detail: String,
    bytes: u64,
    downloaded: bool,
}
#[tauri::command]
fn model_status(app: AppHandle) -> Vec<ModelStatus> {
    models::MODELS
        .iter()
        .filter(|m| cfg!(target_os = "macos") || m.id != "speech")
        .map(|m| ModelStatus {
            id: m.id.into(),
            label: m.label.into(),
            detail: m.detail.into(),
            bytes: m.bytes,
            downloaded: models::is_downloaded(&app, m),
        })
        .collect()
}
#[tauri::command]
async fn download_model(app: AppHandle, id: String) -> CmdResult<()> {
    models::download(&app, &id).await.map_err(err)
}
#[tauri::command]
fn unload_models(state: State<'_, Arc<AppState>>) -> CmdResult<()> {
    if state.busy.load(Ordering::SeqCst) {
        return Err("Stop the Oracle before unloading models".into());
    }
    *state.engine.lock().map_err(err)? = None;
    *state.voice.lock().map_err(err)? = None;
    memory::release_if_empty(state.inner())?;
    Ok(())
}
#[tauri::command]
fn stop_oracle(state: State<'_, Arc<AppState>>) {
    state.cancel.store(true, Ordering::Relaxed);
}
#[tauri::command]
async fn voice_mode(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    enabled: bool,
) -> CmdResult<()> {
    let state = state.inner().clone();
    let busy = if enabled { Some(claim(&state)?) } else { None };
    state.voice_enabled.store(enabled, Ordering::SeqCst);
    if !enabled {
        state.cancel.store(true, Ordering::Relaxed);
    }
    let coordinator = memory::coordinator(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let admission = if enabled {
            Some(coordinator.admit(&state.cancel).map_err(err)?)
        } else {
            None
        };
        if let Some(admission) = &admission {
            memory::resident(&state, &coordinator, admission)?;
        }
        let mut speech = state.speech.lock().map_err(err)?;
        if !enabled {
            *speech = None;
            drop(speech);
            memory::release_if_empty(&state)?;
            return Ok(());
        }
        if speech.is_none() {
            let path = models::model_path(&app, models::spec("voice").unwrap()).map_err(err)?;
            let mut params = whisper_rs::WhisperContextParameters::default();
            params.use_gpu(false);
            let ctx = whisper_rs::WhisperContext::new_with_params(&path, params).map_err(err)?;
            if state.voice_enabled.load(Ordering::SeqCst) {
                *speech = Some(ctx);
            }
        }
        Ok(())
    })
    .await
    .map_err(err)?
}
#[tauri::command]
async fn transcribe_voice(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    samples: Vec<f32>,
) -> CmdResult<String> {
    transcribe_impl(app, state.inner().clone(), samples).await
}
async fn transcribe_impl(app: AppHandle, state: Arc<AppState>, samples: Vec<f32>) -> CmdResult<String> {
    if samples.len() < 8000
        || samples.len() > 480000
        || samples.iter().any(|s| !s.is_finite() || s.abs() > 1.01)
    {
        return Err("Voice input must be 0.5 to 30 seconds of 16 kHz mono audio".into());
    }
    let busy = claim(&state)?;
    let coordinator = memory::coordinator(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let _ = app.emit("oracle-status", "Waiting for local inference");
        let admission = coordinator.admit(&state.cancel).map_err(err)?;
        memory::resident(&state, &coordinator, &admission)?;
        if !state.voice_enabled.load(Ordering::SeqCst) {
            return Err("Voice mode is closed".into());
        }
        let mut ctx = state.speech.lock().map_err(err)?;
        if ctx.is_none() {
            let path = models::model_path(&app, models::spec("voice").unwrap()).map_err(err)?;
            let mut params = whisper_rs::WhisperContextParameters::default();
            params.use_gpu(false);
            *ctx = Some(whisper_rs::WhisperContext::new_with_params(&path, params).map_err(err)?);
        }
        let _ = app.emit("oracle-status", "Transcribing");
        let ctx = ctx
            .as_ref()
            .context("Enter voice mode before speaking")
            .map_err(err)?;
        let mut whisper = ctx.create_state().map_err(err)?;
        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        // The archive's own names ("Baal", "Moloch") as a spelling hint.
        let vocabulary = archive(&app).ok()
            .and_then(|a| a.leading_concepts(None, 16).ok())
            .map(|labels| labels.join(", "))
            .filter(|v| !v.is_empty())
            .map(|v| format!("{}.", v.chars().take(400).collect::<String>()));
        if let Some(vocabulary) = &vocabulary {
            params.set_initial_prompt(vocabulary);
        }
        params.set_language(Some("auto"));
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        params.set_n_threads(
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4)
                .min(6),
        );
        let flag = state.clone();
        params.set_abort_callback_safe(move || {
            flag.cancel.load(Ordering::Relaxed) || !flag.voice_enabled.load(Ordering::SeqCst)
        });
        whisper.full(params, &samples).map_err(err)?;
        let text = (0..whisper.full_n_segments())
            .filter_map(|i| whisper.get_segment(i))
            .filter_map(|s| s.to_str_lossy().ok().map(|s| s.trim().to_string()))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(text)
    })
    .await
    .map_err(err)?
}
#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn ask_oracle(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    question: String,
    source_id: Option<String>,
    history: Vec<(String, String)>,
    lens: Option<Lens>,
    spoken: Option<bool>,
) -> CmdResult<OracleAnswer> {
    if question.trim().is_empty() || question.len() > 2000 {
        return Err("Enter a question below 2,000 characters".into());
    }
    let consultation = Consultation { lens: lens.unwrap_or_default(), spoken: spoken.unwrap_or(false) };
    let state = state.inner().clone();
    let busy = claim(&state)?;
    let root = root(&app)?;
    let chat = models::model_path(&app, models::spec("chat").unwrap()).map_err(err)?;
    let embed = models::model_path(&app, models::spec("embed").unwrap()).map_err(err)?;
    let coordinator = memory::coordinator(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        let _busy = busy;
        let _ = app.emit("oracle-status", "Waiting for local inference");
        let admission = coordinator.admit(&state.cancel).map_err(err)?;
        memory::resident(&state, &coordinator, &admission)?;
        let mut engine = state.engine.lock().map_err(err)?;
        if engine.is_none() {
            let _ = app.emit("oracle-status", "Waking the Oracle");
            *engine = Some(
                TextEngine::load(&chat.to_string_lossy(), &embed.to_string_lossy()).map_err(err)?,
            );
        }
        let engine = engine
            .as_ref()
            .context("Models are unavailable")
            .map_err(err)?;
        let archive = Archive::open(root).map_err(err)?;
        let prepared = prepare(&archive, engine, &question, source_id.as_deref(), &history, consultation, &state.cancel, &mut |status| {
            let _ = app.emit("oracle-status", status);
        }).map_err(err)?;
        let _ = app.emit("oracle-status", "Speaking");
        let count = prepared.hits.len();
        let mut sentences = knowledge_core::SentenceStream::new(count).with_passages(prepared.passages.clone()).with_descriptions(prepared.descriptions.clone());
        let raw = engine
            .chat_with(&prepared.prompt, prepared.max_tokens, prepared.sampling, &state.cancel, &mut |text| {
                let _ = app.emit("oracle-token", text);
                for sentence in sentences.push(text) {
                    let _ = app.emit("oracle-sentence", sentence);
                }
            })
            .map_err(err)?;
        let reply = brief_answer(&raw);
        let complete = complete_sentences(&reply);
        // A sentence cut off by the token budget is not spoken.
        if complete.trim_end_matches('…') == reply.trim() || complete.ends_with('…') {
            if let Some(sentence) = sentences.finish() {
                let _ = app.emit("oracle-sentence", sentence);
            }
        }
        let answer = knowledge_core::mark_quotes(&knowledge_core::grounded_answer(&complete, count), &prepared.passages, &prepared.descriptions);
        Ok(OracleAnswer {
            answer,
            sources: prepared.hits,
            threads: prepared.threads,
        })
    })
    .await
    .map_err(err)?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(speech::init())
        .manage(Arc::new(AppState::new()))
        .setup(|app| {
            memory::watch(
                app.handle().clone(),
                app.state::<Arc<AppState>>().inner().clone(),
            )
            .map_err(std::io::Error::other)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            library,
            source_detail,
            evidence_detail,
            asset_path,
            assess,
            correct_reading,
            evidence_reading,
            correct_source_spellings,
            spelling_suggestions,
            resolve_annotation,
            activate_revision,
            export_library,
            add_concept,
            add_claim,
            add_relationship,
            search,
            begin_import,
            append_import,
            finish_import,
            cancel_import,
            model_status,
            download_model,
            unload_models,
            stop_oracle,
            ask_oracle,
            voice_mode,
            transcribe_voice,
            speech::speak_text,
            speech::stop_speech,
            oracle_voice::speech_engine,
            oracle_voice::synthesize_speech,
            oracle_voice::oracle_voices
        ])
        .run(app_context())
        .expect("Archive failed to start");
}

fn app_context() -> tauri::Context<tauri::Wry> { tauri::generate_context!() }

#[cfg(all(feature="diagnostics",not(target_os="android")))]
pub fn verify_archive_runtime(task:&str,input:Option<String>)->anyhow::Result<serde_json::Value> {
    let mut context=app_context();context.config_mut().app.windows.clear();
    let app=tauri::Builder::default().build(context)?;
    let state=Arc::new(AppState::new());
    tauri::async_runtime::block_on(async {
        match task {
            "models" => { for id in ["chat","embed","voice"] { models::download(app.handle(),id).await?; }
                Ok(serde_json::to_value(model_status(app.handle().clone()))?) },
            "voice" => { state.voice_enabled.store(true,Ordering::SeqCst);
                let path=input.context("Supply a JSON array of 16kHz PCM samples")?;
                let samples=serde_json::from_slice(&std::fs::read(path)?)?;
                let start=std::time::Instant::now();
                let text=transcribe_impl(app.handle().clone(),state.clone(),samples).await.map_err(anyhow::Error::msg)?;
                Ok(serde_json::json!({"text":text,"elapsedSeconds":start.elapsed().as_secs_f64(),"runtime":"Whisper small Q5_1 CPU; native transcription path"})) },
            _=>anyhow::bail!("Choose models or voice"),
        }
    })
}
