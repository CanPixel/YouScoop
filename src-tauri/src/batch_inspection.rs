//! Selective, checkpointed audiovisual inspection, exclusively in YouScoop.
use crate::{desktop_worker::{self, Moment}, state::AppState, store, CmdResult};
use anyhow::{bail, Context, Result};
use serde_json::json;
use std::{path::PathBuf, sync::{atomic::Ordering, Arc}};
use tauri::{AppHandle, Emitter, State};
use tokio::process::Command;

const QUESTION: &str = "Describe audible speech and visible evidence separately. Quote only confidently legible text. Identify quoted or rebuttal voices only when clear. Do not invent names, redacted values, document authenticity or verification.";

struct Clips(PathBuf);
impl Drop for Clips { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

pub(crate) fn checkpoint(app: &AppHandle, id: &str, saved: &[Moment]) -> Result<()> {
    if !store::meta_exists(app, id) { bail!("The source was removed"); }
    let path = store::video_dir_path(app,id)?.join("audiovisual.json");
    let temp = path.with_extension("json.part");
    std::fs::write(&temp,serde_json::to_vec_pretty(saved)?)?;
    std::fs::File::open(&temp)?.sync_all()?;
    std::fs::rename(temp,path)?;
    Ok(())
}

/// Repair the index from raw checkpoints, including work saved before a crash
/// or cancellation. Idempotency is based on preserved moment content.
pub(crate) async fn reconcile(app: &AppHandle, state: &Arc<AppState>, id: &str) -> Result<()> {
    let _ = state;
    let mut index = store::load_index(app,id)?;
    let prior = index.len();
    // Legacy raw model outputs stay in audiovisual.json for inspection. They
    // are not searchable claims until a human writes a reviewed observation.
    index.retain(|c| c.source != "audiovisual");
    if index.len() == prior { return Ok(()); }
    if !store::meta_exists(app,id) { bail!("The source was removed"); }
    store::save_index(app,id,&index)?;
    let mut meta=store::load_meta(app,id)?;meta.chunk_count=index.len();store::save_meta_if_present(app,&meta)?;
    Ok(())
}

#[tauri::command]
pub async fn inspect_source(app: AppHandle,state: State<'_,Arc<AppState>>,video_id:String)->CmdResult<Vec<Moment>> {
    let _job=crate::memory::WorkerJob::begin(state.inner()).map_err(crate::err_str)?;
    batch(&app,state.inner(),&video_id).await.map_err(crate::err_str)
}

pub(crate) async fn batch(app:&AppHandle,state:&Arc<AppState>,id:&str)->Result<Vec<Moment>> {
    batch_with_cancel(app,state,id,None).await
}
pub(crate) async fn batch_with_cancel(app:&AppHandle,state:&Arc<AppState>,id:&str,cancel_after:Option<usize>)->Result<Vec<Moment>> {
    let meta=store::load_meta(app,id)?;
    if meta.status!="ready" { bail!("Finish transcript processing first"); }
    let video=meta.video_path.context("Original media unavailable")?;
    let index=store::load_index(app,id)?;
    let plan=crate::investigation::inspection_plan(&index,meta.duration_secs);
    if plan.is_empty() { bail!("Select a video of at most two hours"); }
    reconcile(app,state,id).await?;
    let mut saved=desktop_worker::moments(app,id)?;
    let todo:Vec<_>=plan.iter().copied().filter(|start| !saved.iter().any(|m|
        (m.start-*start).abs()<0.01 && m.question==QUESTION && !m.text.trim().is_empty() && !m.truncated)).collect();
    let dir=store::video_dir_path(app,id)?;
    std::fs::write(dir.join("inspection-plan.json"),serde_json::to_vec_pretty(&json!({
        "method":"four_timeline_anchors_and_visual_document_moments","clipSeconds":10,
        "starts":plan,"durationSeconds":meta.duration_secs,"completeVideoCoverage":false,
        "question":QUESTION}))?)?;
    if todo.is_empty() { return Ok(saved); }
    let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
    let clips=Clips(dir.join("omni-clips").join(format!("batch-{stamp}")));
    std::fs::create_dir_all(&clips.0)?;
    let ffmpeg=crate::ytdlp::find_ffmpeg().context("ffmpeg is required")?;
    let mut requests=vec![];
    for (i,start) in todo.iter().enumerate() {
        if state.worker_cancel.load(Ordering::Relaxed) { bail!("Cancelled; completed inspections remain saved"); }
        let _=app.emit("worker-progress",json!({"kind":"status","detail":format!("Preparing selected moment {}/{}",i+1,todo.len())}));
        let end=(*start+10.0).min(meta.duration_secs);
        let path=clips.0.join(format!("{i}.mp4"));
        let mut command=Command::new(&ffmpeg);
        command.args(["-v","error","-y","-ss",&start.to_string(),"-i",&video,"-t",&(end-start).to_string(),"-map","0:v:0","-map","0:a:0","-c:v","libx264","-preset","ultrafast","-crf","18","-c:a","aac"])
            .arg(&path).kill_on_drop(true);
        let output=tokio::select! {
            result=command.output()=>result?,
            _=async {while !state.worker_cancel.load(Ordering::Relaxed){tokio::time::sleep(std::time::Duration::from_millis(70)).await;}}=>bail!("Cancelled; completed inspections remain saved"),
        };
        if !output.status.success() { bail!("Could not prepare a synchronized clip: {}",String::from_utf8_lossy(&output.stderr)); }
        requests.push(json!({"path":path,"start":start,"end":end}));
    }
    let cfg=desktop_worker::config(app)?;
    let mut checkpoint_error=None;
    let starting_count=saved.len();
    let result=desktop_worker::run(app,state,json!({"task":"omni","model":cfg.omni_model,"ffmpeg":ffmpeg,"question":QUESTION,"clips":requests,"maxTokens":300}),|event| {
        if event["kind"]!="moment" { return; }
        let Some(text)=event["answer"].as_str().filter(|s|!s.trim().is_empty()) else { return; };
        saved.push(Moment {start:event["start"].as_f64().unwrap_or(0.0),end:event["end"].as_f64().unwrap_or(0.0),
            text:text.into(),question:QUESTION.into(),model:"Qwen2.5-Omni-3B / Transformers MPS FP16".into(),
            generated_at:knowledge_core::now(),elapsed_seconds:event["elapsedSeconds"].as_f64().unwrap_or(0.0),
            truncated:event["truncated"].as_bool().unwrap_or(false)});
        if let Err(error)=checkpoint(app,id,&saved) {checkpoint_error=Some(error);state.worker_cancel.store(true,Ordering::Relaxed);}
        if cancel_after.is_some_and(|n|saved.len()-starting_count>=n) {state.worker_cancel.store(true,Ordering::Relaxed);}
    }).await;
    if let Some(error)=checkpoint_error { return Err(error); }
    result?;
    reconcile(app,state,id).await?;
    Ok(saved)
}
