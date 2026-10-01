//! YouScoop's handoff to the independent SACRiNOMiCON library.
use anyhow::{bail, Context, Result};
use knowledge_core::{stable_id, Archive, AssetInput, Evidence, EvidenceKind, Provenance, Source, SourcePackage};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};
use crate::store::{Chunk, FrameRef, VideoMeta};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedSource {
    pub source_id: String,
    pub package_path: String,
    pub archive_path: String,
    pub evidence_count: usize,
    pub artifact_count: usize,
}

pub fn library_root(app: &AppHandle) -> Result<PathBuf> {
    Ok(app.path().data_dir()?.join("com.canpixel.sacrinomicon").join("library"))
}

/// This builder has no dependency on a running webview, so export is reproducible in CLI tests.
pub fn prepare(dir: &Path) -> Result<(SourcePackage, Vec<AssetInput>)> {
    let meta: VideoMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    if meta.status != "ready" { bail!("Finish processing the video before saving it"); }
    let chunks: Vec<Chunk> = serde_json::from_slice::<Vec<Chunk>>(&std::fs::read(dir.join("index.json"))?)?
        .into_iter().filter(|c| c.source != "audiovisual").collect();
    if chunks.is_empty() { bail!("This video has no indexed evidence"); }
    let source_id = stable_id("source", &if meta.url.is_empty() { format!("local:{}",meta.id) } else { meta.url.clone() });
    let mut inputs = Vec::new();
    let mut add = |path: PathBuf, name: String, kind: &str| -> Result<String> {
        if !path.is_file() { bail!("Original artifact is missing: {}", path.display()); }
        let (sha256, _) = knowledge_core::file_digest(&path)?;
        let id = stable_id("asset", &format!("{source_id}:{name}:{sha256}"));
        inputs.push(AssetInput { id:id.clone(), name, kind:kind.into(), path });
        Ok(id)
    };
    let media_id = if let Some(path) = &meta.video_path {
        let path = PathBuf::from(path);
        let extension = path.extension().and_then(|s|s.to_str()).unwrap_or("mp4");
        Some(add(path.clone(), format!("original.{extension}"), "video")?)
    } else { None };
    let mut cues_id = None;
    for (name, kind) in [("meta.json","source_metadata"),("cues.json","captions"),("index.json","raw_index"),("descriptions.json","visual_output"),("audiovisual.json","audiovisual_output"),("inspection-plan.json","inspection_coverage"),("frames.json","frame_manifest")] {
        let path = dir.join(name);
        if path.is_file() {
            let id = add(path, name.into(), kind)?;
            if name == "cues.json" { cues_id = Some(id); }
        }
    }
    let frames: Vec<FrameRef> = std::fs::read(dir.join("frames.json")).ok()
        .map(|bytes|serde_json::from_slice(&bytes)).transpose()?.unwrap_or_default();
    let mut frame_assets = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let path = PathBuf::from(&frame.file);
        let path = if path.is_absolute() { path } else { dir.join("frames").join(path) };
        let id = add(path, format!("frame-{i:04}.jpg"), "keyframe")?;
        frame_assets.push((frame.ts,id));
    }
    if let Some(path) = &meta.thumb_path {
        if Path::new(path).is_file() { add(PathBuf::from(path),"thumbnail.jpg".into(),"thumbnail")?; }
    }
    let evidence: Vec<Evidence> = chunks.iter().map(|chunk| {
        let visual = chunk.source == "vision";
        let audiovisual = chunk.source == "audiovisual_reviewed";
        let original_asset = if visual {
            frame_assets.iter().min_by(|a,b|(a.0-chunk.start).abs().total_cmp(&(b.0-chunk.start).abs()))
                .filter(|(ts,_)|(*ts-chunk.start).abs()<1.0).map(|(_,id)|id.clone()).or_else(||media_id.clone())
        } else if audiovisual { media_id.clone() } else { cues_id.clone().or_else(||media_id.clone()) };
        Evidence {
            id: stable_id("evidence", &format!("{source_id}:{}:{}:{}:{}", chunk.source,chunk.start,chunk.end,chunk.text)),
            source_id:source_id.clone(), start:chunk.start.min(meta.duration_secs), end:chunk.end.min(meta.duration_secs), text:chunk.text.clone(),
            kind: if visual { EvidenceKind::Visual } else if audiovisual { EvidenceKind::Audiovisual } else { EvidenceKind::Speech },
            provenance:Provenance {
                method:if visual { "sampled_frame_description" } else if audiovisual { "human_reviewed_audio_visual_observation" } else if meta.has_captions { "source_captions" } else { "speech_transcription" }.into(),
                model:if visual { Some("SmolVLM2-2.2B-Instruct Q4_K_M".into()) } else if audiovisual { None } else if meta.has_captions { None } else { Some("whisper-large-v3-turbo Q5_0".into()) },
                generated_at:meta.added_at, original_asset,
            }, vector:chunk.vector.clone(),
        }
    }).collect();
    let asset_ids: Vec<_> = inputs.iter().map(|asset|(&asset.id,&asset.name,&asset.kind)).collect();
    let revision = stable_id("revision", &serde_json::to_string(&(&meta,&evidence,&asset_ids))?);
    Ok((SourcePackage {
        schema_version:knowledge_core::SCHEMA_VERSION, revision, exported_at:knowledge_core::now(),
        source:Source { id:source_id, title:meta.title, uri:meta.url, publisher:meta.channel, duration_seconds:meta.duration_secs,captured_at:meta.added_at,summary:meta.summary },
        embedding_model:knowledge_core::EMBEDDING_MODEL.into(),evidence,
        claims:vec![],concepts:vec![],relationships:vec![],assets:vec![],
    },inputs))
}

pub fn save(app: &AppHandle, video_id: &str) -> Result<SavedSource> {
    let dir = crate::store::video_dir_path(app, video_id)?;
    let exports = app.path().app_data_dir()?.join("exports");
    save_to(&dir, &exports, &library_root(app)?)
}

pub fn save_to(dir: &Path, exports: &Path, root: &Path) -> Result<SavedSource> {
    let (mut package,inputs) = prepare(dir)?;
    knowledge_core::navigation_drafts(&mut package);
    package.revision = stable_id("revision", &serde_json::to_string(&(&package.revision,&package.source,&package.evidence,&package.claims,&package.concepts))?);
    let filename = format!("{}.sacri",package.source.id.strip_prefix("source:").context("Invalid source ID")?);
    let output = exports.join(filename);
    let package = knowledge_core::write_package(package,&inputs,&output)?;
    let mut archive = Archive::open(root)?;
    let source_id = archive.import_file(&output)?;
    if !archive.summary()?.assessments.iter().any(|a|a.entity_id==source_id) {
        archive.assess(&source_id,"inbox","Preserved from YouScoop. Claims await review.")?;
    }
    Ok(SavedSource {source_id,package_path:output.to_string_lossy().into(),archive_path:root.to_string_lossy().into(),evidence_count:package.evidence.len(),artifact_count:package.assets.len()})
}
