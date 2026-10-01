//! Reproducible native storage/inference check, including Android ARM64.
use knowledge_core::Archive;
use std::sync::atomic::AtomicBool;
fn main()->anyhow::Result<()> {
    let args:Vec<_>=std::env::args().collect();
    anyhow::ensure!(args.len()==4 || args.len()==7,"Usage: archive-probe LIBRARY PACKAGE OUTPUT [CHAT EMBED QUESTION]");
    let start=std::time::Instant::now();let mut archive=Archive::open(&args[1])?;
    let report=archive.import_portable(std::path::Path::new(&args[2]))?;
    let summary=archive.summary()?;
    let again=archive.import_portable(std::path::Path::new(&args[2]))?;
    anyhow::ensure!(summary.evidence_count==archive.summary()?.evidence_count,"Repeat import duplicated evidence");
    let source_id=summary.sources.first().context("No source imported")?.id.clone();
    let backup=std::path::Path::new(&args[1]).join("verified.sacri-library");archive.export_library(&backup)?;
    let restored=std::path::Path::new(&args[1]).join("restored");let mut restored=Archive::open(&restored)?;
    restored.import_portable(&backup)?;let restored_summary=restored.summary()?;
    anyhow::ensure!(summary.evidence_count==restored_summary.evidence_count,"Restore changed evidence count");
    let storage_seconds=start.elapsed().as_secs_f64();
    let mut result=serde_json::json!({"runtime":std::env::consts::OS,"architecture":std::env::consts::ARCH,
        "storageSeconds":storage_seconds,"firstImport":report,"repeatImport":again,
        "sources":summary.sources.len(),"evidence":summary.evidence_count,"claims":summary.claims.len(),"concepts":summary.concepts.len(),
        "restoreEvidence":restored_summary.evidence_count,"sourceId":source_id});
    if args.len()==7 {
        let load=std::time::Instant::now();let engine=local_inference::TextEngine::load(&args[4],&args[5])?;
        result["modelLoadSeconds"]=serde_json::json!(load.elapsed().as_secs_f64());
        let vectors=engine.embed_texts(&[args[6].clone()],true)?;
        let hits=archive.search(&args[6],Some(&vectors[0]),Some(&source_id),3)?;
        let mut prompt="<|im_start|>system\nAnswer briefly from these excerpts. Source assertions are not verified facts. Cite only supplied [E1] references.\n<|im_end|>\n<|im_start|>user\n".to_string();
        for(i,h)in hits.iter().enumerate(){prompt.push_str(&format!("[E{}] {}\n",i+1,h.reading.as_deref().unwrap_or(&h.evidence.text).chars().take(500).collect::<String>().replace("<|","< |")));}
        prompt.push_str(&format!("Question: {}<|im_end|>\n<|im_start|>assistant\n",args[6].replace("<|","< |")));
        let began=std::time::Instant::now();let mut first=None;
        let answer=engine.chat(&prompt,120,&AtomicBool::new(false),&mut |_|{if first.is_none(){first=Some(began.elapsed().as_secs_f64());}})?;
        result["firstTokenSeconds"]=serde_json::json!(first);result["generationSeconds"]=serde_json::json!(began.elapsed().as_secs_f64());
        result["answer"]=serde_json::json!(knowledge_core::grounded_answer(&answer,hits.len()));
    }
    std::fs::write(&args[3],serde_json::to_vec_pretty(&result)?)?;println!("{}",serde_json::to_string_pretty(&result)?);
    Ok(())
}
use anyhow::Context;
