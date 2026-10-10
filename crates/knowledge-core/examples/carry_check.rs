//! Import an older package, annotate it, import a reprocessed package, and
//! report which annotations reached the new current passages.
//! Usage: carry_check LIBRARY OLD.sacri NEW.sacri
use knowledge_core::*;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 4, "Usage: carry_check LIBRARY OLD.sacri NEW.sacri");
    let mut archive = Archive::open(&args[1])?;
    let source = archive.import_file(std::path::Path::new(&args[2]))?;
    archive.correct_source_spellings(&source, &[SpellingCorrection { original: "Ball".into(), corrected: "Baal".into() }])?;
    let detail = archive.source_detail(&source)?;
    let find = |needle: &str| detail.evidence.iter().find(|e| e.kind == EvidenceKind::Speech && e.text.contains(needle)).map(|e| e.id.clone()).unwrap();
    let dialogue = find("did not have a bank");
    let claim = find("only time the account is named");
    archive.assess(&dialogue, "contested", "Two voices disagree about the account name.")?;
    let original = archive.evidence(&claim)?.text;
    archive.correct_reading(&claim, Some(&original.replace("Ball", "Baal").replace("ball", "Baal")))?;
    archive.add_concept("Account name", &claim)?;
    let started = std::time::Instant::now();
    archive.import_file(std::path::Path::new(&args[3]))?;
    let elapsed = started.elapsed().as_secs_f64();
    let summary = archive.summary()?;
    let current = archive.source_detail(&source)?;
    let current_ids: std::collections::HashSet<_> = current.evidence.iter().map(|e| e.id.clone()).collect();
    let carried: Vec<_> = summary.assessments.iter().filter(|a| current_ids.contains(&a.entity_id)).map(|a| {
        let text = current.evidence.iter().find(|e| e.id == a.entity_id).unwrap().text.clone();
        serde_json::json!({"status": a.status, "hasCorrectedReading": a.corrected_text.is_some(), "passage": text.chars().take(110).collect::<String>(), "reading": archive.reading(&a.entity_id).ok().map(|r| r.chars().take(110).collect::<String>())})
    }).collect();
    let concept = summary.concepts.iter().find(|c| c.label == "Account name").unwrap();
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "currentPassages": current.evidence.len(), "importSeconds": elapsed, "carriedAnnotations": carried,
        "conceptLinksOnCurrentPassages": concept.evidence_ids.iter().filter(|id| current_ids.contains(*id)).count(),
    }))?);
    Ok(())
}
