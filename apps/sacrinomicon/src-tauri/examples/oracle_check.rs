use knowledge_core::Archive;
use local_inference::TextEngine;
use std::sync::atomic::AtomicBool;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5 || args.len() == 6,
        "Usage: oracle_check LIBRARY CHAT_MODEL EMBEDDING_MODEL QUESTION [CORRECTIONS_JSON]"
    );
    let archive = Archive::open(&args[1])?;
    if let Some(path) = args.get(5) {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Reading {
            evidence_id: String,
            corrected_text: String,
        }
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct SourceSpellings {
            source_id: String,
            spellings: Vec<knowledge_core::SpellingCorrection>,
        }
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Correction {
            Reading(Reading),
            Source(SourceSpellings),
        }
        let corrections: Vec<Correction> = serde_json::from_slice(&std::fs::read(path)?)?;
        for correction in corrections {
            match correction {
                Correction::Reading(reading) => {
                    archive.correct_reading(&reading.evidence_id, Some(&reading.corrected_text))?;
                }
                Correction::Source(source) => {
                    archive.correct_source_spellings(&source.source_id, &source.spellings)?;
                }
            }
        }
    }
    let locks = std::env::var_os("SACRI_INFERENCE_LOCKS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").expect("Home directory"))
                .join("Library/Application Support/.sacrinomicon-inference")
        });
    let coordinator = inference_coordinator::Coordinator::new(locks)?;
    let admission = coordinator.admit(&AtomicBool::new(false))?;
    let _resident = coordinator.resident(&admission)?;
    let model = TextEngine::load(&args[2], &args[3])?;
    let vector = model.embed_texts(&[args[4].clone()], true)?;
    let hits = archive.search(&args[4], Some(&vector[0]), None, 6)?;
    let prompt = sacrinomicon_lib::oracle_prompt(&args[4], &hits, &[]);
    let start = std::time::Instant::now();
    let answer = model.chat(&prompt, 700, &AtomicBool::new(false), &mut |_| {})?;
    let answer = knowledge_core::grounded_answer(&answer, hits.len());
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"question":args[4],"answer":answer,"elapsedSeconds":start.elapsed().as_secs_f64(),"sources":hits})
        )?
    );
    Ok(())
}
