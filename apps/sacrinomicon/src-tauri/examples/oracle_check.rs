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
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Review {
            evidence_id: String,
            status: String,
            note: String,
        }
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Correction {
            Reading(Reading),
            Source(SourceSpellings),
            Review(Review),
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
                Correction::Review(review) => {
                    archive.assess(&review.evidence_id, &review.status, &review.note)?;
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
    // ORACLE_LENS=esoteric, ORACLE_SPOKEN=1 and ORACLE_HISTORY=turns.json
    // ([["question","answer"], ...]) exercise the app's other consultations.
    let consultation = sacrinomicon_lib::Consultation {
        lens: if std::env::var("ORACLE_LENS").is_ok_and(|v| v == "esoteric") { sacrinomicon_lib::Lens::Esoteric } else { sacrinomicon_lib::Lens::Archive },
        spoken: std::env::var_os("ORACLE_SPOKEN").is_some(),
    };
    let history: Vec<(String, String)> = match std::env::var_os("ORACLE_HISTORY") {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Vec::new(),
    };
    let start = std::time::Instant::now();
    let mut statuses = Vec::new();
    let prepared = sacrinomicon_lib::prepare(&archive, &model, &args[4], None, &history, consultation, &AtomicBool::new(false), &mut |s| statuses.push(s.to_string()))?;
    if std::env::var_os("ORACLE_PROMPT").is_some() {
        eprintln!("{}", prepared.prompt);
    }
    let count = prepared.hits.len();
    let mut stream = knowledge_core::SentenceStream::new(count).with_passages(prepared.passages.clone()).with_descriptions(prepared.descriptions.clone());
    let mut sentences: Vec<(f64, String)> = Vec::new();
    let raw = model.chat_with(&prepared.prompt, prepared.max_tokens, prepared.sampling, &AtomicBool::new(false), &mut |text| {
        for sentence in stream.push(text) {
            sentences.push((start.elapsed().as_secs_f64(), sentence));
        }
    })?;
    let reply = sacrinomicon_lib::brief_answer(&raw);
    let complete = sacrinomicon_lib::complete_sentences(&reply);
    if complete.trim_end_matches('…') == reply.trim() || complete.ends_with('…') {
        sentences.extend(stream.finish().map(|s| (start.elapsed().as_secs_f64(), s)));
    }
    let answer = knowledge_core::mark_quotes(&knowledge_core::grounded_answer(&complete, count), &prepared.passages, &prepared.descriptions);
    let hits = prepared.hits;
    let threads = prepared.threads;
    let search_query = prepared.search_query;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"question":args[4],"answer":answer,"elapsedSeconds":start.elapsed().as_secs_f64(),"spokenSentences":sentences,"sources":hits,"threads":threads,"statuses":statuses,"searchQuery":search_query})
        )?
    );
    Ok(())
}
