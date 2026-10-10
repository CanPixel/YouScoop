//! Retrieval benchmark: does the passage that answers a question reach the
//! Oracle? Usage: retrieval_check LIBRARY CHAT_MODEL EMBEDDING_MODEL QUESTIONS_JSON
use knowledge_core::Archive;
use local_inference::TextEngine;
use std::collections::HashSet;

#[derive(serde::Deserialize)]
struct Question {
    q: String,
    gold: Vec<String>,
}
fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(str::to_lowercase).collect()
}
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 5, "Usage: retrieval_check LIBRARY CHAT_MODEL EMBEDDING_MODEL QUESTIONS_JSON");
    let archive = Archive::open(&args[1])?;
    let engine = TextEngine::load(&args[2], &args[3])?;
    let questions: Vec<Question> = serde_json::from_slice(&std::fs::read(&args[4])?)?;
    let k = std::env::var("RETRIEVAL_K").ok().and_then(|v| v.parse().ok()).unwrap_or(sacrinomicon_lib::ORACLE_PASSAGES);
    let (mut hit6, mut hit12, mut mrr, mut duplicates, mut distinct, mut search_ms) = (0, 0, 0.0, 0, 0, 0.0);
    let mut rows = Vec::new();
    for question in &questions {
        let vector = engine.embed_texts(std::slice::from_ref(&question.q), true)?;
        let start = std::time::Instant::now();
        let hits = archive.search(&question.q, Some(&vector[0]), None, k)?;
        search_ms += start.elapsed().as_secs_f64() * 1000.0;
        let relevant: Vec<bool> = hits
            .iter()
            .map(|h| {
                let text = h.reading.clone().unwrap_or_else(|| h.evidence.text.clone()).to_lowercase().replace("ball", "baal");
                question.gold.iter().any(|g| text.contains(g.as_str()))
            })
            .collect();
        let first = relevant.iter().position(|r| *r).map(|i| i + 1);
        if first.is_some_and(|r| r <= 6) { hit6 += 1; }
        if first.is_some() { hit12 += 1; }
        mrr += first.map_or(0.0, |r| 1.0 / r as f64);
        distinct += relevant.iter().filter(|r| **r).count();
        let sets: Vec<_> = hits.iter().map(|h| words(&h.evidence.text)).collect();
        for i in 0..sets.len() {
            for j in i + 1..sets.len() {
                let union = sets[i].union(&sets[j]).count().max(1);
                if sets[i].intersection(&sets[j]).count() as f64 / union as f64 > 0.6 { duplicates += 1; }
            }
        }
        rows.push(serde_json::json!({"question": question.q, "firstRelevantRank": first, "relevantInTop": relevant.iter().filter(|r| **r).count(),
            "kinds": hits.iter().map(|h| format!("{:?}", h.evidence.kind)).collect::<Vec<_>>()}));
    }
    let n = questions.len() as f64;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "passages": k, "questions": questions.len(),
        "hitAt6": hit6, "hitAtK": hit12, "meanReciprocalRank": (mrr / n * 1000.0).round() / 1000.0,
        "relevantPassagesInTopK": distinct, "nearDuplicatePairs": duplicates,
        "meanSearchMs": (search_ms / n * 10.0).round() / 10.0, "rows": rows,
    }))?);
    Ok(())
}
