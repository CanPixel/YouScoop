//! Dump what retrieval actually feeds the chat prompt for a given query.
//! Usage: cargo run --example rag_dump -- <video_dir> "<question>"

use youscoop_lib::llm::{cosine, Engine};

#[derive(serde::Deserialize)]
struct Chunk {
    start: f64,
    text: String,
    source: String,
    vector: Vec<f32>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("video dir");
    let question = args.next().expect("question");

    let home = std::env::var("HOME").unwrap();
    let models = format!("{home}/Library/Application Support/com.canpixel.youscoop/models");
    let engine = Engine::load(
        &format!("{models}/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
        &format!("{models}/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"),
        &format!("{models}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        &format!("{models}/bge-small-en-v1.5-q8_0.gguf"),
    )
    .expect("engine load");

    let index_path =
        format!("{home}/Library/Application Support/com.canpixel.youscoop/videos/{dir}/index.json");
    let chunks: Vec<Chunk> =
        serde_json::from_str(&std::fs::read_to_string(index_path).unwrap()).unwrap();

    let qv = engine.embed_texts(&[question.clone()], true).unwrap().remove(0);
    let mut scored: Vec<(f32, &Chunk)> = chunks
        .iter()
        .filter(|c| !c.vector.is_empty())
        .map(|c| (cosine(&c.vector, &qv) + youscoop_lib::rag::lexical_boost(&question, &c.text), c))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("query: {question}\ntop 12 retrieved chunks:");
    for (score, c) in scored.iter().take(12) {
        let secs = c.start as u64;
        println!("{score:.3} [{}:{:02}] {}: {}", secs / 60, secs % 60, c.source, c.text.trim());
    }
    drop(engine);
}
