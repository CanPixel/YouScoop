//! Replay a chat question through the real RAG + prompt + generation path.
//! Usage: cargo run --example chat_test -- <video_dir> "<question>"

use std::sync::atomic::AtomicBool;

use youscoop_lib::llm::Engine;
use youscoop_lib::rag;
use youscoop_lib::store::{Chunk, VideoMeta};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("video dir");
    let question = args.next().expect("question");

    let home = std::env::var("HOME").unwrap();
    let base = format!("{home}/Library/Application Support/com.canpixel.youscoop");
    let engine = Engine::load(
        &format!("{base}/models/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
        &format!("{base}/models/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"),
        &format!("{base}/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        &format!("{base}/models/bge-small-en-v1.5-q8_0.gguf"),
    )
    .expect("engine load");

    let meta: VideoMeta =
        serde_json::from_str(&std::fs::read_to_string(format!("{base}/videos/{dir}/meta.json")).unwrap())
            .unwrap();
    let chunks: Vec<Chunk> =
        serde_json::from_str(&std::fs::read_to_string(format!("{base}/videos/{dir}/index.json")).unwrap())
            .unwrap();

    let qv = engine.embed_texts(&[question.clone()], true).unwrap().remove(0);
    let excerpts = rag::retrieve(&chunks, &qv, &question, 6);
    println!("--- excerpts fed to model ---");
    for c in &excerpts {
        println!("[{}] {}: {:.80}…", rag::format_ts(c.start), c.source, c.text.trim());
    }
    let prompt = rag::build_chat_prompt(&meta, &excerpts, &[], &question);

    let cancel = AtomicBool::new(false);
    let answer = engine.chat(&prompt, 512, &cancel, &mut |_| {}).expect("chat");
    println!("--- raw answer ---\n{answer}");
    // Mirror ask_video so the harness shows what the UI will actually render.
    let snapped = rag::snap_citations(&answer, &rag::excerpt_sublines(&excerpts));
    println!("--- after citation snapping ---\n{snapped}");
    drop(engine);
}
