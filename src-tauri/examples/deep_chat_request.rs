//! Print the desktop model request that a deep chat question would send, so
//! it can be replayed through runtime/worker.py outside the app.
//! Usage: cargo run --example deep_chat_request -- <video_id> "<question>" MODEL_DIR
//!   | <desktop runtime python> -u runtime/worker.py
use youscoop_lib::store::{Chunk, VideoMeta};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(args.len() == 3, "Usage: deep_chat_request VIDEO_ID QUESTION MODEL_DIR");
    let base = format!("{}/Library/Application Support/com.canpixel.youscoop/videos/{}", std::env::var("HOME")?, args[0]);
    let meta: VideoMeta = serde_json::from_str(&std::fs::read_to_string(format!("{base}/meta.json"))?)?;
    let chunks: Vec<Chunk> = serde_json::from_str(&std::fs::read_to_string(format!("{base}/index.json"))?)?;
    let context = youscoop_lib::investigation::context(&chunks, &args[1]);
    eprintln!("{} passages, {} characters, transcript complete: {}", context.passages.len(), context.text.chars().count(), context.spoken_complete);
    let messages = youscoop_lib::investigation::messages(&meta.title, &meta.channel, &context, &args[1], &[]);
    println!("{}", serde_json::json!({"task": "gemma", "model": args[2], "messages": messages}));
    Ok(())
}
