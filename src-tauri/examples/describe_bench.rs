//! Describe keyframes with a given frame model for side-by-side comparison.
//! Usage: describe_bench VLM.gguf MMPROJ.gguf FRAME.jpg...
use std::sync::atomic::AtomicBool;
use youscoop_lib::llm::Engine;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "Usage: describe_bench VLM.gguf MMPROJ.gguf FRAME.jpg...");
    let base = format!("{}/Library/Application Support/com.canpixel.youscoop/models", std::env::var("HOME")?);
    let mut engine = Engine::load(&args[1], &args[2], &format!("{base}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), &format!("{base}/bge-small-en-v1.5-q8_0.gguf"))?;
    let cancel = AtomicBool::new(false);
    let started = std::time::Instant::now();
    engine.ensure_vision()?;
    let load = started.elapsed().as_secs_f64();
    let mut rows = Vec::new();
    for frame in &args[3..] {
        let t = std::time::Instant::now();
        let text = engine.describe_frame(frame, &cancel)?;
        rows.push(serde_json::json!({"frame": frame.rsplit('/').next(), "seconds": t.elapsed().as_secs_f64(), "description": text}));
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({"model": args[1].rsplit('/').next(), "loadSeconds": load, "frames": rows}))?);
    Ok(())
}
