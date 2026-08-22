//! Smoke test for the duplicate-ggml fix: creating a BGE embed context used to
//! abort in llama_context::sched_reserve (fused-GDN scan misreading GELU nodes).
//! Run with: cargo run --example ggml_smoke

use smoltalk_lib::llm::Engine;
use smoltalk_lib::stt::WhisperEngine;

fn main() {
    let dir = dirs_path();
    let engine = Engine::load(
        &format!("{dir}/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
        &format!("{dir}/mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"),
        &format!("{dir}/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
        &format!("{dir}/bge-small-en-v1.5-q8_0.gguf"),
    )
    .expect("engine load");
 
    let embs = engine
        .embed_texts(&["hello world".to_string(), "a cat on a mat".to_string()], false)
        .expect("embed_texts");
    assert_eq!(embs.len(), 2);
    assert_eq!(embs[0].len(), 384);
    let norm: f32 = embs[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "embedding not normalized: {norm}");
    println!("EMBED OK: dim={} norm={:.4}", embs[0].len(), norm);

    let whisper = WhisperEngine::load(&format!("{dir}/ggml-base-q5_1.bin")).expect("whisper load");
    // 2s of a 440 Hz tone at 16 kHz: just exercises the full pipeline.
    let samples: Vec<f32> = (0..32000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.1)
        .collect();
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cues = whisper.transcribe(&samples, cancel, |_| {}).expect("transcribe");
    println!("WHISPER OK: {} cues", cues.len());

    // Re-create an embed context after whisper ran, mirroring the pipeline order.
    let embs2 = engine.embed_texts(&["query".to_string()], true).expect("embed after whisper");
    println!("EMBED-AFTER-WHISPER OK: dim={}", embs2[0].len());

    // Drop engines before exit (exit-time Metal assert pitfall).
    drop(whisper);
    drop(engine);
    println!("ALL OK");
}

fn dirs_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/Library/Application Support/com.canpixel.smoltalk/models")
}
