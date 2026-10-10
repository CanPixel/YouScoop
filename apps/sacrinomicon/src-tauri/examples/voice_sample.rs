//! Render the Oracle's candidate voices to WAV files for listening checks.
//! Usage: voice_sample KOKORO_DIR OUTPUT_DIR [TEXT]
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("The neural voice renders on macOS only");
}
#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 3, "Usage: voice_sample KOKORO_DIR OUTPUT_DIR [TEXT]");
    let text = args.get(3).cloned().unwrap_or_else(|| {
        "The archive remembers what was said. Follow the thread, and it leads back to the old names, Baal and Moloch, spoken again in a modern ledger.".into()
    });
    let voice = sacrinomicon_lib::OracleVoice::load(std::path::Path::new(&args[1]))?;
    std::fs::create_dir_all(&args[2])?;
    for name in sacrinomicon_lib::ORACLE_VOICES.iter().map(|v| v.id) {
        let start = std::time::Instant::now();
        let wav = voice.render(&text, name)?;
        let path = std::path::Path::new(&args[2]).join(format!("{name}.wav"));
        std::fs::write(&path, &wav)?;
        println!("{name}: {:.2}s -> {}", start.elapsed().as_secs_f64(), path.display());
    }
    Ok(())
}
