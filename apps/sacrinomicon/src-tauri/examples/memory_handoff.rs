fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "Usage: memory_handoff LOCK_DIRECTORY CHAT_MODEL EMBEDDING_MODEL MARKER"
    );
    sacrinomicon_lib::verify_memory_handoff(
        std::path::Path::new(&args[1]),
        &args[2],
        &args[3],
        std::path::Path::new(&args[4]),
    )
}
