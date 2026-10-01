fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len()==4, "Usage: archive_video VIDEO_DIR EXPORT_DIR LIBRARY_DIR");
    let saved = youscoop_lib::archive::save_to(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]),std::path::Path::new(&args[3]))?;
    println!("{}",serde_json::to_string_pretty(&saved)?);
    Ok(())
}
