fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Voice separation relaunches the current binary, which here is this example.
    if args.get(1).map(String::as_str) == Some("--diarize") {
        std::process::exit(youscoop_lib::voices::child_main(&args[2..]));
    }
    anyhow::ensure!(args.len() == 2, "Usage: enrich_video VIDEO_ID");
    let result = youscoop_lib::verify_enrichment(&args[1])?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
