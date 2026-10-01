fn main()->anyhow::Result<()> {
    let args:Vec<_>=std::env::args().collect();
    anyhow::ensure!(args.len()==2||args.len()==3,"Usage: native_runtime models | voice PCM_JSON");
    println!("{}",serde_json::to_string_pretty(&sacrinomicon_lib::verify_archive_runtime(&args[1],args.get(2).cloned())?)?);
    Ok(())
}
