fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 2 || args.len() == 3,
        "Usage: desktop_setup runtime|omni|gemma|status|worker [LOCAL_SNAPSHOT_OR_REQUEST]"
    );
    let result = youscoop_lib::verify_desktop_setup(&args[1], args.get(2).cloned())?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
