use knowledge_core::Archive;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len()>=3,"Usage: archive LIBRARY summary | import FILE | search QUERY | assess ID STATUS NOTE | concept LABEL EVIDENCE_ID");
    let mut archive = Archive::open(&args[1])?;
    let output = match args[2].as_str() {
        "summary" => serde_json::to_value(archive.summary()?)?,
        "import" if args.len() == 4 => {
            serde_json::to_value(archive.import_portable(std::path::Path::new(&args[3]))?)?
        }
        "export" if args.len() == 4 => {
            archive.export_library(std::path::Path::new(&args[3]))?;
            serde_json::json!({"exported":args[3]})
        }
        "search" if args.len() == 4 => {
            serde_json::to_value(archive.search(&args[3], None, None, 20)?)?
        }
        "assess" if args.len() == 6 => {
            serde_json::to_value(archive.assess(&args[3], &args[4], &args[5])?)?
        }
        "concept" if args.len() == 5 => {
            serde_json::to_value(archive.add_concept(&args[3], &args[4])?)?
        }
        _ => anyhow::bail!("Invalid archive operation"),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
