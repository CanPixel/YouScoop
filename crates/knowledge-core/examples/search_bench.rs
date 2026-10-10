//! Search cost at library scale. Usage: search_bench DIR SOURCES PASSAGES_PER_SOURCE
//! Builds a synthetic library, then times hybrid search against parsing every
//! passage's JSON record, which is what search did before vectors had their own table.
use knowledge_core::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 4, "Usage: search_bench DIR SOURCES PASSAGES_PER_SOURCE");
    let (sources, per): (usize, usize) = (args[2].parse()?, args[3].parse()?);
    let mut archive = Archive::open(&args[1])?;
    let words = ["account", "bank", "memo", "deity", "ritual", "email", "archive", "document", "voice", "symbol", "history", "temple", "money", "wire", "letter"];
    let mut seed = 7_u64;
    let mut next = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5 };
    let started = std::time::Instant::now();
    if archive.summary()?.sources.len() < sources {
        for s in 0..sources {
            let source_id = format!("source:bench-{s}");
            let evidence = (0..per).map(|i| {
                let mut vector: Vec<f32> = (0..EMBEDDING_DIM).map(|_| next()).collect();
                let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                vector.iter_mut().for_each(|v| *v /= norm);
                let text = (0..40).map(|j| words[(i * 7 + j * 3 + s) % words.len()]).collect::<Vec<_>>().join(" ");
                Evidence { id: format!("evidence:bench-{s}-{i}"), source_id: source_id.clone(), start: i as f64 * 30.0, end: i as f64 * 30.0 + 30.0, text,
                    kind: EvidenceKind::Speech, provenance: Provenance { method: "bench".into(), model: None, generated_at: 1, original_asset: None }, vector }
            }).collect();
            let package = SourcePackage { schema_version: SCHEMA_VERSION, revision: format!("revision:bench-{s}"), exported_at: 1,
                source: Source { id: source_id, title: format!("Bench {s}"), uri: String::new(), publisher: "bench".into(), duration_seconds: per as f64 * 30.0, captured_at: 1, summary: None },
                embedding_model: EMBEDDING_MODEL.into(), evidence, claims: vec![], concepts: vec![], relationships: vec![], assets: vec![] };
            archive.import(&package, std::path::Path::new(&args[1]))?;
        }
    }
    let built = started.elapsed().as_secs_f64();
    let query: Vec<f32> = { let v: Vec<f32> = (0..EMBEDDING_DIM).map(|_| next()).collect(); let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.into_iter().map(|x| x / n).collect() };
    let first = std::time::Instant::now();
    archive.search("bank account memo", Some(&query), None, 12)?;
    let first_ms = first.elapsed().as_secs_f64() * 1000.0;
    let timed = std::time::Instant::now();
    for _ in 0..5 { archive.search("bank account memo", Some(&query), None, 12)?; }
    let search_ms = timed.elapsed().as_secs_f64() * 1000.0 / 5.0;
    let vector_only = std::time::Instant::now();
    for _ in 0..5 { archive.search("zzqx vvqy", Some(&query), None, 12)?; }
    let vector_only_ms = vector_only.elapsed().as_secs_f64() * 1000.0 / 5.0;
    let text_only = std::time::Instant::now();
    for _ in 0..5 { archive.search("bank account memo", None, None, 12)?; }
    let text_only_ms = text_only.elapsed().as_secs_f64() * 1000.0 / 5.0;
    let scan = std::time::Instant::now();
    let conn = rusqlite::Connection::open(std::path::Path::new(&args[1]).join("archive.sqlite"))?;
    let mut statement = conn.prepare("SELECT data FROM evidence WHERE current=1")?;
    let mut best = 0.0_f32;
    for row in statement.query_map([], |r| r.get::<_, String>(0))? {
        let e: Evidence = serde_json::from_str(&row?)?;
        best = best.max(e.vector.iter().zip(&query).map(|(a, b)| a * b).sum());
    }
    let scan_ms = scan.elapsed().as_secs_f64() * 1000.0;
    println!("{}", serde_json::json!({"passages": sources * per, "buildSeconds": built, "firstSearchMsIncludingIndexBackfill": first_ms,
        "searchMs": search_ms, "vectorOnlyMs": vector_only_ms, "textOnlyMs": text_only_ms, "previousJsonScanMs": scan_ms, "check": best > 0.0}));
    Ok(())
}
