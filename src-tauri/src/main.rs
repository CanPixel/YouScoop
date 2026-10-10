// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Ingestion runs voice separation in a child copy of this binary so it can be cancelled.
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--diarize") {
        std::process::exit(youscoop_lib::voices::child_main(&args[2..]));
    }
    youscoop_lib::run()
}
