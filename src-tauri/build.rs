fn main() {
    build_ocr_helper();
    tauri_build::build()
}

/// Compile the Vision OCR helper (ocr/main.swift) into OUT_DIR. The app embeds
/// the binary and runs it as a child process, so a Vision failure cannot take
/// the app down with it.
fn build_ocr_helper() {
    println!("cargo:rerun-if-changed=ocr/main.swift");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("youscoop-ocr");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        std::fs::write(&out, b"").unwrap();
        return;
    }
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "aarch64" => "arm64",
        other => other,
    }
    .to_string();
    let status = std::process::Command::new("xcrun")
        .args(["swiftc", "-O", "-target", &format!("{arch}-apple-macos13.0"), "ocr/main.swift", "-o"])
        .arg(&out)
        .status()
        .expect("xcrun swiftc is required to build the OCR helper (install the Xcode command line tools)");
    assert!(status.success(), "Compiling ocr/main.swift failed");
}
