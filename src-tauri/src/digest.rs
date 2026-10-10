//! Content identities: SHA-256 digests of files and of strings.
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A stable identity for content within a namespace, e.g. "runtime:3f2a…".
pub fn stable_id(namespace: &str, content: &str) -> String {
    format!("{namespace}:{}", hex(&Sha256::digest(content.as_bytes())))
}

/// A file's SHA-256 and its length, read in 64 KiB blocks.
pub fn file_digest(path: &Path) -> Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = 0;
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok((hex(&hash.finalize()), bytes))
}

/// Seconds since the Unix epoch.
pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    #[test]
    fn ids_are_stable_sha256() {
        assert_eq!(super::stable_id("x", ""), "x:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }
}
