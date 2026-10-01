use crate::{hex, safe_asset_name, Asset, SourcePackage};
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::{NamedTempFile, TempDir};
use zip::write::SimpleFileOptions;

pub struct AssetInput {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub path: PathBuf,
}
pub fn file_digest(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
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
pub fn write_package(
    mut package: SourcePackage,
    inputs: &[AssetInput],
    output: &Path,
) -> Result<SourcePackage> {
    package.assets.clear();
    for input in inputs {
        if !safe_asset_name(&input.name) {
            bail!("Invalid artifact filename");
        }
        let (sha256, bytes) = file_digest(&input.path)?;
        package.assets.push(Asset {
            id: input.id.clone(),
            name: input.name.clone(),
            kind: input.kind.clone(),
            sha256,
            bytes,
        });
    }
    package.validate()?;
    let parent = output
        .parent()
        .context("Package destination has no parent")?;
    std::fs::create_dir_all(parent)?;
    let mut temp = NamedTempFile::new_in(parent)?;
    {
        let mut archive = zip::ZipWriter::new(temp.as_file_mut());
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        archive.start_file("manifest.json", options)?;
        archive.write_all(&serde_json::to_vec(&package)?)?;
        for (input, asset) in inputs.iter().zip(&package.assets) {
            archive.start_file(format!("assets/{}", asset.name), options)?;
            let mut file = File::open(&input.path)?;
            let mut hash = Sha256::new();
            let mut bytes = 0_u64;
            let mut buffer = [0_u8; 65536];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
                bytes += n as u64;
                archive.write_all(&buffer[..n])?;
            }
            if hex(&hash.finalize()) != asset.sha256 || bytes != asset.bytes {
                bail!("Artifact changed during export: {}", asset.name);
            }
        }
        archive.finish()?;
    }
    temp.as_file().sync_all()?;
    temp.persist(output).map_err(|e| e.error)?;
    Ok(package)
}

/// Verify all data before the archive database can change. No untrusted zip path is used directly.
pub fn read_package(path: &Path) -> Result<(SourcePackage, TempDir)> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;
    if zip.len() > 1025 {
        bail!("Too many package entries");
    }
    let mut manifest = zip.by_name("manifest.json")?;
    if manifest.size() > 32 * 1024 * 1024 {
        bail!("Package manifest is too large");
    }
    let mut bytes = Vec::new();
    manifest
        .by_ref()
        .take(32 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    drop(manifest);
    let package: SourcePackage = serde_json::from_slice(&bytes)?;
    package.validate()?;
    if zip.len() != package.assets.len() + 1 {
        bail!("Package contains unlisted artifacts");
    }
    let total: u64 = package.assets.iter().map(|a| a.bytes).sum();
    if total > 8 * 1024_u64.pow(3) {
        bail!("Package exceeds size limit");
    }
    let temp = tempfile::tempdir()?;
    let mut seen = std::collections::HashSet::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let name = entry.name().to_string();
        if !seen.insert(name.clone()) {
            bail!("Duplicate package entry");
        }
        if name == "manifest.json" {
            continue;
        }
        let asset = package
            .assets
            .iter()
            .find(|a| name == format!("assets/{}", a.name))
            .context("Unlisted package artifact")?;
        if entry.size() != asset.bytes || entry.is_dir() {
            bail!("Artifact size mismatch");
        }
        let target = temp.path().join(&asset.name);
        let mut file = File::create(&target)?;
        let copied = std::io::copy(&mut entry.by_ref().take(asset.bytes + 1), &mut file)?;
        file.sync_all()?;
        let (digest, actual_bytes) = file_digest(&target)?;
        if copied != asset.bytes || actual_bytes != asset.bytes || digest != asset.sha256 {
            bail!("Artifact checksum mismatch: {}", asset.name);
        }
    }
    Ok((package, temp))
}
