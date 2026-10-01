//! Portable source preservation, provenance and archive storage shared by both apps.
mod backup;
mod package;
mod store;
mod drafts;
mod answers;
pub use answers::grounded_answer;
pub use drafts::{attributed_reading, navigation_drafts};
use anyhow::{bail, Result};
pub use backup::{AnnotationConflict, ImportReport, LibraryBackup, SourceConflict};
pub use package::{file_digest, read_package, write_package, AssetInput};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
pub use store::{
    corrected_spelling, Archive, ArchiveSummary, Assessment, RevisionInfo, SearchHit, SourceDetail,
    SpellingCorrection,
};

pub const SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_MODEL: &str = "bge-small-en-v1.5-q8_0";
pub const EMBEDDING_DIM: usize = 384;

pub fn stable_id(namespace: &str, content: &str) -> String {
    format!("{namespace}:{}", hex(&Sha256::digest(content.as_bytes())))
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Source {
    pub id: String,
    pub title: String,
    pub uri: String,
    pub publisher: String,
    pub duration_seconds: f64,
    pub captured_at: u64,
    pub summary: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Speech,
    Visual,
    Audiovisual,
    Document,
    Note,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Provenance {
    pub method: String,
    pub model: Option<String>,
    pub generated_at: u64,
    pub original_asset: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Evidence {
    pub id: String,
    pub source_id: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub kind: EvidenceKind,
    pub provenance: Provenance,
    #[serde(default)]
    pub vector: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Claim {
    pub id: String,
    pub source_id: String,
    pub statement: String,
    /// Speaker assertion, observation, interpretation, or metaphysical hypothesis.
    pub kind: String,
    pub evidence_ids: Vec<String>,
    pub provenance: Provenance,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Concept {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub evidence_ids: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    Mentions,
    Supports,
    Contradicts,
    Resembles,
    Influenced,
    Related,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Relationship {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub kind: RelationKind,
    pub evidence_ids: Vec<String>,
    pub explanation: String,
    pub created_by: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Asset {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourcePackage {
    pub schema_version: u32,
    pub revision: String,
    pub exported_at: u64,
    pub source: Source,
    pub embedding_model: String,
    pub evidence: Vec<Evidence>,
    pub claims: Vec<Claim>,
    pub concepts: Vec<Concept>,
    pub relationships: Vec<Relationship>,
    pub assets: Vec<Asset>,
}

impl SourcePackage {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != SCHEMA_VERSION {
            bail!("Unsupported source package version {}", self.schema_version);
        }
        if self.source.id.is_empty()
            || self.source.id.len() > 200
            || self.source.title.trim().is_empty()
            || self.revision.is_empty()
            || !self.source.duration_seconds.is_finite()
            || self.source.duration_seconds < 0.0
            || self.source.duration_seconds > 7200.0
        {
            bail!("Invalid source metadata");
        }
        if self.assets.len() > 1024
            || self.evidence.len() > 100_000
            || self.concepts.len() > 10_000
            || self.claims.len() > 100_000
            || self.relationships.len() > 100_000
        {
            bail!("Package exceeds archive limits");
        }
        let mut ids = HashSet::new();
        ids.insert(self.source.id.as_str());
        for id in self
            .evidence
            .iter()
            .map(|e| e.id.as_str())
            .chain(self.claims.iter().map(|c| c.id.as_str()))
            .chain(self.concepts.iter().map(|c| c.id.as_str()))
            .chain(self.relationships.iter().map(|r| r.id.as_str()))
            .chain(self.assets.iter().map(|a| a.id.as_str()))
        {
            if id.is_empty() || id.len() > 200 || !ids.insert(id) {
                bail!("Invalid or duplicate record ID");
            }
        }
        let evidence_ids: HashSet<_> = self.evidence.iter().map(|e| e.id.as_str()).collect();
        let graph_ids: HashSet<_> = std::iter::once(self.source.id.as_str())
            .chain(self.evidence.iter().map(|e| e.id.as_str()))
            .chain(self.claims.iter().map(|c| c.id.as_str()))
            .chain(self.concepts.iter().map(|c| c.id.as_str()))
            .collect();
        let asset_ids: HashSet<_> = self.assets.iter().map(|a| a.id.as_str()).collect();
        for asset in &self.assets {
            if !safe_asset_name(&asset.name)
                || asset.sha256.len() != 64
                || !asset.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || asset.bytes > 4 * 1024_u64.pow(3)
            {
                bail!("Invalid asset manifest");
            }
        }
        let mut names = HashSet::new();
        for asset in &self.assets {
            if !names.insert(&asset.name) {
                bail!("Duplicate asset filename");
            }
        }
        for e in &self.evidence {
            if e.source_id != self.source.id
                || !e.start.is_finite()
                || !e.end.is_finite()
                || e.start < 0.0
                || e.end < e.start
                || e.end > self.source.duration_seconds + 1.0
                || e.text.trim().is_empty()
                || e.text.len() > 100_000
            {
                bail!("Invalid evidence passage");
            }
            if !e.vector.is_empty()
                && (self.embedding_model != EMBEDDING_MODEL
                    || e.vector.len() != EMBEDDING_DIM
                    || e.vector.iter().any(|v| !v.is_finite()))
            {
                bail!("Incompatible evidence embedding");
            }
            if e.provenance
                .original_asset
                .as_ref()
                .is_some_and(|id| !asset_ids.contains(id.as_str()))
            {
                bail!("Evidence points to a missing original artifact");
            }
        }
        let valid_refs = |refs: &[String]| refs.iter().all(|id| evidence_ids.contains(id.as_str()));
        for c in &self.claims {
            if c.source_id != self.source.id
                || c.statement.trim().is_empty()
                || c.evidence_ids.is_empty()
                || !valid_refs(&c.evidence_ids)
            {
                bail!("Claim lacks preserved source evidence");
            }
        }
        for c in &self.concepts {
            if c.label.trim().is_empty()
                || c.label.len() > 200
                || c.evidence_ids.is_empty()
                || !valid_refs(&c.evidence_ids)
            {
                bail!("Concept lacks preserved source evidence");
            }
        }
        for r in &self.relationships {
            if !graph_ids.contains(r.from_id.as_str())
                || !graph_ids.contains(r.to_id.as_str())
                || r.from_id == r.to_id
                || r.evidence_ids.is_empty()
                || !valid_refs(&r.evidence_ids)
            {
                bail!("Relationship lacks valid nodes or evidence");
            }
        }
        Ok(())
    }
}
pub(crate) fn safe_asset_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() < 200
        && name != "."
        && name != ".."
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
