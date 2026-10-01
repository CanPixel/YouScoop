//! Portable library snapshots contain typed data, never an executable database.
use crate::{
    Archive, Assessment, Asset, Claim, Concept, Evidence, Relationship, Source, SourcePackage,
};
use anyhow::{bail, Context, Result};
use rusqlite::{params, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::File,
    io::{Read, Write},
    path::Path,
};

const FORMAT: &str = "sacrinomicon-library";
const MANIFEST_LIMIT: u64 = 128 * 1024 * 1024;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PreservedEvidence {
    pub evidence: Evidence,
    pub current: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceAsset {
    pub source_id: String,
    pub asset: Asset,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotationConflict {
    pub id: String,
    pub incoming: Assessment,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AnnotationDecision {
    pub id: String,
    pub choice: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SourceConflict {
    pub source_id: String,
    pub revision: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LibraryBackup {
    pub format: String,
    pub schema_version: u32,
    pub exported_at: u64,
    pub sources: Vec<Source>,
    pub revisions: Vec<SourcePackage>,
    pub evidence: Vec<PreservedEvidence>,
    pub claims: Vec<Claim>,
    pub concepts: Vec<Concept>,
    pub relationships: Vec<Relationship>,
    pub assessments: Vec<Assessment>,
    pub assessment_history: Vec<Assessment>,
    pub assets: Vec<SourceAsset>,
    pub annotation_conflicts: Vec<AnnotationConflict>,
    pub heads: BTreeMap<String, String>,
    pub annotation_decisions: Vec<AnnotationDecision>,
    pub source_conflicts: Vec<SourceConflict>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImportReport {
    pub source_ids: Vec<String>,
    pub annotation_conflicts: usize,
    pub source_conflicts: Vec<String>,
}
fn assessment_id(a: &Assessment) -> Result<String> {
    Ok(crate::stable_id("assessment", &serde_json::to_string(a)?))
}
pub(crate) fn keep_assessment(tx: &Transaction<'_>, a: &Assessment) -> Result<()> {
    tx.execute(
        "INSERT OR IGNORE INTO assessment_history VALUES(?1,?2)",
        params![assessment_id(a)?, serde_json::to_string(a)?],
    )?;
    Ok(())
}
impl LibraryBackup {
    fn inventory(&self) -> Result<BTreeMap<String, u64>> {
        let mut files = BTreeMap::new();
        for a in self
            .assets
            .iter()
            .map(|a| &a.asset)
            .chain(self.revisions.iter().flat_map(|r| r.assets.iter()))
        {
            if a.sha256.len() != 64
                || !a.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                || a.bytes > 4 * 1024_u64.pow(3)
            {
                bail!("Invalid backup artifact");
            }
            if files
                .insert(a.sha256.clone(), a.bytes)
                .is_some_and(|n| n != a.bytes)
            {
                bail!("Conflicting artifact lengths");
            }
        }
        if files.len() > 100_000 || files.values().sum::<u64>() > 8 * 1024_u64.pow(3) {
            bail!("Library exceeds portable export limits");
        }
        Ok(files)
    }
    pub fn validate(&self) -> Result<()> {
        if self.format != FORMAT || self.schema_version != 1 {
            bail!("Unsupported library backup format");
        }
        if self.sources.len() > 10_000
            || self.revisions.len() > 100_000
            || self.evidence.len() > 1_000_000
            || self.assessment_history.len() > 1_000_000
            || self.assessments.len() > 1_000_000
        {
            bail!("Library backup is too large");
        }
        let mut ids = HashSet::new();
        for id in self
            .sources
            .iter()
            .map(|s| &s.id)
            .chain(self.evidence.iter().map(|e| &e.evidence.id))
            .chain(self.claims.iter().map(|c| &c.id))
            .chain(self.concepts.iter().map(|c| &c.id))
            .chain(self.relationships.iter().map(|r| &r.id))
        {
            if id.is_empty() || id.len() > 200 || !ids.insert(id.clone()) {
                bail!("Duplicate library record ID");
            }
        }
        let evidence: HashSet<_> = self
            .evidence
            .iter()
            .map(|e| e.evidence.id.as_str())
            .collect();
        let sources: HashSet<_> = self.sources.iter().map(|s| s.id.as_str()).collect();
        let nodes: HashSet<_> = self
            .sources
            .iter()
            .map(|s| s.id.as_str())
            .chain(evidence.iter().copied())
            .chain(self.claims.iter().map(|c| c.id.as_str()))
            .chain(self.concepts.iter().map(|c| c.id.as_str()))
            .collect();
        let mut revisions = HashSet::new();
        for r in &self.revisions {
            r.validate()?;
            if !sources.contains(r.source.id.as_str())
                || !revisions.insert((&r.source.id, &r.revision))
            {
                bail!("Invalid revision history");
            }
        }
        if self.heads.len() != self.sources.len()
            || self.heads.iter().any(|(id, head)| {
                !sources.contains(id.as_str()) || !revisions.contains(&(id, head))
            })
        {
            bail!("Invalid active source revisions");
        }
        if self
            .source_conflicts
            .iter()
            .any(|c| !revisions.contains(&(&c.source_id, &c.revision)))
        {
            bail!("Invalid source revision conflict");
        }
        for source in &self.sources {
            let mut historical_source = source.clone();
            historical_source.duration_seconds = self
                .revisions
                .iter()
                .filter(|r| r.source.id == source.id)
                .map(|r| r.source.duration_seconds)
                .fold(source.duration_seconds, f64::max);
            // History can contain passages beyond a subsequently shortened source.
            SourcePackage {
                schema_version: crate::SCHEMA_VERSION,
                revision: "backup-validation".into(),
                exported_at: 0,
                source: historical_source,
                embedding_model: crate::EMBEDDING_MODEL.into(),
                evidence: self
                    .evidence
                    .iter()
                    .filter(|e| e.evidence.source_id == source.id)
                    .map(|e| e.evidence.clone())
                    .collect(),
                claims: self
                    .claims
                    .iter()
                    .filter(|c| c.source_id == source.id)
                    .cloned()
                    .collect(),
                concepts: vec![],
                relationships: vec![],
                assets: self
                    .assets
                    .iter()
                    .filter(|a| a.source_id == source.id)
                    .map(|a| a.asset.clone())
                    .collect(),
            }
            .validate()?;
        }
        if self
            .evidence
            .iter()
            .any(|e| !sources.contains(e.evidence.source_id.as_str()))
            || self
                .claims
                .iter()
                .any(|c| !sources.contains(c.source_id.as_str()))
            || self
                .assets
                .iter()
                .any(|a| !sources.contains(a.source_id.as_str()))
        {
            bail!("Record has no preserved source");
        }
        for c in &self.concepts {
            if c.label.trim().is_empty()
                || c.label.len() > 200
                || c.evidence_ids.is_empty()
                || c.evidence_ids
                    .iter()
                    .any(|id| !evidence.contains(id.as_str()))
            {
                bail!("Invalid concept anchors");
            }
        }
        for r in &self.relationships {
            if r.from_id == r.to_id
                || !nodes.contains(r.from_id.as_str())
                || !nodes.contains(r.to_id.as_str())
                || r.evidence_ids.is_empty()
                || r.evidence_ids
                    .iter()
                    .any(|id| !evidence.contains(id.as_str()))
            {
                bail!("Invalid relationship anchors");
            }
        }
        let valid_assessment = |a: &Assessment| {
            nodes.contains(a.entity_id.as_str())
                && a.note.len() <= 100_000
                && crate::SpellingCorrection::valid_all(&a.source_spellings)
                && (a.source_spellings.is_empty() || sources.contains(a.entity_id.as_str()))
                && a.corrected_text.as_ref().is_none_or(|text| {
                    evidence.contains(a.entity_id.as_str())
                        && !text.trim().is_empty()
                        && text.len() <= 100_000
                })
                && [
                    "unreviewed",
                    "supported",
                    "contested",
                    "uncertain",
                    "belief",
                    "rejected",
                    "inbox",
                    "collected",
                ]
                .contains(&a.status.as_str())
        };
        let mut assessed = HashSet::new();
        for a in &self.assessments {
            if !valid_assessment(a) || !assessed.insert(&a.entity_id) {
                bail!("Invalid assessment");
            }
        }
        for a in &self.assessment_history {
            if !valid_assessment(a) {
                bail!("Invalid historical assessment");
            }
        }
        for c in &self.annotation_conflicts {
            if !valid_assessment(&c.incoming) || c.id != assessment_id(&c.incoming)? {
                bail!("Invalid annotation conflict");
            }
        }
        for d in &self.annotation_decisions {
            if !d.id.starts_with("assessment:")
                || d.id.len() != 75
                || !d.id[11..].bytes().all(|b| b.is_ascii_hexdigit())
                || !["local", "incoming", "edited"].contains(&d.choice.as_str())
            {
                bail!("Invalid annotation decision");
            }
        }
        self.inventory()?;
        Ok(())
    }
}
impl Archive {
    pub fn source_conflicts(&self) -> Result<Vec<SourceConflict>> {
        let mut q = self.conn.prepare(
            "SELECT source_id,revision FROM source_conflicts ORDER BY source_id,revision",
        )?;
        let rows = q.query_map([], |r| {
            Ok(SourceConflict {
                source_id: r.get(0)?,
                revision: r.get(1)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
    pub fn activate_revision(&mut self, source_id: &str, revision: &str) -> Result<()> {
        let tx = self.conn.transaction()?;
        let manifest: String = tx
            .query_row(
                "SELECT manifest FROM revisions WHERE source_id=?1 AND revision=?2",
                params![source_id, revision],
                |r| r.get(0),
            )
            .optional()?
            .context("Source revision not found")?;
        let p: SourcePackage = serde_json::from_str(&manifest)?;
        p.validate()?;
        tx.execute(
            "UPDATE sources SET data=?2 WHERE id=?1",
            params![source_id, serde_json::to_string(&p.source)?],
        )?;
        tx.execute(
            "UPDATE source_heads SET revision=?2 WHERE source_id=?1",
            params![source_id, revision],
        )?;
        tx.execute(
            "UPDATE evidence SET current=0 WHERE source_id=?1",
            params![source_id],
        )?;
        for e in &p.evidence {
            tx.execute(
                "UPDATE evidence SET current=1,data=?2 WHERE id=?1",
                params![e.id, serde_json::to_string(e)?],
            )?;
        }
        for a in &p.assets {
            tx.execute(
                "UPDATE assets SET data=?3 WHERE source_id=?1 AND id=?2",
                params![source_id, a.id, serde_json::to_string(a)?],
            )?;
        }
        tx.execute(
            "DELETE FROM source_conflicts WHERE source_id=?1",
            params![source_id],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn annotation_conflicts(&self) -> Result<Vec<AnnotationConflict>> {
        let mut q=self.conn.prepare("SELECT c.id,h.data FROM annotation_conflicts c JOIN assessment_history h ON h.id=c.id ORDER BY c.entity_id,c.id")?;
        let rows = q.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        rows.map(|row| {
            let (id, data) = row?;
            Ok(AnnotationConflict {
                id,
                incoming: serde_json::from_str(&data)?,
            })
        })
        .collect()
    }
    pub fn resolve_annotation(&self, id: &str, use_incoming: bool) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        let data:String=tx.query_row("SELECT h.data FROM annotation_conflicts c JOIN assessment_history h ON h.id=c.id WHERE c.id=?1",params![id],|r|r.get(0)).optional()?.context("Annotation conflict not found")?;
        let incoming: Assessment = serde_json::from_str(&data)?;
        if use_incoming {
            tx.execute("INSERT INTO assessments(entity_id,status,note,updated_at,corrected_text,source_spellings) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(entity_id) DO UPDATE SET status=excluded.status,note=excluded.note,updated_at=excluded.updated_at,corrected_text=excluded.corrected_text,source_spellings=excluded.source_spellings",params![incoming.entity_id,incoming.status,incoming.note,incoming.updated_at as i64,incoming.corrected_text,serde_json::to_string(&incoming.source_spellings)?])?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO annotation_decisions VALUES(?1,?2)",
            params![id, if use_incoming { "incoming" } else { "local" }],
        )?;
        tx.execute("DELETE FROM annotation_conflicts WHERE id=?1", params![id])?;
        tx.commit()?;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<LibraryBackup> {
        // One read transaction includes WAL changes consistently during export.
        let tx = self.conn.unchecked_transaction()?;
        let summary = self.summary()?;
        let mut q = self
            .conn
            .prepare("SELECT data,current FROM evidence ORDER BY id")?;
        let evidence = q
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?)))?
            .map(|r| {
                let (data, current) = r?;
                Ok(PreservedEvidence {
                    evidence: serde_json::from_str(&data)?,
                    current,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut q = self
            .conn
            .prepare("SELECT source_id,data FROM assets ORDER BY source_id,id")?;
        let assets = q
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .map(|r| {
                let (source_id, data) = r?;
                Ok(SourceAsset {
                    source_id,
                    asset: serde_json::from_str(&data)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut q = self
            .conn
            .prepare("SELECT source_id,revision FROM source_heads")?;
        let heads = q
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<rusqlite::Result<BTreeMap<_, _>>>()?;
        let mut q = self
            .conn
            .prepare("SELECT id,choice FROM annotation_decisions ORDER BY id")?;
        let annotation_decisions = q
            .query_map([], |r| {
                Ok(AnnotationDecision {
                    id: r.get(0)?,
                    choice: r.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut q = self.conn.prepare(
            "SELECT source_id,revision FROM source_conflicts ORDER BY source_id,revision",
        )?;
        let source_conflicts = q
            .query_map([], |r| {
                Ok(SourceConflict {
                    source_id: r.get(0)?,
                    revision: r.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let backup = LibraryBackup {
            format: FORMAT.into(),
            schema_version: 1,
            exported_at: crate::now(),
            sources: summary.sources,
            revisions: self.all("SELECT manifest FROM revisions ORDER BY rowid")?,
            evidence,
            claims: summary.claims,
            concepts: summary.concepts,
            relationships: summary.relationships,
            assessments: summary.assessments,
            assessment_history: self.all("SELECT data FROM assessment_history ORDER BY id")?,
            assets,
            annotation_conflicts: summary.annotation_conflicts,
            heads,
            annotation_decisions,
            source_conflicts,
        };
        tx.commit()?;
        backup.validate()?;
        Ok(backup)
    }
    pub fn export_library(&self, path: &Path) -> Result<()> {
        let backup = self.snapshot()?;
        let manifest = serde_json::to_vec(&backup)?;
        if manifest.len() as u64 > MANIFEST_LIMIT {
            bail!("Library manifest exceeds export limits");
        }
        let parent = path.parent().context("Export destination has no parent")?;
        std::fs::create_dir_all(parent)?;
        let mut temp = tempfile::NamedTempFile::new_in(parent)?;
        {
            let mut zip = zip::ZipWriter::new(temp.as_file_mut());
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("manifest.json", options)?;
            zip.write_all(&manifest)?;
            for (digest, bytes) in backup.inventory()? {
                let asset = self.root.join("assets").join(&digest);
                let before = crate::package::file_digest(&asset)?;
                if before != (digest.clone(), bytes) {
                    bail!("Preserved artifact failed export verification");
                }
                zip.start_file(format!("assets/{digest}"), options)?;
                let mut input = File::open(&asset)?;
                let mut hash = Sha256::new();
                let mut count = 0_u64;
                let mut buffer = [0_u8; 65536];
                loop {
                    let n = input.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    hash.update(&buffer[..n]);
                    count += n as u64;
                    zip.write_all(&buffer[..n])?;
                }
                if crate::hex(&hash.finalize()) != digest || count != bytes {
                    bail!("Artifact changed during export");
                }
            }
            zip.finish()?;
        }
        temp.as_file().sync_all()?;
        temp.persist(path).map_err(|e| e.error)?;
        Ok(())
    }
    pub fn import_portable(&mut self, path: &Path) -> Result<ImportReport> {
        let mut zip = zip::ZipArchive::new(File::open(path)?)?;
        let mut entry = zip.by_name("manifest.json")?;
        if entry.size() > MANIFEST_LIMIT {
            bail!("Portable manifest is too large");
        }
        let mut bytes = Vec::new();
        entry
            .by_ref()
            .take(MANIFEST_LIMIT + 1)
            .read_to_end(&mut bytes)?;
        drop(entry);
        #[derive(Deserialize)]
        struct Header {
            format: Option<String>,
        }
        let header: Header = serde_json::from_slice(&bytes)?;
        if header.format.is_none() {
            drop(zip);
            return Ok(ImportReport {
                source_ids: vec![self.import_file(path)?],
                annotation_conflicts: 0,
                source_conflicts: vec![],
            });
        }
        drop(header);
        let backup: LibraryBackup = serde_json::from_slice(&bytes)?;
        backup.validate()?;
        let inventory = backup.inventory()?;
        if zip.len() != inventory.len() + 1 {
            bail!("Library contains unlisted artifacts");
        }
        let temp = tempfile::tempdir()?;
        let mut seen = HashSet::new();
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i)?;
            let name = entry.name().to_string();
            if !seen.insert(name.clone()) {
                bail!("Duplicate library entry");
            }
            if name == "manifest.json" {
                continue;
            }
            let digest = name
                .strip_prefix("assets/")
                .context("Unlisted library entry")?;
            let bytes = *inventory.get(digest).context("Unlisted library artifact")?;
            if entry.size() != bytes || entry.is_dir() {
                bail!("Library artifact size mismatch");
            }
            let path = temp.path().join(digest);
            let mut file = File::create(&path)?;
            if std::io::copy(&mut entry.by_ref().take(bytes + 1), &mut file)? != bytes
                || crate::package::file_digest(&path)? != (digest.into(), bytes)
            {
                bail!("Library artifact checksum mismatch");
            }
        }
        // Verify every artifact before the transaction can change knowledge.
        for (digest, bytes) in &inventory {
            let target = self.root.join("assets").join(digest);
            if target.exists() {
                if crate::package::file_digest(&target)? != (digest.clone(), *bytes) {
                    bail!("Existing archive artifact is corrupt");
                }
            } else {
                let mut file = tempfile::NamedTempFile::new_in(self.root.join("assets"))?;
                std::io::copy(
                    &mut File::open(temp.path().join(digest))?,
                    file.as_file_mut(),
                )?;
                file.as_file().sync_all()?;
                file.persist(target).map_err(|e| e.error)?;
            }
        }
        self.merge_backup(&backup)
    }
    fn merge_backup(&mut self, backup: &LibraryBackup) -> Result<ImportReport> {
        let existing = self.snapshot()?;
        let existing_sources: HashMap<_, _> = existing.sources.iter().map(|s| (&s.id, s)).collect();
        let tx = self.conn.transaction()?;
        let mut keep_sources = HashSet::new();
        let mut source_conflicts = Vec::new();
        for source in &backup.sources {
            if existing_sources.contains_key(&source.id) {
                let local = existing.heads.get(&source.id);
                let incoming = backup.heads.get(&source.id);
                let is_descendant = local.zip(incoming).is_some_and(|(l, r)| {
                    l != r
                        && backup
                            .revisions
                            .iter()
                            .any(|p| p.source.id == source.id && &p.revision == l)
                        && !existing
                            .revisions
                            .iter()
                            .any(|p| p.source.id == source.id && &p.revision == r)
                });
                if !is_descendant {
                    keep_sources.insert(source.id.clone());
                }
                if local.zip(incoming).is_some_and(|(l, r)| {
                    l != r
                        && !is_descendant
                        && (backup
                            .revisions
                            .iter()
                            .any(|p| p.source.id == source.id && &p.revision == l)
                            || !existing
                                .revisions
                                .iter()
                                .any(|p| p.source.id == source.id && &p.revision == r))
                }) {
                    source_conflicts.push(source.id.clone());
                }
            }
        }
        for r in &backup.revisions {
            crate::store::import_records(&tx, r)?;
        }
        for id in &source_conflicts {
            tx.execute(
                "INSERT OR IGNORE INTO source_conflicts VALUES(?1,?2)",
                params![id, backup.heads[id]],
            )?;
        }
        for c in &backup.source_conflicts {
            tx.execute(
                "INSERT OR IGNORE INTO source_conflicts VALUES(?1,?2)",
                params![c.source_id, c.revision],
            )?;
        }
        for source in &backup.sources {
            let value = existing_sources
                .get(&source.id)
                .filter(|_| keep_sources.contains(&source.id))
                .copied()
                .unwrap_or(source);
            tx.execute("INSERT INTO sources VALUES(?1,?2,?3) ON CONFLICT(id) DO UPDATE SET data=excluded.data",params![value.id,serde_json::to_string(value)?,crate::now() as i64])?;
            let head = if keep_sources.contains(&source.id) {
                &existing.heads[&source.id]
            } else {
                &backup.heads[&source.id]
            };
            tx.execute("INSERT INTO source_heads VALUES(?1,?2) ON CONFLICT(source_id) DO UPDATE SET revision=excluded.revision",params![source.id,head])?;
            tx.execute(
                "UPDATE evidence SET current=0 WHERE source_id=?1",
                params![source.id],
            )?;
        }
        for row in &backup.evidence {
            let e = &row.evidence;
            let previous: Option<String> = tx
                .query_row(
                    "SELECT data FROM evidence WHERE id=?1",
                    params![e.id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(p) = previous {
                let p: Evidence = serde_json::from_str(&p)?;
                if p.source_id != e.source_id
                    || p.text != e.text
                    || p.start != e.start
                    || p.end != e.end
                    || p.kind != e.kind
                {
                    bail!("Evidence ID collision in backup");
                }
            }
            tx.execute(
                "INSERT OR IGNORE INTO evidence VALUES(?1,?2,?3,0)",
                params![e.id, e.source_id, serde_json::to_string(e)?],
            )?;
            tx.execute("DELETE FROM evidence_fts WHERE id=?1", params![e.id])?;
            tx.execute(
                "INSERT INTO evidence_fts VALUES(?1,?2)",
                params![e.id, e.text],
            )?;
            if row.current && !keep_sources.contains(&e.source_id) {
                tx.execute("UPDATE evidence SET current=1 WHERE id=?1", params![e.id])?;
            }
        }
        for row in &existing.evidence {
            if row.current && keep_sources.contains(&row.evidence.source_id) {
                tx.execute(
                    "UPDATE evidence SET data=?2,current=1 WHERE id=?1",
                    params![row.evidence.id, serde_json::to_string(&row.evidence)?],
                )?;
            }
        }
        for c in &backup.claims {
            crate::store::immutable_record(&tx, "claims", &c.id, c)?;
            tx.execute(
                "INSERT OR IGNORE INTO claims VALUES(?1,?2,?3)",
                params![c.id, c.source_id, serde_json::to_string(c)?],
            )?;
        }
        for c in &backup.concepts {
            let previous: Option<String> = tx
                .query_row(
                    "SELECT data FROM concepts WHERE id=?1",
                    params![c.id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(p) = previous {
                let p: Concept = serde_json::from_str(&p)?;
                if p.label != c.label || p.aliases != c.aliases {
                    bail!("Concept ID collision in backup");
                }
            }
            tx.execute(
                "INSERT OR IGNORE INTO concepts VALUES(?1,?2)",
                params![c.id, serde_json::to_string(c)?],
            )?;
            for id in &c.evidence_ids {
                tx.execute(
                    "INSERT OR IGNORE INTO concept_evidence VALUES(?1,?2)",
                    params![c.id, id],
                )?;
            }
        }
        for r in &backup.relationships {
            crate::store::immutable_record(&tx, "relationships", &r.id, r)?;
            tx.execute(
                "INSERT OR IGNORE INTO relationships VALUES(?1,?2)",
                params![r.id, serde_json::to_string(r)?],
            )?;
        }
        for a in &backup.assets {
            tx.execute("INSERT INTO assets VALUES(?1,?2,?3) ON CONFLICT(source_id,id) DO UPDATE SET data=excluded.data",params![a.source_id,a.asset.id,serde_json::to_string(&a.asset)?])?;
        }
        for a in &existing.assets {
            if keep_sources.contains(&a.source_id) {
                tx.execute(
                    "UPDATE assets SET data=?3 WHERE source_id=?1 AND id=?2",
                    params![a.source_id, a.asset.id, serde_json::to_string(&a.asset)?],
                )?;
            }
        }
        for a in existing
            .assessments
            .iter()
            .chain(&backup.assessment_history)
            .chain(&backup.assessments)
        {
            keep_assessment(&tx, a)?;
        }
        for d in &backup.annotation_decisions {
            tx.execute(
                "INSERT OR IGNORE INTO annotation_decisions VALUES(?1,?2)",
                params![d.id, d.choice],
            )?;
        }
        for a in &backup.assessments {
            let local = existing
                .assessments
                .iter()
                .find(|p| p.entity_id == a.entity_id);
            if local.is_none() {
                tx.execute(
                    "INSERT INTO assessments(entity_id,status,note,updated_at,corrected_text,source_spellings) VALUES(?1,?2,?3,?4,?5,?6)",
                    params![a.entity_id, a.status, a.note, a.updated_at as i64, a.corrected_text,serde_json::to_string(&a.source_spellings)?],
                )?;
            } else if local.is_some_and(|p| {
                p.status != a.status
                    || p.note != a.note
                    || p.corrected_text != a.corrected_text
                    || p.source_spellings != a.source_spellings
            }) {
                tx.execute("INSERT OR IGNORE INTO annotation_conflicts SELECT ?1,?2 WHERE NOT EXISTS(SELECT 1 FROM annotation_decisions WHERE id=?1)",params![assessment_id(a)?,a.entity_id])?;
            }
        }
        for c in &backup.annotation_conflicts {
            keep_assessment(&tx, &c.incoming)?;
            tx.execute("INSERT OR IGNORE INTO annotation_conflicts SELECT ?1,?2 WHERE NOT EXISTS(SELECT 1 FROM annotation_decisions WHERE id=?1)",params![c.id,c.incoming.entity_id])?;
        }
        tx.commit()?;
        let mut q = self
            .conn
            .prepare("SELECT DISTINCT source_id FROM source_conflicts ORDER BY source_id")?;
        let source_conflicts = q
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ImportReport {
            source_ids: backup.sources.iter().map(|s| s.id.clone()).collect(),
            annotation_conflicts: self.annotation_conflicts()?.len(),
            source_conflicts,
        })
    }
}
