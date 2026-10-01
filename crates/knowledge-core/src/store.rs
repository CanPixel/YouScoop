use crate::{now, Claim, Concept, Evidence, Relationship, Source, SourcePackage};
use anyhow::{bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct Archive {
    pub root: PathBuf,
    pub(crate) conn: Connection,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Assessment {
    pub entity_id: String,
    pub status: String,
    pub note: String,
    /// Human reading of an extraction, never a replacement for preserved evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrected_text: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_spellings: Vec<SpellingCorrection>,
    pub updated_at: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SpellingCorrection {
    pub original: String,
    pub corrected: String,
}
impl SpellingCorrection {
    pub(crate) fn valid_all(items: &[Self]) -> bool {
        let mut originals = std::collections::HashSet::new();
        items.len() <= 32
            && items.iter().all(|s| {
                !s.original.trim().is_empty()
                    && !s.corrected.trim().is_empty()
                    && s.original.len() <= 100
                    && s.corrected.len() <= 100
                    && s.original != s.corrected
                    && originals.insert(&s.original)
            })
    }
}
/// Case-sensitive, whole-term, single-pass substitutions. Never cascade edits.
pub fn corrected_spelling(text: &str, spellings: &[SpellingCorrection]) -> String {
    let mut output = String::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let term = spellings
            .iter()
            .filter(|s| !s.original.is_empty() && rest.starts_with(&s.original))
            .filter(|s| {
                let word = |c: char| c.is_alphanumeric() || c == '_';
                !text[..cursor].chars().next_back().is_some_and(word)
                    && !rest[s.original.len()..].chars().next().is_some_and(word)
            })
            .max_by_key(|s| s.original.len());
        if let Some(s) = term {
            output.push_str(&s.corrected);
            cursor += s.original.len();
        } else {
            let c = rest.chars().next().unwrap();
            output.push(c);
            cursor += c.len_utf8();
        }
    }
    output
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveSummary {
    pub sources: Vec<Source>,
    pub evidence_count: usize,
    pub concepts: Vec<Concept>,
    pub claims: Vec<Claim>,
    pub relationships: Vec<Relationship>,
    pub assessments: Vec<Assessment>,
    pub annotation_conflicts: Vec<crate::AnnotationConflict>,
    pub source_conflicts: Vec<crate::SourceConflict>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceDetail {
    pub source: Source,
    pub evidence: Vec<Evidence>,
    pub claims: Vec<Claim>,
    pub assessments: Vec<Assessment>,
    pub assets: Vec<crate::Asset>,
    pub revisions: Vec<RevisionInfo>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RevisionInfo {
    pub revision: String,
    pub exported_at: u64,
    pub evidence_count: usize,
    pub active: bool,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub evidence: Evidence,
    pub source_title: String,
    pub score: f32,
    pub assessment: Option<Assessment>,
    pub reading: Option<String>,
    pub source_spellings: Vec<SpellingCorrection>,
}

impl Archive {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(root.join("assets"))?;
        let mut conn = Connection::open(root.join("archive.sqlite"))?;
        conn.busy_timeout(std::time::Duration::from_secs(10))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY, data TEXT NOT NULL, imported_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions(source_id TEXT NOT NULL REFERENCES sources(id), revision TEXT NOT NULL, manifest TEXT NOT NULL, PRIMARY KEY(source_id,revision));
            CREATE TABLE IF NOT EXISTS source_heads(source_id TEXT PRIMARY KEY REFERENCES sources(id), revision TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_conflicts(source_id TEXT NOT NULL REFERENCES sources(id), revision TEXT NOT NULL, PRIMARY KEY(source_id,revision));
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), data TEXT NOT NULL, current INTEGER NOT NULL DEFAULT 1);
            CREATE INDEX IF NOT EXISTS evidence_source ON evidence(source_id,current);
            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED, text, tokenize='unicode61');
            CREATE VIRTUAL TABLE IF NOT EXISTS assessment_fts USING fts5(id UNINDEXED, note, tokenize='unicode61');
            CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id), data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS concepts(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS concept_evidence(concept_id TEXT NOT NULL REFERENCES concepts(id), evidence_id TEXT NOT NULL REFERENCES evidence(id), PRIMARY KEY(concept_id,evidence_id));
            CREATE TABLE IF NOT EXISTS relationships(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS assessments(entity_id TEXT PRIMARY KEY, status TEXT NOT NULL, note TEXT NOT NULL, updated_at INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS assets(source_id TEXT NOT NULL REFERENCES sources(id), id TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(source_id,id));
            CREATE TABLE IF NOT EXISTS assessment_history(id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS annotation_decisions(id TEXT PRIMARY KEY, choice TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS annotation_conflicts(id TEXT PRIMARY KEY REFERENCES assessment_history(id), entity_id TEXT NOT NULL);")?;
        let migration = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let has_correction = migration
            .prepare("PRAGMA table_info(assessments)")?
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .iter()
            .any(|name| name == "corrected_text");
        if !has_correction {
            migration.execute_batch(
                "ALTER TABLE assessments ADD COLUMN corrected_text TEXT;
                DROP TRIGGER IF EXISTS assessment_search_insert;
                DROP TRIGGER IF EXISTS assessment_search_update;
                DELETE FROM assessment_fts;",
            )?;
        }
        let has_spellings = migration
            .prepare("PRAGMA table_info(assessments)")?
            .query_map([], |r| r.get::<_, String>(1))?
            .collect::<rusqlite::Result<Vec<_>>>()?
            .iter()
            .any(|name| name == "source_spellings");
        if !has_spellings {
            migration.execute_batch(
                "ALTER TABLE assessments ADD COLUMN source_spellings TEXT NOT NULL DEFAULT '[]';",
            )?;
        }
        migration.commit()?;
        // Upgrade libraries written before annotations became searchable. Triggers
        // keep the note and its search index in the same SQLite transaction.
        conn.execute_batch("BEGIN IMMEDIATE;
            CREATE TRIGGER IF NOT EXISTS assessment_search_insert AFTER INSERT ON assessments BEGIN
                INSERT INTO assessment_fts(id,note) VALUES(new.entity_id,new.note || ' ' || coalesce(new.corrected_text,''));
            END;
            CREATE TRIGGER IF NOT EXISTS assessment_search_update AFTER UPDATE ON assessments BEGIN
                DELETE FROM assessment_fts WHERE id=old.entity_id;
                INSERT INTO assessment_fts(id,note) VALUES(new.entity_id,new.note || ' ' || coalesce(new.corrected_text,''));
            END;
            CREATE TRIGGER IF NOT EXISTS assessment_search_delete AFTER DELETE ON assessments BEGIN
                DELETE FROM assessment_fts WHERE id=old.entity_id;
            END;
            INSERT INTO assessment_fts(id,note)
                SELECT a.entity_id,a.note || ' ' || coalesce(a.corrected_text,'') FROM assessments a
                WHERE NOT EXISTS(SELECT 1 FROM assessment_fts f WHERE f.id=a.entity_id);
            INSERT OR IGNORE INTO source_heads(source_id,revision)
                SELECT r.source_id,r.revision FROM revisions r WHERE r.rowid=(SELECT max(p.rowid) FROM revisions p WHERE p.source_id=r.source_id);
            COMMIT;")?;
        Ok(Self { root, conn })
    }
    pub fn import_file(&mut self, file: &Path) -> Result<String> {
        let (package, dir) = crate::read_package(file)?;
        self.import(&package, dir.path())
    }
    pub fn import(&mut self, package: &SourcePackage, asset_dir: &Path) -> Result<String> {
        package.validate()?;
        // Verification also protects callers importing an already-unpacked package.
        for asset in &package.assets {
            let input = asset_dir.join(&asset.name);
            let (digest, bytes) = crate::package::file_digest(&input)?;
            if digest != asset.sha256 || bytes != asset.bytes {
                bail!("Artifact checksum mismatch");
            }
            let target = self.root.join("assets").join(&asset.sha256);
            if !target.exists() {
                let mut temp = tempfile::NamedTempFile::new_in(self.root.join("assets"))?;
                std::io::copy(&mut std::fs::File::open(&input)?, temp.as_file_mut())?;
                temp.as_file().sync_all()?;
                temp.persist(&target).map_err(|e| e.error)?;
            } else {
                let (digest, bytes) = crate::package::file_digest(&target)?;
                if digest != asset.sha256 || bytes != asset.bytes {
                    bail!("Preserved archive asset is corrupt");
                }
            }
        }
        let tx = self.conn.transaction()?;
        let id = import_records(&tx, package)?;
        tx.commit()?;
        Ok(id)
    }
    pub fn assess(&self, entity_id: &str, status: &str, note: &str) -> Result<Assessment> {
        self.save_assessment(entity_id, Some(status), Some(note), None, None)
    }
    pub fn correct_reading(
        &self,
        evidence_id: &str,
        corrected: Option<&str>,
    ) -> Result<Assessment> {
        self.evidence(evidence_id)?;
        if corrected.is_some_and(|text| text.trim().is_empty() || text.len() > 100_000) {
            bail!("Enter a corrected reading below 100,000 bytes, or remove it explicitly");
        }
        self.save_assessment(evidence_id, None, None, Some(corrected), None)
    }
    pub fn correct_source_spellings(
        &self,
        source_id: &str,
        spellings: &[SpellingCorrection],
    ) -> Result<Assessment> {
        self.source_detail(source_id)?;
        if !SpellingCorrection::valid_all(spellings) {
            bail!("Enter up to 32 distinct spelling corrections below 100 bytes each");
        }
        self.save_assessment(source_id, None, None, None, Some(spellings))
    }
    fn save_assessment(
        &self,
        entity_id: &str,
        status: Option<&str>,
        note: Option<&str>,
        correction: Option<Option<&str>>,
        spellings: Option<&[SpellingCorrection]>,
    ) -> Result<Assessment> {
        if status.is_some_and(|s| {
            ![
                "unreviewed",
                "supported",
                "contested",
                "uncertain",
                "belief",
                "rejected",
                "inbox",
                "collected",
            ]
            .contains(&s)
        }) || note.is_some_and(|n| n.len() > 100_000)
        {
            bail!("Invalid assessment");
        }
        let exists: bool = self.conn.query_row("SELECT EXISTS(SELECT id FROM sources WHERE id=?1 UNION SELECT id FROM evidence WHERE id=?1 UNION SELECT id FROM claims WHERE id=?1 UNION SELECT id FROM concepts WHERE id=?1)",params![entity_id],|r|r.get(0))?;
        if !exists {
            bail!("Assessment target is not in the archive");
        }
        let tx = self.conn.unchecked_transaction()?;
        let previous: Option<Assessment> = tx
            .query_row(
                "SELECT entity_id,status,note,updated_at,corrected_text,source_spellings FROM assessments WHERE entity_id=?1",
                params![entity_id],
                |r| {
                    Ok(Assessment {
                        entity_id: r.get(0)?,
                        status: r.get(1)?,
                        note: r.get(2)?,
                        corrected_text: r.get(4)?,
                        source_spellings: read_spellings(r, 5)?,
                        updated_at: r.get::<_, i64>(3)? as u64,
                    })
                },
            )
            .optional()?;
        let corrected_text = match correction {
            Some(text) => text.map(str::to_owned),
            None => previous.as_ref().and_then(|a| a.corrected_text.clone()),
        };
        let status = status.unwrap_or_else(|| {
            previous
                .as_ref()
                .map_or("unreviewed", |a| a.status.as_str())
        });
        let note = note.unwrap_or_else(|| previous.as_ref().map_or("", |a| a.note.as_str()));
        let assessment = Assessment {
            entity_id: entity_id.into(),
            status: status.into(),
            note: note.into(),
            corrected_text,
            source_spellings: spellings
                .map(<[SpellingCorrection]>::to_vec)
                .unwrap_or_else(|| {
                    previous
                        .as_ref()
                        .map_or_else(Vec::new, |a| a.source_spellings.clone())
                }),
            updated_at: now(),
        };
        if let Some(previous) = &previous {
            crate::backup::keep_assessment(&tx, previous)?;
        }
        crate::backup::keep_assessment(&tx, &assessment)?;
        tx.execute("INSERT INTO assessments(entity_id,status,note,updated_at,corrected_text,source_spellings) VALUES(?1,?2,?3,?4,?5,?6) ON CONFLICT(entity_id) DO UPDATE SET status=excluded.status,note=excluded.note,updated_at=excluded.updated_at,corrected_text=excluded.corrected_text,source_spellings=excluded.source_spellings",params![entity_id,status,note,assessment.updated_at as i64,assessment.corrected_text,serde_json::to_string(&assessment.source_spellings)?])?;
        tx.execute("INSERT OR REPLACE INTO annotation_decisions SELECT id,'edited' FROM annotation_conflicts WHERE entity_id=?1",params![entity_id])?;
        tx.execute(
            "DELETE FROM annotation_conflicts WHERE entity_id=?1",
            params![entity_id],
        )?;
        tx.commit()?;
        Ok(assessment)
    }
    pub fn summary(&self) -> Result<ArchiveSummary> {
        let mut concepts: Vec<Concept> = self.all("SELECT data FROM concepts ORDER BY id")?;
        for c in &mut concepts {
            let mut q = self
                .conn
                .prepare("SELECT evidence_id FROM concept_evidence WHERE concept_id=?1")?;
            c.evidence_ids = q
                .query_map(params![c.id], |r| r.get(0))?
                .collect::<rusqlite::Result<_>>()?;
        }
        Ok(ArchiveSummary {
            sources: self.all("SELECT data FROM sources ORDER BY imported_at DESC")?,
            evidence_count: self.conn.query_row(
                "SELECT count(*) FROM evidence WHERE current=1",
                [],
                |r| r.get::<_, i64>(0),
            )? as usize,
            concepts,
            claims: self.all("SELECT data FROM claims")?,
            relationships: self.all("SELECT data FROM relationships")?,
            assessments: self.assessments()?,
            annotation_conflicts: self.annotation_conflicts()?,
            source_conflicts: self.source_conflicts()?,
        })
    }
    pub(crate) fn all<T: serde::de::DeserializeOwned>(&self, sql: &str) -> Result<Vec<T>> {
        let mut q = self.conn.prepare(sql)?;
        let rows = q.query_map([], |r| r.get::<_, String>(0))?;
        rows.map(|row| Ok(serde_json::from_str(&row?)?)).collect()
    }
    pub(crate) fn assessments(&self) -> Result<Vec<Assessment>> {
        let mut q = self
            .conn
            .prepare("SELECT entity_id,status,note,updated_at,corrected_text,source_spellings FROM assessments")?;
        let records = q
            .query_map([], |r| {
                Ok(Assessment {
                    entity_id: r.get(0)?,
                    status: r.get(1)?,
                    note: r.get(2)?,
                    corrected_text: r.get(4)?,
                    source_spellings: read_spellings(r, 5)?,
                    updated_at: r.get::<_, i64>(3)? as u64,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(records)
    }
    pub fn source_detail(&self, id: &str) -> Result<SourceDetail> {
        let source: String = self
            .conn
            .query_row("SELECT data FROM sources WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .optional()?
            .context("Source not found")?;
        let mut q = self.conn.prepare("SELECT data FROM evidence WHERE source_id=?1 AND current=1 ORDER BY json_extract(data,'$.start')")?;
        let evidence = q
            .query_map(params![id], |r| r.get::<_, String>(0))?
            .map(|r| Ok(serde_json::from_str(&r?)?))
            .collect::<Result<Vec<Evidence>>>()?;
        let mut q = self
            .conn
            .prepare("SELECT data FROM claims WHERE source_id=?1")?;
        let claims = q
            .query_map(params![id], |r| r.get::<_, String>(0))?
            .map(|r| Ok(serde_json::from_str(&r?)?))
            .collect::<Result<Vec<Claim>>>()?;
        let mut q = self
            .conn
            .prepare("SELECT data FROM assets WHERE source_id=?1")?;
        let assets = q
            .query_map(params![id], |r| r.get::<_, String>(0))?
            .map(|r| Ok(serde_json::from_str(&r?)?))
            .collect::<Result<Vec<crate::Asset>>>()?;
        let ids: std::collections::HashSet<_> = evidence
            .iter()
            .map(|e| e.id.as_str())
            .chain(claims.iter().map(|c| c.id.as_str()))
            .chain(std::iter::once(id))
            .collect();
        let assessments = self
            .assessments()?
            .into_iter()
            .filter(|a| ids.contains(a.entity_id.as_str()))
            .collect();
        let head: String = self.conn.query_row(
            "SELECT revision FROM source_heads WHERE source_id=?1",
            params![id],
            |r| r.get(0),
        )?;
        let mut q = self
            .conn
            .prepare("SELECT manifest FROM revisions WHERE source_id=?1 ORDER BY rowid DESC")?;
        let revisions = q
            .query_map(params![id], |r| r.get::<_, String>(0))?
            .map(|r| {
                let p: SourcePackage = serde_json::from_str(&r?)?;
                Ok(RevisionInfo {
                    active: p.revision == head,
                    revision: p.revision,
                    exported_at: p.exported_at,
                    evidence_count: p.evidence.len(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SourceDetail {
            source: serde_json::from_str(&source)?,
            evidence,
            claims,
            assessments,
            assets,
            revisions,
        })
    }
    pub fn asset_path(&self, source_id: &str, asset_id: &str) -> Result<PathBuf> {
        let data: String = self
            .conn
            .query_row(
                "SELECT data FROM assets WHERE source_id=?1 AND id=?2",
                params![source_id, asset_id],
                |r| r.get(0),
            )
            .optional()?
            .context("Artifact not found")?;
        let asset: crate::Asset = serde_json::from_str(&data)?;
        Ok(self.root.join("assets").join(asset.sha256))
    }
    pub fn evidence(&self, id: &str) -> Result<Evidence> {
        let data: String = self
            .conn
            .query_row("SELECT data FROM evidence WHERE id=?1", params![id], |r| {
                r.get(0)
            })
            .optional()?
            .context("Evidence not found")?;
        Ok(serde_json::from_str(&data)?)
    }
    pub fn reading(&self, id: &str) -> Result<String> {
        let evidence = self.evidence(id)?;
        let assessments = self.assessments()?;
        let corrected = assessments
            .iter()
            .find(|a| a.entity_id == id)
            .and_then(|a| a.corrected_text.as_deref());
        let terms = assessments
            .iter()
            .find(|a| a.entity_id == evidence.source_id)
            .map_or(&[][..], |a| a.source_spellings.as_slice());
        Ok(corrected_spelling(
            corrected.unwrap_or(&evidence.text),
            terms,
        ))
    }
    /// Human concepts and claims use the same preserved evidence references as imports.
    pub fn add_concept(&mut self, label: &str, evidence_id: &str) -> Result<Concept> {
        let label = label.trim();
        if label.is_empty() || label.len() > 200 {
            bail!("Enter a concept name below 200 characters");
        }
        self.evidence(evidence_id)?;
        let id = crate::stable_id("concept", &label.to_lowercase());
        let concept = Concept {
            id: id.clone(),
            label: label.into(),
            aliases: vec![],
            evidence_ids: vec![evidence_id.into()],
        };
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR IGNORE INTO concepts VALUES(?1,?2)",
            params![id, serde_json::to_string(&concept)?],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO concept_evidence VALUES(?1,?2)",
            params![id, evidence_id],
        )?;
        tx.commit()?;
        Ok(concept)
    }
    pub fn add_claim(&self, evidence_id: &str, statement: &str, kind: &str) -> Result<Claim> {
        let evidence = self.evidence(evidence_id)?;
        if statement.trim().is_empty()
            || statement.len() > 10000
            || ![
                "speaker_assertion",
                "observation",
                "interpretation",
                "metaphysical_hypothesis",
            ]
            .contains(&kind)
        {
            bail!("Invalid claim or interpretation");
        }
        let claim = Claim {
            id: crate::stable_id(
                "claim",
                &format!("{evidence_id}:{kind}:{}", statement.trim()),
            ),
            source_id: evidence.source_id,
            statement: statement.trim().into(),
            kind: kind.into(),
            evidence_ids: vec![evidence_id.into()],
            provenance: crate::Provenance {
                method: "human_annotation".into(),
                model: None,
                generated_at: now(),
                original_asset: None,
            },
        };
        self.conn.execute(
            "INSERT OR IGNORE INTO claims VALUES(?1,?2,?3)",
            params![claim.id, claim.source_id, serde_json::to_string(&claim)?],
        )?;
        Ok(claim)
    }
    pub fn add_relationship(
        &self,
        from: &str,
        to: &str,
        kind: crate::RelationKind,
        evidence_id: &str,
        explanation: &str,
    ) -> Result<Relationship> {
        self.evidence(evidence_id)?;
        if from == to || explanation.trim().is_empty() || explanation.len() > 10000 {
            bail!("A relationship needs two distinct nodes and an explanation");
        }
        for id in [from, to] {
            let exists: bool=self.conn.query_row("SELECT EXISTS(SELECT id FROM sources WHERE id=?1 UNION SELECT id FROM evidence WHERE id=?1 UNION SELECT id FROM concepts WHERE id=?1 UNION SELECT id FROM claims WHERE id=?1)",params![id],|r|r.get(0))?;
            if !exists {
                bail!("Relationship node not found");
            }
        }
        let relationship = Relationship {
            id: crate::stable_id(
                "relationship",
                &format!("{from}:{to}:{kind:?}:{evidence_id}:{explanation}"),
            ),
            from_id: from.into(),
            to_id: to.into(),
            kind,
            evidence_ids: vec![evidence_id.into()],
            explanation: explanation.into(),
            created_by: "human".into(),
        };
        self.conn.execute(
            "INSERT OR IGNORE INTO relationships VALUES(?1,?2)",
            params![relationship.id, serde_json::to_string(&relationship)?],
        )?;
        Ok(relationship)
    }
    pub fn search(
        &self,
        query: &str,
        vector: Option<&[f32]>,
        source_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<SearchHit>> {
        if vector
            .is_some_and(|v| v.len() != crate::EMBEDDING_DIM || v.iter().any(|f| !f.is_finite()))
        {
            bail!("Incompatible query embedding");
        }
        let stopwords = [
            "the",
            "and",
            "for",
            "are",
            "was",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "how",
            "does",
            "did",
            "this",
            "that",
            "there",
            "here",
            "about",
            "video",
            "with",
            "from",
            "they",
            "them",
            "his",
            "her",
            "its",
            "into",
            "have",
            "has",
            "say",
            "speaker",
            "claim",
            "claims",
            "said",
            "parts",
            "remain",
            "interpretation",
            "rather",
            "than",
            "visible",
            "evidence",
        ];
        let terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2 && !stopwords.contains(&s.to_lowercase().as_str()))
            .take(24)
            .map(|s| format!("\"{}\"", s))
            .collect();
        let mut lexical = HashMap::new();
        if !terms.is_empty() {
            let mut q = self.conn.prepare("SELECT f.id FROM evidence_fts f JOIN evidence e ON e.id=f.id WHERE evidence_fts MATCH ?1 AND e.current=1 AND (?2 IS NULL OR e.source_id=?2) ORDER BY bm25(evidence_fts) LIMIT 300")?;
            for (rank, row) in q
                .query_map(params![terms.join(" OR "), source_id], |r| {
                    r.get::<_, String>(0)
                })?
                .enumerate()
            {
                lexical.insert(row?, 1.0 / (1.0 + rank as f32 * 0.1));
            }
            let mut q=self.conn.prepare("SELECT a.id FROM assessment_fts a JOIN evidence e ON e.id=a.id WHERE assessment_fts MATCH ?1 AND e.current=1 AND (?2 IS NULL OR e.source_id=?2) ORDER BY bm25(assessment_fts) LIMIT 100")?;
            for row in q.query_map(params![terms.join(" OR "), source_id], |r| {
                r.get::<_, String>(0)
            })? {
                lexical.insert(row?, 1.5);
            }
        }
        let assessments: HashMap<_, _> = self
            .assessments()?
            .into_iter()
            .map(|a| (a.entity_id.clone(), a))
            .collect();
        let titles: HashMap<String, String> = self
            .all::<Source>("SELECT data FROM sources")?
            .into_iter()
            .map(|s| (s.id, s.title))
            .collect();
        let mut q = self.conn.prepare(
            "SELECT data FROM evidence WHERE current=1 AND (?1 IS NULL OR source_id=?1)",
        )?;
        let mut hits = Vec::new();
        for row in q.query_map(params![source_id], |r| r.get::<_, String>(0))? {
            let mut evidence: Evidence = serde_json::from_str(&row?)?;
            let source_spellings = assessments
                .get(&evidence.source_id)
                .map_or_else(Vec::new, |a| a.source_spellings.clone());
            let corrected = assessments
                .get(&evidence.id)
                .and_then(|a| a.corrected_text.as_deref());
            let reading = if corrected.is_some() || !source_spellings.is_empty() {
                Some(corrected_spelling(
                    corrected.unwrap_or(&evidence.text),
                    &source_spellings,
                ))
            } else {
                None
            };
            let corrected_match = reading.as_ref().is_some_and(|r| {
                let words: std::collections::HashSet<_> = r
                    .split(|c: char| !c.is_alphanumeric())
                    .map(str::to_lowercase)
                    .collect();
                terms
                    .iter()
                    .any(|t| words.contains(&t.trim_matches('"').to_lowercase()))
            });
            let text_score = lexical
                .get(&evidence.id)
                .copied()
                .unwrap_or(0.0)
                .max(if corrected_match { 1.0 } else { 0.0 });
            let similarity = vector.map(|v| cosine(v, &evidence.vector)).unwrap_or(0.0);
            if text_score <= 0.0 && similarity <= 0.0 {
                continue;
            }
            let score = similarity + text_score * 0.25;
            evidence.vector.clear();
            hits.push(SearchHit {
                source_title: titles.get(&evidence.source_id).cloned().unwrap_or_default(),
                assessment: assessments.get(&evidence.id).cloned(),
                reading,
                source_spellings,
                evidence,
                score,
            });
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit.min(30));
        Ok(hits)
    }
}
fn read_spellings(
    row: &rusqlite::Row<'_>,
    column: usize,
) -> rusqlite::Result<Vec<SpellingCorrection>> {
    let data: String = row.get(column)?;
    serde_json::from_str(&data).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(column, rusqlite::types::Type::Text, Box::new(e))
    })
}
pub(crate) fn immutable_record<T: Serialize>(
    tx: &rusqlite::Transaction<'_>,
    table: &str,
    id: &str,
    record: &T,
) -> Result<()> {
    let previous: Option<String> = tx
        .query_row(
            &format!("SELECT data FROM {table} WHERE id=?1"),
            params![id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(previous) = previous {
        if serde_json::from_str::<serde_json::Value>(&previous)? != serde_json::to_value(record)? {
            bail!("{table} ID collision");
        }
    }
    Ok(())
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let aa: f32 = a.iter().map(|v| v * v).sum();
    let bb: f32 = b.iter().map(|v| v * v).sum();
    if aa <= 0.0 || bb <= 0.0 {
        return 0.0;
    }
    a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>() / (aa * bb).sqrt()
}

pub(crate) fn import_records(
    tx: &rusqlite::Transaction<'_>,
    package: &SourcePackage,
) -> Result<String> {
    let previous_revision: Option<String> = tx
        .query_row(
            "SELECT manifest FROM revisions WHERE source_id=?1 AND revision=?2",
            params![package.source.id, package.revision],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(previous) = previous_revision {
        let mut previous: SourcePackage = serde_json::from_str(&previous)?;
        let mut incoming = package.clone();
        previous.exported_at = 0;
        incoming.exported_at = 0;
        if serde_json::to_value(previous)? != serde_json::to_value(incoming)? {
            bail!("Revision ID collision: preserved contents differ");
        }
        return Ok(package.source.id.clone());
    }
    tx.execute(
        "INSERT INTO sources VALUES(?1,?2,?3) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
        params![
            package.source.id,
            serde_json::to_string(&package.source)?,
            now() as i64
        ],
    )?;
    tx.execute(
        "INSERT INTO revisions VALUES(?1,?2,?3)",
        params![
            package.source.id,
            package.revision,
            serde_json::to_string(package)?
        ],
    )?;
    tx.execute("INSERT INTO source_heads VALUES(?1,?2) ON CONFLICT(source_id) DO UPDATE SET revision=excluded.revision",params![package.source.id,package.revision])?;
    tx.execute(
        "UPDATE evidence SET current=0 WHERE source_id=?1",
        params![package.source.id],
    )?;
    for e in &package.evidence {
        // Stable IDs must never be reused for different evidence contents.
        let previous: Option<String> = tx
            .query_row(
                "SELECT data FROM evidence WHERE id=?1",
                params![e.id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(previous) = previous {
            let p: Evidence = serde_json::from_str(&previous)?;
            if p.source_id != e.source_id
                || p.text != e.text
                || p.start != e.start
                || p.end != e.end
                || p.kind != e.kind
            {
                bail!("Evidence ID collision");
            }
        }
        tx.execute("INSERT INTO evidence VALUES(?1,?2,?3,1) ON CONFLICT(id) DO UPDATE SET data=excluded.data,current=1",params![e.id,e.source_id,serde_json::to_string(e)?])?;
        tx.execute("DELETE FROM evidence_fts WHERE id=?1", params![e.id])?;
        tx.execute(
            "INSERT INTO evidence_fts VALUES(?1,?2)",
            params![e.id, e.text],
        )?;
    }
    for c in &package.claims {
        immutable_record(tx, "claims", &c.id, c)?;
        tx.execute(
            "INSERT OR IGNORE INTO claims VALUES(?1,?2,?3)",
            params![c.id, c.source_id, serde_json::to_string(c)?],
        )?;
    }
    for c in &package.concepts {
        let previous: Option<String> = tx
            .query_row(
                "SELECT data FROM concepts WHERE id=?1",
                params![c.id],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(previous) = previous {
            let p: Concept = serde_json::from_str(&previous)?;
            if p.label != c.label || p.aliases != c.aliases {
                bail!("Concept ID collision");
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
    for r in &package.relationships {
        immutable_record(tx, "relationships", &r.id, r)?;
        tx.execute(
            "INSERT OR IGNORE INTO relationships VALUES(?1,?2)",
            params![r.id, serde_json::to_string(r)?],
        )?;
    }
    for a in &package.assets {
        tx.execute("INSERT INTO assets VALUES(?1,?2,?3) ON CONFLICT(source_id,id) DO UPDATE SET data=excluded.data",params![package.source.id,a.id,serde_json::to_string(a)?])?;
    }
    Ok(package.source.id.clone())
}
