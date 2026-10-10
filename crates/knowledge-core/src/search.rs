//! Hybrid passage search: meaning (embeddings), whole words (FTS5), substrings
//! (FTS5 trigram) and human annotations. Embedding similarity leads and text
//! matches add to it; near-duplicates are skipped so a small context window
//! holds different information. Measured on the retrieval benchmark in
//! experiments/model-trial/oracle-eval, this kept the best passage first more
//! often than reciprocal rank fusion, which let common words outvote a strong
//! semantic match.
//!
//! Embeddings live in `evidence_vectors` as little-endian f32 blobs, so a query
//! scores every passage without parsing its JSON record; only the passages it
//! returns are loaded in full. The vector and trigram tables are filled lazily
//! for any evidence row missing from them, which upgrades existing libraries
//! and covers every import and restore path. Evidence rows are immutable per
//! ID, so a filled row never goes stale.
use crate::store::{corrected_spelling, cosine, Archive, Assessment, SearchHit};
use crate::{Evidence, Source};
use anyhow::{bail, Result};
use rusqlite::params;
use std::collections::{HashMap, HashSet};

/// Weight of text matches relative to embedding similarity (0.15 to 0.5 tried).
const TEXT_WEIGHT: f32 = 0.25;
/// A term in at most this many passages is distinctive and counts fully.
const RARE: f32 = 3.0;
/// Terms in more passages than this carry no useful weight.
const COMMON: usize = 200;
/// Passages at least this similar to one already chosen add nothing new.
/// 0.92 removed every near-duplicate pair on the benchmark (0.88 to 0.95 tried).
const NEAR_DUPLICATE: f32 = 0.92;

const STOPWORDS: [&str; 44] = [
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "which", "who", "why", "how", "does", "did", "this",
    "that", "there", "here", "about", "video", "with", "from", "they", "them", "his", "her", "its", "into", "have", "has",
    "say", "speaker", "claim", "claims", "said", "parts", "remain", "interpretation", "rather", "than", "visible",
    "evidence", "according",
];

fn encode(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
fn fts_phrase(term: &str) -> String {
    format!("\"{}\"", term.replace('"', ""))
}

impl Archive {
    /// Every current passage whose text starts with a label such as "[Email]",
    /// in source and time order, for questions that ask for all of a kind.
    pub fn labeled_passages(&self, label: &str, source_id: Option<&str>, limit: usize) -> Result<Vec<SearchHit>> {
        let pattern = format!("[{}]%", label.replace(['%', '_'], ""));
        let mut statement = self.conn.prepare_cached(
            "SELECT data FROM evidence WHERE current=1 AND (?2 IS NULL OR source_id=?2) AND json_extract(data,'$.text') LIKE ?1 ORDER BY source_id, json_extract(data,'$.start') LIMIT ?3",
        )?;
        let rows = statement.query_map(params![pattern, source_id, limit as i64], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        let assessments: HashMap<String, Assessment> = self.assessments()?.into_iter().map(|a| (a.entity_id.clone(), a)).collect();
        let titles: HashMap<String, String> = self.all::<Source>("SELECT data FROM sources")?.into_iter().map(|s| (s.id, s.title)).collect();
        rows.iter().map(|data| Ok(hit(serde_json::from_str(data)?, 1.0, &assessments, &titles))).collect()
    }

    /// The speech just before and after a passage in the same source, so a
    /// transcript chunk that starts mid-thought arrives with its setting.
    /// Returns the readings (human corrections applied) of the nearest
    /// preceding and following current speech passages within `seconds`.
    pub fn surrounding_speech(&self, evidence: &Evidence, seconds: f64) -> Result<(Option<String>, Option<String>)> {
        let assessments: HashMap<String, Assessment> = self.assessments()?.into_iter().map(|a| (a.entity_id.clone(), a)).collect();
        let reading = |data: String| -> Result<String> {
            let e: Evidence = serde_json::from_str(&data)?;
            let spellings = assessments.get(&e.source_id).map_or(&[][..], |a| a.source_spellings.as_slice());
            let corrected = assessments.get(&e.id).and_then(|a| a.corrected_text.as_deref()).unwrap_or(&e.text);
            Ok(corrected_spelling(corrected, spellings))
        };
        let near = |sql: &str| -> Result<Option<String>> {
            let mut statement = self.conn.prepare_cached(sql)?;
            let row = statement
                .query_map(params![evidence.source_id, evidence.id, evidence.start, evidence.end, seconds], |r| r.get::<_, String>(0))?
                .next()
                .transpose()?;
            row.map(reading).transpose()
        };
        let before = near("SELECT data FROM evidence WHERE current=1 AND source_id=?1 AND id<>?2 AND json_extract(data,'$.kind')='speech' AND json_extract(data,'$.start')<?3 AND json_extract(data,'$.end')>=?3-?5 ORDER BY json_extract(data,'$.start') DESC LIMIT 1")?;
        let after = near("SELECT data FROM evidence WHERE current=1 AND source_id=?1 AND id<>?2 AND json_extract(data,'$.kind')='speech' AND json_extract(data,'$.start')>?3 AND json_extract(data,'$.start')<=?4+?5 ORDER BY json_extract(data,'$.start') LIMIT 1")?;
        Ok((before, after))
    }

    /// Every source record, without its passages: titles and summaries for
    /// orienting the Oracle and for describing the archive.
    pub fn sources(&self) -> Result<Vec<Source>> {
        self.all("SELECT data FROM sources ORDER BY imported_at DESC")
    }

    /// The most anchored concept labels in the archive, for opening threads.
    pub fn leading_concepts(&self, source_id: Option<&str>, limit: usize) -> Result<Vec<String>> {
        let mut statement = self.conn.prepare_cached(
            "SELECT json_extract(c.data,'$.label') FROM concepts c JOIN concept_evidence ce ON ce.concept_id=c.id JOIN evidence e ON e.id=ce.evidence_id WHERE e.current=1 AND (?1 IS NULL OR e.source_id=?1) GROUP BY c.id ORDER BY count(*) DESC, c.id LIMIT ?2",
        )?;
        let labels = statement.query_map(params![source_id, limit as i64], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(labels)
    }

    /// Labels of the concepts a reader has anchored to any of these passages.
    pub fn concept_labels_for(&self, evidence_ids: &[&str]) -> Result<Vec<String>> {
        let mut statement = self.conn.prepare_cached(
            "SELECT DISTINCT json_extract(c.data,'$.label') FROM concept_evidence ce JOIN concepts c ON c.id=ce.concept_id WHERE ce.evidence_id=?1",
        )?;
        let mut labels: Vec<String> = Vec::new();
        for id in evidence_ids {
            for label in statement.query_map(params![id], |r| r.get::<_, String>(0))? {
                let label = label?;
                if !labels.contains(&label) {
                    labels.push(label);
                }
            }
        }
        Ok(labels)
    }

    /// Fill the vector and trigram tables for evidence rows that lack them.
    pub(crate) fn ensure_search_index(&self) -> Result<()> {
        let missing: Vec<(String, String)> = self
            .conn
            .prepare("SELECT e.id,e.data FROM evidence e WHERE NOT EXISTS(SELECT 1 FROM evidence_vectors v WHERE v.id=e.id)")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        if missing.is_empty() {
            return Ok(());
        }
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut vectors = tx.prepare_cached("INSERT OR REPLACE INTO evidence_vectors(id,vector) VALUES(?1,?2)")?;
            let mut trigram = tx.prepare_cached("INSERT INTO evidence_trigram(id,text) VALUES(?1,?2)")?;
            for (id, data) in missing {
                let evidence: Evidence = serde_json::from_str(&data)?;
                vectors.execute(params![id, encode(&evidence.vector)])?;
                trigram.execute(params![id, evidence.text])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn search(&self, query: &str, vector: Option<&[f32]>, source_id: Option<&str>, limit: usize) -> Result<Vec<SearchHit>> {
        if vector.is_some_and(|v| v.len() != crate::EMBEDDING_DIM || v.iter().any(|f| !f.is_finite())) {
            bail!("Incompatible query embedding");
        }
        self.ensure_search_index()?;
        let assessments: HashMap<String, Assessment> = self.assessments()?.into_iter().map(|a| (a.entity_id.clone(), a)).collect();

        let mut terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.chars().count() > 2 && !STOPWORDS.contains(&s.to_lowercase().as_str()))
            .take(24)
            .map(str::to_lowercase)
            .collect();
        // Human spelling corrections: a search for "Baal" also finds caption text that says "Ball".
        for a in assessments.values() {
            for s in &a.source_spellings {
                if terms.iter().any(|t| t.eq_ignore_ascii_case(&s.corrected)) && !terms.iter().any(|t| t.eq_ignore_ascii_case(&s.original)) {
                    terms.push(s.original.to_lowercase());
                }
            }
        }
        // Plural questions ("emails", "headlines") also find singular words and labels.
        let singular: Vec<String> = terms
            .iter()
            .filter_map(|t| match t.strip_suffix("ies") {
                Some(stem) if t.len() > 5 => Some(format!("{stem}y")),
                _ => t.strip_suffix('s').filter(|stem| stem.len() >= 4 && !stem.ends_with('s')).map(str::to_string),
            })
            .collect();
        terms.extend(singular);
        terms.sort();
        terms.dedup();

        // Text evidence per passage: the sum of its matched terms' rarity
        // weights. A term found in at most RARE passages counts fully; a common
        // one ("Epstein" in a source about him) counts little. A substring-only
        // match, such as a name inside "larrysummers.com", counts at 60%.
        // None when a term is in more than COMMON passages: it weighs under 1.5%,
        // and SQLite stops reading at the limit instead of listing every match.
        let matching = |sql: &str, term: &str| -> Result<Option<Vec<String>>> {
            let mut statement = self.conn.prepare_cached(&format!("{sql} LIMIT {}", COMMON + 1))?;
            let ids = statement.query_map(params![fts_phrase(term), source_id], |r| r.get::<_, String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            Ok((ids.len() <= COMMON).then_some(ids))
        };
        const WORDS: &str = "SELECT f.id FROM evidence_fts f JOIN evidence e ON e.id=f.id WHERE evidence_fts MATCH ?1 AND e.current=1 AND (?2 IS NULL OR e.source_id=?2)";
        const SUBSTRINGS: &str = "SELECT t.id FROM evidence_trigram t JOIN evidence e ON e.id=t.id WHERE evidence_trigram MATCH ?1 AND e.current=1 AND (?2 IS NULL OR e.source_id=?2)";
        const NOTES: &str = "SELECT a.id FROM assessment_fts a JOIN evidence e ON e.id=a.id WHERE assessment_fts MATCH ?1 AND e.current=1 AND (?2 IS NULL OR e.source_id=?2)";
        let mut text: HashMap<String, f32> = HashMap::new();
        for term in &terms {
            let Some(whole) = matching(WORDS, term)? else { continue };
            let whole: HashSet<String> = whole.into_iter().collect();
            // Substrings always include whole-word matches, so a common word ends here too.
            let partial: HashSet<String> = if term.chars().count() >= 4 {
                match matching(SUBSTRINGS, term)? {
                    Some(ids) => ids.into_iter().collect(),
                    None => continue,
                }
            } else {
                HashSet::new()
            };
            // Notes and corrected readings match too: "Baal" may exist only in a human correction.
            let annotated = matching(NOTES, term)?.unwrap_or_default();
            let frequency = whole.union(&partial).count().max(annotated.len());
            if frequency == 0 {
                continue;
            }
            let weight = (RARE / frequency as f32).min(1.0);
            for id in &partial {
                let factor = if whole.contains(id) { 1.0 } else { 0.6 };
                *text.entry(id.clone()).or_default() += weight * factor;
            }
            for id in whole.difference(&partial) {
                *text.entry(id.clone()).or_default() += weight;
            }
            for id in annotated {
                *text.entry(id).or_default() += 1.5 * weight.max(0.5);
            }
        }

        // Score every current passage from its vector blob, keeping only the
        // best candidates (with room for skipped near-duplicates) in memory.
        let limit = limit.min(30);
        let keep = limit * 4;
        let mut candidates: Vec<(String, f32, Vec<f32>)> = Vec::with_capacity(keep * 2);
        let mut floor = f32::NEG_INFINITY;
        {
            let mut statement = self.conn.prepare_cached("SELECT e.id,v.vector FROM evidence e JOIN evidence_vectors v ON v.id=e.id WHERE e.current=1 AND (?1 IS NULL OR e.source_id=?1)")?;
            let mut rows = statement.query(params![source_id])?;
            while let Some(row) = rows.next()? {
                let id: String = row.get(0)?;
                let bytes = row.get_ref(1)?.as_blob()?;
                let matched = text.get(&id).copied().unwrap_or(0.0).min(1.5);
                let similarity = match vector {
                    Some(q) if bytes.len() == q.len() * 4 => bytes
                        .chunks_exact(4)
                        .zip(q)
                        .map(|(b, x)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * x)
                        .sum::<f32>(),
                    _ => 0.0,
                };
                let score = similarity + matched * TEXT_WEIGHT;
                if (matched <= 0.0 && similarity <= 0.0) || score <= floor {
                    continue;
                }
                candidates.push((id, score, decode(bytes)));
                if candidates.len() >= keep * 2 {
                    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    candidates.truncate(keep);
                    floor = candidates.last().map_or(f32::NEG_INFINITY, |c| c.1);
                }
            }
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(keep);
        let vectors: HashMap<String, Vec<f32>> = candidates.iter().map(|(id, _, v)| (id.clone(), v.clone())).collect();
        let scored: Vec<(&str, f32)> = candidates.iter().map(|(id, score, _)| (id.as_str(), *score)).collect();

        let chosen = distinct(&scored, &vectors, limit);
        let titles: HashMap<String, String> = self.all::<Source>("SELECT data FROM sources")?.into_iter().map(|s| (s.id, s.title)).collect();
        let mut hits = Vec::with_capacity(chosen.len());
        for (id, score) in chosen {
            let data: String = self.conn.query_row("SELECT data FROM evidence WHERE id=?1", params![id], |r| r.get(0))?;
            hits.push(hit(serde_json::from_str(&data)?, score, &assessments, &titles));
        }
        Ok(hits)
    }
}

fn hit(mut evidence: Evidence, score: f32, assessments: &HashMap<String, Assessment>, titles: &HashMap<String, String>) -> SearchHit {
    evidence.vector.clear();
    let source_spellings = assessments.get(&evidence.source_id).map_or_else(Vec::new, |a| a.source_spellings.clone());
    let corrected = assessments.get(&evidence.id).and_then(|a| a.corrected_text.as_deref());
    let reading = (corrected.is_some() || !source_spellings.is_empty()).then(|| corrected_spelling(corrected.unwrap_or(&evidence.text), &source_spellings));
    SearchHit {
        source_title: titles.get(&evidence.source_id).cloned().unwrap_or_default(),
        assessment: assessments.get(&evidence.id).cloned(),
        reading,
        source_spellings,
        evidence,
        score,
    }
}

/// Take passages in score order, skipping any that nearly duplicates one
/// already taken (the same document shown twice, overlapping descriptions).
fn distinct<'a>(scored: &[(&'a str, f32)], vectors: &HashMap<String, Vec<f32>>, limit: usize) -> Vec<(&'a str, f32)> {
    let mut chosen: Vec<(&str, f32)> = Vec::new();
    for &(id, score) in scored {
        if chosen.len() == limit {
            break;
        }
        if chosen.iter().all(|(c, _)| cosine(&vectors[id], &vectors[*c]) < NEAR_DUPLICATE) {
            chosen.push((id, score));
        }
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn near_duplicates_are_skipped_and_relevance_order_is_kept() {
        let vectors: HashMap<String, Vec<f32>> = [
            ("a", vec![1.0, 0.0, 0.0]),
            ("a-copy", vec![0.999, 0.04, 0.0]),
            ("b", vec![0.0, 1.0, 0.0]),
            ("c", vec![0.6, 0.0, 0.8]),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let pool = [("a", 0.9), ("a-copy", 0.89), ("b", 0.7), ("c", 0.6)];
        let chosen: Vec<_> = distinct(&pool, &vectors, 3).into_iter().map(|(id, _)| id).collect();
        assert_eq!(chosen, ["a", "b", "c"]);
    }
    #[test]
    fn vectors_round_trip_through_blobs() {
        let v = vec![0.25_f32, -1.5, 3.0e-7];
        assert_eq!(decode(&encode(&v)), v);
    }
}
