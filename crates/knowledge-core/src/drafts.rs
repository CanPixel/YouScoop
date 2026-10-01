//! Extractive navigation drafts. These never classify an allegation as fact.
use crate::{Claim, Concept, EvidenceKind, Provenance, SourcePackage, stable_id};
use std::collections::{BTreeMap, BTreeSet};

/// Preserve caption turn boundaries without guessing speakers from >> markers.
pub fn attributed_reading(text: &str) -> String {
    if !text.contains(">>") {
        return format!("SOURCE EXCERPT (may contain reported speech; no independent verification):\n{text}");
    }
    let mut output=String::from("MIXED VOICES. Speaker identities are not established by these captions. Do not attribute denial or agreement to the main presenter.\n");
    for (i,part) in text.split(">>").enumerate() {
        if !part.trim().is_empty() {
            output.push_str(&format!("UNATTRIBUTED TURN {}:\n{}\n",i+1,part.trim()));
        }
    }
    output
}

fn meaningful(word:&str)->bool {
    word.len()>=4 && !["this","that","with","from","have","they","them","there","their","what","when","where","which","about","would","could","should","because","really","very","been","were","will","into","just","like","then","than","some","said","says","saying","know","think","going","people","also","actually","only","does","even","your","these","those","here","such","other","more","most","much","dont","didnt","want","being","while","after","before","over","under","video","named","name","called","call","well","make","made","time","thing","things","something","right","take","used","using","basically","look","looks","looking","come","comes"].contains(&word)
}

/// Exact excerpt claims and recurring phrases make a fresh archive browsable
/// immediately. Model synthesis and human annotations remain separate records.
pub fn navigation_drafts(package:&mut SourcePackage) {
    if !package.claims.is_empty() || !package.concepts.is_empty() { return; }
    let mut phrases:BTreeMap<String,(String,BTreeSet<String>)>=BTreeMap::new();
    for e in &package.evidence {
        if e.kind!=EvidenceKind::Speech { continue; }
        let words:Vec<_>=e.text.split_whitespace().map(|w|w.trim_matches(|c:char|!c.is_alphanumeric())).collect();
        for pair in words.windows(2) {
            let a=pair[0].to_lowercase();let b=pair[1].to_lowercase();
            if !meaningful(&a) || !meaningful(&b) || a==b { continue; }
            let key=format!("{a} {b}");
            if key.len()>60 { continue; }
            let record=phrases.entry(key).or_insert_with(||(format!("{} {}",pair[0],pair[1]),BTreeSet::new()));
            record.1.insert(e.id.clone());
        }
        let statement=e.text.chars().take(600).collect::<String>();
        if statement.trim().len()<40 { continue; }
        let provenance=Provenance{method:"extractive_navigation_draft_unreviewed".into(),model:None,
            generated_at:package.source.captured_at,original_asset:e.provenance.original_asset.clone()};
        package.claims.push(Claim{id:stable_id("claim",&format!("{}:extractive_draft",e.id)),source_id:e.source_id.clone(),
            statement,kind:if e.text.contains(">>"){ "unattributed_dialogue_excerpt" }else{"source_excerpt"}.into(),evidence_ids:vec![e.id.clone()],provenance});
    }
    let mut ranked:Vec<_>=phrases.into_iter().filter(|(_,(_,ids))|ids.len()>=3).collect();
    ranked.sort_by(|a,b|b.1.1.len().cmp(&a.1.1.len()).then_with(||a.0.cmp(&b.0)));
    package.concepts=ranked.into_iter().take(24).map(|(key,(_label,ids))|Concept{
        id:stable_id("concept",&format!("draft:{key}")),label:key,aliases:vec![],evidence_ids:ids.into_iter().collect()}).collect();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caption_turns_do_not_invent_a_presenter_identity() {
        let s=attributed_reading("Quoted denial >> Narrator disagreement >> Another turn");
        assert!(s.contains("UNATTRIBUTED TURN 1:\nQuoted denial"));
        assert!(s.contains("UNATTRIBUTED TURN 2:\nNarrator disagreement"));
        assert!(!s.contains("MAIN SPEAKER:"));
    }
}
