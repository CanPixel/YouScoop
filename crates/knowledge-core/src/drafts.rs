//! Extractive navigation drafts. These never classify an allegation as fact.
use crate::{Claim, Concept, EvidenceKind, Provenance, SourcePackage, stable_id};
use std::collections::{BTreeMap, BTreeSet};

/// The voice label YouScoop puts at the start of voice-separated speech.
pub fn voice_label(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix("[Voice ")?;
    let close = rest.find(']')?;
    let name = &rest[..close];
    (!name.is_empty() && name.len() <= 3 && name.chars().all(|c| c.is_ascii_alphanumeric()))
        .then(|| (&text[1..close + 7], text[close + 8..].trim_start()))
}

/// Screen kinds YouScoop labels on-screen text with ("[Email] From: ...").
pub const SCREEN_KINDS: [&str; 7] = ["Email", "Memo", "Search results", "Wikipedia page", "News article", "Government document", "Creator's note"];

/// The screen kind a question asks to enumerate ("Which emails are shown?").
pub fn listing_kind(question: &str) -> Option<&'static str> {
    let q = format!(" {} ", question.to_lowercase());
    let asks_for_all = [" which ", " list ", " all ", " every ", " what are the ", " how many ", " name the ", " show me the "].iter().any(|w| q.contains(w));
    if !asks_for_all {
        return None;
    }
    let kinds: [(&[&str], &str); 7] = [
        (&[" email", " e-mail", " mails "], "Email"),
        (&[" memo"], "Memo"),
        (&[" wikipedia", " wiki "], "Wikipedia page"),
        (&[" news", " headline", " articles "], "News article"),
        (&[" search result", " google search"], "Search results"),
        (&[" government document", " court document", ".gov"], "Government document"),
        (&[" creator's note", " creators note", " notes from the creator", " notes the creator", " on-screen notes"], "Creator's note"),
    ];
    kinds.iter().find(|(words, _)| words.iter().any(|w| q.contains(w))).map(|(_, kind)| *kind)
}

/// A leading "[Kind] " label: a screen kind, "Screen" for keyframes identified
/// as screens of text, or a voice. Returns the label and the rest of the text.
pub fn leading_label(text: &str) -> Option<(&str, &str)> {
    let rest = text.strip_prefix('[')?;
    let close = rest.find(']')?;
    let label = &rest[..close];
    (label == "Screen" || SCREEN_KINDS.contains(&label) || voice_label(text).is_some()).then(|| (label, rest[close + 1..].trim_start()))
}

/// Frame a passage for the Oracle: on-screen documents, voice-separated
/// speech, or caption turns preserved without guessing speakers from >> marks.
pub fn attributed_reading(text: &str, kind: &EvidenceKind) -> String {
    if *kind == EvidenceKind::Visual {
        if let Some(("Screen", rest)) = leading_label(text) {
            return format!("SCREEN FRAME, identified by OCR of this frame:\n{rest}");
        }
        return format!("AUTOMATIC FRAME DESCRIPTION by a small vision model. It can describe the scene, but it often misreads or invents on-screen text, names, numbers and addresses; take wording from ON-SCREEN TEXT passages instead:\n{text}");
    }
    if *kind == EvidenceKind::Document {
        let (shown, body) = match leading_label(text) {
            Some((label, rest)) if SCREEN_KINDS.contains(&label) => (format!(" ({}, by its layout)", label.to_lowercase()), rest),
            _ => (String::new(), text),
        };
        return format!("ON-SCREEN TEXT{shown}, machine OCR:\n{body}");
    }
    if let Some((voice, rest)) = voice_label(text) {
        let mut out = format!("SOURCE AUDIO (voice cluster {}; A is main narration):\n{rest}", voice.trim_start_matches("Voice "));
        if rest.contains(">>") {
            out.push_str("\nCaption marks (>>) inside this passage can mean brief interjections by another voice.");
        }
        return out;
    }
    if !text.contains(">>") {
        return format!("SOURCE AUDIO:\n{text}");
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
        let s=attributed_reading("Quoted denial >> Narrator disagreement >> Another turn",&EvidenceKind::Speech);
        assert!(s.contains("UNATTRIBUTED TURN 1:\nQuoted denial"));
        assert!(s.contains("UNATTRIBUTED TURN 2:\nNarrator disagreement"));
        assert!(!s.contains("MAIN SPEAKER:"));
    }
    #[test]
    fn voice_labels_and_on_screen_text_are_framed_for_the_oracle() {
        let s=attributed_reading("[Voice B] Jeffrey Epstein did not have a bank account named Baal.",&EvidenceKind::Speech);
        assert!(s.starts_with("SOURCE AUDIO (voice cluster B; A is main narration):"));
        assert!(s.ends_with(":\nJeffrey Epstein did not have a bank account named Baal."));
        assert_eq!(voice_label("[Voice AB] x"),Some(("Voice AB","x")));
        assert_eq!(voice_label("[Voices of the past] x"),None);
        let d=attributed_reading("Memorandum\nPlease wire",&EvidenceKind::Document);
        assert!(d.starts_with("ON-SCREEN TEXT") && d.contains("machine OCR"));
        let e=attributed_reading("[Email] From: Jeffrey Epstein\nSubject: Re:",&EvidenceKind::Document);
        assert!(e.starts_with("ON-SCREEN TEXT (email, by its layout)") && e.ends_with(":\nFrom: Jeffrey Epstein\nSubject: Re:"));
        let f=attributed_reading("[Screen] The frame shows an email.",&EvidenceKind::Visual);
        assert_eq!(f,"SCREEN FRAME, identified by OCR of this frame:\nThe frame shows an email.");
        assert_eq!(leading_label("[Not a label] text"),None);
    }
    #[test]
    fn listing_questions_name_a_screen_kind() {
        assert_eq!(listing_kind("Which emails from the Epstein files are shown on screen?"), Some("Email"));
        assert_eq!(listing_kind("List all the Wikipedia pages shown"), Some("Wikipedia page"));
        assert_eq!(listing_kind("What are the news headlines in the video?"), Some("News article"));
        assert_eq!(listing_kind("What does the Summers email say?"), None);
        assert_eq!(listing_kind("Which deity is linked to the account?"), None);
    }
}
