//! Spelling suggestions from on-screen text, checked by listening.
//!
//! Auto-captions mangle names ("Ball" for "Baal", "Moolok" for "Moloch")
//! that the video often shows correctly on screen nearby. A caption word
//! becomes a candidate when it sounds like a capitalized word shown on at
//! least three screens within ninety seconds. Whisper then re-transcribes a
//! few of its caption moments twice, without and with the on-screen spelling
//! as a hint; a candidate is kept only if Whisper writes the on-screen
//! spelling. Suggestions are proposals for the archive's spelling panel,
//! never applied automatically.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::stt::WhisperEngine;
use crate::vtt::Cue;

/// Frequent English words never suggested as, or replaced by, a name.
const COMMON: &[&str] = &[
    "the", "be", "to", "of", "and", "in", "that", "have", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no", "just", "him",
    "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", "look",
    "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "are", "was", "were", "been", "has", "had",
    "did", "said", "says", "four", "read", "real", "here", "more", "while", "whole", "still", "mean", "moment", "style", "mount",
    "tue", "mon", "wed", "thu", "fri", "sat", "sun", "man", "men", "old", "same", "many", "much", "very", "made", "where", "why",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub original: String,
    pub suggested: String,
    /// Caption moments using the original spelling near a matching screen.
    pub moments: Vec<f64>,
    pub screens: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub at: f64,
    pub without_hint: String,
    pub with_hint: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub original: String,
    pub suggested: String,
    /// "heard without a hint" or "heard with the on-screen spelling as a hint".
    pub evidence: String,
    pub caption_uses: usize,
    pub screens: usize,
    /// Checked moments where Whisper kept the caption word even with the hint
    /// (for example "the vampire's ball"): applying the spelling source-wide
    /// would change those too.
    pub kept_original_at: Vec<f64>,
    pub checks: Vec<Check>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpellingRecord {
    pub method: String,
    pub suggestions: Vec<Suggestion>,
    pub rejected: Vec<Candidate>,
}

pub const METHOD: &str = "phonetic match to on-screen text, confirmed by Whisper large-v3-turbo re-transcription";

#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub fn load(dir: &Path) -> Option<SpellingRecord> {
    serde_json::from_slice(&std::fs::read(dir.join("spellings.json")).ok()?).ok()
}

/// A rough sound key: consonant skeleton after common English spelling
/// equivalences, so "Ball" and "Baal" share "bl" and "Moolok" and "Moloch" "mlk".
pub fn sound_key(word: &str) -> String {
    let mut s: String = word.to_lowercase().chars().filter(|c| c.is_ascii_alphabetic()).collect();
    for (from, to) in [("ph", "f"), ("ch", "k"), ("ck", "k"), ("gh", ""), ("sh", "x"), ("th", "0"), ("qu", "kw"), ("wh", "w")] {
        s = s.replace(from, to);
    }
    if s.starts_with("kn") {
        s.remove(0);
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        let mapped = match c {
            'c' if matches!(chars.get(i + 1), Some('e' | 'i' | 'y')) => 's',
            'c' | 'q' | 'x' => 'k',
            'z' => 's',
            'v' => 'f',
            'j' => 'g',
            other => *other,
        };
        if i == 0 {
            out.push(if "aeiou".contains(mapped) { 'a' } else { mapped });
        } else if !"aeiouyhw".contains(mapped) && !out.ends_with(mapped) {
            out.push(mapped);
        }
    }
    out
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut previous = row[0];
        row[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let current = row[j + 1];
            row[j + 1] = (current + 1).min(row[j] + 1).min(previous + usize::from(x != *y));
            previous = current;
        }
    }
    row[b.len()]
}

fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphabetic()).filter(|w| w.chars().count() >= 3)
}

/// Caption words that sound like a name the screens show; see the module note.
pub fn candidates(cues: &[Cue], screens: &[(f64, f64, String)]) -> Vec<Candidate> {
    let mut on_screen: HashMap<String, (usize, usize)> = HashMap::new(); // lower -> (screens, capitalized uses)
    for (_, _, text) in screens {
        let mut seen = HashSet::new();
        for w in words(text) {
            let lower = w.to_lowercase();
            let entry = on_screen.entry(lower.clone()).or_default();
            if seen.insert(lower) {
                entry.0 += 1;
            }
            if w.chars().next().is_some_and(char::is_uppercase) {
                entry.1 += 1;
            }
        }
    }
    let mut in_captions: HashMap<String, usize> = HashMap::new();
    // Spelling corrections match exact letter case, so suggest the caption's own form.
    let mut caption_forms: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for cue in cues {
        for w in words(&cue.text) {
            *in_captions.entry(w.to_lowercase()).or_default() += 1;
            *caption_forms.entry(w.to_lowercase()).or_default().entry(w.to_string()).or_default() += 1;
        }
    }
    let usual_form = |lower: &str| {
        caption_forms.get(lower).and_then(|forms| forms.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|(f, _)| f.clone())).unwrap_or_else(|| lower.to_string())
    };
    let common = |w: &str| COMMON.contains(&w);
    // (original, suggested) -> (moments, screens, display spelling)
    let mut found: BTreeMap<(String, String), (Vec<f64>, usize, String)> = BTreeMap::new();
    for (start, end, text) in screens {
        let targets: HashSet<&str> = words(text).collect();
        for target in targets {
            let t = target.to_lowercase();
            let Some(&(screen_count, capitalized)) = on_screen.get(&t) else { continue };
            // Terms shown in lowercase ("goyim") qualify too; the Whisper check filters them.
            let _ = capitalized;
            if screen_count < 3 || common(&t) {
                continue;
            }
            for cue in cues.iter().filter(|c| c.start >= start - 90.0 && c.start <= end + 90.0) {
                for word in words(&cue.text) {
                    let o = word.to_lowercase();
                    let inflection = o.starts_with(&t) || t.starts_with(&o);
                    if o == t || common(&o) || inflection || o.chars().next() != t.chars().next() || sound_key(&o) != sound_key(&t) {
                        continue;
                    }
                    if o.chars().count().abs_diff(t.chars().count()) > 2 || distance(&o, &t) > (t.chars().count() / 2).max(2) {
                        continue;
                    }
                    // The caption spelling must be rare on screen and more common in speech.
                    if on_screen.get(&o).map_or(0, |e| e.0) * 3 >= screen_count || in_captions.get(&t).copied().unwrap_or(0) >= in_captions[&o] {
                        continue;
                    }
                    let display = if target.chars().all(|c| c.is_uppercase()) { capitalize(&t) } else { target.to_string() };
                    let entry = found.entry((o.clone(), t.clone())).or_insert((Vec::new(), screen_count, display));
                    if !entry.0.contains(&cue.start) {
                        entry.0.push(cue.start);
                    }
                }
            }
        }
    }
    let mut out: Vec<Candidate> = found
        .into_iter()
        .map(|((original, _), (mut moments, screens, suggested))| {
            moments.sort_by(f64::total_cmp);
            let original = usual_form(&original);
            // Match the caption's case: "goyam" becomes "goyim", "Ball" becomes "Baal".
            let suggested = if original.chars().next().is_some_and(char::is_lowercase) { suggested.to_lowercase() } else { capitalize(&suggested) };
            Candidate { original, suggested, moments, screens }
        })
        .collect();
    out.sort_by(|a, b| b.moments.len().cmp(&a.moments.len()).then(a.original.cmp(&b.original)));
    out.truncate(8);
    out
}
fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    chars.next().map_or_else(String::new, |c| c.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect())
}
fn contains_word(text: &str, word: &str) -> bool {
    words(text).any(|w| w.eq_ignore_ascii_case(word))
}

/// Re-transcribe up to three caption moments per candidate, without and with
/// the on-screen spelling as a hint. `clip` returns 16 kHz mono audio for a
/// time range. Keeps candidates Whisper writes with the on-screen spelling.
pub fn confirm(
    whisper: &WhisperEngine,
    candidates: &[Candidate],
    cancel: Arc<AtomicBool>,
    mut clip: impl FnMut(f64, f64) -> Result<Vec<f32>>,
    in_captions: impl Fn(&str) -> usize,
) -> Result<SpellingRecord> {
    let mut record = SpellingRecord { method: METHOD.into(), ..Default::default() };
    for candidate in candidates {
        let step = (candidate.moments.len() / 3).max(1);
        let mut checks = Vec::new();
        for &at in candidate.moments.iter().step_by(step).take(3) {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                bail!("cancelled");
            }
            let samples = clip((at - 1.0).max(0.0), 8.0)?;
            let text = |cues: Vec<Cue>| cues.into_iter().map(|c| c.text).collect::<Vec<_>>().join(" ");
            let without_hint = text(whisper.transcribe_clip(&samples, None, cancel.clone())?);
            let with_hint = text(whisper.transcribe_clip(&samples, Some(&format!("Names and terms shown on screen: {}.", candidate.suggested)), cancel.clone())?);
            checks.push(Check { at, without_hint, with_hint });
        }
        let heard = checks.iter().filter(|c| contains_word(&c.without_hint, &candidate.suggested)).count();
        let hinted = checks.iter().filter(|c| contains_word(&c.with_hint, &candidate.suggested)).count();
        let evidence = if heard > 0 {
            Some("heard without a hint")
        } else if hinted * 2 > checks.len() {
            Some("heard with the on-screen spelling as a hint")
        } else {
            None
        };
        match evidence {
            Some(evidence) => record.suggestions.push(Suggestion {
                original: candidate.original.clone(),
                suggested: candidate.suggested.clone(),
                evidence: evidence.into(),
                caption_uses: in_captions(&candidate.original.to_lowercase()),
                screens: candidate.screens,
                kept_original_at: checks.iter().filter(|c| contains_word(&c.with_hint, &candidate.original) && !contains_word(&c.with_hint, &candidate.suggested)).map(|c| c.at).collect(),
                checks,
            }),
            None => record.rejected.push(candidate.clone()),
        }
    }
    // One caption word, two spellings ("Moolok" as Moloch or Molek): keep the
    // one heard without a hint and drop alternatives only the hint produced.
    let strong: Vec<String> = record.suggestions.iter().filter(|s| s.evidence == "heard without a hint").map(|s| s.original.clone()).collect();
    let (kept, dropped): (Vec<_>, Vec<_>) = record.suggestions.drain(..).partition(|s| s.evidence == "heard without a hint" || !strong.contains(&s.original));
    record.suggestions = kept;
    record.rejected.extend(dropped.into_iter().map(|s| Candidate { original: s.original, suggested: s.suggested, moments: s.checks.iter().map(|c| c.at).collect(), screens: s.screens }));
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sound_keys_match_caption_mishearings_and_not_other_words() {
        for (a, b) in [("Ball", "Baal"), ("Bal", "Baal"), ("Moolok", "Moloch"), ("Seemetic", "Semitic"), ("goyam", "goyim")] {
            assert_eq!(sound_key(a), sound_key(b), "{a} {b}");
        }
        for (a, b) in [("the", "Tue"), ("here", "Hebrew"), ("real", "Read")] {
            assert_ne!(sound_key(a), sound_key(b), "{a} {b}");
        }
    }
    #[test]
    fn candidates_need_a_repeated_capitalized_name_on_screen() {
        let cue = |start: f64, text: &str| Cue { start, end: start + 3.0, text: text.into() };
        let cues = vec![cue(386.0, "an account that he has named Ball."), cue(395.0, "named Ball after the deity Bal"), cue(400.0, "can you see the style"), cue(410.0, "it is two years old"), cue(415.0, "at least for the goyam")];
        let screen = |start: f64, text: &str| (start, start + 4.0, text.to_string());
        let screens = vec![
            screen(390.0, "Baal. name: Wachovia Bank\nTue 9/25/2012"), screen(398.0, "account name is indeed BAAL\nCanaan style"),
            screen(420.0, "Epstein's Baal bank account\nTue Mar 10\nfor the goyim"), screen(430.0, "Baal worship\nCanaan Tue goyim"), screen(440.0, "GOYIM in abundance"),
        ];
        let found = candidates(&cues, &screens);
        let pairs: Vec<_> = found.iter().map(|c| (c.original.as_str(), c.suggested.as_str(), c.moments.len())).collect();
        assert_eq!(pairs, [("Ball", "Baal", 2), ("Bal", "Baal", 1), ("goyam", "goyim", 1)]);
    }
}
