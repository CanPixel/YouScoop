//! Bounded long-source selection and exact passage references for desktop chat.
use crate::{rag, store::Chunk};
use std::collections::BTreeSet;

pub struct Context {
    pub text: String,
    pub passages: Vec<rag::SubLine>,
    pub complete: bool,
}

/// Include a whole short source; combine query matches and chronological
/// coverage for long sources. Never label selected excerpts a complete read.
pub fn context(chunks: &[Chunk], question: &str) -> Context {
    let refs: Vec<_> = chunks.iter().filter(|c| c.source != "audiovisual").collect();
    let mut all = rag::excerpt_sublines(&refs);
    all.sort_by(|a, b| a.start.total_cmp(&b.start));
    const BUDGET: usize = 12000;
    let total: usize = all.iter().map(|p| p.text.chars().count() + 90).sum();
    let mut selected = BTreeSet::new();
    if total <= BUDGET {
        selected.extend(0..all.len());
    } else {
        // Query matches go first, then distribute coverage over the timeline.
        let mut ranked: Vec<_> = all.iter().enumerate().map(|(i, p)|
            (i, rag::lexical_boost(question, &p.text))).collect();
        ranked.sort_by(|a,b| b.1.total_cmp(&a.1));
        let mut used = 0;
        for (i, score) in ranked.into_iter().filter(|(_, score)| *score > 0.0).take(16) {
            let _ = score;
            let cost = all[i].text.chars().count() + 90;
            if used + cost <= BUDGET / 2 { selected.insert(i); used += cost; }
        }
        for n in 0..80 {
            if all.is_empty() { break; }
            let i = n * (all.len() - 1) / 79;
            if selected.contains(&i) { continue; }
            let cost = all[i].text.chars().count() + 90;
            if used + cost <= BUDGET { selected.insert(i); used += cost; }
        }
    }
    let complete = selected.len() == all.len();
    let mut passages = Vec::new();
    let mut text: String = if complete { "Coverage: supplied passages cover the complete indexed source.\n".into() }
        else { "Coverage: selected query matches and chronological samples from a longer source. The intervening material was NOT supplied. Do not claim a complete argument audit.\n".into() };
    for (i, p) in all.drain(..).enumerate() {
        if selected.contains(&i) {
            text.push_str(&format!("[E{}] [{}] {}: {}\n", passages.len()+1,
                rag::format_ts(p.start), p.label, p.text.replace("<|", "< |")));
            passages.push(p);
        }
    }
    Context { text, passages, complete }
}

/// Resolve only supplied evidence IDs. Never guess or move a citation to a
/// nearby timestamp based on the generated sentence's wording.
pub fn resolve(answer: &str, passages: &[rag::SubLine]) -> String {
    let normalized = knowledge_core::grounded_answer(answer, passages.len());
    let mut output = String::new();
    let mut rest = normalized.as_str();
    while let Some(open) = rest.find('[') {
        output.push_str(&rest[..open]);
        rest = &rest[open..];
        let Some(close) = rest.find(']') else { break; };
        let label = &rest[1..close];
        if let Some(id) = label.strip_prefix('E') {
            match id.parse::<usize>().ok().and_then(|n| n.checked_sub(1)).and_then(|n| passages.get(n)) {
                Some(p) => output.push_str(&format!("[{}]", rag::format_ts(p.start))),
                None => output.push_str("(unavailable evidence reference)"),
            }
        } else if label.contains(':') && label.chars().all(|c| c.is_ascii_digit() || c == ':') {
            output.push_str("(timestamp was not supplied as an evidence reference)");
        } else { output.push_str(&rest[..=close]); }
        rest = &rest[close+1..];
    }
    output.push_str(rest);
    output
}

/// At most twelve ten-second inspections: four timeline anchors plus visual
/// document moments. This is selective audiovisual coverage, not every frame.
pub fn inspection_plan(chunks: &[Chunk], duration: f64) -> Vec<f64> {
    if !duration.is_finite() || duration <= 0.0 || duration > 7200.0 { return vec![]; }
    let mut starts = vec![];
    let add = |starts: &mut Vec<f64>, start: f64| {
        let start = start.max(0.0).min((duration - 10.0).max(0.0));
        if start.is_finite() && starts.iter().all(|s| (s-start).abs() >= 10.0) { starts.push(start); }
    };
    for ratio in [0.0, 0.33, 0.66, 0.98] { add(&mut starts, duration * ratio); }
    let terms = ["document", "email", "letter", "account", "bank", "chart", "text", "diagram", "screen", "table"];
    let mut visual: Vec<_> = chunks.iter().filter(|c| c.source == "vision").map(|c| {
        let lower = c.text.to_lowercase();
        (c, terms.iter().filter(|t| lower.contains(**t)).count())
    }).collect();
    visual.sort_by(|a,b| b.1.cmp(&a.1).then_with(|| a.0.start.total_cmp(&b.0.start)));
    for (c, _) in visual { if starts.len() == 12 { break; } add(&mut starts, c.start); }
    starts.sort_by(f64::total_cmp);
    starts
}

#[cfg(test)]
mod tests {
    use super::*;
    fn chunk(i: usize) -> Chunk { Chunk { start: i as f64 * 30.0, end: i as f64 * 30.0 + 30.0,
        text: format!("passage {i} {}", "source words ".repeat(40)), source:"speech".into(), vector:vec![], cues:vec![] } }
    #[test]
    fn two_hour_source_is_bounded_and_retains_query_matches() {
        let mut chunks: Vec<_> = (0..240).map(chunk).collect();
        chunks[110].text = "The rare phrase quartz tribunal appears here.".into();
        let c = context(&chunks, "quartz tribunal");
        assert!(!c.complete); assert!(c.text.chars().count() < 12500);
        assert!(c.text.contains("quartz tribunal"));
        assert!(c.passages.iter().any(|p| p.start == 0.0));
    }
    #[test]
    fn exact_references_cannot_invent_or_snap_a_timestamp() {
        let c = context(&[chunk(0), chunk(1)], "overview");
        assert!(c.complete);
        assert_eq!(resolve("Grouped [E1, E2]", &c.passages), "Grouped [0:00] [0:30]");
        assert_eq!(resolve("Assertion [E2], invented [E99], guessed [12:34]", &c.passages),
            "Assertion [0:30], invented (unavailable evidence reference), guessed (timestamp was not supplied as an evidence reference)");
    }
    #[test]
    fn inspection_plan_is_bounded_non_overlapping_and_rejects_longer_sources() {
        let mut chunks: Vec<_> = (0..240).map(chunk).collect();
        for c in &mut chunks { c.source="vision".into(); c.text="bank document".into(); }
        let p = inspection_plan(&chunks,7200.0);
        assert_eq!(p.len(),12); assert!(p.iter().all(|s| *s <= 7190.0));
        assert!(p.windows(2).all(|w| w[1]-w[0]>=10.0));
        assert!(inspection_plan(&chunks,7201.0).is_empty());
        assert_eq!(inspection_plan(&[],5.0),vec![0.0]);
    }
}
