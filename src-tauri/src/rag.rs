//! Retrieval + prompt construction for "talk to the video".

use crate::llm::cosine;
use crate::store::{Chunk, VideoMeta};

pub fn format_ts(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

const STOPWORDS: [&str; 32] = [
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "which", "who", "why",
    "how", "does", "did", "this", "that", "there", "here", "about", "video", "with", "from",
    "they", "them", "his", "her", "its", "into", "have", "has", "say",
];

/// Lowercased query terms worth matching lexically (rare-ish words like names).
fn query_terms(question: &str) -> Vec<String> {
    question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(str::to_string)
        .collect()
}

fn boost_for(terms: &[String], text: &str) -> f32 {
    let text = text.to_lowercase();
    let matched = terms.iter().filter(|t| text.contains(t.as_str())).count() as f32;
    let mut boost = (matched * 0.05).min(0.15);
    // Consecutive query terms appearing as a phrase ("fun virus", "uy scuti")
    // are a much stronger signal than scattered single words.
    if terms.windows(2).any(|w| text.contains(&format!("{} {}", w[0], w[1]))) {
        boost += 0.08;
    }
    boost
}

/// The full hybrid boost for one chunk text against a question (for debug tools).
pub fn lexical_boost(question: &str, text: &str) -> f32 {
    boost_for(&query_terms(question), text)
}

/// Select the most relevant chunks, capped by count and size.
///
/// Ranking is hybrid: cosine similarity plus a lexical boost for chunks that
/// literally contain the query's terms. Embedding scores on conversational
/// transcripts are packed very tightly (~0.03 spread), so an exact phrase like
/// "fun virus" must outrank chunks that merely talk about similar things.
pub fn retrieve<'a>(
    chunks: &'a [Chunk],
    query_vec: &[f32],
    question: &str,
    top_k: usize,
) -> Vec<&'a Chunk> {
    let terms = query_terms(question);
    let mut scored: Vec<(f32, &Chunk)> = chunks
        .iter()
        .filter(|c| c.source != "audiovisual" && !c.vector.is_empty())
        .map(|c| (cosine(&c.vector, query_vec) + boost_for(&terms, &c.text), c))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<&Chunk> = Vec::new();
    let mut total_chars = 0usize;
    for (_, c) in scored.into_iter().take(top_k * 2) {
        // Keep the excerpt budget modest: small chat models start
        // cross-attributing facts between excerpts when the context grows too
        // large. 3600 chars fits `top_k` full-size (600-char) chunks so ranked
        // chunks aren't silently dropped. Check before adding so the budget is
        // a real cap (the first chunk is always taken).
        if selected.len() >= top_k
            || (!selected.is_empty() && total_chars + c.text.len() > 3600)
        {
            break;
        }
        total_chars += c.text.len();
        selected.push(c);
    }
    // Present excerpts in chronological order.
    selected.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
    selected
}

pub struct HistoryTurn {
    pub question: String,
    pub answer: String,
}

/// One `[ts] speech:` / `[ts] visual:` line exactly as the prompt shows it.
pub struct SubLine {
    pub start: f64,
    /// "speech" or "visual" — the label used in the prompt.
    pub label: &'static str,
    pub text: String,
}

/// Split retrieved excerpts into the lines the model actually sees. Citation
/// validation scores against these, so prompt and validator must share one
/// implementation or a citation could be judged against text nobody showed it.
pub fn excerpt_sublines(excerpts: &[&Chunk]) -> Vec<SubLine> {
    let mut out: Vec<SubLine> = Vec::new();
    let mut emitted_cues: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for c in excerpts {
        if c.source == "audiovisual" { continue; }
        if c.source == "vision" || c.source == "audiovisual_reviewed" {
            out.push(SubLine { start: c.start, label: if c.source == "vision" { "visual" } else { "reviewed audiovisual observation" }, text: c.text.trim().to_string() });
            continue;
        }
        if c.cues.is_empty() {
            // Old index without cue-level timestamps.
            out.push(SubLine { start: c.start, label: "speech", text: c.text.trim().to_string() });
            continue;
        }
        // Render the chunk as ~12s sub-lines so citations land on the actual
        // sentence. Adjacent chunks overlap by one cue: skip already-emitted ones.
        let mut line_start: Option<f64> = None;
        let mut line_text = String::new();
        for cue in &c.cues {
            let key = (cue.start * 1000.0) as i64;
            if !emitted_cues.insert(key) {
                continue;
            }
            // Close the current line before this cue if it would stretch it
            // past ~12s: each line's timestamp must stay close to its content.
            if let Some(start) = line_start {
                if cue.start - start >= 12.0 || line_text.len() >= 260 {
                    out.push(SubLine { start, label: "speech", text: line_text.clone() });
                    line_text.clear();
                    line_start = None;
                }
            }
            line_start.get_or_insert(cue.start);
            if !line_text.is_empty() {
                line_text.push(' ');
            }
            line_text.push_str(cue.text.trim());
        }
        if let (Some(start), false) = (line_start, line_text.is_empty()) {
            out.push(SubLine { start, label: "speech", text: line_text });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Citation validation
//
// The chat model reliably retrieves the right content but is unreliable about
// which excerpt line a fact came from — it tends to reuse the first timestamp
// in the block. Rather than hope a prompt rule fixes that, re-point each
// citation after generation by measuring how much the citing sentence actually
// overlaps each line the model was shown.
// ---------------------------------------------------------------------------

/// A cited line needs at least this much weighted overlap with the citing
/// sentence to be considered grounded. Deliberately low — roughly one shared
/// content word. Anything at or above it is left alone.
///
/// This validator is intentionally conservative: it repairs citations with no
/// lexical grounding at all and leaves everything else untouched. An earlier,
/// greedier version scored each sentence and re-pointed every citation at the
/// single best line, which flattened a compound sentence's three correct,
/// distinct citations into three copies of one timestamp. Destroying good
/// citations is far worse than leaving a marginal one in place.
const MIN_SUPPORT: f64 = 0.75;

/// Crude suffix stripping so "production lines" matches "production line" and
/// "logging" matches "logged". Not a real stemmer — stems are only ever
/// compared against each other, so consistency beats linguistic correctness.
fn stem(w: &str) -> &str {
    for suf in ["ing", "ed", "es", "s"] {
        if w.len() > suf.len() + 3 && w.ends_with(suf) {
            return &w[..w.len() - suf.len()];
        }
    }
    w
}

fn content_stems(text: &str) -> std::collections::HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(|w| stem(w).to_string())
        .collect()
}

/// Rare words are what actually identify a line: matching "wormhole" says far
/// more about provenance than matching "system".
fn idf_table(lines: &[SubLine]) -> std::collections::HashMap<String, f64> {
    let n = lines.len().max(1) as f64;
    let mut df: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for l in lines {
        for w in content_stems(&l.text) {
            *df.entry(w).or_insert(0) += 1;
        }
    }
    df.into_iter().map(|(w, d)| (w, (1.0 + n / d as f64).ln())).collect()
}

fn support(
    sentence: &std::collections::HashSet<String>,
    line: &SubLine,
    idf: &std::collections::HashMap<String, f64>,
) -> f64 {
    content_stems(&line.text)
        .intersection(sentence)
        .map(|w| idf.get(w).copied().unwrap_or(0.0))
        .sum()
}

fn is_timestamp(s: &str) -> bool {
    let seg: Vec<&str> = s.split(':').collect();
    (2..=3).contains(&seg.len())
        && seg.iter().enumerate().all(|(i, p)| {
            !p.is_empty()
                && p.len() <= 2
                && p.chars().all(|c| c.is_ascii_digit())
                && (i == 0 || p.len() == 2)
        })
}

/// Byte range of a `[m:ss]` / `[m:ss, m:ss]` citation and the stamps inside it.
struct Citation {
    start: usize,
    end: usize,
    stamps: Vec<String>,
}

fn find_citations(text: &str) -> Vec<Citation> {
    let mut out = Vec::new();
    let mut open: Option<usize> = None;
    for (i, ch) in text.char_indices() {
        match ch {
            '[' => open = Some(i),
            ']' => {
                if let Some(s) = open.take() {
                    let parts: Vec<&str> = text[s + 1..i].split(',').map(str::trim).collect();
                    if !parts.is_empty() && parts.iter().all(|p| is_timestamp(p)) {
                        let stamps = parts.iter().map(|p| p.to_string()).collect();
                        out.push(Citation { start: s, end: i + 1, stamps });
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Sentence spans. A '.' only ends a sentence when followed by whitespace and
/// not preceded by a lone letter, so "e.g." doesn't shred the sentence and cost
/// us the very words that identify the source line.
fn sentence_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    for (i, ch) in text.char_indices() {
        let terminator = match ch {
            '\n' | '!' | '?' => true,
            '.' => {
                let next_is_break = text[i + 1..].chars().next().is_none_or(char::is_whitespace);
                let lone_initial = i >= 1
                    && bytes[i - 1].is_ascii_alphanumeric()
                    && (i < 2 || !bytes[i - 2].is_ascii_alphanumeric());
                next_is_break && !lone_initial
            }
            _ => false,
        };
        if terminator {
            let end = i + ch.len_utf8();
            if !text[start..end].trim().is_empty() {
                out.push(start..end);
            }
            start = end;
        }
    }
    if !text[start..].trim().is_empty() {
        out.push(start..text.len());
    }
    out
}

/// Re-point (or remove) each citation in `answer` so it names the excerpt line
/// that actually supports the sentence citing it.
pub fn snap_citations(answer: &str, lines: &[SubLine]) -> String {
    if lines.is_empty() {
        return answer.to_string();
    }
    let idf = idf_table(lines);
    let citations = find_citations(answer);
    let mut edits: Vec<(usize, usize, String)> = Vec::new();

    for sent in sentence_ranges(answer) {
        let in_sentence: Vec<&Citation> =
            citations.iter().filter(|c| sent.contains(&c.start)).collect();
        if in_sentence.is_empty() {
            continue;
        }
        // Score against the prose only: timestamps carry no content, and a
        // neighbouring citation's digits shouldn't sway the match.
        let mut prose = answer[sent.clone()].to_string();
        for c in &in_sentence {
            let (lo, hi) = (c.start - sent.start, c.end - sent.start);
            prose.replace_range(lo..hi, &" ".repeat(hi - lo));
        }
        let words = content_stems(&prose);
        let scored: Vec<f64> = lines.iter().map(|l| support(&words, l, &idf)).collect();

        // One sentence must never cite the same instant twice, whether the
        // model repeated itself or two repairs converged on the same line.
        let mut used: Vec<String> = Vec::new();
        for cite in in_sentence {
            let mut kept: Vec<String> = Vec::new();
            for stamp in &cite.stamps {
                let grounded = lines
                    .iter()
                    .position(|l| format_ts(l.start) == *stamp)
                    .is_some_and(|i| scored[i] >= MIN_SUPPORT);
                let resolved = if grounded {
                    Some(stamp.clone())
                } else {
                    // Ungrounded: hand it the best line this sentence hasn't
                    // already claimed, or drop it if nothing supports it.
                    lines
                        .iter()
                        .zip(&scored)
                        .filter(|(l, sc)| {
                            **sc >= MIN_SUPPORT && !used.contains(&format_ts(l.start))
                        })
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(l, _)| format_ts(l.start))
                };
                match resolved {
                    Some(r) if !used.contains(&r) => {
                        used.push(r.clone());
                        kept.push(r);
                    }
                    _ => {}
                }
            }
            let replacement =
                if kept.is_empty() { String::new() } else { format!("[{}]", kept.join(", ")) };
            if replacement != answer[cite.start..cite.end] {
                edits.push((cite.start, cite.end, replacement));
            }
        }
    }

    if edits.is_empty() {
        return answer.to_string();
    }
    edits.sort_by_key(|e| e.0);
    let mut out = answer.to_string();
    for (start, end, replacement) in edits.into_iter().rev() {
        out.replace_range(start..end, &replacement);
    }
    // Dropping a citation can strand " ." or a double space.
    while out.contains("  ") {
        out = out.replace("  ", " ");
    }
    out.replace(" .", ".").replace(" ,", ",").trim().to_string()
}

pub fn build_chat_prompt(
    meta: &VideoMeta,
    excerpts: &[&Chunk],
    history: &[HistoryTurn],
    question: &str,
) -> String {
    let mut excerpt_lines = String::new();
    for l in excerpt_sublines(excerpts) {
        excerpt_lines.push_str(&format!("[{}] {}: {}\n", format_ts(l.start), l.label, l.text));
    }

    // A whole-video overview (generated at ingest time) grounds global
    // questions that six retrieved excerpts can't answer on their own.
    let overview = match &meta.summary {
        Some(s) if !s.trim().is_empty() => format!(
            "\nOverview of the whole video (auto-generated; excerpts below are the \
             authoritative source for specific facts):\n{}\n",
            s.trim()
        ),
        _ => String::new(),
    };

    // ChatML prompt for the Qwen3 chat model.
    let mut prompt = format!(
        "<|im_start|>system\n\
You are YouScoop, an assistant that answers questions about one specific video. \
You are given indexed excerpts. \"speech\" lines are quotes from the spoken transcript; names in them may be \
misspelled by the transcriber. \"visual\" lines are automatic frame descriptions: they may misread details, \
and quoted screen text can also be wrong. Reviewed audiovisual observations are human notes, not independent verification. Raw Omni output is excluded from this prompt. \
Attribute statements to the speaker and separate visible observations from interpretation.\n\
Video: \"{}\" by {} (duration {}).\n{}\nExcerpts:\n{}\n\n\
Rules:\n\
- Answer using only the excerpts and overview above. Do not add facts from your own knowledge, and do not silently \
correct names — quote them as the transcript spells them.\n\
- Cite the timestamp of the specific line a fact comes from in square brackets after each claim, e.g. …the largest star [{}]. Pick the line where the fact is actually said, not a neighboring one.\n\
- Only combine information from different excerpts when each part keeps its own citation.\n\
- If the excerpts don't contain the answer, say so plainly.\n\
- Keep answers short and direct.<|im_end|>\n",
        meta.title,
        meta.channel,
        format_ts(meta.duration_secs),
        overview,
        excerpt_lines.trim_end(),
        format_ts(excerpts.first().map(|c| c.start).unwrap_or(0.0)),
    );

    for turn in history.iter().rev().take(2).rev() {
        prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n",
            turn.question.trim(),
            turn.answer.trim()
        ));
    }
    prompt.push_str(&format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        question.trim()
    ));
    prompt
}

// ---- Ingest-time summarization ----

/// Digest lines ("[ts] text") for the summary pass. Speech chunks carry the
/// narrative; vision lines are only used when there is no speech at all
/// (e.g. a silent screen recording).
pub fn summary_digest(chunks: &[Chunk]) -> Vec<String> {
    let speech: Vec<String> = chunks
        .iter()
        .filter(|c| c.source == "speech")
        .map(|c| format!("[{}] {}", format_ts(c.start), c.text.trim()))
        .collect();
    if !speech.is_empty() {
        return speech;
    }
    chunks
        .iter()
        .filter(|c| c.source == "vision")
        .map(|c| format!("[{}] (visual) {}", format_ts(c.start), c.text.trim()))
        .collect()
}

/// Split digest lines into parts of roughly `max_chars`, aiming for at most
/// `max_parts` (parts grow past `max_chars` if needed — better a coarse
/// summary than none).
pub fn split_digest(lines: &[String], max_chars: usize, max_parts: usize) -> Vec<String> {
    let total: usize = lines.iter().map(|l| l.len() + 1).sum();
    let budget = max_chars.max(total.div_ceil(max_parts));
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in lines {
        if !cur.is_empty() && cur.len() + line.len() + 1 > budget {
            parts.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }
    if !cur.is_empty() {
        parts.push(cur);
    }
    parts
}

/// Map pass: condense one transcript section into timestamped bullets.
pub fn build_map_prompt(meta: &VideoMeta, part: &str) -> String {
    format!(
        "<|im_start|>system\n\
You condense sections of video transcripts. Be factual; use only the transcript given.<|im_end|>\n\
<|im_start|>user\nVideo: \"{}\" by {}. One section of the transcript, with [timestamp] markers:\n\n{}\n\n\
Write 3-6 short bullet points covering this section's main points, in order. \
Start each bullet with the [m:ss] timestamp where that point begins (copy it from the transcript).<|im_end|>\n\
<|im_start|>assistant\n",
        meta.title, meta.channel, part
    )
}

/// Final pass: produce the overview + chapter outline stored in the library.
/// `material` is either the whole digest (short videos) or the concatenated
/// map-pass bullets (long ones).
pub fn build_final_summary_prompt(meta: &VideoMeta, material: &str) -> String {
    format!(
        "<|im_start|>system\n\
You summarize videos from their transcripts. Be factual; use only the material given.<|im_end|>\n\
<|im_start|>user\nVideo: \"{}\" by {} (duration {}). Timestamped material from the video:\n\n{}\n\n\
Write exactly this, nothing else:\n\
1. A 2-3 sentence overview of what the video covers and concludes.\n\
2. A line \"Chapters:\" followed by 3-8 chapter lines, each formatted as \"[m:ss] Chapter title\" \
using timestamps copied from the material.<|im_end|>\n\
<|im_start|>assistant\n",
        meta.title,
        meta.channel,
        format_ts(meta.duration_secs),
        material
    )
}

// ---- Follow-up query rewriting ----

/// Ask the chat model to turn a context-dependent follow-up ("what did he say
/// about it?") into a standalone retrieval query. The rewritten query is used
/// only for retrieval; the user's original wording still goes in the prompt.
pub fn build_rewrite_prompt(history: &[HistoryTurn], question: &str) -> String {
    let mut convo = String::new();
    for turn in history.iter().rev().take(2).rev() {
        convo.push_str(&format!("Q: {}\nA: {}\n", turn.question.trim(), turn.answer.trim()));
    }
    format!(
        "<|im_start|>system\n\
You rewrite follow-up questions about a video as standalone search queries. \
Resolve pronouns and references using the conversation. Keep the user's key terms. \
Output only the rewritten query on a single line — no quotes, no explanation.<|im_end|>\n\
<|im_start|>user\nConversation so far:\n{}\nFollow-up question: {}\n\nStandalone query:<|im_end|>\n\
<|im_start|>assistant\n",
        convo,
        question.trim()
    )
}

/// Clean the model's rewrite output; None means "use the original question".
pub fn clean_rewrite(raw: &str, original: &str) -> Option<String> {
    let line = raw.lines().find(|l| !l.trim().is_empty())?.trim();
    let line = line.trim_matches(|c| c == '"' || c == '\'' || c == '`').trim();
    if line.is_empty() || line.len() > 200 || line.eq_ignore_ascii_case(original.trim()) {
        return None;
    }
    Some(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The real lines from "Now you can build your own Raspberry Pi factory"
    // that Qwen3 mis-cited as [15:10].
    fn pi_lines() -> Vec<SubLine> {
        [
            (910.8, "this and get it to work correctly with an image. But you can imagine once you do get the bug sorted out and you get a green light at the end of this, being able to do this and pull out a pie, put in a new one,"),
            (923.5, "clamp it down, and it's flashing automatically. And it's logging all of these pies into a database. So when you're building your product, you can have a little sticker that has the Mac address and have it automatically print out, all that kind of stuff."),
            (935.4, "You can integrate this into your production line and make Raspberry Pi compute module-based products much quicker. If you someday are in a factory"),
            (434.0, "for my particular needs, but it does have one of the faster Intel mobile chips right now. Let me get this going. I'm going to download Raspberry Pi Imager. I'm going to plug this device into my network."),
            (663.6, "I didn't want to go through all that stuff. So I set it up using PiGen and I set up a magic wormhole and I made sure this is connected to Raspberry Pi Connect. So I should just be able to go into screen sharing on here."),
            (687.5, "right into here. so I'll say wormhole. It looks like it was already installed on here. Wormhole receive. So that should receive this file from my Mac directly onto the CM5 programming jig."),
        ]
        .into_iter()
        .map(|(start, text)| SubLine { start, label: "speech", text: text.to_string() })
        .collect()
    }

    #[test]
    fn snaps_a_citation_onto_the_line_that_supports_it() {
        let answer = "Additionally, integration with production lines and automated logging \
                      (e.g., MAC address tracking) enables scalable manufacturing [15:10].";
        let out = snap_citations(answer, &pi_lines());
        assert!(!out.contains("[15:10]"), "should not keep the unsupported stamp:\n{out}");
        assert!(
            out.contains("[15:23]") || out.contains("[15:35]"),
            "should snap onto the logging/production-line content:\n{out}"
        );
    }

    #[test]
    fn snaps_a_distinctive_term_to_its_own_line() {
        // [15:10] shares nothing with this sentence, so it is ungrounded and
        // gets re-pointed; "magic wormhole" is rare enough to win outright.
        let answer = "He sets up a magic wormhole to transfer files [15:10].";
        let out = snap_citations(answer, &pi_lines());
        assert!(
            out.contains("[11:03]") || out.contains("[11:27]"),
            "expected a wormhole line, got:\n{out}"
        );
    }

    #[test]
    fn keeps_distinct_citations_in_a_compound_sentence() {
        // Regression: an earlier version scored the whole sentence and pointed
        // every citation at one winner, collapsing three correct, distinct
        // citations into three copies of the same timestamp.
        let answer = "You flash an OS to the jig with Raspberry Pi Imager, set up a magic \
                      wormhole to move the image, and log each Mac address into a database \
                      [7:14][11:03][15:23].";
        let out = snap_citations(answer, &pi_lines());
        assert!(out.contains("[7:14]"), "Imager citation lost:\n{out}");
        assert!(out.contains("[11:03]"), "wormhole citation lost:\n{out}");
        assert!(out.contains("[15:23]"), "logging citation lost:\n{out}");
    }

    #[test]
    fn never_emits_the_same_stamp_twice_in_one_sentence() {
        let answer = "It logs every pi into a database with its Mac address [15:10][15:10].";
        let out = snap_citations(answer, &pi_lines());
        assert_eq!(out.matches("15:23").count(), 1, "duplicate repair:\n{out}");
    }

    #[test]
    fn leaves_a_well_supported_citation_alone() {
        let answer = "You download Raspberry Pi Imager onto the laptop with the Intel mobile chip [7:14].";
        let out = snap_citations(answer, &pi_lines());
        assert_eq!(out, answer);
    }

    #[test]
    fn drops_a_citation_nothing_supports() {
        let answer = "The presenter mentions his favourite pizza toppings [15:10].";
        let out = snap_citations(answer, &pi_lines());
        assert!(!out.contains('['), "unsupported citation should go:\n{out}");
        assert!(out.ends_with("toppings."), "punctuation should tidy up:\n{out}");
    }

    #[test]
    fn collapses_a_group_that_resolves_to_one_line() {
        let answer = "It logs every pi into a database with its Mac address [15:10, 15:23].";
        let out = snap_citations(answer, &pi_lines());
        assert_eq!(out.matches("15:23").count(), 1, "should dedupe, got:\n{out}");
        assert!(!out.contains("15:10"), "should re-point the bad stamp:\n{out}");
    }

    #[test]
    fn each_sentence_is_scored_independently() {
        let answer = "You download Raspberry Pi Imager onto the laptop with the Intel mobile chip \
                      [7:14]. It logs every pi into a database with its Mac address [7:14].";
        let out = snap_citations(answer, &pi_lines());
        let (first, second) = out.split_once(". ").expect("two sentences");
        assert!(first.contains("[7:14]"), "first should stay:\n{out}");
        assert!(second.contains("[15:23]"), "second should snap:\n{out}");
    }

    #[test]
    fn abbreviations_do_not_split_the_sentence() {
        let r = sentence_ranges("Uses e.g. Docker here. Then more.");
        assert_eq!(r.len(), 2, "e.g. must not end a sentence: {r:?}");
    }

    #[test]
    fn non_timestamps_in_brackets_are_left_alone() {
        let answer = "See [note] and [99] and [abc] here [15:10].";
        let out = snap_citations(answer, &pi_lines());
        assert!(out.contains("[note]") && out.contains("[99]") && out.contains("[abc]"));
    }

    #[test]
    fn empty_excerpts_are_a_no_op() {
        let answer = "Nothing to check here [1:23].";
        assert_eq!(snap_citations(answer, &[]), answer);
    }

    use crate::store::SubCue;

    fn speech_chunk(start: f64, end: f64, cues: Vec<(f64, &str)>) -> Chunk {
        let text = cues.iter().map(|(_, t)| *t).collect::<Vec<_>>().join(" ");
        Chunk {
            start,
            end,
            text,
            source: "speech".into(),
            vector: vec![1.0],
            cues: cues.into_iter().map(|(s, t)| SubCue { start: s, text: t.into() }).collect(),
        }
    }

    fn meta() -> VideoMeta {
        VideoMeta { title: "T".into(), channel: "C".into(), duration_secs: 100.0, ..Default::default() }
    }

    #[test]
    fn prompt_cites_cue_level_timestamps() {
        let chunk = speech_chunk(
            10.0,
            55.0,
            vec![(10.0, "intro talk."), (25.0, "the fun virus appears."), (40.0, "outro talk.")],
        );
        let prompt = build_chat_prompt(&meta(), &[&chunk], &[], "q");
        // 25s cue must get its own [0:25] line, not be buried under [0:10].
        assert!(prompt.contains("[0:25] speech:"), "missing cue-level line:\n{prompt}");
        assert!(prompt.contains("the fun virus appears."));
    }

    #[test]
    fn prompt_dedupes_one_cue_overlap() {
        let a = speech_chunk(0.0, 42.0, vec![(0.0, "one."), (20.0, "two."), (40.0, "three.")]);
        let b = speech_chunk(40.0, 80.0, vec![(40.0, "three."), (60.0, "four.")]);
        let prompt = build_chat_prompt(&meta(), &[&a, &b], &[], "q");
        assert_eq!(prompt.matches("three.").count(), 1, "overlap cue duplicated:\n{prompt}");
        assert!(prompt.contains("four."));
    }

    #[test]
    fn prompt_falls_back_without_cues() {
        let mut chunk = speech_chunk(90.0, 100.0, vec![(90.0, "legacy text.")]);
        chunk.cues.clear();
        let prompt = build_chat_prompt(&meta(), &[&chunk], &[], "q");
        assert!(prompt.contains("[1:30] speech: legacy text."));
    }

    #[test]
    fn prompt_includes_overview_when_present() {
        let chunk = speech_chunk(0.0, 10.0, vec![(0.0, "hello.")]);
        let mut m = meta();
        m.summary = Some("An overview.\nChapters:\n[0:00] Start".into());
        let prompt = build_chat_prompt(&m, &[&chunk], &[], "q");
        assert!(prompt.contains("An overview."));
        assert!(prompt.contains("Overview of the whole video"));

        m.summary = None;
        let prompt = build_chat_prompt(&m, &[&chunk], &[], "q");
        assert!(!prompt.contains("Overview of the whole video"));
    }

    #[test]
    fn retrieve_budget_is_a_hard_cap() {
        let big_text = "x".repeat(600);
        let chunks: Vec<Chunk> = (0..10)
            .map(|i| {
                let mut c = speech_chunk(i as f64, i as f64 + 1.0, vec![(i as f64, "t")]);
                c.text = big_text.clone();
                c.vector = vec![1.0];
                c
            })
            .collect();
        let selected = retrieve(&chunks, &[1.0], "q", 8);
        let total: usize = selected.iter().map(|c| c.text.len()).sum();
        assert!(total <= 3600, "budget overshot: {total}");
        assert!(!selected.is_empty());
    }

    #[test]
    fn fabricated_raw_omni_text_cannot_enter_chat_retrieval() {
        let mut raw = speech_chunk(10.0, 20.0, vec![]);
        raw.source = "audiovisual".into();
        raw.text = "Invented account number 123-456-7890".into();
        raw.vector = vec![1.0];
        let mut reviewed = speech_chunk(10.0, 20.0, vec![]);
        reviewed.source = "audiovisual_reviewed".into();
        reviewed.text = "A document appears beside the presenter; account details remain unconfirmed.".into();
        reviewed.vector = vec![1.0];
        let chunks = vec![raw, reviewed];
        let selected = retrieve(&chunks, &[1.0], "account number", 4);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].source, "audiovisual_reviewed");
    }

    #[test]
    fn split_digest_respects_caps() {
        let lines: Vec<String> = (0..100).map(|i| format!("[0:{i:02}] line {i} padding")).collect();
        let parts = split_digest(&lines, 300, 8);
        assert!(parts.len() <= 8 + 1, "too many parts: {}", parts.len());
        assert!(parts.len() > 1);
        // Nothing lost.
        let joined: usize = parts.iter().map(|p| p.lines().count()).sum();
        assert_eq!(joined, 100);
    }

    #[test]
    fn clean_rewrite_filters_junk() {
        assert_eq!(
            clean_rewrite("\"fun virus first appearance\"\n", "what about it?"),
            Some("fun virus first appearance".into())
        );
        assert_eq!(clean_rewrite("   \n\n", "q"), None);
        assert_eq!(clean_rewrite("what about it?", "what about it?"), None);
        let long = "x".repeat(300);
        assert_eq!(clean_rewrite(&long, "q"), None);
    }

    #[test]
    fn lexical_boost_prefers_exact_phrases() {
        let with_phrase = lexical_boost("where is the fun virus", "how about the fun virus? ok");
        let scattered = lexical_boost("where is the fun virus", "virus stuff is fun to watch a virus");
        let unrelated = lexical_boost("where is the fun virus", "completely different words");
        assert!(with_phrase > scattered, "{with_phrase} vs {scattered}");
        assert!(scattered > unrelated);
        assert_eq!(unrelated, 0.0);
    }
}
