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
        .filter(|c| !c.vector.is_empty())
        .map(|c| (cosine(&c.vector, query_vec) + boost_for(&terms, &c.text), c))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<&Chunk> = Vec::new();
    let mut total_chars = 0usize;
    for (_, c) in scored.into_iter().take(top_k * 2) {
        // Keep the excerpt budget modest: SmolVLM2 2.2B starts cross-attributing
        // facts between excerpts when the context grows too large. 3600 chars
        // fits `top_k` full-size (600-char) chunks so ranked chunks aren't
        // silently dropped.
        if selected.len() >= top_k || total_chars > 3600 {
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

pub fn build_chat_prompt(
    meta: &VideoMeta,
    excerpts: &[&Chunk],
    history: &[HistoryTurn],
    question: &str,
) -> String {
    let mut excerpt_lines = String::new();
    let mut emitted_cues: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for c in excerpts {
        if c.source == "vision" {
            excerpt_lines.push_str(&format!("[{}] visual: {}\n", format_ts(c.start), c.text.trim()));
            continue;
        }
        if c.cues.is_empty() {
            // Old index without cue-level timestamps.
            excerpt_lines.push_str(&format!("[{}] speech: {}\n", format_ts(c.start), c.text.trim()));
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
                    excerpt_lines.push_str(&format!("[{}] speech: {line_text}\n", format_ts(start)));
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
            excerpt_lines.push_str(&format!("[{}] speech: {line_text}\n", format_ts(start)));
        }
    }

    // ChatML prompt for the Qwen3 chat model.
    let mut prompt = format!(
        "<|im_start|>system\n\
You are SmolTalk, an assistant that answers questions about one specific video. \
You are given indexed excerpts. \"speech\" lines are quotes from the spoken transcript; names in them may be \
misspelled by the transcriber. \"visual\" lines are automatic frame descriptions: they may misread details, \
but text they quote from the screen is usually really shown there. Prefer speech lines for facts; use visual \
lines for what is shown or written on screen.\n\
Video: \"{}\" by {} (duration {}).\n\nExcerpts:\n{}\n\n\
Rules:\n\
- Answer using only the excerpts above. Do not add facts from your own knowledge, and do not silently \
correct names — quote them as the transcript spells them.\n\
- Cite the timestamp of the specific line a fact comes from in square brackets after each claim, e.g. …the largest star [{}]. Pick the line where the fact is actually said, not a neighboring one.\n\
- Only combine information from different excerpts when each part keeps its own citation.\n\
- If the excerpts don't contain the answer, say so plainly.\n\
- Keep answers short and direct.<|im_end|>\n",
        meta.title,
        meta.channel,
        format_ts(meta.duration_secs),
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

#[cfg(test)]
mod tests {
    use super::*;
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
    fn lexical_boost_prefers_exact_phrases() {
        let with_phrase = lexical_boost("where is the fun virus", "how about the fun virus? ok");
        let scattered = lexical_boost("where is the fun virus", "virus stuff is fun to watch a virus");
        let unrelated = lexical_boost("where is the fun virus", "completely different words");
        assert!(with_phrase > scattered, "{with_phrase} vs {scattered}");
        assert!(scattered > unrelated);
        assert_eq!(unrelated, 0.0);
    }
}
