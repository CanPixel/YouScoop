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

/// Select the most relevant chunks for a query vector, capped by count and size.
pub fn retrieve<'a>(chunks: &'a [Chunk], query_vec: &[f32], top_k: usize) -> Vec<&'a Chunk> {
    let mut scored: Vec<(f32, &Chunk)> = chunks
        .iter()
        .filter(|c| !c.vector.is_empty())
        .map(|c| (cosine(&c.vector, query_vec), c))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected: Vec<&Chunk> = Vec::new();
    let mut total_chars = 0usize;
    for (_, c) in scored.into_iter().take(top_k * 2) {
        if selected.len() >= top_k || total_chars > 4000 {
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
    for c in excerpts {
        let kind = if c.source == "vision" { "visual" } else { "speech" };
        excerpt_lines.push_str(&format!("[{}] {kind}: {}\n", format_ts(c.start), c.text.trim()));
    }

    let mut prompt = format!(
        "<|im_start|>User: You are SmolTalk, an assistant that answers questions about one specific video. \
You are given indexed excerpts: \"speech\" lines are from the spoken transcript, \"visual\" lines describe what is on screen.\n\
Video: \"{}\" by {} (duration {}).\n\nExcerpts:\n{}\n\
Answer questions using only these excerpts. When you reference a moment, cite its timestamp in square brackets, e.g. [{}]. \
If the excerpts don't contain the answer, say so plainly. Keep answers short and direct.<end_of_utterance>\n\
Assistant: Understood. Ask me anything about this video and I'll answer with timestamps.<end_of_utterance>\n",
        meta.title,
        meta.channel,
        format_ts(meta.duration_secs),
        excerpt_lines.trim_end(),
        format_ts(excerpts.first().map(|c| c.start).unwrap_or(0.0)),
    );

    for turn in history.iter().rev().take(2).rev() {
        prompt.push_str(&format!(
            "User: {}<end_of_utterance>\nAssistant: {}<end_of_utterance>\n",
            turn.question.trim(),
            turn.answer.trim()
        ));
    }
    prompt.push_str(&format!("User: {}<end_of_utterance>\nAssistant:", question.trim()));
    prompt
}
