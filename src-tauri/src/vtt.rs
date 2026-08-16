//! WebVTT parsing tuned for YouTube captions (including rolling auto-captions),
//! plus transcript chunking for the semantic index.

use serde::{Deserialize, Serialize};

use crate::store::Chunk;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Cue {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

fn parse_timestamp(s: &str) -> Option<f64> {
    // "HH:MM:SS.mmm" or "MM:SS.mmm"
    let s = s.trim();
    let parts: Vec<&str> = s.split(':').collect();
    let (h, m, sec) = match parts.len() {
        3 => (parts[0].parse::<f64>().ok()?, parts[1].parse::<f64>().ok()?, parts[2].parse::<f64>().ok()?),
        2 => (0.0, parts[0].parse::<f64>().ok()?, parts[1].parse::<f64>().ok()?),
        _ => return None,
    };
    Some(h * 3600.0 + m * 60.0 + sec)
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#39;", "'")
        .replace("&quot;", "\"")
        .replace("&nbsp;", " ")
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn parse_vtt(content: &str) -> Vec<Cue> {
    let mut cues: Vec<Cue> = Vec::new();
    let mut lines = content.lines().peekable();
    while let Some(line) = lines.next() {
        if !line.contains("-->") {
            continue;
        }
        let mut halves = line.split("-->");
        let start = halves.next().and_then(parse_timestamp);
        let end = halves
            .next()
            .map(|rest| rest.trim())
            .map(|rest| rest.split_whitespace().next().unwrap_or(""))
            .and_then(parse_timestamp);
        let (Some(start), Some(end)) = (start, end) else { continue };

        let mut text_lines: Vec<String> = Vec::new();
        while let Some(peek) = lines.peek() {
            if peek.trim().is_empty() || peek.contains("-->") {
                break;
            }
            text_lines.push(lines.next().unwrap().to_string());
        }
        let text = collapse_ws(&strip_tags(&text_lines.join(" ")));
        if !text.is_empty() {
            cues.push(Cue { start, end, text });
        }
    }
    dedupe_rolling(cues)
}

/// YouTube auto-captions "roll": each cue repeats the previous line and adds a new one.
/// Keep only the newly-added portion of each cue.
fn dedupe_rolling(cues: Vec<Cue>) -> Vec<Cue> {
    let mut out: Vec<Cue> = Vec::new();
    let mut tail = String::new();
    for cue in cues {
        let novel = novel_suffix(&tail, &cue.text);
        if novel.is_empty() {
            continue;
        }
        tail = cue.text.clone();
        out.push(Cue { start: cue.start, end: cue.end, text: novel });
    }
    out
}

/// The portion of `cur` that is not an overlap with the end of `prev`.
fn novel_suffix(prev: &str, cur: &str) -> String {
    if prev.is_empty() {
        return cur.to_string();
    }
    if prev == cur {
        return String::new();
    }
    // Longest suffix of `prev` that is a prefix of `cur` (word-aligned).
    let prev_words: Vec<&str> = prev.split(' ').collect();
    let cur_words: Vec<&str> = cur.split(' ').collect();
    let max_overlap = prev_words.len().min(cur_words.len());
    for k in (1..=max_overlap).rev() {
        if prev_words[prev_words.len() - k..] == cur_words[..k] {
            return cur_words[k..].join(" ");
        }
    }
    cur.to_string()
}

/// Merge cues into ~`window_secs` transcript chunks for embedding.
/// Consecutive chunks overlap by one cue so sentences (and the claims in
/// them) aren't cut in half at chunk boundaries.
pub fn chunk_cues(cues: &[Cue], window_secs: f64, max_chars: usize) -> Vec<Chunk> {
    use crate::store::SubCue;

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cur_text = String::new();
    let mut cur_cues: Vec<SubCue> = Vec::new();
    let mut cur_start = 0.0_f64;
    let mut cur_end = 0.0_f64;
    let mut seed_only = false;

    for cue in cues {
        if cur_text.is_empty() {
            cur_start = cue.start;
        }
        if !cur_text.is_empty() {
            cur_text.push(' ');
        }
        cur_text.push_str(&cue.text);
        cur_cues.push(SubCue { start: cue.start, text: cue.text.clone() });
        cur_end = cue.end.max(cur_start);
        seed_only = false;

        if cur_end - cur_start >= window_secs || cur_text.len() >= max_chars {
            chunks.push(Chunk {
                start: cur_start,
                end: cur_end,
                text: std::mem::take(&mut cur_text),
                source: "speech".into(),
                vector: Vec::new(),
                cues: std::mem::take(&mut cur_cues),
            });
            // Seed the next chunk with this cue for a one-cue overlap.
            cur_text = cue.text.clone();
            cur_cues = vec![SubCue { start: cue.start, text: cue.text.clone() }];
            cur_start = cue.start;
            cur_end = cue.end.max(cue.start);
            seed_only = true;
        }
    }
    if !seed_only && !cur_text.trim().is_empty() {
        chunks.push(Chunk {
            start: cur_start,
            end: cur_end,
            text: cur_text,
            source: "speech".into(),
            vector: Vec::new(),
            cues: cur_cues,
        });
    }
    chunks
}
