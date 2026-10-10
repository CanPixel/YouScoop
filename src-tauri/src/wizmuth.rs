//! YouScoop's handoff to WiZMUTH: a video's transcript, or one chat answer,
//! written as a Markdown note into the WiZMUTH vault's inbox. WiZMUTH indexes
//! it on its next read of the vault. Notes are plain files: open, edit or
//! delete them like any other note.
use anyhow::{bail, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

use crate::rag;
use crate::store::{Chunk, VideoMeta};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SentNote {
    pub path: String,
    pub vault: String,
}

/// WiZMUTH's chosen vault, or the default it creates in Documents.
fn vault(app: &AppHandle) -> Result<PathBuf> {
    let support = app.path().data_dir()?;
    let chosen = std::fs::read(support.join("com.canpixel.wizmuth").join("sources.json")).ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|v| v["vault"].as_str().map(PathBuf::from))
        .filter(|p| p.is_dir());
    let vault = match chosen {
        Some(v) => v,
        None => app.path().document_dir()?.join("WiZMUTH"),
    };
    std::fs::create_dir_all(&vault)?;
    Ok(vault)
}

fn inbox(vault: &Path) -> Result<PathBuf> {
    let dir = vault.join("WiZMUTH").join("Inbox");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn file_title(title: &str) -> String {
    let cleaned: String = title.chars().map(|c| if "/\\:*?\"<>|#^[]\n\r\t".contains(c) { ' ' } else { c }).collect();
    let cleaned: String = cleaned.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(90).collect();
    let cleaned = cleaned.trim_matches(['.', ' ']).to_string();
    if cleaned.is_empty() { "Untitled".into() } else { cleaned }
}
fn unique(dir: &Path, stem: &str) -> PathBuf {
    let mut path = dir.join(format!("{stem}.md"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem} {n}.md"));
        n += 1;
    }
    path
}
fn yaml(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', " "))
}

/// A link that opens the video at a moment, when the source is on YouTube.
fn moment(meta: &VideoMeta, seconds: f64) -> String {
    let label = rag::format_ts(seconds);
    if meta.url.contains("youtube.com") || meta.url.contains("youtu.be") {
        format!("[{label}](https://youtu.be/{}?t={})", meta.id, seconds.max(0.0) as u64)
    } else {
        format!("[{label}]")
    }
}

fn front_matter(meta: &VideoMeta, today: &str, extra: &str) -> String {
    format!(
        "---\ntitle: {}\nsource: {}\nchannel: {}\nduration: {}\nadded: {today}\norigin: YouScoop\n{extra}tags: [youscoop]\n---\n\n",
        yaml(&meta.title), yaml(&meta.url), yaml(&meta.channel), rag::format_ts(meta.duration_secs)
    )
}

/// The whole spoken transcript in paragraphs of about a minute, each opening
/// at its moment in the video, then the text shown on screen.
pub fn video_note(meta: &VideoMeta, chunks: &[Chunk], today: &str) -> String {
    let refs: Vec<&Chunk> = chunks.iter().collect();
    let mut lines = rag::excerpt_sublines(&refs);
    lines.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut note = front_matter(meta, today, "");
    note.push_str(&format!("# {}\n\n", meta.title));
    if !meta.url.is_empty() { note.push_str(&format!("{} · [Watch]({})\n\n", meta.channel, meta.url)); }
    if let Some(summary) = meta.summary.as_deref().filter(|s| !s.trim().is_empty()) {
        note.push_str(&format!("## Summary\n\n{}\n\n", summary.trim()));
    }
    note.push_str("## Transcript\n\n");
    let mut paragraph = String::new();
    let mut opened = 0.0;
    for line in lines.iter().filter(|l| l.label == "speech") {
        if paragraph.is_empty() { opened = line.start; }
        if !paragraph.is_empty() { paragraph.push(' '); }
        paragraph.push_str(line.text.trim());
        if line.start - opened >= 55.0 || paragraph.len() > 900 {
            note.push_str(&format!("{} {}\n\n", moment(meta, opened), paragraph));
            paragraph.clear();
        }
    }
    if !paragraph.is_empty() { note.push_str(&format!("{} {}\n\n", moment(meta, opened), paragraph)); }
    let screens: Vec<_> = lines.iter().filter(|l| l.label == "on-screen text").collect();
    if !screens.is_empty() {
        note.push_str("## On screen\n\nRead by OCR; characters can be misread.\n\n");
        for s in screens {
            note.push_str(&format!("- {} {}\n", moment(meta, s.start), s.text.split_whitespace().collect::<Vec<_>>().join(" ")));
        }
        note.push('\n');
    }
    note
}

/// One question and its answer, with the answer's timestamps made into links.
pub fn answer_note(meta: &VideoMeta, question: &str, answer: &str, today: &str) -> String {
    let mut linked = String::with_capacity(answer.len());
    let mut rest = answer;
    while let Some(open) = rest.find('[') {
        linked.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        match after.find(']').map(|close| (close, &after[..close])) {
            Some((close, label)) if !label.is_empty() && label.chars().all(|c| c.is_ascii_digit() || c == ':') => {
                let seconds = label.split(':').fold(0.0, |acc, part| acc * 60.0 + part.parse::<f64>().unwrap_or(0.0));
                linked.push_str(&moment(meta, seconds));
                rest = &after[close + 1..];
            }
            _ => {
                linked.push('[');
                rest = after;
            }
        }
    }
    linked.push_str(rest);
    format!(
        "{}# {}\n\nFrom [{}]({}) · {}\n\n> {}\n\n{}\n",
        front_matter(meta, today, &format!("question: {}\n", yaml(question))),
        question.trim(),
        meta.title, meta.url, meta.channel,
        question.trim().replace('\n', "\n> "),
        linked.trim()
    )
}

pub fn send_video(app: &AppHandle, video_id: &str, today: &str) -> Result<SentNote> {
    let dir = crate::store::video_dir_path(app, video_id)?;
    let meta: VideoMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    if meta.status != "ready" { bail!("Finish processing the video first"); }
    let chunks: Vec<Chunk> = serde_json::from_slice(&std::fs::read(dir.join("index.json"))?)?;
    let vault = vault(app)?;
    let path = unique(&inbox(&vault)?, &file_title(&format!("YouScoop - {}", meta.title)));
    std::fs::write(&path, video_note(&meta, &chunks, today))?;
    Ok(SentNote { path: path.to_string_lossy().into(), vault: vault.to_string_lossy().into() })
}

pub fn send_answer(app: &AppHandle, video_id: &str, question: &str, answer: &str, today: &str) -> Result<SentNote> {
    if answer.trim().is_empty() { bail!("Nothing to send"); }
    let dir = crate::store::video_dir_path(app, video_id)?;
    let meta: VideoMeta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let vault = vault(app)?;
    let words: String = question.split_whitespace().take(9).collect::<Vec<_>>().join(" ");
    let path = unique(&inbox(&vault)?, &file_title(&format!("{} - {words}", meta.title.chars().take(40).collect::<String>())));
    std::fs::write(&path, answer_note(&meta, question, answer, today))?;
    Ok(SentNote { path: path.to_string_lossy().into(), vault: vault.to_string_lossy().into() })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn meta() -> VideoMeta {
        serde_json::from_value(serde_json::json!({
            "id": "p5BoJRLOm8I", "url": "https://www.youtube.com/watch?v=p5BoJRLOm8I", "title": "TEN Bands That PEAKED Early",
            "channel": "That Metal Bloke", "durationSecs": 1759.0, "addedAt": 1, "status": "ready"
        })).unwrap()
    }
    #[test]
    fn answers_link_their_moments_into_the_video() {
        let note = answer_note(&meta(), "which 10 bands?", "Down [1:10] and Possessed [4:22]; see [note].", "2026-10-10");
        assert!(note.contains("Down [1:10](https://youtu.be/p5BoJRLOm8I?t=70) and Possessed [4:22](https://youtu.be/p5BoJRLOm8I?t=262); see [note]."));
        assert!(note.starts_with("---\ntitle: \"TEN Bands That PEAKED Early\""));
        assert!(note.contains("question: \"which 10 bands?\""));
    }
    #[test]
    fn transcripts_become_timestamped_paragraphs() {
        let chunk = |start: f64, text: &str, source: &str| Chunk { start, end: start + 5.0, text: text.into(), source: source.into(), vector: vec![], cues: vec![] };
        let chunks = vec![chunk(0.0, "Number ten is Down.", "speech"), chunk(70.0, "Next up we have Possessed.", "speech"), chunk(72.0, "POSSESSED - SEVEN CHURCHES", "ocr")];
        let note = video_note(&meta(), &chunks, "2026-10-10");
        assert!(note.contains("## Transcript\n\n[0:00](https://youtu.be/p5BoJRLOm8I?t=0) Number ten is Down. Next up we have Possessed."));
        assert!(note.contains("## On screen") && note.contains("[1:12](https://youtu.be/p5BoJRLOm8I?t=72) POSSESSED - SEVEN CHURCHES"));
        assert_eq!(file_title("YouScoop - a/b: c?"), "YouScoop - a b c");
    }
}
