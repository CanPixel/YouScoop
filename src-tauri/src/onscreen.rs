//! On-screen text: full-resolution frames sampled at a fixed interval, read by
//! Apple's Vision framework through the embedded `youscoop-ocr` helper.
//!
//! Scene keyframes are 512 px wide and miss documents shown between cuts, so
//! this stage samples the original video instead. Raw recognition results are
//! kept in `ocr.json`; repeated frames of one screen are merged into a single
//! passage whose clearest frame is saved as its original artifact.
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

pub const ENGINE: &str = "Apple Vision VNRecognizeTextRequest (accurate)";
/// Seconds between sampled frames.
pub const INTERVAL: f64 = 2.0;
/// Seconds of video decoded and read per checkpoint.
const WINDOW: f64 = 120.0;
static HELPER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/youscoop-ocr"));

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OcrSample {
    pub ts: f64,
    pub lines: Vec<OcrLine>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Screen {
    pub start: f64,
    pub end: f64,
    /// Timestamp of the clearest sampled frame, preserved as `frame`.
    pub frame_ts: f64,
    pub text: String,
    #[serde(default)]
    pub frame: String,
}
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrRecord {
    pub engine: String,
    pub interval_secs: f64,
    pub sampled_until: f64,
    pub complete: bool,
    pub samples: Vec<OcrSample>,
    pub screens: Vec<Screen>,
}

pub fn load(dir: &Path) -> Option<OcrRecord> {
    serde_json::from_slice(&std::fs::read(dir.join("ocr.json")).ok()?).ok()
}
fn save(dir: &Path, record: &OcrRecord) -> Result<()> {
    let path = dir.join("ocr.json");
    let temp = path.with_extension("json.part");
    std::fs::write(&temp, serde_json::to_vec(record)?)?;
    std::fs::rename(temp, path)?;
    Ok(())
}

/// Write the embedded helper to `bin_dir` once per build of it.
fn helper(bin_dir: &Path) -> Result<PathBuf> {
    if HELPER.is_empty() {
        bail!("On-screen text recognition requires macOS");
    }
    let hash = HELPER.iter().fold(0xcbf29ce484222325_u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3));
    let path = bin_dir.join(format!("youscoop-ocr-{hash:016x}"));
    if !path.is_file() {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(bin_dir)?;
        let temp = path.with_extension("part");
        std::fs::write(&temp, HELPER)?;
        std::fs::set_permissions(&temp, std::fs::Permissions::from_mode(0o755))?;
        std::fs::rename(&temp, &path)?;
    }
    Ok(path)
}

#[derive(Deserialize)]
struct HelperPage {
    path: String,
    lines: Vec<OcrLine>,
}

/// Sample, read and group on-screen text, resuming from `ocr.json`.
/// `progress` receives the fraction of the video read and a short detail.
pub async fn read(
    bin_dir: &Path,
    video: &Path,
    dir: &Path,
    duration: f64,
    cancel: &AtomicBool,
    progress: impl Fn(f32, String),
) -> Result<OcrRecord> {
    let mut record = load(dir).filter(|r| r.interval_secs == INTERVAL).unwrap_or_else(|| OcrRecord {
        engine: ENGINE.into(),
        interval_secs: INTERVAL,
        ..Default::default()
    });
    // Screens are always regrouped from the raw samples, so improved grouping
    // rules apply to videos read earlier without decoding them again.
    let helper = if record.sampled_until < duration { Some(helper(bin_dir)?) } else { None };
    let scratch = dir.join("ocr-tmp");
    while record.sampled_until < duration {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let start = record.sampled_until;
        let length = WINDOW.min(duration - start);
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch)?;
        let status = crate::media::ffmpeg_cmd()?
            .arg("-ss").arg(format!("{start:.3}"))
            .arg("-t").arg(format!("{length:.3}"))
            .arg("-i").arg(video)
            .arg("-vf").arg(format!("fps=1/{INTERVAL}"))
            .arg("-q:v").arg("2")
            .arg(scratch.join("%05d.jpg"))
            .status()
            .await?;
        if !status.success() {
            bail!("ffmpeg could not sample frames for on-screen text");
        }
        let mut frames: Vec<PathBuf> = std::fs::read_dir(&scratch)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
            .collect();
        frames.sort();
        let times: HashMap<String, f64> = frames
            .iter()
            .enumerate()
            .map(|(i, p)| (p.to_string_lossy().into_owned(), start + i as f64 * INTERVAL))
            .collect();
        if let (false, Some(helper)) = (frames.is_empty(), &helper) {
            let output = tokio::process::Command::new(helper)
                .args(&frames)
                .kill_on_drop(true)
                .output()
                .await
                .context("The on-screen text helper could not start")?;
            if !output.status.success() {
                bail!("On-screen text recognition failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            let pages: Vec<HelperPage> = serde_json::from_slice(&output.stdout)?;
            for page in pages {
                let ts = *times.get(&page.path).ok_or_else(|| anyhow!("Unexpected OCR page"))?;
                if ts < duration {
                    record.samples.push(OcrSample { ts, lines: page.lines });
                }
            }
        }
        record.sampled_until = start + length;
        save(dir, &record)?;
        let screens = group_screens(&record.samples, INTERVAL).len();
        progress((record.sampled_until / duration) as f32, format!("Read on-screen text to {} ({screens} screens with text)", crate::rag::format_ts(record.sampled_until)));
    }
    let _ = std::fs::remove_dir_all(&scratch);

    let frames_dir = dir.join("ocr-frames");
    std::fs::create_dir_all(&frames_dir)?;
    let mut screens = group_screens(&record.samples, INTERVAL);
    for screen in &mut screens {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let name = format!("ocr-{:08}.jpg", (screen.frame_ts * 1000.0).round() as u64);
        let path = frames_dir.join(&name);
        if !path.is_file() {
            let status = crate::media::ffmpeg_cmd()?
                .arg("-ss").arg(format!("{:.3}", screen.frame_ts))
                .arg("-i").arg(video)
                .arg("-frames:v").arg("1")
                .arg("-q:v").arg("2")
                .arg(&path)
                .status()
                .await?;
            if !status.success() || !path.is_file() {
                bail!("ffmpeg could not preserve the frame at {}", crate::rag::format_ts(screen.frame_ts));
            }
        }
        screen.frame = name;
    }
    // Remove preserved frames that no screen uses any more.
    let used: HashSet<&str> = screens.iter().map(|s| s.frame.as_str()).collect();
    for entry in std::fs::read_dir(&frames_dir)?.flatten() {
        if !used.contains(entry.file_name().to_string_lossy().as_ref()) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    record.screens = screens;
    record.complete = true;
    save(dir, &record)?;
    Ok(record)
}

/// Letters of non-overlay text that make a keyframe a screen of text. Measured
/// on the test source: screens had 426 to 1,231, talking-head frames at most 51.
const SCREEN_LETTERS: usize = 200;

/// OCR the exact keyframes the frame-description model saw, at full
/// resolution, cached in keyframe-ocr.json by timestamp.
pub async fn read_keyframes(bin_dir: &Path, video: &Path, dir: &Path, times: &[f64]) -> Result<Vec<OcrSample>> {
    let path = dir.join("keyframe-ocr.json");
    let mut cached: Vec<OcrSample> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let key = |t: f64| (t * 1000.0).round() as i64;
    let missing: Vec<f64> = times.iter().copied().filter(|t| !cached.iter().any(|s| key(s.ts) == key(*t))).collect();
    if !missing.is_empty() {
        let helper = helper(bin_dir)?;
        let scratch = dir.join("keyframe-ocr-tmp");
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch)?;
        let mut frames = Vec::new();
        for (i, t) in missing.iter().enumerate() {
            let frame = scratch.join(format!("{i:05}.jpg"));
            let status = crate::media::ffmpeg_cmd()?
                .arg("-ss").arg(format!("{t:.3}"))
                .arg("-i").arg(video)
                .arg("-frames:v").arg("1")
                .arg("-q:v").arg("2")
                .arg(&frame)
                .status()
                .await?;
            if status.success() && frame.is_file() {
                frames.push((frame, *t));
            }
        }
        for batch in frames.chunks(60) {
            let output = tokio::process::Command::new(&helper).args(batch.iter().map(|(f, _)| f)).kill_on_drop(true).output().await?;
            if !output.status.success() {
                bail!("On-screen text recognition failed: {}", String::from_utf8_lossy(&output.stderr));
            }
            let pages: Vec<HelperPage> = serde_json::from_slice(&output.stdout)?;
            for (page, (_, t)) in pages.into_iter().zip(batch) {
                cached.push(OcrSample { ts: *t, lines: page.lines });
            }
        }
        let _ = std::fs::remove_dir_all(&scratch);
        let temp = path.with_extension("json.part");
        std::fs::write(&temp, serde_json::to_vec(&cached)?)?;
        std::fs::rename(temp, &path)?;
    }
    Ok(cached)
}

/// A factual description for a keyframe that is mostly a screen of text, used
/// instead of the frame model's guess, which tends to misread or invent it.
pub fn screen_frame_label(sample: &OcrSample, overlays: &HashSet<String>) -> Option<String> {
    let text: Vec<&str> = sample
        .lines
        .iter()
        .filter(|l| l.confidence >= 0.5 && normalized(&l.text).chars().count() >= 3 && !overlays.contains(&normalized(&l.text)))
        .map(|l| l.text.as_str())
        .collect();
    let text = text.join("\n");
    if letters(&text) < SCREEN_LETTERS {
        return None;
    }
    let shown = match screen_kind(&text) {
        Some("Email") => "an email",
        Some("Memo") => "a memo",
        Some("Search results") => "web search results",
        Some("Wikipedia page") => "a Wikipedia page",
        Some("News article") => "a news article",
        Some("Government document") => "a government document",
        Some("Creator's note") => "a note added by the video's creator",
        _ => "a page of text",
    };
    Some(format!("[Screen] The frame shows {shown}. Its wording is in the on-screen text passage for this moment."))
}

/// What kind of screen a passage shows, from layout markers OCR reliably
/// reads. Used as a label such as "[Email]" so a question about "emails"
/// finds email screens that never contain the word, and so a listing question
/// can return every screen of a kind. None when no marker is clear.
pub fn screen_kind(text: &str) -> Option<&'static str> {
    let lower = text.to_lowercase();
    let lines: Vec<&str> = lower.lines().map(str::trim).collect();
    let starts = |prefixes: &[&str]| lines.iter().filter(|l| prefixes.iter().any(|p| l.starts_with(p))).count();
    // A memorandum also has From/To/Date headers, so it is checked first.
    if lines.iter().any(|l| l.starts_with("memorandum") || *l == "memo") {
        return Some("Memo");
    }
    let headers = starts(&["from:", "to:", "subject:", "sent:", "date:", "cc:"]);
    if headers >= 2 || lower.contains("sent from my iphone") || (lower.contains(" wrote:") && headers >= 1) {
        return Some("Email");
    }
    if lines.first().is_some_and(|l| *l == "google") || (lower.contains("images") && lower.contains("videos") && lower.contains("shopping")) {
        return Some("Search results");
    }
    // OCR often garbles the tagline ("From Vilkipodia, the froe encyclopeda").
    if lower.contains("wikipedia") || lower.contains("view history") || (lines.contains(&"article") && lines.contains(&"talk")) || lower.contains("encyclop") {
        return Some("Wikipedia page");
    }
    let outlets = ["reuters", "associated press", "bbc", "cnn", "fox news", "nbc", "cbs", "abc news", "the guardian", "new york times", "washington post", "daily mail", "the sun", "telegraph", "newsweek", "politico", "axios", "bloomberg"];
    if outlets.iter().any(|o| lines.iter().any(|l| l == o || l.starts_with(&format!("{o} ")))) || lines.iter().any(|l| l.starts_with("published:") || l.contains("reporter")) {
        return Some("News article");
    }
    if lower.contains(".gov") {
        return Some("Government document");
    }
    // The creator's own annotations are marked with an asterisk.
    if lines.iter().any(|l| l.starts_with('*') && l.chars().filter(|c| c.is_alphabetic()).count() >= 10) {
        return Some("Creator's note");
    }
    None
}

/// The passage text YouScoop indexes for a screen: its kind label, then the text read.
pub fn passage_text(screen: &Screen) -> String {
    match screen_kind(&screen.text) {
        Some(kind) => format!("[{kind}] {}", screen.text),
        None => screen.text.clone(),
    }
}

fn letters(s: &str) -> usize {
    s.chars().filter(|c| c.is_alphabetic()).count()
}
fn normalized(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}
fn words(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 { 0.0 } else { a.intersection(b).count() as f64 / union as f64 }
}
/// Two readings of one screen: mostly the same words, or one partial reading
/// (a page still scrolling or zooming in) mostly contained in the other.
/// Different emails share header words, so containment needs a high bar.
fn same_screen(a: &HashSet<String>, b: &HashSet<String>) -> bool {
    let smaller = a.len().min(b.len());
    jaccard(a, b) >= 0.5 || (smaller >= 8 && a.intersection(b).count() * 10 >= smaller * 6)
}

#[derive(Default)]
struct Recurrence {
    samples: usize,
    runs: usize,
    last: f64,
    centers: Vec<(f64, f64)>,
    height: f64,
}
impl Recurrence {
    /// Large text that never moves: a graphic laid over the video.
    fn fixed_graphic(&self) -> bool {
        let n = self.centers.len() as f64;
        let mean = |f: fn(&(f64, f64)) -> f64| self.centers.iter().map(f).sum::<f64>() / n;
        let (mx, my) = (mean(|c| c.0), mean(|c| c.1));
        let spread = self.centers.iter().map(|c| (c.0 - mx).abs().max((c.1 - my).abs())).fold(0.0, f64::max);
        spread <= 0.02 && self.height / n >= 0.06
    }
}

/// Merge consecutive samples showing the same text into screens.
///
/// Lines below 0.5 confidence or with fewer than three letters and digits are
/// ignored. Overlays are lines that return in at least three separate runs and
/// either cover 8% of samples, contain no letters (watermarks, timecodes), or
/// are large text at a fixed position (banners such as "available on").
/// Document text recurs too, but it is small or moves with zoom and layout.
/// A sample needs twelve letters of remaining text to count. Consecutive
/// samples join a screen when their word sets overlap by half; the screen
/// keeps its clearest sample's text. Back-to-back short screens, such as
/// burned-in subtitles, are then joined into one passage.
/// Normalized texts of recurring overlays (watermarks, banners) across samples.
pub fn overlays(samples: &[OcrSample], interval: f64) -> HashSet<String> {
    let kept = confident_lines(samples);
    let mut recurrence: HashMap<String, Recurrence> = HashMap::new();
    for (ts, lines) in &kept {
        for line in lines {
            let entry = recurrence.entry(normalized(&line.text)).or_insert_with(|| Recurrence { last: f64::NEG_INFINITY, ..Default::default() });
            entry.samples += 1;
            if ts - entry.last > interval * 1.5 {
                entry.runs += 1;
            }
            entry.last = *ts;
            entry.centers.push((line.x + line.w / 2.0, line.y + line.h / 2.0));
            entry.height += line.h;
        }
    }
    let total = kept.len();
    recurrence
        .into_iter()
        .filter(|(text, r)| r.runs >= 3 && (r.samples * 100 >= total * 8 || !text.chars().any(char::is_alphabetic) || r.fixed_graphic()))
        .map(|(text, _)| text)
        .collect()
}
fn confident_lines(samples: &[OcrSample]) -> Vec<(f64, Vec<&OcrLine>)> {
    samples
        .iter()
        .map(|s| {
            let mut seen = HashSet::new();
            let lines = s
                .lines
                .iter()
                .filter(|l| l.confidence >= 0.5 && normalized(&l.text).chars().count() >= 3)
                .filter(|l| seen.insert(normalized(&l.text)))
                .collect();
            (s.ts, lines)
        })
        .collect()
}

pub fn group_screens(samples: &[OcrSample], interval: f64) -> Vec<Screen> {
    let kept = confident_lines(samples);
    let overlay_set = overlays(samples, interval);
    let overlay = |text: &str| overlay_set.contains(&normalized(text));
    let kept: Vec<(f64, Vec<(String, f32)>)> = kept
        .into_iter()
        .map(|(ts, lines)| (ts, lines.into_iter().map(|l| (l.text.trim().to_string(), l.confidence)).collect()))
        .collect();

    let mut screens: Vec<(Screen, HashSet<String>, f64, f64)> = Vec::new(); // screen, best words, best score, last ts
    for (ts, lines) in &kept {
        let lines: Vec<_> = lines.iter().filter(|(text, _)| !overlay(text)).collect();
        let text = lines.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join("\n");
        if letters(&text) < 12 {
            continue;
        }
        let score: f64 = lines.iter().map(|(t, c)| letters(t) as f64 * *c as f64).sum();
        let sample_words = words(&text);
        match screens.last_mut() {
            Some((screen, best, best_score, last)) if ts - *last <= interval * 2.01 && same_screen(&sample_words, best) => {
                screen.end = ts + interval;
                *last = *ts;
                if score > *best_score {
                    screen.text = text;
                    screen.frame_ts = *ts;
                    *best = sample_words;
                    *best_score = score;
                }
            }
            _ => screens.push((
                Screen { start: *ts, end: ts + interval, frame_ts: *ts, text, frame: String::new() },
                sample_words,
                score,
                *ts,
            )),
        }
    }
    let mut joined: Vec<Screen> = Vec::new();
    for (screen, ..) in screens {
        if let Some(previous) = joined.last_mut() {
            let short = |s: &Screen| s.end - s.start <= interval * 3.01;
            if screen.start - previous.end <= interval * 0.01 && short(&screen) && (short(previous) || previous.text.len() < 700) && previous.text.len() + screen.text.len() <= 900 {
                previous.end = screen.end;
                for line in screen.text.lines() {
                    // Persistent lines such as a broadcaster's name appear once.
                    if !previous.text.lines().any(|l| l == line) {
                        previous.text.push('\n');
                        previous.text.push_str(line);
                    }
                }
                continue;
            }
        }
        joined.push(screen);
    }
    joined
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(ts: f64, lines: &[(&str, f32)]) -> OcrSample {
        OcrSample {
            ts,
            lines: lines
                .iter()
                .map(|(text, confidence)| OcrLine { text: (*text).into(), confidence: *confidence, x: 0.0, y: 0.0, w: 1.0, h: 0.1 })
                .collect(),
        }
    }
    // Shapes taken from the real 32:45 source: a memo shown for several
    // samples, a channel watermark that changes slightly between frames, and
    // a recurring banner.
    fn source() -> Vec<OcrSample> {
        let mut samples = Vec::new();
        for i in 0..40 {
            let ts = i as f64 * 2.0;
            let mut lines = vec![];
            if i % 4 == 0 { lines.push((["AVALIABLE", "ON", "PATREON"][0], 0.9)); lines.push(("PATREON", 0.9)); }
            if i % 3 == 0 { lines.push((if i % 2 == 0 { ".25. + 61" } else { "25.7 61" }, 0.6)); }
            if (10..15).contains(&i) {
                lines.push(("Memorandum", 0.9));
                lines.push(("From: Jeffrey Epstein / COUQ Foundation", if i == 12 { 0.95 } else { 0.7 }));
                lines.push(("Please wire Eleven Thousand Four Hundred Thirty Eight Dollars", 0.8));
                if i == 12 { lines.push(("To: William Doherty @ JP Morgan", 0.9)); }
            }
            if (20..22).contains(&i) { lines.push(("Epstein's Baal", 0.9)); lines.push(("bank account", 0.9)); }
            samples.push(sample(ts, &lines));
        }
        samples
    }
    #[test]
    fn repeated_frames_become_one_screen_with_the_clearest_text() {
        let screens = group_screens(&source(), 2.0);
        assert_eq!(screens.len(), 2, "{screens:#?}");
        assert_eq!((screens[0].start, screens[0].end, screens[0].frame_ts), (20.0, 30.0, 24.0));
        assert!(screens[0].text.contains("To: William Doherty @ JP Morgan"));
        assert_eq!(screens[1].text, "Epstein's Baal\nbank account");
    }
    #[test]
    fn recurring_overlays_and_letterless_watermarks_are_removed() {
        let screens = group_screens(&source(), 2.0);
        assert!(screens.iter().all(|s| !s.text.contains("PATREON") && !s.text.contains("61")), "{screens:#?}");
    }
    #[test]
    fn large_fixed_banners_are_removed_but_recurring_document_text_is_kept() {
        let line = |text: &str, x: f64, y: f64, h: f64| OcrLine { text: text.into(), confidence: 0.9, x, y, w: 0.2, h };
        let mut samples = Vec::new();
        for (i, ts) in [0.0, 100.0, 200.0, 300.0].iter().enumerate() {
            let shift = i as f64 * 0.1;
            samples.push(OcrSample { ts: *ts, lines: vec![
                line("AVALIABLE", 0.4, 0.27, 0.15),
                line("Subject: Re: Question about the estate", 0.05, 0.1 + shift, 0.02),
                line(&format!("Body text of email number {i} that differs"), 0.05, 0.3 + shift, 0.02),
            ]});
        }
        // A realistic share of frames: the repeated heading stays under 8% coverage.
        samples.extend((0..60).map(|i| sample(400.0 + i as f64 * 10.0, &[(&format!("Unrelated slide number {i} of the talk"), 0.9)])));
        let screens: Vec<_> = group_screens(&samples, 2.0).into_iter().filter(|s| s.start < 400.0).collect();
        assert_eq!(screens.len(), 4);
        assert!(screens.iter().all(|s| !s.text.contains("AVALIABLE") && s.text.contains("Subject: Re: Question")), "{screens:#?}");
    }
    #[test]
    fn back_to_back_subtitles_join_into_one_passage() {
        let samples: Vec<_> = ["in the western elites", "is a very strong desire", "to freeze the existing situation"]
            .iter()
            .enumerate()
            .map(|(i, text)| sample(i as f64 * 2.0, &[(text, 0.9), ("RIA NOVOSTI broadcast", 0.9)]))
            .collect();
        let screens = group_screens(&samples, 2.0);
        assert_eq!(screens.len(), 1, "{screens:#?}");
        assert_eq!(screens[0].text, "in the western elites\nRIA NOVOSTI broadcast\nis a very strong desire\nto freeze the existing situation");
        assert_eq!((screens[0].start, screens[0].end), (0.0, 6.0));
    }
    #[test]
    fn screen_kinds_come_from_layout_markers_on_the_test_source() {
        let cases = [
            ("From: Jeffrey Epstein\nDate: Fri, 24 Apr 2009 11:36:38 -0400\nSubject:\nwhere are you?", Some("Email")),
            ("Subject:\nRe: Question\nOh come on.\nSent from my iPhone", Some("Email")),
            ("WikiLeaks\nShop\nView email\nFrom:podesta@podesta.com\nTo: john.podesta@gmail.com", Some("Email")),
            ("JPM FINANCIAL SPONSORS Jul 29 2009\nMemorandum\nFrom: Jeffrey Epstein / COUQ Foundation\nTo: William Doherty @ JP Morgan\nDate: 'July 27, 2009", Some("Memo")),
            ("Google\nAl Mode\nAll\nchina visit\nImages\nVideos\nShort videos\nNews\nShopping", Some("Search results")),
            ("Isaac Kappy\nAnticie\nTalk\nFrom Vilkipodia, the froe encyclopeda\nIsaac Kappy was an American actor.", Some("Wikipedia page")),
            ("Poland to probe possible links\nAlan Charlish, Pawel Florkiewicz\nReuters\nFeb. 3, 2026", Some("News article")),
            ("MOSCOW'S PAEDO PLOT\nClaudia Lee, Foreign Nees Reporter\nPublished: T233, 3 Feb 2026", Some("News article")),
            ("1 of 1\nAEL DEFENSE FOND\njustice.gov", Some("Government document")),
            ("*(l address lan Carroll refuting the\nfact that this account name is", Some("Creator's note")),
            ("Yes, Phoenicians were essentially the late-period, Iron Age\nCanaanites of the Levant coast", None),
        ];
        for (text, kind) in cases {
            assert_eq!(screen_kind(text), kind, "{text}");
        }
    }
    #[test]
    fn keyframes_full_of_text_get_a_factual_label() {
        let line = |text: &str| OcrLine { text: text.into(), confidence: 0.9, x: 0.1, y: 0.1, w: 0.5, h: 0.02 };
        let email = OcrSample { ts: 1676.0, lines: [
            "Subject:", "Re: Question", "Oh come on.", "Sent from my iPhone", "Please direct all scheduling inquiries to my office at",
            "Follow me on twitter", "On Dec 8, 2017, at 8:16 PM, jeffrey E. wrote:", "I dont know of any. they did a great job of closing most, at least for the goyim",
            "On Sat, Dec 9, 2017 at 2:01 AM wrote: I'm writing another anti tax bill screed and I don't want to use rifle shot benefits",
        ].iter().map(|t| line(t)).collect() };
        assert_eq!(screen_frame_label(&email, &HashSet::new()).as_deref(), Some("[Screen] The frame shows an email. Its wording is in the on-screen text passage for this moment."));
        let banner = OcrSample { ts: 418.0, lines: vec![line("AVALIABLE"), line("ON"), line("PATREON")] };
        assert_eq!(screen_frame_label(&banner, &HashSet::new()), None);
    }
    #[test]
    fn low_confidence_and_tiny_text_is_not_a_screen() {
        let samples = vec![sample(0.0, &[("Blurry headline text here", 0.3)]), sample(2.0, &[("TP", 0.9), ("OK", 0.9)])];
        assert!(group_screens(&samples, 2.0).is_empty());
    }
    #[test]
    fn a_gap_or_a_different_document_starts_a_new_screen() {
        let memo = [("Memorandum about the transfer request", 0.9)];
        let email = [("Subject: Re: Question from the office", 0.9)];
        let samples = vec![sample(0.0, &memo), sample(2.0, &memo), sample(10.0, &memo), sample(12.0, &email), sample(14.0, &email), sample(16.0, &email), sample(18.0, &email)];
        let screens = group_screens(&samples, 2.0);
        assert_eq!(screens.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(), [(0.0, 4.0), (10.0, 12.0), (12.0, 20.0)]);
    }
    #[test]
    fn a_page_read_partly_then_fully_is_one_screen() {
        // Shape of the 29:18 source page: a few lines while it zooms in, then the whole page.
        let partial = [("He had kind eyes but didnt speak directly to me.", 1.0), ("She said to push all the pain away. I don't understand", 1.0)];
        let full = [("Close your eyes close your eyes. Dont speak she doesnt talk.", 1.0), ("I think from Israel. He had kind eyes but didnt speak directly to me.", 1.0),
                    ("She said to push all the pain away. don t understand", 1.0), ("I cant stop shaking and its been a week.", 1.0)];
        let samples = vec![sample(0.0, &partial), sample(2.0, &partial), sample(4.0, &full), sample(6.0, &full)];
        let screens = group_screens(&samples, 2.0);
        assert_eq!(screens.len(), 1, "{screens:#?}");
        assert!(screens[0].text.starts_with("Close your eyes"));
    }
}
