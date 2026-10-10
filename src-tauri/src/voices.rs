//! Speaker separation ("voices"): pyannote segmentation, WeSpeaker embeddings
//! and clustering through sherpa-onnx, run offline on the CPU.
//!
//! sherpa-onnx ignores a progress callback's request to stop, so diarization
//! runs in a child process (this binary with `--diarize`) that ingestion can
//! kill on cancel. The child also releases the decoded audio, about 460 MB for
//! two hours, as soon as it exits.
//!
//! Labels are automatic voice clusters ordered by speaking time ("Voice A"
//! speaks most). They are not identities.
use std::ffi::c_void;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncBufReadExt;

use crate::vtt::Cue;

pub const ENGINE: &str = "sherpa-onnx 1.13.8: pyannote segmentation 3.0, WeSpeaker ResNet34-LM, fast clustering";
/// Cosine clustering threshold, measured on the 32:45 test source.
pub const THRESHOLD: f32 = 0.5;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub start: f64,
    pub end: f64,
    pub voice: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VoiceSummary {
    pub voice: String,
    pub seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceRecord {
    pub engine: String,
    pub threshold: f32,
    pub turns: Vec<Turn>,
    pub voices: Vec<VoiceSummary>,
}

pub fn load(dir: &Path) -> Option<VoiceRecord> {
    serde_json::from_slice(&std::fs::read(dir.join("voices.json")).ok()?).ok()
}

/// Name raw cluster indices "Voice A", "Voice B"... by descending speaking time.
pub fn label(segments: &[(f64, f64, i32)]) -> (Vec<Turn>, Vec<VoiceSummary>) {
    let mut totals: Vec<(i32, f64)> = Vec::new();
    for &(start, end, speaker) in segments {
        match totals.iter_mut().find(|(s, _)| *s == speaker) {
            Some((_, total)) => *total += end - start,
            None => totals.push((speaker, end - start)),
        }
    }
    totals.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let name = |i: usize| {
        let letters = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        if i < 26 { format!("Voice {}", letters[i] as char) } else { format!("Voice {}", i + 1) }
    };
    let names: Vec<(i32, String)> = totals.iter().enumerate().map(|(i, (s, _))| (*s, name(i))).collect();
    let voice = |speaker: i32| names.iter().find(|(s, _)| *s == speaker).unwrap().1.clone();
    let mut turns: Vec<Turn> = segments.iter().map(|&(start, end, s)| Turn { start, end, voice: voice(s) }).collect();
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    let summary = totals.iter().map(|(s, seconds)| VoiceSummary { voice: voice(*s), seconds: (seconds * 10.0).round() / 10.0 }).collect();
    (turns, summary)
}

/// The voice for each cue: the turn overlapping most with the time from the
/// cue's start to the next cue's start (at most six seconds; caption end
/// times are unreliable for rolling captions). Captions mark a change of
/// speaker with ">>"; for those cues a turn that begins within a second of
/// the cue wins even when it is short, because overlap alone lets a brief
/// retort be absorbed by the voice around it. Voices with under five seconds
/// in the whole source are too small to trust for this. Cues without any
/// overlapping turn, such as music, inherit the previous cue's voice, or the
/// next one's at the start of the video.
pub fn assign(cues: &[Cue], turns: &[Turn]) -> Vec<Option<String>> {
    let mut totals: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for t in turns {
        *totals.entry(t.voice.as_str()).or_default() += t.end - t.start;
    }
    let mut voices: Vec<Option<String>> = cues
        .iter()
        .enumerate()
        .map(|(i, cue)| {
            if cue.text.trim_start().starts_with(">>") {
                let onset = turns
                    .iter()
                    .filter(|t| (t.start - cue.start).abs() <= 1.0 && totals[t.voice.as_str()] >= 5.0)
                    .min_by(|a, b| (a.start - cue.start).abs().total_cmp(&(b.start - cue.start).abs()));
                if let Some(t) = onset {
                    return Some(t.voice.clone());
                }
            }
            let end = cues.get(i + 1).map_or(cue.end, |next| next.start).max(cue.start + 0.5).min(cue.start + 6.0);
            turns
                .iter()
                .map(|t| (t.end.min(end) - t.start.max(cue.start), t))
                .filter(|(overlap, _)| *overlap > 0.0)
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .map(|(_, t)| t.voice.clone())
        })
        .collect();
    for i in 1..voices.len() {
        if voices[i].is_none() {
            voices[i] = voices[i - 1].clone();
        }
    }
    if let Some(first) = voices.iter().find_map(|v| v.clone()) {
        for v in voices.iter_mut().take_while(|v| v.is_none()) {
            *v = Some(first.clone());
        }
    }
    voices
}

#[derive(Serialize, Deserialize)]
struct ChildOutput {
    segments: Vec<(f64, f64, i32)>,
}

/// Run diarization in a child process, resuming from `voices.json`.
pub async fn diarize(
    segmentation: &Path,
    voiceprint: &Path,
    video: &Path,
    dir: &Path,
    cancel: &AtomicBool,
    progress: impl Fn(f32),
) -> Result<VoiceRecord> {
    if let Some(record) = load(dir).filter(|r| r.engine == ENGINE && r.threshold == THRESHOLD) {
        return Ok(record);
    }
    let ffmpeg = crate::ytdlp::find_ffmpeg().ok_or_else(|| anyhow!("ffmpeg not found: install it with `brew install ffmpeg`"))?;
    let output = dir.join("voices-raw.json");
    let _ = std::fs::remove_file(&output);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).clamp(1, 12);
    let mut child = tokio::process::Command::new(std::env::current_exe()?)
        .arg("--diarize")
        .arg(&ffmpeg)
        .arg(video)
        .arg(segmentation)
        .arg(voiceprint)
        .arg(THRESHOLD.to_string())
        .arg(threads.to_string())
        .arg(&output)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("Voice separation could not start")?;
    let mut lines = tokio::io::BufReader::new(child.stderr.take().unwrap()).lines();
    let mut tail = Vec::new();
    loop {
        tokio::select! {
            line = lines.next_line() => match line? {
                Some(line) => {
                    if let Some((done, total)) = line.strip_prefix("progress ").and_then(|p| p.split_once('/')) {
                        if let (Ok(done), Ok(total)) = (done.parse::<f32>(), total.parse::<f32>()) {
                            progress(done / total.max(1.0));
                        }
                    } else if !line.trim().is_empty() {
                        tail.push(line);
                        if tail.len() > 8 { tail.remove(0); }
                    }
                }
                None => break,
            },
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if cancel.load(Ordering::Relaxed) {
                    child.kill().await?;
                    bail!("cancelled");
                }
            }
        }
    }
    let status = child.wait().await?;
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    if !status.success() {
        bail!("Voice separation failed: {}", tail.join(" | "));
    }
    let raw: ChildOutput = serde_json::from_slice(&std::fs::read(&output)?)?;
    let _ = std::fs::remove_file(&output);
    let (turns, voices) = label(&raw.segments);
    let record = VoiceRecord { engine: ENGINE.into(), threshold: THRESHOLD, turns, voices };
    let path = dir.join("voices.json");
    let temp = path.with_extension("json.part");
    std::fs::write(&temp, serde_json::to_vec(&record)?)?;
    std::fs::rename(temp, path)?;
    Ok(record)
}

use sherpa_onnx_sys as sys;
extern "C" {
    // Exported by the sherpa-onnx C API; the bindings omit it.
    fn SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
        sd: *const sys::OfflineSpeakerDiarization,
        samples: *const f32,
        n: i32,
        callback: extern "C" fn(i32, i32, *mut c_void) -> i32,
        arg: *mut c_void,
    ) -> *const sys::OfflineSpeakerDiarizationResult;
}
extern "C" fn report(done: i32, total: i32, _: *mut c_void) -> i32 {
    eprintln!("progress {done}/{total}");
    0
}

/// `youscoop --diarize FFMPEG VIDEO SEGMENTATION VOICEPRINT THRESHOLD THREADS OUTPUT`
pub fn child_main(args: &[String]) -> i32 {
    match child(args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e:#}");
            1
        }
    }
}
fn child(args: &[String]) -> Result<()> {
    let [ffmpeg, video, segmentation, voiceprint, threshold, threads, output] = args else {
        bail!("Usage: --diarize FFMPEG VIDEO SEGMENTATION VOICEPRINT THRESHOLD THREADS OUTPUT");
    };
    let decoded = std::process::Command::new(ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .output()?;
    if !decoded.status.success() {
        bail!("ffmpeg could not decode the audio: {}", String::from_utf8_lossy(&decoded.stderr));
    }
    let samples: Vec<f32> = decoded.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    drop(decoded);
    if samples.len() > i32::MAX as usize || samples.len() < 16000 {
        bail!("The audio is too short or too long for voice separation");
    }
    let threads: i32 = threads.parse()?;
    let mut config = sherpa_onnx::OfflineSpeakerDiarizationConfig::default();
    config.segmentation.pyannote.model = Some(segmentation.clone());
    config.segmentation.num_threads = threads;
    config.embedding.model = Some(voiceprint.clone());
    config.embedding.num_threads = threads;
    config.clustering.threshold = threshold.parse()?;
    let diarizer = sherpa_onnx::OfflineSpeakerDiarization::create(&config).context("Voice separation models could not load")?;
    // The wrapper's only field is the C pointer; it hides it, so read it directly.
    const _: () = assert!(std::mem::size_of::<sherpa_onnx::OfflineSpeakerDiarization>() == std::mem::size_of::<*const c_void>());
    let raw: *const sys::OfflineSpeakerDiarization = unsafe { std::mem::transmute_copy(&diarizer) };
    let result = unsafe {
        SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(raw, samples.as_ptr(), samples.len() as i32, report, std::ptr::null_mut())
    };
    if result.is_null() {
        bail!("Voice separation produced no result");
    }
    let segments = unsafe { read_result(result) };
    std::fs::write(output, serde_json::to_vec(&ChildOutput { segments })?)?;
    Ok(())
}
unsafe fn read_result(result: *const sys::OfflineSpeakerDiarizationResult) -> Vec<(f64, f64, i32)> {
    let n = sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
    let segments = sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);
    let out = if segments.is_null() || n <= 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(segments, n as usize).iter().map(|s| (s.start as f64, s.end as f64, s.speaker)).collect()
    };
    if !segments.is_null() {
        sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments);
    }
    sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cue(start: f64, end: f64) -> Cue {
        Cue { start, end, text: format!("cue at {start}") }
    }
    #[test]
    fn voices_are_lettered_by_speaking_time() {
        let (turns, voices) = label(&[(0.0, 10.0, 3), (10.0, 12.0, 0), (12.0, 30.0, 3), (30.0, 33.0, 5)]);
        let summary: Vec<_> = voices.iter().map(|v| (v.voice.as_str(), v.seconds)).collect();
        assert_eq!(summary, [("Voice A", 28.0), ("Voice B", 3.0), ("Voice C", 2.0)]);
        assert_eq!(turns[1].voice, "Voice C");
        assert_eq!(turns[3].voice, "Voice B");
    }
    #[test]
    fn cues_take_the_most_overlapping_voice_and_gaps_inherit() {
        let turns = vec![
            Turn { start: 0.0, end: 9.0, voice: "Voice A".into() },
            Turn { start: 9.0, end: 20.0, voice: "Voice B".into() },
        ];
        // The second cue's caption end is far too late; the next cue's start bounds it.
        let cues = vec![cue(2.0, 3.0), cue(8.5, 30.0), cue(12.0, 13.0), cue(40.0, 41.0)];
        assert_eq!(assign(&cues, &turns), [Some("Voice A"), Some("Voice B"), Some("Voice B"), Some("Voice B")].map(|v| v.map(String::from)));
        let early = vec![cue(0.0, 0.1), cue(10.0, 11.0)];
        let late = vec![Turn { start: 9.5, end: 12.0, voice: "Voice B".into() }];
        assert_eq!(assign(&early, &late), [Some("Voice B".to_string()), Some("Voice B".to_string())]);
    }
    #[test]
    fn a_caption_speaker_change_takes_the_turn_that_starts_there() {
        // From the 32:45 source: a half-second retort inside a longer clip of another voice.
        let turns = vec![
            Turn { start: 300.0, end: 420.0, voice: "Voice A".into() },
            Turn { start: 435.95, end: 449.92, voice: "Voice B".into() },
            Turn { start: 438.56, end: 439.08, voice: "Voice A".into() },
            Turn { start: 443.29, end: 444.06, voice: "Voice C".into() },
        ];
        let cue = |start: f64, text: &str| Cue { start, end: start + 3.0, text: text.into() };
        let cues = vec![cue(436.0, ">> Jeffrey Epstein did not have a bank"), cue(439.0, ">> Yes, he did. This premise is hobgoblins."), cue(443.0, ">> Wait, what?"), cue(445.0, "no marker here")];
        // Voice C has under five seconds in total, so its onset is not trusted.
        assert_eq!(assign(&cues, &turns), [Some("Voice B"), Some("Voice A"), Some("Voice B"), Some("Voice B")].map(|v| v.map(String::from)));
    }
}
