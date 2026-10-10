//! The Oracle's spoken voice. On macOS a Kokoro neural model renders each
//! sentence to WAV, falling back to the system voice before it is installed;
//! the page plays it through its own low-register chain. Android
//! speaks through its native plugin in `speech.rs`.
use crate::{err, models, AppState, CmdResult};
use std::sync::Arc;
use tauri::{AppHandle, State};

pub struct VoiceSpec {
    pub id: &'static str,
    pub label: &'static str,
    /// Kokoro v0.19 speaker index.
    pub sid: i32,
    /// Above 1.0 is faster. The page plays it back at 0.95 speed, lowering
    /// pitch by about a semitone, so the heard pace is close to natural.
    pub speed: f32,
    /// macOS system voice used until the neural model is installed.
    pub system: &'static str,
}

pub const ORACLE_VOICES: [VoiceSpec; 4] = [
    VoiceSpec { id: "lewis", label: "Lewis · grave, low", sid: 10, speed: 1.03, system: "Daniel" },
    VoiceSpec { id: "george", label: "George · elder, measured", sid: 9, speed: 1.05, system: "Daniel" },
    VoiceSpec { id: "adam", label: "Adam · dark, resonant", sid: 5, speed: 1.06, system: "Ralph" },
    VoiceSpec { id: "michael", label: "Michael · calm, warm", sid: 6, speed: 1.06, system: "Fred" },
];
pub const KOKORO_DIR: &str = "kokoro-en-v0_19";

fn voice(id: &str) -> &'static VoiceSpec {
    ORACLE_VOICES.iter().find(|v| v.id == id).unwrap_or(&ORACLE_VOICES[0])
}

/// Text as it should be heard: no citations, markup or verification tags.
pub fn speakable(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(['[', '(']) {
        out.push_str(&rest[..open]);
        let closer = if rest[open..].starts_with('[') { ']' } else { ')' };
        let Some(close) = rest[open..].find(closer) else { rest = &rest[open..]; break };
        let inner = &rest[open + 1..open + close];
        let annotation = inner.starts_with('E') && inner[1..].chars().all(|c| c.is_ascii_digit() || c == ',' || c == ' ' || c == 'E')
            || inner.starts_with("not verbatim") || inner.starts_with("from an automatic") || inner.starts_with("unavailable evidence");
        if !annotation { out.push_str(&rest[open..=open + close]); }
        rest = &rest[open + close + 1..];
    }
    out.push_str(rest);
    let out = out.replace(['*', '_', '#', '`', '>'], "").replace(" — ", ", ").replace('—', ", ");
    let mut clean = out.split_whitespace().collect::<Vec<_>>().join(" ");
    for p in [" .", " ,", " ;", " :", " ?", " !"] { clean = clean.replace(p, &p[1..]); }
    clean.trim().to_string()
}

/// A 16-bit mono WAV, which every webview decodes.
fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    // Short fades keep sentence joins free of clicks.
    let fade = (rate as usize / 200).min(samples.len() / 2);
    for (i, s) in samples.iter().enumerate() {
        let gain = if i < fade { i as f32 / fade as f32 } else if i + fade >= samples.len() { (samples.len() - i) as f32 / fade as f32 } else { 1.0 };
        out.extend_from_slice(&(((s * gain).clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

#[cfg(target_os = "macos")]
pub struct OracleVoice {
    tts: sherpa_onnx::OfflineTts,
}
#[cfg(target_os = "macos")]
impl OracleVoice {
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        let path = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());
        let config = sherpa_onnx::OfflineTtsConfig {
            model: sherpa_onnx::OfflineTtsModelConfig {
                kokoro: sherpa_onnx::OfflineTtsKokoroModelConfig {
                    model: path("model.onnx"),
                    voices: path("voices.bin"),
                    tokens: path("tokens.txt"),
                    data_dir: path("espeak-ng-data"),
                    ..Default::default()
                },
                // The text model runs on the GPU, so speech can use the CPU cores.
                // fp32 renders about five times faster than the int8 model on Apple CPUs.
                num_threads: std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4).clamp(2, 6),
                provider: Some("cpu".into()),
                ..Default::default()
            },
            ..Default::default()
        };
        let tts = sherpa_onnx::OfflineTts::create(&config).ok_or_else(|| anyhow::anyhow!("The Oracle's voice model could not load"))?;
        Ok(Self { tts })
    }
    pub fn render(&self, text: &str, voice_id: &str) -> anyhow::Result<Vec<u8>> {
        let spec = voice(voice_id);
        let config = sherpa_onnx::GenerationConfig { sid: spec.sid, speed: spec.speed, ..Default::default() };
        let audio = self.tts
            .generate_with_config::<fn(&[f32], f32) -> bool>(text, &config, None)
            .ok_or_else(|| anyhow::anyhow!("The Oracle's voice could not render this sentence"))?;
        Ok(wav(audio.samples(), audio.sample_rate() as u32))
    }
}

/// The system voice rendered to a file, so it passes through the same chain.
#[cfg(target_os = "macos")]
async fn system_voice(text: &str, voice_id: &str) -> CmdResult<Vec<u8>> {
    let dir = std::env::temp_dir().join("sacrinomicon-voice");
    std::fs::create_dir_all(&dir).map_err(err)?;
    let file = dir.join(format!("{}.wav", knowledge_core::stable_id("say", &format!("{text}{:?}", std::time::Instant::now())).replace(':', "-")));
    let status = tokio::process::Command::new("/usr/bin/say")
        .args(["-v", voice(voice_id).system, "-r", "175", "--file-format=WAVE", "--data-format=LEI16@22050", "-o"])
        .arg(&file)
        .arg("--")
        .arg(text)
        .kill_on_drop(true)
        .status()
        .await
        .map_err(err)?;
    let bytes = std::fs::read(&file).map_err(err);
    let _ = std::fs::remove_file(&file);
    if !status.success() {
        return Err("The local system voice failed".into());
    }
    bytes
}

#[cfg(target_os = "macos")]
fn installed_dir(app: &AppHandle) -> CmdResult<Option<std::path::PathBuf>> {
    let spec = models::spec("speech").ok_or("Voice model is not registered")?;
    if !models::is_downloaded(app, spec) {
        return Ok(None);
    }
    let dir = models::models_dir(app).map_err(err)?;
    let target = dir.join(KOKORO_DIR);
    if !target.join("model.onnx").is_file() || !target.join("espeak-ng-data").is_dir() {
        let archive = models::model_path(app, spec).map_err(err)?;
        let staging = dir.join(format!("{KOKORO_DIR}.extract"));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).map_err(err)?;
        let status = std::process::Command::new("/usr/bin/tar")
            .arg("-xjf").arg(&archive).arg("-C").arg(&staging)
            .status().map_err(err)?;
        if !status.success() || !staging.join(KOKORO_DIR).join("model.onnx").is_file() {
            let _ = std::fs::remove_dir_all(&staging);
            return Err("The voice model archive could not be unpacked; download it again".into());
        }
        let _ = std::fs::remove_dir_all(&target);
        std::fs::rename(staging.join(KOKORO_DIR), &target).map_err(err)?;
        let _ = std::fs::remove_dir_all(&staging);
    }
    Ok(Some(target))
}

/// Which voice the page will hear: "neural", "system", or "native" (spoken by the OS itself).
#[tauri::command]
pub fn speech_engine(app: AppHandle) -> String {
    #[cfg(target_os = "macos")]
    {
        match models::spec("speech") {
            Some(spec) if models::is_downloaded(&app, spec) => "neural".into(),
            _ => "system".into(),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        "native".into()
    }
}

#[tauri::command]
pub async fn synthesize_speech(
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
    text: String,
    voice: String,
) -> CmdResult<tauri::ipc::Response> {
    let text = speakable(&text);
    if text.is_empty() || text.len() > 4000 {
        return Err("Speech text must be between 1 and 4,000 characters".into());
    }
    #[cfg(target_os = "macos")]
    {
        let Some(dir) = tauri::async_runtime::spawn_blocking({
            let app = app.clone();
            move || installed_dir(&app)
        }).await.map_err(err)?? else {
            return Ok(tauri::ipc::Response::new(system_voice(&text, &voice).await?));
        };
        let state = state.inner().clone();
        let coordinator = crate::memory::coordinator(&app)?;
        let bytes = tauri::async_runtime::spawn_blocking(move || {
            let mut engine = state.voice.lock().map_err(err)?;
            if engine.is_none() {
                // The voice is small and loads beside the text models under the same lease.
                let never = std::sync::atomic::AtomicBool::new(false);
                let admission = coordinator.admit(&never).map_err(err)?;
                crate::memory::resident(&state, &coordinator, &admission)?;
                *engine = Some(OracleVoice::load(&dir).map_err(err)?);
            }
            engine.as_ref().ok_or("The Oracle's voice is unavailable")?.render(&text, &voice).map_err(err)
        }).await.map_err(err)??;
        Ok(tauri::ipc::Response::new(bytes))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (app, state, voice);
        Err("This device speaks through its native voice".into())
    }
}

#[derive(serde::Serialize)]
pub struct VoiceChoice {
    id: &'static str,
    label: &'static str,
}
#[tauri::command]
pub fn oracle_voices() -> Vec<VoiceChoice> {
    ORACLE_VOICES.iter().map(|v| VoiceChoice { id: v.id, label: v.label }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speech_drops_citations_markup_and_verification_tags() {
        assert_eq!(
            speakable("The account was **Baal** [E1] [E2], \"named\" (not verbatim in the cited passages) — so it is said (and repeated)."),
            "The account was Baal, \"named\", so it is said (and repeated)."
        );
        assert_eq!(speakable("[E3]"), "");
    }
    #[test]
    fn wav_header_matches_sample_count() {
        let bytes = wav(&[0.0, 0.5, -0.5, 1.0], 24000);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(bytes.len(), 44 + 8);
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 8);
    }
}
