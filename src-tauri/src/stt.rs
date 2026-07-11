//! Local speech-to-text fallback via whisper.cpp (whisper-rs).

use anyhow::{anyhow, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::vtt::Cue;

pub struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    pub fn load(model_path: &str) -> Result<Self> {
        let mut params = WhisperContextParameters::default();
        params.use_gpu(true);
        let ctx = WhisperContext::new_with_params(model_path, params)
            .map_err(|e| anyhow!("failed to load whisper model: {e}"))?;
        Ok(Self { ctx })
    }

    /// Transcribe 16 kHz mono f32 samples into timestamped cues.
    /// `on_progress` receives 0..=1.
    pub fn transcribe(
        &self,
        samples: &[f32],
        mut on_progress: impl FnMut(f32) + Send + 'static,
    ) -> Result<Vec<Cue>> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| anyhow!("whisper state error: {e}"))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));
        params.set_translate(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_special(false);
        params.set_print_timestamps(false);
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8);
        params.set_n_threads(threads);
        params.set_progress_callback_safe(move |p: i32| {
            on_progress((p as f32 / 100.0).clamp(0.0, 1.0));
        });

        state
            .full(params, samples)
            .map_err(|e| anyhow!("whisper transcription failed: {e}"))?;

        let n = state.full_n_segments();
        let mut cues = Vec::with_capacity(n as usize);
        for i in 0..n {
            let Some(segment) = state.get_segment(i) else { continue };
            let text = segment.to_str_lossy().map(|s| s.trim().to_string()).unwrap_or_default();
            if !text.is_empty() {
                // Timestamps are in centiseconds (10ms units).
                cues.push(Cue {
                    start: segment.start_timestamp() as f64 / 100.0,
                    end: segment.end_timestamp() as f64 / 100.0,
                    text,
                });
            }
        }
        Ok(cues)
    }
}
