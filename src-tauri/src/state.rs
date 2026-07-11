//! Shared application state.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use crate::llm::Engine;
use crate::stt::WhisperEngine;

pub struct AppState {
    /// The inference engine (SmolVLM2 + embeddings). One lock serializes GPU work.
    pub engine: Arc<Mutex<Option<Engine>>>,
    /// Whisper is loaded lazily the first time a caption-less video shows up.
    pub whisper: Arc<Mutex<Option<WhisperEngine>>>,
    /// Per-video ingestion cancel flags.
    pub cancels: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// Cancels the in-flight chat generation.
    pub chat_cancel: Arc<AtomicBool>,
    /// Guards against double engine loads.
    pub engine_loading: AtomicBool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            engine: Arc::new(Mutex::new(None)),
            whisper: Arc::new(Mutex::new(None)),
            cancels: Mutex::new(HashMap::new()),
            chat_cancel: Arc::new(AtomicBool::new(false)),
            engine_loading: AtomicBool::new(false),
        }
    }

    pub fn engine_loaded(&self) -> bool {
        self.engine.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    pub fn require_engine(&self) -> Result<Arc<Mutex<Option<Engine>>>> {
        if !self.engine_loaded() {
            return Err(anyhow!("The AI engine is not loaded yet: finish setup first."));
        }
        Ok(self.engine.clone())
    }
}
