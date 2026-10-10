//! Portable text inference: Metal on macOS, bounded CPU inference on Android.
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// Chat context in tokens. The Mac has the memory and Metal throughput for a
/// wider evidence window; phones keep the smaller, measured budget.
pub const CHAT_CONTEXT: u32 = if cfg!(target_os = "macos") { 8192 } else { 4096 };
fn n_threads()->i32 { std::thread::available_parallelism().map(|n|n.get() as i32).unwrap_or(4).min(6) }
/// How the next token is chosen. Greedy keeps evidence answers literal;
/// a low, seeded temperature lets associative answers vary their phrasing
/// while the same question still yields the same reply.
#[derive(Clone, Copy, Debug)]
pub enum Sampling {
    Greedy,
    Warm { temperature: f32, seed: u32 },
}
pub struct TextEngine {
    chat_model: LlamaModel,
    embed: LlamaModel,
    backend: LlamaBackend,
}
impl TextEngine {
    pub fn load(chat_path: &str, embed_path: &str) -> Result<Self> {
        let backend = LlamaBackend::init().context("Inference backend init failed")?;
        let gpu_layers = if cfg!(target_os = "macos") { 1_000_000 } else { 0 };
        let chat_model = LlamaModel::load_from_file(&backend, chat_path, &LlamaModelParams::default().with_n_gpu_layers(gpu_layers)).context("Chat model could not load")?;
        let embed = LlamaModel::load_from_file(&backend, embed_path, &LlamaModelParams::default().with_n_gpu_layers(0)).context("Embedding model could not load")?;
        Ok(Self {chat_model,embed,backend})
    }
    pub fn chat(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.chat_with(prompt, max_tokens, Sampling::Greedy, cancel, on_token)
    }
    pub fn chat_with(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: Sampling,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CHAT_CONTEXT))
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_n_threads(n_threads())
            .with_n_threads_batch(n_threads());
        let mut ctx = self
            .chat_model
            .new_context(&self.backend, ctx_params)
            .context("chat context")?;

        let tokens = self
            .chat_model
            .str_to_token(prompt, AddBos::Never)
            .map_err(|e| anyhow!("chat tokenize: {e}"))?;
        if tokens.len() + max_tokens >= CHAT_CONTEXT as usize { return Err(anyhow!("Context is too large. Start a new conversation or narrow the source scope.")); }
        let n_batch = ctx.n_batch() as usize;
        let mut n_past: i32 = 0;
        for window in tokens.chunks(n_batch) {
            if cancel.load(Ordering::Relaxed) { return Err(anyhow!("Cancelled")); }
            let mut batch = LlamaBatch::new(window.len(), 1);
            for (i, &tok) in window.iter().enumerate() {
                let is_last = n_past as usize + i + 1 == tokens.len();
                batch.add(tok, n_past + i as i32, &[0], is_last)?;
            }
            ctx.decode(&mut batch).context("chat prompt decode failed")?;
            n_past += window.len() as i32;
        }

        // Greedy by default: evidence answers are grounded in retrieved
        // excerpts, and sampling temperature just adds paraphrase drift.
        let sampler = match sampling {
            Sampling::Greedy => LlamaSampler::chain_simple([
                LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
                LlamaSampler::greedy(),
            ]),
            // Qwen3's recommended top-k/top-p, with a cooler temperature for grounding.
            Sampling::Warm { temperature, seed } => LlamaSampler::chain_simple([
                LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
                LlamaSampler::top_k(20),
                LlamaSampler::top_p(0.8, 1),
                LlamaSampler::temp(temperature),
                LlamaSampler::dist(seed),
            ]),
        };
        generate(&mut ctx, &self.chat_model, n_past, max_tokens, sampler, cancel, on_token)
    }

    /// Embed texts with BGE-small (CLS pooling, L2-normalized).
    pub fn embed_texts(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(512))
            .with_n_batch(512)
            .with_n_ubatch(512)
            .with_n_threads(n_threads())
            .with_n_threads_batch(n_threads())
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Cls)
            // Skip the auto flash-attention resolution pass; it reserves extra
            // graphs that encoder-only models can't build.
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_DISABLED);
        let mut ctx = self
            .embed
            .new_context(&self.backend, ctx_params)
            .context("embed context")?;

        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let text = if is_query {
                format!("Represent this sentence for searching relevant passages: {text}")
            } else {
                text.clone()
            };
            let mut tokens = self
                .embed
                .str_to_token(&text, AddBos::Always)
                .map_err(|e| anyhow!("embed tokenize: {e}"))?;
            tokens.truncate(506);
            let mut batch = LlamaBatch::new(512, 1);
            batch.add_sequence(&tokens, 0, true)?;
            ctx.clear_kv_cache();
            // BERT-style models are encoder-only.
            if ctx.encode(&mut batch).is_err() {
                ctx.decode(&mut batch).context("embed decode failed")?;
            }
            let emb = ctx
                .embeddings_seq_ith(0)
                .map_err(|e| anyhow!("embeddings: {e}"))?;
            out.push(l2_normalize(emb));
        }
        Ok(out)
    }
}
fn generate(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    mut n_past: i32,
    max_tokens: usize,
    mut sampler: LlamaSampler,
    cancel: &AtomicBool,
    on_token: &mut dyn FnMut(&str),
) -> Result<String> {
    let mut batch = LlamaBatch::new(8, 1);
    let mut out = String::new();
    let mut pending: Vec<u8> = Vec::new();

    for _ in 0..max_tokens {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let token = sampler.sample(ctx, -1);
        sampler.accept(token);
        if model.is_eog_token(token) {
            break;
        }
        let bytes = model
            .token_to_piece_bytes(token, 64, false, None)
            .unwrap_or_default();
        pending.extend_from_slice(&bytes);
        flush_utf8(&mut pending, &mut out, on_token);

        batch.clear();
        batch.add(token, n_past, &[0], true)?;
        ctx.decode(&mut batch).context("decode failed")?;
        n_past += 1;
    }
    if !pending.is_empty() {
        let rest = String::from_utf8_lossy(&pending).into_owned();
        on_token(&rest);
        out.push_str(&rest);
    }
    Ok(out.trim().to_string())
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // Vectors are already L2-normalized, so cosine == dot product.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Emit the maximal valid UTF-8 prefix of `pending` and keep the remainder buffered.
fn flush_utf8(pending: &mut Vec<u8>, out: &mut String, on_token: &mut dyn FnMut(&str)) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                if !s.is_empty() {
                    on_token(s);
                    out.push_str(s);
                }
                pending.clear();
                return;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    let s = std::str::from_utf8(&pending[..valid]).unwrap().to_string();
                    on_token(&s);
                    out.push_str(&s);
                    pending.drain(..valid);
                }
                match e.error_len() {
                    // Truly invalid byte: drop it.
                    Some(len) => {
                        pending.drain(..len);
                    }
                    // Incomplete sequence: wait for the next token.
                    None => return,
                }
            }
        }
    }
}
