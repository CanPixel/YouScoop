//! Local inference engine: SmolVLM2 (frame descriptions), Qwen3 (chat) and
//! BGE-small (embeddings), all running through llama.cpp with Metal
//! acceleration.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Context, Result};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;

fn n_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8)
}

pub struct Engine {
    // Field order matters: mtmd holds a raw pointer into vlm, so it must drop first.
    mtmd: MtmdContext,
    vlm: LlamaModel,
    chat_model: LlamaModel,
    embed: LlamaModel,
    backend: LlamaBackend,
}

impl Engine {
    pub fn load(
        vlm_path: &str,
        mmproj_path: &str,
        chat_path: &str,
        embed_path: &str,
    ) -> Result<Self> {
        let backend = LlamaBackend::init().context("llama backend init failed")?;

        let gpu_params = LlamaModelParams::default().with_n_gpu_layers(1_000_000);
        let vlm = LlamaModel::load_from_file(&backend, vlm_path, &gpu_params)
            .context("failed to load SmolVLM2 model")?;

        let chat_model = LlamaModel::load_from_file(&backend, chat_path, &gpu_params)
            .context("failed to load chat model")?;

        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            print_timings: false,
            n_threads: n_threads(),
            media_marker: CString::new(mtmd_default_marker())?,
            image_min_tokens: -1,
            image_max_tokens: -1,
        };
        let mtmd = MtmdContext::init_from_file(mmproj_path, &vlm, &mtmd_params)
            .map_err(|e| anyhow!("failed to load vision projector: {e}"))?;

        // The embedding model is tiny; CPU-only keeps it off the GPU scheduler
        // entirely, which llama.cpp's context-reserve passes have historically
        // been touchy about for encoder-only (BERT) graphs.
        let cpu_params = LlamaModelParams::default().with_n_gpu_layers(0);
        let embed = LlamaModel::load_from_file(&backend, embed_path, &cpu_params)
            .context("failed to load embedding model")?;

        Ok(Self { mtmd, vlm, chat_model, embed, backend })
    }

    /// Describe a single video frame in 1-2 sentences.
    pub fn describe_frame(&self, image_path: &str, cancel: &AtomicBool) -> Result<String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096))
            .with_n_batch(4096)
            .with_n_ubatch(4096)
            .with_n_threads(n_threads())
            .with_n_threads_batch(n_threads());
        let mut ctx = self
            .vlm
            .new_context(&self.backend, ctx_params)
            .context("vlm context")?;

        let bitmap = MtmdBitmap::from_file(&self.mtmd, image_path, false)
            .map_err(|e| anyhow!("could not read frame image: {e}"))?;

        let prompt = format!(
            "<|im_start|>User: {}Describe this video frame in one or two short sentences. \
             Mention the subjects, what is happening, any visible text, and the setting. \
             Describe only what is actually visible; transcribe on-screen text exactly. \
             Do not guess names of people, places, or objects, and do not add facts that \
             are not shown in the image.<end_of_utterance>\nAssistant:",
            mtmd_default_marker()
        );
        let n_past = self.eval_prompt(&mut ctx, &prompt, &[&bitmap])?;

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.15, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);
        self.generate(&mut ctx, &self.vlm, n_past, 96, sampler, cancel, &mut |_| {})
    }

    /// Text-only chat completion on the dedicated chat model (ChatML prompt),
    /// streaming pieces through `on_token`.
    pub fn chat(
        &self,
        prompt: &str,
        max_tokens: usize,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(8192))
            .with_n_batch(2048)
            .with_n_ubatch(2048)
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
        let n_batch = ctx.n_batch() as usize;
        let mut n_past: i32 = 0;
        for window in tokens.chunks(n_batch) {
            let mut batch = LlamaBatch::new(window.len(), 1);
            for (i, &tok) in window.iter().enumerate() {
                let is_last = n_past as usize + i + 1 == tokens.len();
                batch.add(tok, n_past + i as i32, &[0], is_last)?;
            }
            ctx.decode(&mut batch).context("chat prompt decode failed")?;
            n_past += window.len() as i32;
        }

        // Greedy decoding: answers are grounded in retrieved excerpts, and
        // sampling temperature just adds paraphrase drift.
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.05, 0.0, 0.0),
            LlamaSampler::greedy(),
        ]);
        self.generate(&mut ctx, &self.chat_model, n_past, max_tokens, sampler, cancel, on_token)
    }

    /// Tokenize (text + optional media) and evaluate the full prompt. Returns n_past.
    fn eval_prompt(
        &self,
        ctx: &mut LlamaContext,
        prompt: &str,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<i32> {
        let input = MtmdInputText {
            text: prompt.to_string(),
            add_special: false,
            parse_special: true,
        };
        let chunks = self
            .mtmd
            .tokenize(input, bitmaps)
            .map_err(|e| anyhow!("prompt tokenization failed: {e}"))?;
        let n_batch = ctx.n_batch() as i32;
        let n_past = chunks
            .eval_chunks(&self.mtmd, ctx, 0, 0, n_batch, true)
            .map_err(|e| anyhow!("prompt evaluation failed: {e}"))?;
        Ok(n_past)
    }

    fn generate(
        &self,
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
