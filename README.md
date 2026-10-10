<p align="center">
  <img src="branding/youscoop-icon-1024.png" width="128" alt="YouScoop icon" />
</p>

<h1 align="center">YouScoop</h1>
<p align="center"><em>Talk to any YouTube video in full privacy.</em></p>

---

YouScoop is a macOS desktop app that downloads a YouTube video, reads its transcript, *watches* its frames with a vision-language model, and builds a local semantic index. Then you can talk to the video! Answers cite clickable timestamps that seek the player to the exact moment.

Everything runs on-device (Apple Silicon / Metal). No accounts, no API keys, no cloud calls after the one-time model download.

## Features

- **Paste a URL, get a conversation** - ingestion is a staged pipeline with live progress: download → transcript → keyframes → vision pass → semantic index.
- **Captions first, Whisper fallback** - YouTube captions are used when available; caption-less videos are transcribed locally (the Whisper model auto-downloads on first need).
- **Vision pass** - SmolVLM2 describes ~48 keyframes per video, so you can ask about things that are only *shown*, including on-screen text.
- **Grounded chat** - hybrid retrieval (embeddings + lexical phrase matching) feeds excerpt lines with sentence-level timestamps to the chat model, which must cite `[m:ss]` per claim. Timestamps render as chips that seek the player.
- **Timeline of indexed moments** - speech (blue) and visual (purple) markers under the player; click to jump. Hover for a preview.
- **Local library** - videos, thumbnails, and indexes persist in the app data dir; chats persist per video.
- One-click copy of responses, Esc to stop generation, "Watch on YouTube" link, resume-capable model downloads.

## Models

Downloaded once from Hugging Face into `~/Library/Application Support/com.canpixel.youscoop/models/` (~4.3 GB total):

| Role | Model | Size |
|---|---|---|
| Frame descriptions | SmolVLM2-2.2B-Instruct Q4_K_M (+ mmproj Q8_0) | ~1.6 GB |
| Chat | Qwen3-4B-Instruct-2507 Q4_K_M | ~2.5 GB |
| Embeddings | bge-small-en-v1.5 Q8_0 (CPU) | ~35 MB |
| Transcription fallback | Whisper large-v3-turbo Q5_0 (on demand) | ~570 MB |

The original small-model pipeline uses `llama.cpp` (via `llama-cpp-2`, Metal + mtmd) and `whisper.cpp` (via a vendored `whisper-rs` fork - see below). Optional Gemma chat uses MLX.

## Requirements

- macOS on Apple Silicon (developed on M-series; 16 GB+ RAM recommended)
- [Bun](https://bun.sh) and a Rust toolchain
- `ffmpeg` on your system: `brew install ffmpeg`
- `yt-dlp` ships inside the app as a Tauri sidecar (`src-tauri/binaries/`)

## Development

```bash
bun install
bun tauri dev              # run with hot reload
bun tauri build            # release bundle
bun tauri build --debug --bundles app   # fast debug .app
```

### Debug harnesses

Replay the RAG pipeline against an already-ingested video without the UI:

```bash
cd src-tauri
cargo run --example rag_dump  -- <video_id> "your question"   # retrieval ranking only
cargo run --example chat_test -- <video_id> "your question"   # full prompt + generation
cargo run --example ggml_smoke                                # load all models, embed, transcribe
```

`<video_id>` is a directory name under `~/Library/Application Support/com.canpixel.youscoop/videos/`.

## Architecture notes

- **Pipeline** ([src-tauri/src/pipeline.rs](src-tauri/src/pipeline.rs)): metadata → 720p download → captions (VTT with rolling-caption dedupe) or Whisper → ffmpeg keyframes (interval `clamp(duration/48, 4s, 30s)`) → SmolVLM2 descriptions → BGE embeddings, stored as JSON per video. Videos are capped at **1:30:00**.
- **Retrieval** ([src-tauri/src/rag.rs](src-tauri/src/rag.rs)): brute-force cosine over chunk vectors plus a lexical boost for literal query-term and phrase matches. Speech chunks carry per-cue timestamps so prompt excerpts are ~12-second lines and citations land on the actual sentence.
- **Engine** ([src-tauri/src/llm.rs](src-tauri/src/llm.rs)): one mutex-serialized engine owns all llama.cpp models; GPU work runs on blocking threads. The embedding model runs CPU-only deliberately.
- **Vendored whisper-rs** (`src-tauri/vendor/`): `whisper-rs-sys` and `llama-cpp-sys-2` each statically link their own ggml, and the linker keeps only one copy - the two ggml vintages must match or op enums skew and the process aborts. The fork pins whisper.cpp to a ggml-compatible commit; see `src-tauri/vendor/whisper-rs-sys/README-fork.md` before bumping either crate.

## Privacy

Video files, frames, transcripts, indexes, and chats never leave your machine. Network access is limited to: Hugging Face (one-time model downloads), and YouTube via yt-dlp (video/caption fetch).
