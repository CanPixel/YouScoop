# Run YouScoop

YouScoop is the desktop video workspace. It can send a transcript or a chat answer as a Markdown note into the vault of WiZMUTH, a separate notes app that lives in its own project.

## Desktop development

Install Bun, Rust, and ffmpeg. From the repository root:

```sh
bun install
bun run tauri dev
```

YouScoop's development server uses port 1420. This is a native Tauri window with development assets, not a browser-only demonstration. Close the window to stop the app; stop the development terminal with Ctrl+C.

## YouScoop desktop models

The initial setup downloads the small chat, embedding and frame-description models. For the optional Gemma investigative chat, expand **Desktop model** inside YouScoop. The tested large-model runtime requires Apple Silicon and macOS 26 or later, plus uv available on PATH, in `~/.local/bin`, or in a standard Homebrew directory. The setup panel links to uv's installation documentation when it is missing.

Choose **Install desktop runtime**, then install Gemma. An existing local snapshot can be reused. Otherwise model installation downloads approximately 15.4 GB, plus Python dependencies. Every installed model file is verified against the bundled checksums before its path becomes active. Setup can be cancelled and retried. Runtime repair reinstalls the pinned dependencies and waits for active desktop model workers to finish.

Python, its base interpreter, and model snapshots persist under `~/Library/Application Support/com.canpixel.youscoop/desktop-runtime/`. They do not depend on temporary experiment environments. Inference is offline after installation. Gemma runs in its own worker process and releases its memory when that process exits.

## First usable workflow

1. In YouScoop, select an existing processed video or add a YouTube URL. The maximum supported source duration is two hours; full-length processing speed is still being measured.
2. Click a timeline chip to navigate the scrollable transcript, on-screen text and frame descriptions. Frame descriptions come from a small vision model and can misread or invent details; on-screen wording comes only from OCR.
3. Choose Gemma in the investigative chat model selector when installed. It selects relevant source passages for long questions and resolves only supplied evidence IDs to timestamps.
3a. YouScoop also separates voices and reads on-screen text. New videos get both at the end of processing; for a video processed earlier, choose **Update analysis** (it stays usable meanwhile, and can be cancelled). The update also re-describes keyframes so that on-screen wording comes only from OCR. It labels on-screen passages by kind (email, memo, Wikipedia page, news article), marks keyframes that are screens of text, and checks caption words that sound like on-screen names by re-listening with Whisper; confirmed spellings are kept with the video's analysis. The two small voice models (about 32 MB) download on first use. The transcript then names the voice where speech changes hands ("Speech · Voice B") and shows "On-screen text · OCR" rows. Voice labels are automatic clusters ordered by speaking time, not identities; OCR can misread characters, and each on-screen text passage keeps the frame it was read from.
4. Choose **Send to WiZMUTH** to write the video's transcript (and its on-screen text) as a Markdown note into `WiZMUTH/Inbox` in WiZMUTH's vault, each paragraph linked to its moment on YouTube. Under any chat answer, **→ WiZMUTH** sends just that answer with its question. **Show note** reveals the file.
5. Open WiZMUTH. It picks the note up on its next read of the vault (it re-reads when its window gains focus).

The vault is whichever folder you chose in WiZMUTH, or `~/Documents/WiZMUTH` by default. Notes are ordinary files: edit, move or delete them in Finder or Obsidian.

## Demonstrated limits

Gemma generated answers through the native worker using the installed runtime on the M5 Pro with 24 GB RAM.
