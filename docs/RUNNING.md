# Run the two applications

YouScoop is the desktop video workspace. SACRiNOMiCON is a separate desktop and Android archive with Oracle, Explore, and Library screens. Only YouScoop plays and processes videos. The archive preserves source artifacts and their provenance without requiring the ingestion application to run.

## Desktop development

Install Bun, Rust, and ffmpeg. From the repository root:

```sh
bun install
bun run tauri dev
```

In a separate terminal, start the independent archive:

```sh
cd apps/sacrinomicon
bun install
bun run tauri dev
```

YouScoop uses port 1420; SACRiNOMiCON uses 1430. These are native Tauri windows with development assets, not browser-only demonstrations. Close each window to stop its application; stop its development terminal with Ctrl+C.

## YouScoop desktop models

The initial setup downloads the small chat, embedding and frame-description models. For the optional Gemma investigative chat, expand **Desktop model** inside YouScoop. The tested large-model runtime requires Apple Silicon and macOS 26 or later, plus uv available on PATH, in `~/.local/bin`, or in a standard Homebrew directory. The setup panel links to uv's installation documentation when it is missing.

Choose **Install desktop runtime**, then install Gemma. An existing local snapshot can be reused. Otherwise model installation downloads approximately 15.4 GB, plus Python dependencies. Every installed model file is verified against the bundled checksums before its path becomes active. Setup can be cancelled and retried. Runtime repair reinstalls the pinned dependencies and waits for active desktop model workers to finish.

Python, its base interpreter, and model snapshots persist under `~/Library/Application Support/com.canpixel.youscoop/desktop-runtime/`. They do not depend on temporary experiment environments. Inference is offline after installation. Gemma runs in its own worker process and releases its memory when that process exits.

## First usable workflow

1. In YouScoop, select an existing processed video or add a YouTube URL. The maximum supported source duration is two hours; full-length processing speed is still being measured.
2. Click a timeline chip to navigate the scrollable transcript, on-screen text and frame descriptions. Frame descriptions come from a small vision model and can misread or invent details; on-screen wording comes only from OCR.
3. Choose Gemma in the investigative chat model selector when installed. It selects relevant source passages for long questions and resolves only supplied evidence IDs to timestamps.
3a. YouScoop also separates voices and reads on-screen text. New videos get both at the end of processing; for a video processed earlier, choose **Update analysis** (it stays usable meanwhile, and can be cancelled). The update also re-describes keyframes so that on-screen wording comes only from OCR. It labels on-screen passages by kind (email, memo, Wikipedia page, news article), marks keyframes that are screens of text, and checks caption words that sound like on-screen names by re-listening with Whisper; confirmed spellings appear as suggestions in SACRiNOMiCON under a passage's spelling corrections. The two small voice models (about 32 MB) download on first use. The transcript then names the voice where speech changes hands ("Speech · Voice B") and shows "On-screen text · OCR" rows. Voice labels are automatic clusters ordered by speaking time, not identities; OCR can misread characters, and each on-screen text passage keeps the frame it was read from, which SACRiNOMiCON shows next to the text.
Saving a reprocessed source to SACRiNOMiCON keeps your notes, judgements, concept links and word corrections on the matching new passages, marked as carried forward; the earlier revision stays preserved.
4. Choose **Save to SACRiNOMiCON**. This preserves the source into the independent desktop archive's Inbox and writes a portable `.sacri` package. **Show portable package** reveals it for transfer to another device. If you update a source's analysis after saving, save it again to make the newer version current. Older archive revisions remain preserved.
5. Open SACRiNOMiCON independently. Browse Library, inspect original evidence, add assessments or corrected readings, and explore concept connections. Install its separate small local models through its setup screen to use Oracle. It does not use Gemma.

The desktop archive lives under `~/Library/Application Support/com.canpixel.sacrinomicon/library/`. Phone imports use the portable package, so they do not rely on desktop file paths. The archive also supports complete library export and restore. Conversation history is currently device-local and is not included in library backups.

## Demonstrated limits

Gemma generated answers through the native worker using the installed runtime on the M5 Pro with 24 GB RAM. The archive Oracle answers in a few seconds on the Mac and starts reading aloud at its first finished sentence, but a small model can still frame or attribute passages imperfectly; follow its citations. Android ARM64 release packaging and emulator import/search tests pass. Real microphone flows and Fold performance remain unverified. See [verification](VERIFICATION.md) and [model decision](MODEL-DECISION.md).
