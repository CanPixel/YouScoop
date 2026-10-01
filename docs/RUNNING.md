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

The initial setup downloads the small chat, embedding and frame-description models. For Omni and Gemma, expand **Desktop models** inside YouScoop. The tested large-model runtime requires Apple Silicon and macOS 26 or later, plus uv available on PATH, in `~/.local/bin`, or in a standard Homebrew directory. The setup panel links to uv's installation documentation when it is missing.

Choose **Install desktop runtime**, then install Omni or Gemma. Existing local snapshots can be reused. Otherwise model installation downloads approximately 12 GB for Omni and 15.4 GB for Gemma, plus Python dependencies. Every installed model file is verified against the bundled checksums before its path becomes active. Setup can be cancelled and retried. Runtime repair reinstalls the pinned dependencies and waits for active desktop model workers to finish.

Python, its base interpreter, and model snapshots persist under `~/Library/Application Support/com.canpixel.youscoop/desktop-runtime/`. They do not depend on temporary experiment environments. Inference is offline after installation. Omni loads its audiovisual Thinker directly; speech generation is excluded. Large models run one at a time and release their memory when their worker process exits.

## First usable workflow

1. In YouScoop, select an existing processed video or add a YouTube URL. The maximum supported source duration is two hours; full-length processing speed is still being measured.
2. Click a timeline chip to navigate the scrollable transcript and raw frame descriptions. Seek to a moment and choose **Inspect playback position** to examine its next ten seconds with Omni. The default **Visible scene** question avoids guessing document text; choose **Speech and visuals** or edit the question for deeper joint analysis. Raw outputs can misread text, invent details, or omit relevant on-screen material. Expand **Compare with the speech transcript and inspection question** beneath a raw output and replay the source before using it. Omni stops after its first output line, before it can continue a simulated conversation; multi-line visual text may be cut short.
3. Choose **Experimental 12-moment sweep / resume** for a capped sweep of up to twelve ten-second clips. Cancelling retains completed outputs. Raw Omni text stays visible in the transcript and preserved in the package, but cannot enter search or the archive's evidence by itself. Replay a moment and choose **Write reviewed observation** to write a short description you checked. That observation then becomes searchable and exportable, labeled as yours. You can remove or rewrite it later. Choose Gemma in the investigative chat model selector when installed. It selects relevant source passages for long questions and resolves only supplied evidence IDs to timestamps.
4. Choose **Save to SACRiNOMiCON**. This preserves the source into the independent desktop archive's Inbox and writes a portable `.sacri` package. **Show portable package** reveals it for transfer to another device. If you inspect or edit a source after saving, choose **Update SACRiNOMiCON** to make the newer version current. Older archive revisions remain preserved; removing a reviewed observation from YouScoop does not erase past revisions.
5. Open SACRiNOMiCON independently. Browse Library, inspect original evidence, add assessments or corrected readings, and explore concept connections. Install its separate small local models through its setup screen to use Oracle. It does not use Omni or Gemma.

The desktop archive lives under `~/Library/Application Support/com.canpixel.sacrinomicon/library/`. Phone imports use the portable package, so they do not rely on desktop file paths. The archive also supports complete library export and restore. Conversation history is currently device-local and is not included in library backups.

## Demonstrated limits

Both large models generated answers through the native worker using the installed runtime on the M5 Pro with 24 GB RAM. The new generation stop removed the observed repetitive continuation; it cannot establish that the first answer is true. An earlier Omni run misread text, and a broad archive Oracle answer still made an overbroad interpretation. Android ARM64 release packaging and emulator import/search tests pass. Real microphone flows and Fold performance remain unverified. See [verification](VERIFICATION.md) and [model decision](MODEL-DECISION.md).
