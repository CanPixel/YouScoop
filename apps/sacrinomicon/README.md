# SACRiNOMiCON

SACRiNOMiCON is the independent local archive for sources saved from YouScoop. It has its own native desktop app, with **Oracle**, **Explore**, and **Library / Inbox** screens. YouScoop handles video playback and processing; SACRiNOMiCON opens and searches the preserved archive without YouScoop running.

## Run it on this Mac

From the repository root:

```sh
cd apps/sacrinomicon
bun install                 # only needed if dependencies are missing
bun run tauri dev
```

This opens a native Tauri window. The development frontend uses port 1430 internally. Leave the terminal running while you use the app, then press Ctrl+C to stop it. You can close YouScoop before starting SACRiNOMiCON.

For a standalone `.app` that does not need the development server:

```sh
cd apps/sacrinomicon
bun run tauri build --debug --bundles app
open src-tauri/target/debug/bundle/macos/SACRiNOMiCON.app
```

The debug bundle is useful for local testing. It is not a signed distribution build. A release bundle can be made with `bun run tauri build --bundles app`.

## Find what YouScoop saved

Open **Library / Inbox** in the left sidebar and choose a source. A source saved on this Mac with YouScoop's **Save to SACRiNOMiCON** button appears automatically because both apps use the same local archive. You do not need to import its `.sacri` package on the same Mac. If SACRiNOMiCON was already open when you saved, click **Refresh** in its top bar.

The Mac library is stored at:

```text
~/Library/Application Support/com.canpixel.sacrinomicon/library/
```

From a source record, open a passage to inspect its text, provenance, and preserved artifacts. Use **Converse with this source** to ask the Oracle about that source. **Explore** shows concepts and their linked passages. Source assertions and model descriptions are preserved with provenance; inclusion in the archive does not verify them.

The Oracle needs its separate local **Qwen3 4B** chat model and **BGE-small** embedding model. Its setup buttons appear in the Oracle screen if either model is missing. The Library remains usable while they download.

## Conversing with the Oracle

- **Lens.** *Archive* answers only from preserved passages. *Esoteric* also lets the Oracle draw on its own knowledge of symbolism, mythology, occult history and conspiracy lore to trace patterns and correspondences. Archive material always carries a numbered citation; the Oracle's own lore never does and is introduced as tradition or association.
- **Citations.** Hover a numbered mark to preview its passage; click it to open the evidence. *Threads to follow* under a reply are concepts you have anchored to the passages it used.
- **Follow-ups.** A short or referring question ("why would he say that?") is first rewritten into a self-contained one, so the archive search and the answer both know what "he" means.
- **Voice.** The microphone button opens a half-duplex voice conversation: speak, pause, and the Oracle answers aloud in shorter, speakable replies. Press Space to interrupt and Esc to end. *Speak replies* reads written answers aloud too. Transcription uses the optional Whisper download. On macOS the Oracle speaks with an offline neural voice (Kokoro, about 320 MB, chosen in the ⚙ menu; Lewis is the lowest). Until that is installed, the system voice speaks. Both pass through the app's own warm, slightly lowered voice chain. Android uses its system voice.

## Move the archive to another device

In YouScoop, **Show portable package** reveals the saved `.sacri` file. On another Mac or Android device, open SACRiNOMiCON's **Library / Inbox** and choose **Import source or library**. For a complete backup of the archive, including annotations and revision history, use **Export library** and import the resulting `.sacri-library` file on the other device.

If a saved source changes in YouScoop, choose **Update SACRiNOMiCON** there. The new revision becomes current; older revisions remain preserved. Click **Refresh** in SACRiNOMiCON to see the update if it was already open.

For the wider two-app workflow, see [Running the two applications](../../docs/RUNNING.md).
