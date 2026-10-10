# YouScoop verification

What has been demonstrated on the Mac (M5 Pro, 24 GB), with open quality gates stated separately. The test sources are a 32:45 video about the Epstein files and a 29:19 "ten bands" list video. See [model decision](MODEL-DECISION.md) and [running](RUNNING.md).

| Goal | Demonstrated | Still open |
|---|---|---|
| Transcript and visuals | Scrollable timestamp-bound transcript under the video, chip and row seeking, original speech plus frame descriptions; two-hour cap | A complete two-hour performance run |
| Voice separation | Offline pyannote segmentation 3.0 and WeSpeaker ResNet34-LM through sherpa-onnx, in a cancellable child process. On the 32:45 source it took 155 seconds on 12 CPU threads and found the presenter (30:59) and a second voice (1:07) covering a quoted clip. Speech passages split at voice turns and carry labels; a caption ">>" speaker change takes the voice turn starting within a second of it. The UI names voices only when at least two hold a tenth of the speaking time or three minutes; a single narrator split into small clusters shows no voice labels | Labels are clusters, not identities. Some boundaries are a few seconds off. Two-hour speed is projected at about 8.5 minutes, not measured |
| On-screen text | Apple Vision OCR of full-resolution frames every 2 seconds, through an embedded Swift helper. On the 32:45 source: 983 frames read in about 110 seconds, merged into passages with recurring watermarks and banners removed and burned-in subtitles joined. Each passage links to its full-resolution frame | Stylized or blurred text can come out garbled. Text shown for under two seconds can be missed |
| Desktop Gemma (deep chat) | Optional MLX Q4 Gemma 4 26B-A4B chat. The complete transcript and all on-screen text go in first (up to about 44,000 characters, roughly 11k tokens of the worker's 24k), frame descriptions are sampled into a capped share of the rest, and only supplied evidence IDs resolve to timestamps. Asked "which 10 bands are in the video", it listed all ten in order with citations (Down, Possessed, Fear Factory, Entombed, Rage Against The Machine, Slipknot, Coal Chamber, Linkin Park, Drowning Pool, Mudvayne), about 29 seconds including model load | Sources whose transcript exceeds the budget (roughly 45 minutes and up) fall back to query matches plus chronological samples. Citation presence does not prove each sentence |
| Send to WiZMUTH | A video's transcript (paragraphs of about a minute, each linked to its YouTube moment) and on-screen text, or a single chat answer with its question, are written as Markdown notes into WiZMUTH's vault inbox | WiZMUTH is a separate project and reads the note on its next vault refresh |
| Install and memory | Persistent checksummed desktop Python and Gemma snapshots; idle small models release their weights before the heavy worker loads (see [memory](MEMORY.md)) | Fresh-machine setup and thermal endurance untested |

## Retrieval and analysis (2026-10-02)

| Change | Measured | Limit |
|---|---|---|
| Follow-up questions | A short follow-up or one that refers back ("who sent it?") is rewritten by the chat model into a standalone query | |
| Frame descriptions | The frame prompt no longer asks for on-screen text, and descriptions that still report text where OCR read the screen are left out of the index (16 of 60 on the test source, including an invented email sender). Existing videos re-describe their keyframes through Update analysis | The model still describes some scenes wrongly |
| Screen kinds and listing questions | On-screen passages are labeled by layout (Email, Memo, Wikipedia page, News article, Search results, Government document, Creator's note). On the test source all 14 email screens were labeled correctly. A question asking for every item of a kind ("Which emails are shown?") gets every passage of that kind in time order, in both the quick chat and Gemma | Listing works for these kinds only. The 4B quick-chat model skipped one of the 14 supplied emails |
| Screen keyframes | Each described keyframe is also read by OCR at full resolution. Frames with at least 200 letters of non-banner text get a factual label such as "The frame shows an email" instead of the model's guess: 6 of 60 on the test source | The remaining descriptions still come from the frame model |
| Spelling suggestions | Caption words that sound like a name shown on screen nearby are re-transcribed by Whisper, without and with the on-screen spelling as a hint (Moolok to Moloch, Bal and Ball to Baal, goyam to goyim on the test source; 19 seconds). Suggestions keep the captions' own letter case and are stored with the video's analysis | A hint can bias Whisper, which is why hint-only suggestions are labeled with what was heard |
| Frame model | Qwen2.5-VL 3B (Apache-2.0) replaces SmolVLM2 for frame descriptions when installed; SmolVLM2 remains the fallback. On 20 hand-checked frames it invented about 3 details against 11 for SmolVLM2 and 9 for Gemma 3 4B. Its image encoder runs on the CPU because the Metal path produced invalid output in this llama.cpp build; 2.3 seconds per frame, about 3 minutes per video | It still mislabels some clothing and poses |

Results: `youscoop-enrichment.json` and `youscoop-enrichment-spellings.json` (native enrichment runs), `frame-models.json` (three frame models on 20 frames, with reference facts and scoring notes), all under `experiments/model-trial/results/`.

### First user test on the 32:45 source (2026-10-02)

- Qwen2.5-VL produced no refusals in 60 frames, but 16 descriptions stopped mid-sentence at the 96-token limit. Frame descriptions now get 160 tokens and are cut back to the last whole sentence.
- Keyframes were thinned from 558 scene-change candidates by list position, so busy stretches kept frames one second apart. Scene changes within 3 seconds of a kept frame are now skipped, and thinning keeps frames evenly spaced: on this source the 60 kept frames are 24.7 to 71.8 seconds apart.
- Partial and full readings of the same page now merge when 60% of the smaller reading's words occur in the other one (at least 8 words); two different neighbouring emails sharing 52% stay separate.
- Chat and Gemma prompts report disturbing or offensive source material accurately without refusing, softening or moralizing; reporting is not endorsing.
- Omni was removed from YouScoop (inspection panel, sweep, worker task, model catalog entry and the 11 GB model files). Its earlier trial outputs remain under `experiments/model-trial/results/`. The pinned Python runtime is unchanged because Gemma's MLX stack also needs it.
- "Update analysis" applies these to saved videos as `onscreen:3` (regrouping only, no new OCR) and `scenes:4` (new keyframes and descriptions).

## Repeatable checks

```sh
cd src-tauri && cargo test --lib          # 52 tests: chunking, OCR grouping, voices, citations, deep-chat context, notes
python3 -m unittest discover runtime      # Python runtime and worker lifecycle
bun run lint && bun run build
```

`src-tauri/examples/enrich_video.rs` runs voice separation and on-screen text on a library video without a window (`--features diagnostics`). `deep_chat_request.rs` prints the exact Gemma request for a question so it can be replayed through `runtime/worker.py`.

No model output or source allegation is labeled independently verified by these checks. Citations identify where text was found; they do not authenticate a document or establish that its speaker is correct.
