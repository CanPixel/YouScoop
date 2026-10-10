# YouScoop and SACRiNOMiCON implementation contract

## Objective

Test, experiment, and verify the stack, then finish the desktop ingestion app and the separate desktop/mobile archive. This contract retains the full scope of the user's request.

## Product boundary

YouScoop is a desktop investigation workspace: capture up to two hours of video, preserve original media and exact captions, inspect synchronized audiovisual segments, investigate arguments, and save to the archive without manual transcription or re-indexing.

SACRiNOMiCON is a local conversational archive on desktop and Android. Separate Oracle and Explore screens. Oracle is minimalist, dark, mythical, and restrained; text, read-aloud, and voice conversation are distinct modes. Explore browses sources, claims, concepts, and explicitly typed relationships. Preservation never implies endorsement. Speculation stays available with provenance and uncertainty.

## Selected stack and remaining measurements

- Bun, Vite, React, TypeScript; separate Tauri applications for investigation and archive.
- Omni 3B native Transformers/PyTorch MPS on the Mac, speech output disabled, bounded segments, original-resolution media with bounded visual tokens. Tested audio/video together. Quantized 7B llama.cpp audio was poor and is not selected for ingestion.
- Gemma 4 26B A4B Q4 with MLX-LM as optional desktop investigative chat. Thinking off by default. Exact evidence IDs rather than generated chapter timestamps. Model unloaded before other large inference.
- Small portable Qwen3 4B Q4 text model and BGE-small embeddings for archive retrieval. Mac Metal and Android CPU initially. No archive dependency on the desktop or remote providers.
- Shared versioned source packages and a transactional local SQLite archive with lexical and vector retrieval. Models and embeddings are replaceable; originals, annotations, and provenance persist.
- Local STT, endpoint detection, cancellation, and native offline TTS for half-duplex voice. Voice components load on entering voice mode and unload on leaving. Sentence streaming speech and Qwen speech remain optional enhancements requiring their own verification, not phone dependencies.

Native Omni/Gemma trials are documented in experiments/model-trial/REPORT.md. (2026-10-02: Omni was later removed after it added little accuracy over the transcript, OCR and frame descriptions; see MODEL-DECISION.md.) Complete two-hour ingestion throughput and Fold thermals/latency remain measured gates, not promises. Do not remove audiovisual understanding to make a benchmark pass. Use selective bounded inspection and make processing coverage visible.

## Required deliverables and completion gates

1. Two-hour cap, accurate transcript navigation, preserved original media, inspectable Omni outputs, checkpointed/cancellable ingestion, and model/runtime setup.
2. Desktop Gemma investigative chat with streaming, cancellation, evidence references, and enforced memory separation.
3. A versioned portable package, checksummed artifacts, stable evidence IDs, repeatable imports, and no overwrite of user assessments during reprocessing.
4. One-action archive save from YouScoop. Archive imports on desktop and Android without dependency on original desktop filesystem paths.
5. A separate SACRiNOMiCON app with Oracle, Explore, Library/Inbox, source evidence inspection, annotations, claim assessments, typed concepts/links, and search.
6. Grounded local archive chat, bounded context/history, provenance inspection, uncertainty, and replaceable mobile-sized models.
7. Text, read-aloud, and interactive voice modes. Permission/error states, cancellation, interruption, local voice behavior, and clean memory transitions.
8. Responsive phone/foldable and desktop layouts, accessibility, keyboard navigation, and native builds.
9. Storage/import/model/runtime tests plus native smoke tests covering actual save/import/retrieval/chat flows. Android build and emulator verification; physical Fold performance remains explicitly unverified until a device is available.
10. User-facing setup/run documentation, portable backup/export, verified results, and an honest requirement-by-requirement completion audit.

## Implementation sequence

A. Shared archive domain/store/package, preservation tests, desktop save.
B. Isolated desktop worker, bounded Omni inspection, Gemma chat, memory/cancel lifecycle.
C. Separate native archive app and portable inference.
D. Oracle, Explore, evidence and assessment workflows.
E. Mobile/native voice and packaging; end-to-end verification and repair.

An item is not complete merely because a build passes. Retain open gates until behavior is demonstrated. No Space Grotesk and no em dashes in authored UI or documentation.
