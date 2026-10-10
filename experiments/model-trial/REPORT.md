# YouScoop model feasibility trial

This report records the exploratory trials. Its early recommendation predates the implemented two-app workflow. See [current model decision](../../docs/MODEL-DECISION.md) and [verification](../../docs/VERIFICATION.md) for the later native batch and Android results.

Tested on 2026-09-30 on the user's Apple M5 Pro with 24 GB unified memory. Metal reports a recommended working set of about 19.1 GB. These are bounded single-source trials, not a general benchmark ranking.

## Findings

Gemma 4 26B A4B Q4 is a feasible desktop chat candidate. Qwen2.5-Omni can process synchronized audio and video on this Mac through a native Transformers/MPS reference, but the tested quantized llama.cpp audio path produced poor transcription. Neither candidate should replace the working stack yet.

| Case | Time | Memory observation | Result |
|---|---:|---:|---|
| Existing Qwen3 4B Q4, excerpt argument audit | 29.8 s total process | 3.2 GB process peak RSS | Fast generation, but verbose quotation exhausted the 1,100-token answer budget before completing the audit |
| Gemma A4B Q4, excerpt audit, thinking disabled | 13.0 s generation including prompt processing; 5.0 s to first token | 14.76 GB MLX peak | Completed a useful separation of assertions, verification needs, and interpretive leaps |
| Gemma A4B Q4, unconventional concept inquiry | 11.8 s; 0.8 s to first token | 14.76 GB MLX peak | Discussed egregores without refusal, but offered discourse categories rather than strong empirical discriminators |
| Gemma A4B Q4, complete 33-minute captions | 27.1 s; 12.3 s to first token | 15.20 GB MLX peak | Broad argument synthesis worked; chapter timestamps were materially inaccurate and output hit the token limit |
| Omni 7B Q4, ten-second audio plus one image | 8.1 s total process | 6.85 GB process peak RSS | Accepted both modalities, but garbled speech and misread document details |
| Omni 7B Q4, audio-only | 2.45 s total process | 6.90 GB process peak RSS | Poor transcription; unsuitable as tested |
| Omni 7B Q4, original-resolution document image | 7.23 s total process | 7.27 GB process peak RSS | Correct bank and account-name fields, with mistakes in other text and inadequate uncertainty marking |
| Omni 3B native FP16, synchronized ten-second audiovisual clip | 12.8 s after 5.9 s model load | 13.27 GB process peak RSS; 12.04 GB MPS driver allocation at completion | Processed audio and video jointly but output document OCR, omitted speech, and invented zeros for a redacted routing number |
| Omni 3B native FP16, audio-only | 4.23 s after 6.4 s model load | 15.31 GB process peak RSS; 9.79 GB MPS driver allocation at completion | Faithful spoken transcription matching the existing captions closely |

| Omni 3B native FP16, focused audiovisual extraction | 5.17 s after 6.17 s model load | 14.13 GB process peak RSS; 12.05 GB MPS driver allocation at completion | Correctly transcribed speech and extracted bank/account-name fields without filling redactions; omitted the requested final assessment |

Memory metrics differ: MLX allocator peaks, process RSS, and Metal driver allocations are not interchangeable. Process timings include startup; the native and MLX generation timings exclude model loading. MLX-LM loaded Gemma's text portion only, excluding the vision tower. Do not generalize these results to full multimodal Gemma use.

## Accuracy observations

The speaker claims that Epstein's bank account was named "Baal," not "Ball." The user clarified this spelling after reviewing the report. The original captions and saved model transcriptions rendered it as "Ball" and remain unchanged as raw evaluation artifacts. This is an example of a phonetically plausible transcription losing the intended conceptual reference. The visible screenshot separately labels the recipient Account Name as "One Clearlake Centre, LLC." The distorted bank-label text and the speaker's interpretation require scrutiny. The video screenshot is not an authenticated original document.

Omni 7B read the bank and account-name fields from a larger image, but its spoken transcription was unreliable in llama.cpp. The native 3B audio test was substantially more faithful. A subsequent focused audiovisual prompt produced the spoken transcript and the correct bank/account-name fields in the same answer, without inventing redacted numbers. It omitted the requested final assessment, so instruction following remains incomplete. Because size, precision, input treatment, and runtime all differ, this does not isolate the cause or establish that 3B is intrinsically better than 7B.

Gemma's short-excerpt analysis was useful. Its full-transcript outline put the account-document discussion at 3:46-5:05, whereas the tested passage is around 6:30. Other topic ranges were also compressed or misplaced. Its summary must not become a source of authoritative timestamps. Source citations should be selected from actual cues or verified spans.



Default Gemma thinking consumed the small output budget before producing complete final answers. The saved thinking-mode runs demonstrate that configuration issue; subsequent runs explicitly disable thinking. Full-transcript output still reached the 1,100-token cap, so that answer is incomplete.

## Runtime work

- MLX-LM 0.31.3 and MLX-VLM 0.7.4 support Gemma 4. No Qwen2.5-Omni implementation was found in the installed MLX packages.
- Quantized Omni used the existing Homebrew llama.cpp build 11146 (7fe450e19). The published model card excludes native video and speech generation; the tests do not establish temporally synchronized video support for that path.
- Native Omni used Transformers 5.17.0, PyTorch 2.14.0, and MPS FP16. Direct MPS weight loading crashed; CPU loading followed by Metal transfer worked.
- The native speech-disabled wrapper attempted to access an absent talker. Calling the Thinker generation component directly worked.
- Audio/video preprocessing uses PyAV and ffmpeg on the same local clip, with 1 FPS frames and 16 kHz audio. Qwen's processor receives both streams with `use_audio_in_video=True`, preserving native temporal interleaving. The experiment bypasses reference decoder utilities incompatible with the installed library versions.
- MPS high and low memory watermarks remain enabled, at 0.85 and 0.75 of the recommended working set respectively.

## Recommendation

Keep Whisper and the current ingestion stack while developing an optional bounded Omni inspection path. Use separate transcript and visual outputs, prohibit filling redactions, and retain original artifacts for verification. Evaluate a supported quantized native audiovisual runtime before selecting Omni 7B for production.

Gemma A4B is worth a desktop-only synthesis/chat option. Load it separately from ingestion models. It does not automatically provide reliable citations or verified truth.

Do not promise the two-hour/30-minute ingestion target from these tests. The native audiovisual examples took roughly 0.52 to 1.28 seconds per second of source media, depending on prompt and output length. Naive extrapolations are about 62 to 154 minutes for two hours, excluding loading, downloads, and other stages. This short-clip ratio is not a measured full-video throughput figure; longer batching, fewer output tokens, and optimized runtimes may change it. Selective deep inspection remains the practical direction to test.

## Review artifacts

- [Gemma completed excerpt answers](results/gemma-answers.md)
- [Gemma full-transcript answer](results/gemma-whole-video-answer.md)
- [Native Omni audiovisual result](results/omni-3b-native.json)
- [Native Omni focused audiovisual extraction](results/omni-3b-native-focused.json)
- [Native Omni spoken transcription](results/omni-3b-native-audio.json)
- [Omni 7B audio/image answer](results/omni-7b-audio-image.txt)
- [Omni 7B original-resolution OCR answer](results/omni-7b-document-original.txt)
- [Reproduction instructions](README.md)

The initial Gemma/Omni trials did not change application code, existing indexes, or original media. Roughly 31 GB of trial model artifacts are in the ignored `out/model-trial` directory. Python dependencies are isolated in `/private/tmp/youscoop-model-trial`.
