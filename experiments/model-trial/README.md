# Isolated model trial

This experiment compares desktop ingestion and investigative chat candidates on an Apple M5 Pro with 24 GB unified memory. It does not change application code, model settings, or stored videos.

Weights and scratch media live in the ignored `out/model-trial/` directory. Revisions are recorded in `models.json`. The temporary Python environment is `/private/tmp/youscoop-model-trial`; dependencies are recorded in `requirements.lock.txt`.

## Cases

- Qwen3 4B baseline: audit captions from 330 to 470 seconds of the already-ingested video, with 1,100 output tokens.
- Gemma 4 26B A4B Q4 through MLX-LM: the same excerpt audit, followed by an open inquiry into egregores. Text-only loading excludes the vision tower. A separate optional full-transcript case tests global synthesis.
- Qwen2.5-Omni 7B Q4_K_M through llama.cpp build 11146: audio/image analysis, audio transcription, and higher-resolution document reading.
- Qwen2.5-Omni 3B through Transformers/MPS FP16: native synchronized video and audio, with speech generation disabled. This is a different size, precision, and backend from the 7B test; it is a capability reference, not a controlled accuracy comparison.

The source is the user's existing local copy of YouTube video `kVReoK-UBJA`. The evidence clip covers 390 to 400 seconds. These tests evaluate faithful extraction and argument analysis, not the truth of the video's allegations.

## Run Gemma

```sh
HF_HUB_OFFLINE=1 /private/tmp/youscoop-model-trial/bin/python experiments/model-trial/gemma_chat.py \
  --model out/model-trial/hf/hub/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d \
  --source "$HOME/Library/Application Support/com.canpixel.youscoop/videos/kVReoK-UBJA" \
  --output experiments/model-trial/results/gemma-short.json
```

Thinking is disabled by default. Append `--thinking` to test it with a larger output budget. Append `--full-transcript` to test the complete captions. Run GPU trials one at a time. Cold-start, output limits, context size, and runtime differ, so results are preliminary rather than a broad model ranking.

## Run native Omni

```sh
HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.75 \
  /private/tmp/youscoop-model-trial/bin/python experiments/model-trial/native_omni.py \
  --model out/model-trial/hf/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e \
  --video out/model-trial/evidence-390-400-original.mp4 \
  --output experiments/model-trial/results/omni-3b-native.json
```

The script decodes local frames with PyAV and audio with ffmpeg, then passes both to Qwen's native synchronized processor. Append `--audio-only` for the speech-only case and `--prompt` to supply an explicit extraction instruction. The MPS memory limit is intentionally retained. Do not disable allocation limits to force a model onto this machine.

## Interpretation

Inspect answers alongside the actual source frames and captions. Record incorrect readings, unsupported inferences, unjustified refusals, and truncation. Process RSS, MLX allocator peak, and Metal driver memory are different metrics and should not be compared as if identical.

## Recreate the temporary Python environment

```sh
UV_CACHE_DIR=/private/tmp/youscoop-uv-cache uv venv --python 3.13 /private/tmp/youscoop-model-trial
UV_CACHE_DIR=/private/tmp/youscoop-uv-cache uv pip install \
  --python /private/tmp/youscoop-model-trial/bin/python \
  -r experiments/model-trial/requirements.lock.txt
```

The weights already downloaded remain in `out/model-trial`. Network access is only needed for setup and model downloads; completed inference trials ran with Hugging Face offline mode enabled.
