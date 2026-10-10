"""Native synchronized audiovisual reference on Apple MPS, speech output disabled."""
import argparse
import json
import resource
import subprocess

import av
import numpy as np
import time
from pathlib import Path

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--audio-only", action="store_true")
    p.add_argument("--prompt")
    a = p.parse_args()
    torch.set_num_threads(6)
    start = time.perf_counter()
    print("Loading native Omni with speech output disabled", flush=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.float16, device_map="cpu", attn_implementation="eager",
        enable_audio_output=False, local_files_only=True,
    ).eval()
    model = model.to("mps")
    processor = Qwen2_5OmniProcessor.from_pretrained(a.model, local_files_only=True)
    load_seconds = time.perf_counter() - start
    print(f"Loaded in {load_seconds:.1f}s", flush=True)
    question = a.prompt or "Analyze this clip using BOTH its audio and visuals. Describe what is said and what the visible document actually shows. Transcribe only legible relevant text, mark uncertain readings, and distinguish the speaker's interpretation from the document. Do not infer authenticity or events not visible."
    messages = [{"role": "user", "content": [
        ({"type": "audio", "audio": str(a.video.resolve())} if a.audio_only else
         {"type": "video", "video": str(a.video.resolve()), "fps": 1, "max_pixels": 256 * 28 * 28}),
        {"type": "text", "text": question},
    ]}]
    start = time.perf_counter()
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Decode both streams from the same local clip. Native processor interleaves
    # their tokens using the explicitly supplied 1 FPS timeline.
    frames = []
    next_second = 0.0
    with av.open(str(a.video)) as container:
        for frame in container.decode(video=0):
            if frame.time is not None and frame.time + 1e-6 >= next_second:
                frames.append(frame.to_ndarray(format="rgb24"))
                next_second += 1.0
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    wave = subprocess.run(["ffmpeg", "-v", "error", "-i", str(a.video), "-vn",
                           "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                          check=True, capture_output=True).stdout
    audio = np.frombuffer(wave, dtype=np.float32).copy()
    inputs = processor(text=prompt, audio=[audio], videos=None if a.audio_only else [video],
                       return_tensors="pt", padding=True, use_audio_in_video=not a.audio_only,
                       videos_kwargs={"fps": 1.0, "do_sample_frames": False,
                                      "size": {"shortest_edge": 128 * 28 * 28,
                                               "longest_edge": 256 * 28 * 28}})
    inputs = inputs.to("mps").to(model.dtype)
    token_count = inputs.input_ids.shape[-1]
    mode = "audio-only" if a.audio_only else "synchronized audio/video"
    print(f"Prepared {mode} input: {token_count} tokens", flush=True)
    with torch.inference_mode():
        ids = model.thinker.generate(**inputs, use_audio_in_video=not a.audio_only,
                             max_new_tokens=a.max_tokens, do_sample=False)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    answer = processor.batch_decode(ids[:, token_count:], skip_special_tokens=True)[0]
    result = {"model": a.model, "runtime": "transformers-mps-fp16", "video": str(a.video),
              "audio_in_video": not a.audio_only, "question": question, "speech_output": False, "load_seconds": load_seconds,
              "processing_and_generation_seconds": elapsed, "input_tokens": token_count,
              "output_tokens": int(ids.shape[-1] - token_count), "answer": answer,
              "mps_allocated_gb": torch.mps.current_allocated_memory()/1e9,
              "mps_driver_gb": torch.mps.driver_allocated_memory()/1e9,
              "process_peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
