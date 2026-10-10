"""Isolated desktop inference. One JSON request on stdin; JSON events on stdout.

The parent owns GPU exclusivity and kills this process on cancellation. Model
weights and cache cannot survive process exit. Source content is never code.
"""
import contextlib
import json
import os
import sys
import time
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.75")

PROTOCOL = sys.stdout

def emit(kind, **payload):
    PROTOCOL.write(json.dumps({"kind": kind, **payload}) + "\n")
    PROTOCOL.flush()

def watch_parent(owner):
    def watch():
        while True:
            if os.getppid() != owner:
                os._exit(0)
            time.sleep(0.2)
    threading.Thread(target=watch, daemon=True).start()

def gemma(request):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_cache_limit(256 * 1024**2)
    start = time.perf_counter()
    emit("status", detail="Loading Gemma 4 on MLX")
    model, tokenizer = load(request["model"], tokenizer_config={"trust_remote_code": False})
    messages = request["messages"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if len(tokenizer.encode(prompt)) > 24000:
        raise ValueError("The investigative context exceeds 24,000 tokens. Narrow the question or select fewer passages.")
    emit("status", detail="Examining source evidence")
    answer = ""
    last = None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=min(request.get("maxTokens", 1200), 1600), sampler=make_sampler(temp=0), prefill_step_size=256):
        answer += response.text
        last = response
        emit("token", text=response.text)
    emit("result", answer=answer, elapsedSeconds=time.perf_counter()-start,
         peakMemoryGb=last.peak_memory if last else 0,
         truncated=bool(last and last.finish_reason == "length"))

def main():
    try:
        request = json.loads(sys.stdin.readline())
        watch_parent(int(request.get("ownerPid", os.getppid())))
        # Libraries occasionally print progress; keep the protocol stream clean.
        with contextlib.redirect_stdout(sys.stderr):
            if request["task"] == "gemma":
                gemma(request)
            else:
                raise ValueError("Unknown worker task")
    except Exception as error:
        emit("error", message=str(error))
        sys.exit(1)

if __name__ == "__main__":
    main()
