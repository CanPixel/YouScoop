"""Omni 3B on the frame benchmark, image only, loaded as runtime/worker.py does."""
import json, os, sys, time
import torch
from PIL import Image
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, StoppingCriteria, StoppingCriteriaList

model_dir, out = sys.argv[1], sys.argv[2]
frames = sys.argv[3:]
INSTRUCTION = "Describe this video frame in one or two short sentences: who is visible and what they are wearing, what they are doing, the setting, and notable objects. If text, a document, a web page or an email is shown, say what kind it is, but do not quote or transcribe its words. Mention only what you can clearly see and leave out anything you are unsure of. Do not guess names of people, places, or objects."

class FirstParagraph(StoppingCriteria):
    def __init__(self, tokenizer, prefix): self.tokenizer, self.prefix = tokenizer, prefix
    def __call__(self, input_ids, scores, **kwargs):
        return "\n" in self.tokenizer.decode(input_ids[0, self.prefix:], skip_special_tokens=True)

torch.set_num_threads(6)
t = time.perf_counter()
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16, device_map="cpu", attn_implementation="eager", local_files_only=True).eval().to("mps")
processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, local_files_only=True)
load = time.perf_counter() - t
rows = []
for path in frames:
    image = Image.open(path).convert("RGB")
    messages = [{"role": "system", "content": [{"type": "text", "text": "You are a careful visual describer."}]},
                {"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": INSTRUCTION}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", padding=True).to("mps").to(model.dtype)
    n = inputs.input_ids.shape[-1]
    began = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=96, do_sample=False, stopping_criteria=StoppingCriteriaList([FirstParagraph(processor.tokenizer, n)]))
    torch.mps.synchronize()
    text = processor.batch_decode(ids[:, n:], skip_special_tokens=True)[0].split("\n", 1)[0].strip()
    rows.append({"frame": os.path.basename(path), "seconds": time.perf_counter() - began, "inputTokens": n, "description": text})
    del inputs, ids
    torch.mps.empty_cache()
json.dump({"model": "Qwen2.5-Omni-3B Thinker (fp16, MPS)", "loadSeconds": load, "peakMemoryGb": torch.mps.driver_allocated_memory() / 1e9, "frames": rows}, open(out, "w"), indent=1)
