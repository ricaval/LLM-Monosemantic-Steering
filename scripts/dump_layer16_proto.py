#!/usr/bin/env python
import os, json, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ——— CONFIG ———
# path to the flattened snapshot dir
MODEL_DIR = os.path.expanduser("../models/mistral_cache")
PROMPTS = [
    {"id": 0, "prompt": "What is aspirin for?"},
    {"id": 1, "prompt": "Define diabetic ketoacidosis."},
]
OUT_PATH = os.path.expanduser("../data/activations/layer16_proto.jsonl")
# —————————————————

# 1) Load tokenizer + model once
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",    # Auto offloads & pins shards
)
# No .to("cuda")

# 2) Hook layer 16
act_buffer = {}
def hook_fn(module, inp, out):
    # out: [1, T, D]
    act_buffer["raw"] = out.detach().cpu().numpy()[0]
model.model.layers[16].post_attention_layernorm.register_forward_hook(hook_fn)

# 3) Prepare output
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fout = open(OUT_PATH, "w")

# 4) Loop prompts
for ex in PROMPTS:
    # Tokenize (stay on CPU!)
    enc = tokenizer(ex["prompt"], return_tensors="pt")
    token_ids = enc["input_ids"][0].tolist()  # record raw token IDs

    # Forward‐generate
    with torch.no_grad():
        # Do NOT move enc to GPU; model will pull shards as needed
        gen_ids = model.generate(**enc, max_new_tokens=20)
    text_out = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    # Capture + pool layer16 activations
    raw = act_buffer.pop("raw")            # [T,4096]
    vec = raw.mean(axis=0).astype("float16")  # pool + quantize

    # Write JSONL (float16 vector → list of floats)
    record = {
        "id":      ex["id"],
        "prompt":  ex["prompt"],
        "output":  text_out,
        "tokens":  token_ids,
        "vector":  vec.tolist(),
    }
    fout.write(json.dumps(record) + "\n")
    fout.flush()  # ensure it’s on disk

    # Clean up
    del enc, gen_ids, raw, vec
    torch.cuda.empty_cache()

fout.close()
print(f"Done! Wrote {len(PROMPTS)} records to {OUT_PATH}")
