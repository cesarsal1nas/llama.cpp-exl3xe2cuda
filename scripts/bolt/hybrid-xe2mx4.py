#!/usr/bin/env python3
# One card file: NVIDIA's 24GB GGUF with the shared experts and Mamba in/out projections
# requantized to XE2MX4. Routed experts (IQ2_XXS) and every other tensor are copied byte for byte.
#
#   scripts/bolt/hybrid-xe2mx4.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf out.gguf
import argparse
import re
import tempfile
from pathlib import Path

from common import check_layout, gguf, quantize_bin, run_quantize

CONVERT = re.compile(r"\.(ssm_in|ssm_out|ffn_up_shexp|ffn_down_shexp)\.weight$")

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--quantize", help="llama-quantize binary, default build-sycl/bin/llama-quantize")
p.add_argument("--device", type=int, default=0, help="Level Zero device used for packing")
a = p.parse_args()

tensors = gguf.GGUFReader(a.src).tensors
types, keep = {}, []
for t in tensors:
    if CONVERT.search(t.name):
        types[t.name] = "xe2mx4"
    else:
        types[t.name] = t.tensor_type.name.lower()
        keep.append(t.name)
n_conv = len(tensors) - len(keep)
print(f"{len(tensors)} tensors, {n_conv} to XE2MX4, {len(keep)} copied", flush=True)

with tempfile.NamedTemporaryFile("w", suffix=".types", delete=False) as f:
    f.write("".join(f"^{re.escape(n)}$={t}\n" for n, t in types.items()))
    policy = f.name
try:
    run_quantize(quantize_bin(a.quantize), policy, a.src, a.out, a.device)
finally:
    Path(policy).unlink()

n = check_layout(a.src, a.out, types, keep)
print(f"ok: {n} tensors, {len(keep)} copied tensors byte-identical", flush=True)
