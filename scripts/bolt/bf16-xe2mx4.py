#!/usr/bin/env python3
# Two card file: BF16 GGUF from convert-bf16.py packed to XE2MX4 with the tensor types in
# bolt-bf16-xe2mx4.types (Mamba in/out, routed and shared experts XE2MX4; latent and attention
# projections, token embedding and output Q8_0; norms, router and conv F32). No imatrix.
#
#   scripts/bolt/bf16-xe2mx4.py Bolt-BF16.gguf Nemotron-3.5-Bolt-74B-A9B-XE2MX4.gguf
import argparse
from pathlib import Path

from common import check_layout, quantize_bin, read_types, run_quantize

TYPES = Path(__file__).resolve().parent / "bolt-bf16-xe2mx4.types"

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("out")
p.add_argument("--quantize", help="llama-quantize binary, default build-sycl/bin/llama-quantize")
p.add_argument("--device", type=int, default=0, help="Level Zero device used for packing")
a = p.parse_args()

run_quantize(quantize_bin(a.quantize), TYPES, a.src, a.out, a.device)
n = check_layout(a.src, a.out, read_types(TYPES))
print(f"ok: {n} tensors match bolt-bf16-xe2mx4.types", flush=True)
