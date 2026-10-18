#!/usr/bin/env python3
# Drafter: NVIDIA's LiLiCorr GGUF with every original tensor kept (lattice, candidate selector,
# transformer head), plus token_embd and output copied from the two card target so the drafter
# can sit on one card next to a split target. Then its large matrices go to XE2MX4 with the
# types in lilicorr-xe2mx4.types; F32 and Q8_0 tensors are copied.
#
#   scripts/bolt/drafter-xe2mx4.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf \
#       Nemotron-3.5-Bolt-74B-A9B-XE2MX4.gguf Nemotron-3.5-Bolt-74B-A9B-LiLiCorr-XE2MX4.gguf
import argparse
import tempfile
from pathlib import Path

from common import check_layout, gguf, quantize_bin, read_types, run_quantize, tensor_digest

TYPES = Path(__file__).resolve().parent / "lilicorr-xe2mx4.types"
HEADS = ("token_embd.weight", "output.weight")

p = argparse.ArgumentParser()
p.add_argument("lilicorr")
p.add_argument("target")
p.add_argument("out")
p.add_argument("--quantize", help="llama-quantize binary, default build-sycl/bin/llama-quantize")
p.add_argument("--device", type=int, default=0, help="Level Zero device used for packing")
p.add_argument("--tmpdir", help="where the grafted intermediate GGUF goes, default next to out")
a = p.parse_args()

src = gguf.GGUFReader(a.lilicorr)
target = gguf.GGUFReader(a.target)
heads = [t for t in target.tensors if t.name in HEADS]
assert len(heads) == 2 and not {t.name for t in heads} & {t.name for t in src.tensors}
for k, f in src.fields.items():
    if k.startswith("tokenizer.") and k != "tokenizer.ggml.mask_token_id":
        assert k in target.fields and f.contents() == target.fields[k].contents(), f"tokenizer mismatch: {k}"

tmpdir = a.tmpdir or str(Path(a.out).resolve().parent)
with tempfile.TemporaryDirectory(dir=tmpdir) as d:
    grafted = Path(d) / "lilicorr-heads.gguf"
    w = gguf.GGUFWriter(grafted, "dflash")
    for k, f in src.fields.items():
        if not k.startswith("GGUF."):
            w.add_key_value(k, f.contents(), f.types[0], f.types[1] if len(f.types) > 1 else None)
    for t in [*src.tensors, *heads]:
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    got = {t.name: t for t in gguf.GGUFReader(grafted).tensors}
    for t in [*src.tensors, *heads]:
        assert tensor_digest(t) == tensor_digest(got[t.name]), t.name
    print(f"grafted: {len(src.tensors)} LiLiCorr tensors kept, {len(heads)} target heads added", flush=True)

    types = read_types(TYPES)
    keep = [n for n, t in got.items() if t.tensor_type.name.lower() == types[n]]
    run_quantize(quantize_bin(a.quantize), TYPES, grafted, a.out, a.device)
    n = check_layout(grafted, a.out, types, keep)
print(f"ok: {n} tensors, {len(keep)} copied tensors byte-identical", flush=True)
