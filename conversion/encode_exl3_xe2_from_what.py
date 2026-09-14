#!/usr/bin/env python3
# Re-encode packed EXL3 .trellis tensors from MMA/v3 walk to xe2-16 (K-column).
# Needs CUDA + ExLlamaV3 (quantize_tiles, pack_trellis, reconstruct).
# suh/svh unchanged. PPL usually rises vs v3. CUDA llama.cpp aborts xe2-16 GGUFs.

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BATCH = 256


def xe2_16_perm(device) -> torch.Tensor:
    perm = torch.empty(256, dtype=torch.int, device=device)
    for n in range(16):
        for k in range(16):
            perm[n * 16 + k] = k * 16 + n
    return perm


def reencode_trellis(trellis, bits, perm, quantize_tiles, pack_trellis, ext):
    device = perm.device
    k_tiles, n_tiles, packed = trellis.shape
    if packed != 16 * bits:
        raise SystemExit(f"trellis last dim {packed} != 16*bits {16 * bits}")
    K = k_tiles * 16
    N = n_tiles * 16
    packed_t = trellis.contiguous().view(torch.int16).to(device)
    what = torch.empty((K, N), device=device, dtype=torch.float16)
    ext.reconstruct(what, packed_t, bits, False, True)
    tiles = (
        what.float()
        .reshape(k_tiles, 16, n_tiles, 16)
        .permute(0, 2, 1, 3)
        .reshape(-1, 256)
        .contiguous()
    )
    tiles = tiles[:, perm]
    qa = {"K": bits, "mul1": True}
    n = tiles.shape[0]
    out_i = torch.empty((n, 256), dtype=torch.short, device=device)
    for i0 in range(0, n, BATCH):
        i1 = min(n, i0 + BATCH)
        _, q_i = quantize_tiles(tiles[i0:i1].contiguous(), qa)
        out_i[i0:i1] = q_i
    packed_out = pack_trellis(out_i.view(n, 1, 256).contiguous(), qa)
    return packed_out.view(k_tiles, n_tiles, 16 * bits).to(dtype=trellis.dtype, device="cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="packed EXL3 safetensors dir (v3 / MMA walk)")
    ap.add_argument("dst", help="output dir (created)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("need CUDA", file=sys.stderr)
        return 2
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib.quantize import pack_trellis, quantize_tiles

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    idx_path = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        print("missing model.safetensors.index.json", file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    perm = xe2_16_perm(device)
    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(idx_path))
    shards = sorted(set(idx["weight_map"].values()))
    for name in os.listdir(src):
        src_p = os.path.join(src, name)
        dst_p = os.path.join(dst, name)
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        if os.path.isfile(src_p) and not os.path.exists(dst_p):
            shutil.copy2(src_p, dst_p)
    t0 = time.time()
    n_tr = 0
    for shard in shards:
        src_p = os.path.join(src, shard)
        dst_p = os.path.join(dst, shard)
        if os.path.exists(dst_p) and os.path.getsize(dst_p) > 1_000_000:
            print(f"skip existing {shard}", flush=True)
            continue
        print(f"encode {shard}", flush=True)
        tensors = {}
        with safe_open(src_p, framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            for i, key in enumerate(keys):
                t = sf.get_tensor(key)
                if key.endswith(".trellis") and "visual." not in key:
                    bits = int(t.shape[2] // 16)
                    print(f"  {key} {tuple(t.shape)} bits={bits}", flush=True)
                    t = reencode_trellis(t, bits, perm, quantize_tiles, pack_trellis, ext)
                    n_tr += 1
                tensors[key] = t
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(keys)} keys", flush=True)
        save_file(tensors, dst_p)
        print(f"  wrote {dst_p}  elapsed {time.time()-t0:.0f}s  trellis={n_tr}", flush=True)
    shutil.copy2(idx_path, os.path.join(dst, "model.safetensors.index.json"))
    print(f"done {dst} trellis={n_tr} {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
