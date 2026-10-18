#!/usr/bin/env python3
# Shared helpers for the Nemotron 3.5 Bolt XE2MX4 conversion scripts.
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # noqa: E402


def quantize_bin(arg):
    return Path(arg) if arg else ROOT / "build-sycl" / "bin" / "llama-quantize"


def quantize_env(device):
    env = os.environ.copy()
    for k in list(env):
        if k.startswith(("GGML_", "XE2_", "LLAMA_", "ZE_")):
            env.pop(k)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{device}"
    return env


def run_quantize(bin_path, types_file, src, out, device, threads=8, buffer_mib=256):
    if Path(out).exists():
        sys.exit(f"{out} exists, refusing to overwrite")
    cmd = [str(bin_path), "--allow-requantize", "--max-buffer-size", str(buffer_mib),
           "--tensor-type-file", str(types_file), str(src), str(out), "XE2MX4", str(threads)]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=quantize_env(device), check=True)


def tensor_digest(t):
    return hashlib.sha256(t.data).digest()


def check_layout(src, out, types=None, unchanged=()):
    a = {t.name: t for t in gguf.GGUFReader(src).tensors}
    b = {t.name: t for t in gguf.GGUFReader(out).tensors}
    assert set(a) == set(b), "tensor names differ"
    for name, t in b.items():
        assert list(t.shape) == list(a[name].shape), name
        if types is not None:
            assert t.tensor_type.name.lower() == types[name], (name, t.tensor_type.name, types[name])
    for name in unchanged:
        assert tensor_digest(a[name]) == tensor_digest(b[name]), f"{name} changed"
    return len(b)


def read_types(path):
    types = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            pat, typ = line.rsplit("=", 1)
            types[pat.replace("\\.", ".")] = typ
    return types
