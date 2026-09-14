from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import os
import re

import numpy as np
import torch

from .base import LazyTorchTensor, logger

if TYPE_CHECKING:
    from torch import Tensor
    from .base import ModelBase


_EXL3_SUFFIXES = (".trellis", ".suh", ".svh", ".mul1", ".mcg", ".su", ".sv")

_EXL3_BITS = frozenset(range(1, 9))

# After filter_tensors: language_model. stripped, mtp.* remapped to model.layers.{n+i}.*
_EXPERT_BASE = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)$"
)


def _eager(value) -> Tensor:
    tensor = value() if callable(value) else value
    return LazyTorchTensor.to_eager(tensor)


def _reorder_v_heads(tensor: Tensor, dim: int, num_k_heads: int, num_v_per_k: int, head_dim: int) -> Tensor:
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def _tile_perm(elem_perm: Tensor) -> Tensor:
    """Map a 16-aligned element permutation to a 16-wide tile permutation."""
    elem_perm = elem_perm.to(dtype=torch.long)
    assert int(elem_perm.numel()) % 16 == 0
    starts = elem_perm.view(-1, 16)[:, 0]
    assert torch.equal(elem_perm.view(-1, 16), starts.unsqueeze(1) + torch.arange(16, dtype=starts.dtype))
    assert torch.all(starts % 16 == 0)
    return starts // 16


def _apply_linear_attn_v_reorder(
    name: str,
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    hparams: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    num_k_heads = hparams.get("linear_num_key_heads", 0)
    num_v_heads = hparams.get("linear_num_value_heads", 0)
    if num_k_heads <= 0 or num_v_heads <= 0 or num_k_heads == num_v_heads:
        return trellis, suh, svh
    if "linear_attn." not in name:
        return trellis, suh, svh

    head_k_dim = hparams["linear_key_head_dim"]
    head_v_dim = hparams["linear_value_head_dim"]
    num_v_per_k = num_v_heads // num_k_heads

    def perm_n(count: int, head_dim: int) -> Tensor:
        return _tile_perm(_reorder_v_heads(
            torch.arange(count, dtype=torch.long),
            0, num_k_heads, num_v_per_k, head_dim,
        ))

    if name.endswith(".linear_attn.in_proj_qkv"):
        q_dim = head_k_dim * num_k_heads
        k_dim = head_k_dim * num_k_heads
        v_dim = head_v_dim * num_v_heads
        q_tiles = q_dim // 16
        k_tiles = k_dim // 16
        v_perm = perm_n(v_dim, head_v_dim) + q_tiles + k_tiles
        n_perm = torch.cat([
            torch.arange(q_tiles + k_tiles, dtype=torch.long),
            v_perm,
        ])
        trellis = trellis[:, n_perm]
        svh = svh.view(-1, 16)[n_perm].reshape(-1)
    elif name.endswith(".linear_attn.in_proj_z"):
        n_perm = perm_n(head_v_dim * num_v_heads, head_v_dim)
        trellis = trellis[:, n_perm]
        svh = svh.view(-1, 16)[n_perm].reshape(-1)
    elif name.endswith(".linear_attn.out_proj"):
        k_perm = perm_n(head_v_dim * num_v_heads, head_v_dim)
        trellis = trellis[k_perm]
        suh = suh.view(-1, 16)[k_perm].reshape(-1)

    return trellis.contiguous(), suh.contiguous(), svh.contiguous()


def pack_exl3_weight(
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if trellis.ndim != 3:
        raise ValueError(f"EXL3 trellis must be 3D, got {tuple(trellis.shape)}")
    k_tiles, n_tiles, packed = trellis.shape
    if packed % 16 != 0:
        raise ValueError(f"EXL3 packed dim must be 16*bits, got {packed}")
    bits = packed // 16
    if bits not in _EXL3_BITS:
        raise ValueError(f"Unsupported EXL3 bit width {bits} (packed={packed})")

    K = k_tiles * 16
    N = n_tiles * 16
    suh = suh.reshape(-1)
    svh = svh.reshape(-1)
    if suh.numel() != K or svh.numel() != N:
        raise ValueError(
            f"EXL3 scale size mismatch: suh {tuple(suh.shape)} vs K={K}, svh {tuple(svh.shape)} vs N={N}"
        )

    weight = np.ascontiguousarray(
        trellis.contiguous().view(torch.int16).cpu().numpy().view(np.uint8).reshape(-1)
    )
    suh_f32 = np.ascontiguousarray(suh.to(dtype=torch.float32).cpu().numpy().reshape(-1))
    svh_f32 = np.ascontiguousarray(svh.to(dtype=torch.float32).cpu().numpy().reshape(-1))
    return weight, suh_f32, svh_f32, bits


def _is_excluded_src(name: str, *, drop_vision: bool, drop_mtp: bool) -> bool:
    if "visual." in name:
        return drop_vision
    if name.startswith("mtp.") or ".mtp." in name:
        return drop_mtp
    return False


def _qtype_of(gguf_mod, bits: int):
    return {
        1: gguf_mod.GGMLQuantizationType.EXL3_1,
        2: gguf_mod.GGMLQuantizationType.EXL3_2,
        3: gguf_mod.GGMLQuantizationType.EXL3_3,
        4: gguf_mod.GGMLQuantizationType.EXL3_4,
        5: gguf_mod.GGMLQuantizationType.EXL3_5,
        6: gguf_mod.GGMLQuantizationType.EXL3_6,
        7: gguf_mod.GGMLQuantizationType.EXL3_7,
        8: gguf_mod.GGMLQuantizationType.EXL3_8,
    }[bits]


def _detect_codebook(tensors: dict[str, Callable], hparams: dict) -> str:
    has_mcg = any(name.endswith(".mcg") for name in tensors)
    has_mul1 = any(name.endswith(".mul1") for name in tensors)
    if has_mcg and has_mul1:
        raise ValueError("EXL3 folder mixes .mcg and .mul1 tensors")
    qc = hparams.get("quantization_config") or {}
    cfg = str(qc.get("codebook") or "").strip().lower()
    if has_mcg:
        cb = "mcg"
    elif has_mul1:
        cb = "mul1"
    elif cfg in ("mcg", "mul1"):
        cb = cfg
    else:
        cb = "mul1"
    if cfg and cfg not in ("", cb):
        logger.warning("quantization_config codebook=%s but tensors say %s; using %s", cfg, cb, cb)
    return cb


def _pop_exl3_sidecars(tensors: dict[str, Callable], base: str) -> None:
    for suffix in _EXL3_SUFFIXES:
        tensors.pop(base + suffix, None)


def _load_linear(tensors: dict[str, Callable], base: str, hparams: dict):
    trellis = _eager(tensors[base + ".trellis"])
    suh = _eager(tensors[base + ".suh"]).reshape(-1)
    svh = _eager(tensors[base + ".svh"]).reshape(-1)
    return _apply_linear_attn_v_reorder(base, trellis, suh, svh, hparams)


def _write_dense_exl3(model: ModelBase, gguf_mod, base: str, tensors: dict[str, Callable]) -> None:
    trellis, suh, svh = _load_linear(tensors, base, model.hparams)
    weight, suh_f32, svh_f32, bits = pack_exl3_weight(trellis, suh, svh)
    K = trellis.shape[0] * 16
    N = trellis.shape[1] * 16
    qtype = _qtype_of(gguf_mod, bits)
    new_name = model.map_tensor_name(base + ".weight")
    scale_base = new_name.removesuffix(".weight")
    # int16 view so the writer keeps raw_shape (N, K) -> GGUF ne = {K, N}
    model.gguf_writer.add_tensor(
        new_name, weight.view(np.int16), raw_shape=(N, K), raw_dtype=qtype,
    )
    model.gguf_writer.add_tensor(
        scale_base + ".suh", suh_f32, raw_dtype=gguf_mod.GGMLQuantizationType.F32,
    )
    model.gguf_writer.add_tensor(
        scale_base + ".svh", svh_f32, raw_dtype=gguf_mod.GGMLQuantizationType.F32,
    )
    logger.info("Packed %s as %s (%d-bit EXL3) + %s.suh/.svh", new_name, qtype.name, bits, scale_base)
    _pop_exl3_sidecars(tensors, base)


def _write_expert_stack(
    model: ModelBase,
    gguf_mod,
    prefix: str,
    proj: str,
    bases_by_id: dict[int, str],
    tensors: dict[str, Callable],
    n_experts: int,
) -> None:
    ids = sorted(bases_by_id)
    if ids != list(range(n_experts)):
        raise ValueError(
            f"EXL3 experts {prefix}.{proj} have ids {ids[:8]}... (n={len(ids)}), expected 0..{n_experts - 1}"
        )

    weights: list[np.ndarray] = []
    suh_rows: list[np.ndarray] = []
    svh_rows: list[np.ndarray] = []
    bits0 = None
    K0 = None
    N0 = None
    for xid in ids:
        base = bases_by_id[xid]
        trellis, suh, svh = _load_linear(tensors, base, model.hparams)
        weight, suh_f32, svh_f32, bits = pack_exl3_weight(trellis, suh, svh)
        K = trellis.shape[0] * 16
        N = trellis.shape[1] * 16
        if bits0 is None:
            bits0, K0, N0 = bits, K, N
        elif (bits, K, N) != (bits0, K0, N0):
            raise ValueError(
                f"EXL3 expert {base} is {bits}-bit {K}x{N}, stack is {bits0}-bit {K0}x{N0}"
            )
        weights.append(weight)
        suh_rows.append(suh_f32)
        svh_rows.append(svh_f32)
        _pop_exl3_sidecars(tensors, base)

    assert bits0 is not None and K0 is not None and N0 is not None
    stacked = np.concatenate(weights, axis=0)
    suh_2d = np.stack(suh_rows, axis=0)
    svh_2d = np.stack(svh_rows, axis=0)
    qtype = _qtype_of(gguf_mod, bits0)
    new_name = model.map_tensor_name(f"{prefix}.{proj}.weight")
    scale_base = new_name.removesuffix(".weight")
    # raw_shape (n_expert, N, K) -> GGUF ne = {K, N, n_expert}
    model.gguf_writer.add_tensor(
        new_name, stacked.view(np.int16), raw_shape=(n_experts, N0, K0), raw_dtype=qtype,
    )
    model.gguf_writer.add_tensor(
        scale_base + ".suh", suh_2d, raw_dtype=gguf_mod.GGMLQuantizationType.F32,
    )
    model.gguf_writer.add_tensor(
        scale_base + ".svh", svh_2d, raw_dtype=gguf_mod.GGMLQuantizationType.F32,
    )
    logger.info(
        "Packed %s as %s (%d-bit, %d experts, %dx%d) + %s.suh/.svh",
        new_name, qtype.name, bits0, n_experts, K0, N0, scale_base,
    )


def pack_and_write_exl3(model: ModelBase) -> None:
    import gguf

    tensors: dict[str, Callable] = model.model_tensors
    drop_mtp = bool(getattr(model, "no_mtp", False))
    codebook = _detect_codebook(tensors, model.hparams)
    bases = sorted({name[:-len(".trellis")] for name in tensors if name.endswith(".trellis")})

    expert_groups: dict[tuple[str, str], dict[int, str]] = {}
    dense_bases: list[str] = []
    for base in bases:
        if _is_excluded_src(base, drop_vision=True, drop_mtp=drop_mtp):
            continue
        m = _EXPERT_BASE.match(base)
        if m:
            expert_groups.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = base
        else:
            dense_bases.append(base)

    n_experts = None
    if expert_groups:
        n_experts = int(model.find_hparam(["num_local_experts", "num_experts"]))
        if n_experts <= 0:
            raise ValueError(f"Invalid expert count {n_experts}")

    written = 0
    for (prefix, proj), by_id in sorted(expert_groups.items()):
        assert n_experts is not None
        _write_expert_stack(model, gguf, prefix, proj, by_id, tensors, n_experts)
        written += 1

    for base in dense_bases:
        _write_dense_exl3(model, gguf, base, tensors)
        written += 1

    drop = [
        name for name in tensors
        if name.endswith(_EXL3_SUFFIXES)
        or _is_excluded_src(name, drop_vision=True, drop_mtp=drop_mtp)
    ]
    for name in drop:
        del tensors[name]

    pack = os.environ.get("EXL3_PACK", "").strip().lower()
    if pack in ("xe2-16", "xe2", "xe2_16"):
        model.gguf_writer.add_string(gguf.Keys.Exl3.PACK, "xe2-16")
        logger.info("GGUF exl3.pack=xe2-16")
    else:
        model.gguf_writer.add_string(gguf.Keys.Exl3.PACK, "mma")
    model.gguf_writer.add_string(gguf.Keys.Exl3.CODEBOOK, codebook)
    logger.info("GGUF exl3.codebook=%s", codebook)

    logger.info(
        "Wrote %d EXL3 tensors (%d expert stacks, %d dense); dropped %d leftover keys",
        written, len(expert_groups), len(dense_bases), len(drop),
    )


def _sylvester_h128(device, dtype):
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h * (128.0 ** -0.5)


def _reconstruct_what(trellis: Tensor) -> Tensor:
    if trellis.ndim != 3:
        raise ValueError(f"EXL3 trellis must be 3D, got {tuple(trellis.shape)}")
    k_tiles, n_tiles, packed = trellis.shape
    bits = packed // 16
    K = k_tiles * 16
    N = n_tiles * 16
    try:
        from exllamav3.ext import exllamav3_ext as ext
    except Exception as exc:
        raise RuntimeError("EXL3 vision dequant needs exllamav3 + CUDA") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("EXL3 vision dequant needs a CUDA device")
    device = torch.device("cuda")
    packed_t = trellis.contiguous().view(torch.int16).to(device)
    what = torch.empty((K, N), device=device, dtype=torch.float16)
    ext.reconstruct(what, packed_t, bits, False, True)
    return what


def _apply_had_scales(what: Tensor, suh: Tensor, svh: Tensor) -> Tensor:
    # W = diag(suh) H128 What H128 diag(svh)
    device = what.device
    H = _sylvester_h128(device, torch.float32)
    K, N = what.shape
    w = what.to(dtype=torch.float32)
    suh = suh.to(device=device, dtype=torch.float32).reshape(-1)
    svh = svh.to(device=device, dtype=torch.float32).reshape(-1)
    out = torch.empty_like(w)
    for kb in range(0, K, 128):
        for nb in range(0, N, 128):
            blk = w[kb:kb + 128, nb:nb + 128]
            blk = H @ blk @ H
            out[kb:kb + 128, nb:nb + 128] = suh[kb:kb + 128, None] * blk * svh[None, nb:nb + 128]
    return out


def dequant_exl3_to_dense(model: ModelBase) -> int:
    """Replace EXL3 trellis linears with F16 dense weights for mmproj / CLIP."""
    tensors: dict[str, Callable] = model.model_tensors
    bases = sorted({name[:-len(".trellis")] for name in tensors if name.endswith(".trellis")})
    inter = None
    hv = getattr(model, "hparams_vision", None) or getattr(model, "hparams", {})
    if isinstance(hv, dict):
        inter = hv.get("intermediate_size")

    written = 0
    for base in bases:
        qkv_name = base.rsplit(".", 1)[0] + ".qkv.weight"
        skip_split = (
            base.endswith((".attn.q_proj", ".attn.k_proj", ".attn.v_proj"))
            and qkv_name in tensors
        )
        if skip_split or (base + ".weight") in tensors:
            for suffix in _EXL3_SUFFIXES:
                tensors.pop(base + suffix, None)
            logger.info("Keep dense %s, drop EXL3 sidecars", base)
            continue

        trellis = _eager(tensors[base + ".trellis"])
        suh = _eager(tensors[base + ".suh"]).reshape(-1)
        svh = _eager(tensors[base + ".svh"]).reshape(-1)
        what = _reconstruct_what(trellis)
        w_kn = _apply_had_scales(what, suh, svh)
        weight = w_kn.T.contiguous().to(dtype=torch.float16).cpu()
        if inter and base.endswith(".mlp.linear_fc1") and weight.shape[0] > inter:
            weight = weight[:inter].contiguous()
            bias_name = base + ".bias"
            if bias_name in tensors:
                bias = _eager(tensors[bias_name]).reshape(-1)
                if bias.numel() > inter:
                    tensors[bias_name] = lambda t=bias[:inter].contiguous(): t
        if inter and base.endswith(".mlp.linear_fc2") and weight.shape[1] > inter:
            weight = weight[:, :inter].contiguous()
        tensors[base + ".weight"] = lambda t=weight: t
        for suffix in _EXL3_SUFFIXES:
            tensors.pop(base + suffix, None)
        logger.info("Dequant %s EXL3 -> F16 %s", base, tuple(weight.shape))
        written += 1

    for name in list(tensors):
        if not any(s in name for s in (".attn.q_proj.", ".attn.k_proj.", ".attn.v_proj.")):
            continue
        qkv = name.split(".attn.")[0] + ".attn.qkv.weight"
        if qkv in tensors:
            del tensors[name]
            logger.info("Dropped split-attn leftover %s", name)

    return written
