# NVIDIA Nemotron 3.5 Bolt on Intel Arc Pro B65

Kernels and launch recipes for NVIDIA Nemotron 3.5 Bolt 74B-A9B (hybrid Mamba2 + attention + MoE)
on Intel Xe2 dGPUs through the SYCL backend. Tested on Arc Pro B65 (32 GB) only. Experimental.

The Bolt kernels are default off; `scripts/bolt-b65.sh` turns them on. Without the launcher's
environment the binary runs the normal llama.cpp SYCL paths. Three changes in this tree are always
on, for every model:

- a startup memory check that refuses to load when buffers exceed a device's total memory (see
  Memory below; `LLAMA_ALLOW_OVERCOMMIT=1` turns the refusal into a warning)
- the oneDNN flash attention causal graph is only used when the real valid KV length is known;
  otherwise it falls back to the explicit mask
- one flash attention shape (q8_0 KV, 1025 tokens, 2 KV heads, GQA 16, head 256) skips the oneDNN
  and MKL kernels, which returned wrong results there, and uses the tile kernel

## Models

Two target GGUFs were used. They are different files and the numbers are not interchangeable.
Both, plus the drafter, are at https://huggingface.co/cesarsal1nas/Nemotron-3.5-Bolt-74B-A9B-XE2MX4-GGUF
and can be rebuilt from NVIDIA's files with `scripts/bolt/` (see Conversion below).

- one card: hybrid file built from NVIDIA's 24GB GGUF. The 80 routed expert tensors stay IQ2_XXS,
  the 160 shared expert and Mamba tensors are requantized to XE2MX4. About 21 GiB.
- two cards: XE2MX4 file converted from the BF16 checkpoint. About 39.8 GiB, does not fit one card.

Drafter: NVIDIA's LiLiCorr head converted to GGUF with its original lattice, candidate selector and
transformer head, used through `--spec-type draft-dflash`. Its large matrices are XE2MX4 and the
small F32 heads are unchanged; target embedding and output tables are grafted in so the drafter can
run on one card next to a split target. The same drafter file was used for `one-draft`,
`dual-draft` and `dual-fast`.

Sources:

- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16
- https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF

The models are under the OpenMDW 1.1 license, https://openmdw.ai/license/1-1/. Derived GGUFs
follow the source model license, not the MIT license of this code.

Files behind the numbers below:

| File | Bytes | SHA256 |
| --- | --- | --- |
| Nemotron-3.5-Bolt-74B-A9B-hybrid-IQ2_XXS-XE2MX4.gguf | 22555066368 | 4cb5e3b65a57887dddce6313e62f823980098f6ead1ddbb215a6441967122020 |
| Nemotron-3.5-Bolt-74B-A9B-XE2MX4.gguf | 42708363968 | 2e81379e92574dfce080b125dd2f179e53ae23b82ed4f40d7120c824ce9e9a36 |
| Nemotron-3.5-Bolt-74B-A9B-LiLiCorr-XE2MX4.gguf | 1822554496 | 18801d9080f67c86407f8b901c7cb20d6a8a0b31bea00e542fb7fb411ad187f1 |

## Conversion

Same build tree, `build-sycl/bin/llama-quantize`, one B65 for packing. Each script refuses to
overwrite its output and checks tensor types, shapes, and that copied tensors are byte-identical.

```
# one card hybrid, from NVIDIA's 24GB GGUF
python3 scripts/bolt/hybrid-xe2mx4.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
    Nemotron-3.5-Bolt-74B-A9B-hybrid-IQ2_XXS-XE2MX4.gguf

# two card file, from NVIDIA's BF16 checkpoint (BF16 GGUF is about 148 GB)
python3 scripts/bolt/convert-bf16.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/ \
    --metadata-from NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf \
    --outtype bf16 --use-temp-file --outfile Bolt-BF16.gguf
python3 scripts/bolt/bf16-xe2mx4.py Bolt-BF16.gguf Nemotron-3.5-Bolt-74B-A9B-XE2MX4.gguf

# drafter, from NVIDIA's LiLiCorr GGUF plus the two card file for its embedding and output
python3 scripts/bolt/drafter-xe2mx4.py NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf \
    Nemotron-3.5-Bolt-74B-A9B-XE2MX4.gguf Nemotron-3.5-Bolt-74B-A9B-LiLiCorr-XE2MX4.gguf
```

`convert-bf16.py` needs the usual `convert_hf_to_gguf.py` Python packages (torch, numpy,
safetensors). It copies the tokenizer and hyperparameter metadata from NVIDIA's 24GB GGUF so
both files describe the same model.

## Build

Requirements, as tested:

- Intel oneAPI 2026.1 (icx/icpx, MKL); `source /opt/intel/oneapi/setvars.sh` before configuring
- Linux 7.0 with the `xe` driver, Intel compute runtime 26.35.39758.10, Level Zero loader 1.32.0
- Resizable BAR on, so each B65 exposes its full 32 GiB

SYCL build and the patched oneDNN are the same as `docs/xe2mx4.md`; put the patched oneDNN from
`docs/xe2mx4-onednn/` first on `LD_LIBRARY_PATH` at runtime. Stock oneDNN works but prefill is much
slower. `git am` on the oneDNN patches needs a git identity configured.

## Run

`scripts/bolt-b65.sh` sets the tested flags:

```
BOLT_MODEL=/path/bolt-hybrid.gguf scripts/bolt-b65.sh one
BOLT_MODEL=/path/bolt-hybrid.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh one-draft
BOLT_MODEL=/path/bolt-xe2mx4.gguf scripts/bolt-b65.sh dual
BOLT_MODEL=/path/bolt-xe2mx4.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh dual-draft
BOLT_MODEL=/path/bolt-xe2mx4.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh dual-fast
```

`one` is target only at 262144 context with `-ub 8192`. The other 262144 recipes use `-ub 2048`,
because a larger ubatch does not fit next to the drafter or across the layer split. `dual-fast`
overlaps prefill across the two cards with `-ub 8192`, so it runs at 16384 context.

Two card recipes default to Level Zero devices 0 and 1 and place the drafter on the first one. Use
`BOLT_DEVICES` if your cards enumerate differently; `sycl-ls` lists them. Keep a GPU that drives a
display out of the split.

The launcher also sets `XE2_KEEP_HYBRID_SLOT=1`, so a follow-up request reuses the cached prompt
instead of reprocessing the whole conversation. The code default is off; see Results.

## What the kernels do

- device side MoE routing: the GPU counts, sorts and tiles the routed rows, so there is no host
  readback or wait per MoE layer
- fused scatter: the MoE GEMM tiles write each row straight to the output, no temp buffer or
  scatter kernel; only enabled for tile variants that support it
- Mamba2 chunked scan (SSD) with vectorized packing and a fused tail
- XE2MX4 MoE GEMM tiles for the two card file, IQ2_XXS DPAS tiles for the hybrid file
- prefill pair: two schedulers alternate ubatches on a two card layer split so both cards work

## Memory

The server refuses to start when the buffers it allocates exceed a device's memory, and warns
when less than about 3 GiB is left. On Intel xe an over-committed device does not fail the
allocation; it evicts VRAM to system memory and runs around 100x slower. Driver, kernel binaries
and oneDNN scratch use roughly 1 to 3 GiB that is not part of the logged buffers. Unified memory
setups that intentionally exceed device memory can set `LLAMA_ALLOW_OVERCOMMIT=1`.

## Quality

WikiText-2 test, first 100 chunks, ctx 512, q8_0 KV, same binary with and without these kernels:

```
model           without          with
two card XE2MX4 5.7486           5.7467, 5.7475
one card hybrid 6.6590 to 6.6674 6.6666, 6.6667
```

The difference is inside the run to run spread of the baseline path. This is one PPL slice, not a
general accuracy claim. Outputs are not bit reproducible between launches even at temperature 0,
with or without these kernels, so other runs of the same slice land on slightly different values.

## Results

### Clean release candidate checks

llama-server HTTP, reasoning enabled, Q8 KV. First request of each fresh server,
5252 prompt tokens, short retrieval answer. These are smoke measurements, not
sustained-generation benchmarks. Context allocation is not prompt length.

| Recipe | Cards | Context allocation | Prompt tokens | PP tok/s | TG tok/s |
| --- | --- | --- | --- | --- | --- |
| one | 1, internal | 262144 | 5252 | 1520.6 | 36.1 |
| one-draft | 1, internal | 262144 | 5252 | 1168.2 | 37.8 |
| dual | 2 | 262144 | 5252 | 2026.2 | 41.0 |
| dual-draft | 2 | 262144 | 5252 | 1683.4 | 63.2 |
| dual-fast | 2 | 16384 | 5252 | 1747.4 | 65.7 |

All five recipes retrieved both keys. Separate cache tests confirmed that follow-up
requests can reuse approximately 5300 cached tokens rather than reprocess the
whole history. The launcher enables `XE2_KEEP_HYBRID_SLOT=1`; the underlying
switch remains default off outside the launcher. The original instant-EOS case
that motivated clearing hybrid slots was not reproduced in these checks, not
proven impossible.

### Historical lab sustained generation

Same model family and Q8 KV, 262144 context allocation, long coding responses
until EOS. These are lab-build medians, not measurements of the clean release
candidate. The model files differ between one-card and two-card rows.

| Configuration | Completed runs | Median TG tok/s |
| --- | --- | --- |
| Hybrid, internal B65 | 2 | 33.4 |
| Hybrid + LiLiCorr, internal B65 | 3 | 44.6 |
| Hybrid, dock B65 | 3 | 33.4 |
| Hybrid + LiLiCorr, dock B65 | 3 | 42.0 |
| BF16-derived XE2MX4, two B65s | 3 | 39.1 |
| BF16-derived XE2MX4 + LiLiCorr, two B65s | 3 | 73.2 |

One target-only run was interrupted and excluded. These rows measure throughput,
not generated-code correctness. Short-answer TG and sustained TG are different
workloads and should not be compared as regressions.

## Tests

`test-backend-ops` on one B65, release build:

- FLASH_ATTN_EXT: 4037/4037
- MUL_MAT: 1115/1120. The 5 failures are `iq4_xs` at m=16, n=5 to 8. The previous release of
  this fork fails the same 5 cases, and they fail with `GGML_SYCL_XE2_PROFILE=0` too. Bolt does
  not use IQ4_XS. Not fixed here.

## Limits

- Arc Pro B65 only. Other Xe2 parts and other systems are untested.
- The test box has one B65 on a chipset PCIe 4.0 x4 link and one in a Thunderbolt 3 dock; the
  dock card is 5 to 8 percent slower on prefill. Two card numbers are capped by the slower card.
- No peer to peer between the cards; the layer split boundary goes through host memory.
- 262144 context with `-ub 8192` on two cards does not fit; the memory check refuses it.
