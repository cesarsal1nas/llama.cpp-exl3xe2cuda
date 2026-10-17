# NVIDIA Nemotron 3.5 Bolt on Intel Arc Pro B65

Kernels and launch recipes for NVIDIA Nemotron 3.5 Bolt 74B-A9B (hybrid Mamba2 + attention + MoE)
on Intel Xe2 dGPUs through the SYCL backend. Tested on Arc Pro B65 (32 GB) only.

Everything here is default off. Without the environment variables below the binary runs the
normal llama.cpp SYCL paths.

## Models

Two GGUFs were used, they are different files and the numbers are not interchangeable:

- one card: hybrid file built from NVIDIA's 24GB GGUF, routed experts stay IQ2_XXS, shared experts
  and Mamba projections are XE2MX4, about 21 GiB
- two cards: XE2MX4 file converted from the BF16 checkpoint, about 39.8 GiB, does not fit one card

The optional drafter is NVIDIA's LiLiCorr head converted to GGUF with its original lattice,
candidate selector and transformer head, used through `--spec-type draft-dflash`.

## Build

Same as `docs/xe2mx4.md`: SYCL build with icx/icpx, and the patched oneDNN from
`docs/xe2mx4-onednn/` first on `LD_LIBRARY_PATH`. Stock oneDNN works but prefill is much slower.

## Run

`scripts/bolt-b65.sh` sets the tested flags:

```
BOLT_MODEL=/path/bolt-hybrid.gguf scripts/bolt-b65.sh one
BOLT_MODEL=/path/bolt-hybrid.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh one-draft
BOLT_MODEL=/path/bolt-xe2mx4.gguf scripts/bolt-b65.sh dual
BOLT_MODEL=/path/bolt-xe2mx4.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh dual-draft
BOLT_MODEL=/path/bolt-xe2mx4.gguf BOLT_DRAFT=/path/lilicorr.gguf scripts/bolt-b65.sh dual-fast
```

`dual-fast` overlaps prefill across the two cards and uses `-ub 8192`; at 262144 context that
ubatch does not fit next to the drafter, so it runs at 16384 context. The 262144 recipes use
`-ub 2048`.

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
and oneDNN scratch use roughly 1 to 3 GiB that is not part of the logged buffers.

## Quality

WikiText-2 test, first 100 chunks, ctx 512, q8_0 KV, same binary with and without these kernels:

```
model           without          with
two card XE2MX4 5.7486           5.7467, 5.7475
one card hybrid 6.6590 to 6.6674 6.6666, 6.6667
```

The difference is inside the run to run spread of the baseline path. This is one PPL slice, not a
general accuracy claim. Outputs are not bit reproducible between launches even at temperature 0,
with or without these kernels.

## Results

Measured on the release build; see the table below once filled in.

## Limits

- Arc Pro B65 only. Other Xe2 parts and other systems are untested.
- The test box has one B65 on a chipset PCIe 4.0 x4 link and one in a Thunderbolt 3 dock; the
  dock card is 5 to 8 percent slower on prefill. Two card numbers are capped by the slower card.
- No peer to peer between the cards; the layer split boundary goes through host memory.
