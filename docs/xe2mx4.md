# XE2MX4

XE2MX4 is a 4-bit GGUF type for Intel Xe2 dGPUs (Battlemage, Arc B-series).
It runs on the SYCL backend only. CUDA and Vulkan do not load it.

## Build

Normal SYCL build:

```
cmake -B build-sycl -G Ninja -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl --target llama-server llama-quantize llama-perplexity
```

## oneDNN

The fast path needs oneDNN v3.11.4 with the two patches in `docs/xe2mx4-onednn/`.
The stock oneDNN that ships with oneAPI works, but XE2MX4 prefill then uses
the stock s4 matmul and is about 4x slower. Keep the default `-ub 512` on stock
oneDNN; `-ub 3072` is slower there, not faster.

26k prompt, B65, same GGUF:

```
oneDNN    -ub    PP     TG
patched   3072   2030   38.4
stock      512    466   35.5
stock     3072    138   34.3
```

The backend checks which library got loaded and picks the path, so the same
binary runs on both.

```
git clone https://github.com/uxlfoundation/oneDNN && cd oneDNN
git checkout v3.11.4
git am /path/to/llama.cpp-exl3xe2cuda/docs/xe2mx4-onednn/*.patch
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DDNNL_CPU_RUNTIME=NONE -DDNNL_GPU_RUNTIME=SYCL -DDNNL_GPU_VENDOR=INTEL \
  -DDNNL_LIBRARY_TYPE=SHARED -DDNNL_ENABLE_WORKLOAD=INFERENCE \
  -DDNNL_BUILD_TESTS=OFF -DDNNL_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_GRAPH=ON \
  -DCMAKE_INSTALL_PREFIX=$HOME/onednn-xe2mx4
cmake --build build && cmake --install build
```

Put that `lib` in front of `LD_LIBRARY_PATH` when you run llama-server.

## Run

```
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build-sycl/bin/llama-server \
  -m model-XE2MX4.gguf -ngl 99 --flash-attn on \
  --ctx-size 262144 -ctk q4_0 -ctv q4_0 -b 3072 -ub 3072 \
  --spec-type draft-mtp --spec-mtp-cheap --spec-draft-p-min 0.00 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0
```

On an Xe2 dGPU the backend sets its own defaults for this path (fuses, attention,
pack layout, MTP batch). Any `GGML_SYCL_*` variable you set wins over them.
`GGML_SYCL_XE2_PROFILE=0` turns the defaults off. `GGML_SYCL_XE2_LOG=1` prints
the per-tensor load and kernel lines.

If the GPU also drives your monitor, leave it out (`level_zero:1` for a second card).

## Make one

```
./build-sycl/bin/llama-quantize --imatrix imatrix.gguf model-BF16.gguf model-XE2MX4.gguf XE2MX4
```

The quantizer runs on the Xe2 GPU. An already quantized GGUF also works as the
source with `--allow-requantize`, at some extra quality cost.
