#!/usr/bin/env bash
# Launch llama-server with the tested NVIDIA Nemotron 3.5 Bolt recipes for Intel Arc Pro B65 (Xe2) GPUs.
#
#   scripts/bolt-b65.sh <recipe> [extra llama-server args...]
#
# Recipes (all use q8_0 KV, flash attention, and the default-off kernels listed below):
#   one        one B65, hybrid IQ2 GGUF, target only, ctx 262144, -ub 8192
#   one-draft  one B65, hybrid IQ2 GGUF + LiLiCorr drafter, ctx 262144, -ub 2048
#   dual       two B65, XE2MX4 GGUF, layer split, ctx 262144, -ub 2048
#   dual-draft two B65, XE2MX4 GGUF + LiLiCorr drafter, ctx 262144, -ub 2048
#   dual-fast  two B65, XE2MX4 GGUF + LiLiCorr drafter, ctx 16384, -ub 8192, prefill pair
#
# Environment:
#   BOLT_MODEL   target GGUF (required)
#   BOLT_DRAFT   LiLiCorr drafter GGUF (required for *-draft and dual-fast)
#   BOLT_DEVICES Level Zero device ids, default "0" for one card and "0,1" for two
#   BOLT_BIN     llama-server binary, default build-sycl/bin/llama-server
#   BOLT_HOST / BOLT_PORT  default 127.0.0.1 / 8080
#
# The patched oneDNN from docs/xe2mx4-onednn must be first on LD_LIBRARY_PATH for full prefill speed.
# Do not point an IQ2 recipe at an XE2MX4 file or the other way around; the MoE kernels differ.
set -euo pipefail

recipe="${1:-}"
[ -n "$recipe" ] || { sed -n '2,20p' "$0"; exit 1; }
shift

: "${BOLT_MODEL:?set BOLT_MODEL to the target GGUF}"
bin="${BOLT_BIN:-$(dirname "$0")/../build-sycl/bin/llama-server}"
host="${BOLT_HOST:-127.0.0.1}"
port="${BOLT_PORT:-8080}"

export CUDA_VISIBLE_DEVICES=
common_env=(
    GGML_SYCL_SSM_SSD=1 GGML_SYCL_SSM_SG=1 GGML_SYCL_SSM_PREPARE=1 GGML_SYCL_SSM_SSD_PACK3D=2
    GGML_SYCL_BOLT_RELU_SQR=1 GGML_SYCL_BOLT_EXPERT_SUM=1 GGML_SYCL_BOLT_CONV=1 GGML_SYCL_BOLT_TAIL=1
    GGML_SYCL_XE2MX_MOE_DEVROUTE=1 GGML_SYCL_MOE_FUSE_SCATTER=1
    XE2_KEEP_HYBRID_SLOT=1
)
iq2_env=(
    GGML_SYCL_IQ2_MOE_DPAS=1 GGML_SYCL_IQ2_MOE_DIRECT=1 GGML_SYCL_IQ2_MOE_SLM=48 GGML_SYCL_IQ2_MOE_TIGHT=1
    GGML_SYCL_Q8_XMX=1
)
xe2mx_env=(
    GGML_SYCL_XE2MX_LO_HOSTPACK=0 GGML_SYCL_XE2MX_MOE_GEMM=1 GGML_SYCL_XE2MX_MOE_DP4A=1
    GGML_SYCL_XE2MX_MOE_NT=8 GGML_SYCL_XE2MX_MOE_TILE_MB=2
)
draft_env=(
    XE2_DFLASH_FULL_INJECT=1 XE2_DFLASH_INJECT_TAIL=1 XE2_LAYER_INP_TAIL=1100 GGML_MTP_DRAFT_NUBATCH=2048
)

args=(-m "$BOLT_MODEL" --host "$host" --port "$port" -ngl 999 -fa on
      --cache-type-k q8_0 --cache-type-v q8_0 --jinja --parallel 1 --fit off)
env_list=("${common_env[@]}")
draft=0

case "$recipe" in
    one)
        devices="${BOLT_DEVICES:-0}"
        env_list+=("${iq2_env[@]}" XE2_LAB_RS_SEQ=1)
        args+=(--device SYCL0 --split-mode none -c 262144 -b 8192 -ub 8192)
        ;;
    one-draft)
        devices="${BOLT_DEVICES:-0}"
        env_list+=("${iq2_env[@]}" "${draft_env[@]}")
        args+=(--device SYCL0 --split-mode none -c 262144 -b 2048 -ub 2048)
        draft=1
        ;;
    dual|dual-draft)
        devices="${BOLT_DEVICES:-0,1}"
        env_list+=("${xe2mx_env[@]}")
        args+=(--device SYCL0,SYCL1 --split-mode layer --tensor-split 1,1 -c 262144 -b 2048 -ub 2048)
        if [ "$recipe" = dual-draft ]; then env_list+=("${draft_env[@]}"); draft=1; fi
        ;;
    dual-fast)
        devices="${BOLT_DEVICES:-0,1}"
        env_list+=("${xe2mx_env[@]}" "${draft_env[@]}"
                   LLAMA_XE2_PREFILL_PAIR=1 LLAMA_XE2_PREFILL_THREAD=1 XE2_CKPT_STREAM=1)
        args+=(--device SYCL0,SYCL1 --split-mode layer --tensor-split 1,1 -c 16384 -b 16384 -ub 8192)
        draft=1
        ;;
    *)
        echo "unknown recipe: $recipe" >&2
        exit 1
        ;;
esac

if [ "$draft" = 1 ]; then
    : "${BOLT_DRAFT:?set BOLT_DRAFT to the LiLiCorr drafter GGUF}"
    args+=(--spec-type draft-dflash --spec-draft-model "$BOLT_DRAFT" --spec-draft-device SYCL0
           --spec-draft-ngl 999 --spec-draft-n-max 7 --spec-draft-p-min 0)
fi

export ONEAPI_DEVICE_SELECTOR="level_zero:${devices}"
exec env "${env_list[@]}" "$bin" "${args[@]}" "$@"
