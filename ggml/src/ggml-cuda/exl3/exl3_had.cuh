#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 Hadamard butterflies
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// 128-pt Sylvester FWHT, same butterflies as fwht.cu. scale = 1/sqrt(128).

#include <cuda_fp16.h>
#include <cstdint>

#define EXL3_HAD_ISQ 0.08838834764831843f

__device__ __forceinline__ void exl3_fwht128_reg(float reg[4], int lane) {
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFFu, val, h, 32);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = 32; h < 128; h *= 2) {
        const int step = h / 32;
#pragma unroll
        for (int j = 0; j < 4; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

// One warp. x,suh,dst are 128-wide. suh may be null.
__device__ __forceinline__ void exl3_had_warp_f32(const float * __restrict__ x,
                                                 const float * __restrict__ suh,
                                                 half * __restrict__ dst) {
    const int lane = threadIdx.x & 31;
    float reg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float v = x[col];
        if (suh) {
            v *= suh[col];
        }
        reg[i] = v * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(reg, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[i * 32 + lane] = __float2half_rn(reg[i]);
    }
}

__device__ __forceinline__ void exl3_f32_to_f16_128(const float * __restrict__ x,
                                                    const float * __restrict__ suh,
                                                    half * __restrict__ dst) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float v = x[col];
        if (suh) {
            v *= suh[col];
        }
        dst[col] = __float2half_rn(v);
    }
}

// A is [size_m, size_k] row-major F32. suh is [size_k] or null.
__device__ __forceinline__ void exl3_had_warp_row(const float * __restrict__ A,
                                                 const float * __restrict__ suh,
                                                 int size_k, int size_m, int row, int k0,
                                                 half * __restrict__ dst) {
    const int lane = threadIdx.x & 31;
    float reg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float v = 0.0f;
        if (row < size_m) {
            v = A[(size_t) row * size_k + k0 + col];
            if (suh) {
                v *= suh[k0 + col];
            }
        }
        reg[i] = v * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(reg, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[i * 32 + lane] = __float2half_rn(reg[i]);
    }
}

// 128-pt out FWHT + optional svh. tmp/dst/svh are 128-wide. tmp may alias dst.
__device__ __forceinline__ void exl3_had_out_f32(const float * __restrict__ tmp,
                                                float * __restrict__ dst,
                                                const float * __restrict__ svh) {
    const int lane = threadIdx.x & 31;
    float reg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reg[i] = tmp[i * 32 + lane] * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(reg, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float v = reg[i];
        if (svh) {
            v *= svh[col];
        }
        dst[col] = v;
    }
}

__device__ __forceinline__ void exl3_had_swiglu_f32(
        const float * __restrict__ Cg, const float * __restrict__ Cu, float * __restrict__ dst,
        const float * __restrict__ svh_g, const float * __restrict__ svh_u) {
    const int lane = threadIdx.x & 31;
    float rg[4];
    float ru[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rg[i] = Cg[i * 32 + lane] * EXL3_HAD_ISQ;
        ru[i] = Cu[i * 32 + lane] * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(rg, lane);
    exl3_fwht128_reg(ru, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float g = rg[i] * svh_g[col];
        float u = ru[i] * svh_u[col];
        dst[col] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ void exl3_post_had_svh_kernel(float * __restrict__ C, const float * __restrict__ svh, int size_n) {
    const int row  = (int) blockIdx.y;
    const int base = (int) blockIdx.x * 128;
    exl3_had_out_f32(C + (size_t) row * size_n + base, C + (size_t) row * size_n + base,
                     svh ? svh + base : nullptr);
}

static void exl3_post_had_svh_cuda(float * C, const float * svh, int size_m, int size_n, cudaStream_t stream) {
    if (size_n % 128 != 0 || size_m < 1) {
        return;
    }
    const dim3 grid((unsigned) (size_n / 128), (unsigned) size_m);
    exl3_post_had_svh_kernel<<<grid, 32, 0, stream>>>(C, svh, size_n);
}

__global__ void exl3_post_had_svh_id_kernel(
        float * __restrict__ C, const float * __restrict__ svh, const int32_t * __restrict__ ids,
        int size_n, int n_expert_used, int64_t ids_nb0, int64_t ids_nb1) {
    const int inst = (int) blockIdx.y;
    const int t = inst / n_expert_used;
    const int s = inst - t * n_expert_used;
    const int32_t e = *(const int32_t *) ((const char *) ids + t * ids_nb1 + s * ids_nb0);
    const int base = (int) blockIdx.x * 128;
    exl3_had_out_f32(C + (size_t) inst * size_n + base, C + (size_t) inst * size_n + base,
                     svh + (size_t) e * size_n + base);
}

static void exl3_post_had_svh_id_cuda(
        float * C, const float * svh, const int32_t * ids,
        int n_inst, int n_expert_used, int size_n,
        int64_t ids_nb0, int64_t ids_nb1, cudaStream_t stream) {
    if (size_n % 128 != 0 || n_inst < 1) {
        return;
    }
    const dim3 grid((unsigned) (size_n / 128), (unsigned) n_inst);
    exl3_post_had_svh_id_kernel<<<grid, 32, 0, stream>>>(C, svh, ids, size_n, n_expert_used, ids_nb0, ids_nb1);
}

__device__ __forceinline__ int32_t exl3_moe_id_at(
        const int32_t * ids, int t, int s, int64_t ids_nb0, int64_t ids_nb1) {
    return *(const int32_t *) ((const char *) ids + t * ids_nb1 + s * ids_nb0);
}

__global__ void exl3_moe_pre_had_kernel(
        const float * __restrict__ x, const float * __restrict__ suh, half * __restrict__ A,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ ids_dst,
        int n_inst, int size_k, int n_expert_used, int x_ne1, int64_t ids_nb0, int64_t ids_nb1) {
    const int r = (int) blockIdx.x * (int) blockDim.y + (int) threadIdx.y;
    if (r >= n_inst) {
        return;
    }
    const int inst = ids_dst[r];
    const int t = inst / n_expert_used;
    const int s = inst - t * n_expert_used;
    const int32_t e = exl3_moe_id_at(ids, t, s, ids_nb0, ids_nb1);
    const int x_inst = (x_ne1 <= 1) ? t : inst;
    const float * xr = x + (size_t) x_inst * size_k;
    const float * sr = suh + (size_t) e * size_k;
    half * ar = A + (size_t) r * size_k;
    for (int k0 = 0; k0 < size_k; k0 += 128) {
        exl3_had_warp_row(xr, sr, size_k, 1, 0, k0, ar + k0);
    }
}

static void exl3_moe_pre_had_cuda(
        const float * x, const float * suh, half * A,
        const int32_t * ids, const int32_t * ids_dst,
        int n_inst, int size_k, int n_expert_used, int x_ne1,
        int64_t ids_nb0, int64_t ids_nb1, cudaStream_t stream) {
    if (size_k % 128 != 0 || n_inst < 1) {
        return;
    }
    const dim3 block(32, 8);
    const dim3 grid((unsigned) ((n_inst + 7) / 8));
    exl3_moe_pre_had_kernel<<<grid, block, 0, stream>>>(
        x, suh, A, ids, ids_dst, n_inst, size_k, n_expert_used, x_ne1, ids_nb0, ids_nb1);
}

__global__ void exl3_moe_post_had_scatter_kernel(
        const float * __restrict__ C, const float * __restrict__ svh, float * __restrict__ dst,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ ids_dst,
        int size_n, int n_expert_used, int64_t ids_nb0, int64_t ids_nb1) {
    const int r = (int) blockIdx.y;
    const int n0 = (int) blockIdx.x * 128;
    const int inst = ids_dst[r];
    const int t = inst / n_expert_used;
    const int s = inst - t * n_expert_used;
    const int32_t e = exl3_moe_id_at(ids, t, s, ids_nb0, ids_nb1);
    exl3_had_out_f32(C + (size_t) r * size_n + n0, dst + (size_t) inst * size_n + n0,
                     svh + (size_t) e * size_n + n0);
}

static void exl3_moe_post_had_scatter_cuda(
        const float * C, const float * svh, float * dst,
        const int32_t * ids, const int32_t * ids_dst,
        int n_inst, int size_n, int n_expert_used,
        int64_t ids_nb0, int64_t ids_nb1, cudaStream_t stream) {
    if (size_n % 128 != 0 || n_inst < 1) {
        return;
    }
    const dim3 grid((unsigned) (size_n / 128), (unsigned) n_inst);
    exl3_moe_post_had_scatter_kernel<<<grid, 32, 0, stream>>>(
        C, svh, dst, ids, ids_dst, size_n, n_expert_used, ids_nb0, ids_nb1);
}

__global__ void exl3_moe_post_had_swiglu_kernel(
        const float * __restrict__ Cg, const float * __restrict__ Cu,
        const float * __restrict__ svh_g, const float * __restrict__ svh_u, float * __restrict__ dst,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ ids_dst,
        int size_n, int n_expert_used, int64_t ids_nb0, int64_t ids_nb1) {
    const int r = (int) blockIdx.y;
    const int n0 = (int) blockIdx.x * 128;
    const int inst = ids_dst[r];
    const int t = inst / n_expert_used;
    const int s = inst - t * n_expert_used;
    const int32_t e = exl3_moe_id_at(ids, t, s, ids_nb0, ids_nb1);
    exl3_had_swiglu_f32(
        Cg + (size_t) r * size_n + n0, Cu + (size_t) r * size_n + n0,
        dst + (size_t) inst * size_n + n0,
        svh_g + (size_t) e * size_n + n0, svh_u + (size_t) e * size_n + n0);
}

static void exl3_moe_post_had_swiglu_cuda(
        const float * Cg, const float * Cu,
        const float * svh_g, const float * svh_u, float * dst,
        const int32_t * ids, const int32_t * ids_dst,
        int n_inst, int size_n, int n_expert_used,
        int64_t ids_nb0, int64_t ids_nb1, cudaStream_t stream) {
    if (size_n % 128 != 0 || n_inst < 1) {
        return;
    }
    const dim3 grid((unsigned) (size_n / 128), (unsigned) n_inst);
    exl3_moe_post_had_swiglu_kernel<<<grid, 32, 0, stream>>>(
        Cg, Cu, svh_g, svh_u, dst, ids, ids_dst, size_n, n_expert_used, ids_nb0, ids_nb1);
}

__global__ void exl3_swiglu_f32_kernel(const float * __restrict__ g, const float * __restrict__ u,
                                       float * __restrict__ d, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        d[i] = (x / (1.0f + expf(-x))) * u[i];
    }
}

static void exl3_swiglu_f32_cuda(const float * g, const float * u, float * d, int64_t n, cudaStream_t stream) {
    if (n < 1) {
        return;
    }
    const int block = 256;
    const int grid = (int) ((n + block - 1) / block);
    exl3_swiglu_f32_kernel<<<grid, block, 0, stream>>>(g, u, d, n);
}

__device__ __forceinline__ void exl3_had_swiglu_pre_down(
        const float * __restrict__ Cg, const float * __restrict__ Cu,
        const float * __restrict__ svh_g, const float * __restrict__ svh_u,
        const float * __restrict__ suh_d, half * __restrict__ A) {
    const int lane = threadIdx.x & 31;
    float rg[4];
    float ru[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        rg[i] = Cg[i * 32 + lane] * EXL3_HAD_ISQ;
        ru[i] = Cu[i * 32 + lane] * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(rg, lane);
    exl3_fwht128_reg(ru, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        const float g = rg[i] * svh_g[col];
        const float u = ru[i] * svh_u[col];
        rg[i] = (g / (1.0f + expf(-g))) * u * suh_d[col] * EXL3_HAD_ISQ;
    }
    exl3_fwht128_reg(rg, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        A[i * 32 + lane] = __float2half_rn(rg[i]);
    }
}

__global__ void exl3_moe_swiglu_pre_down_kernel(
        const float * __restrict__ Cg, const float * __restrict__ Cu,
        const float * __restrict__ svh_g, const float * __restrict__ svh_u,
        const float * __restrict__ suh_d, half * __restrict__ A,
        const int32_t * __restrict__ ids, const int32_t * __restrict__ ids_dst,
        int size_n, int n_expert_used, int64_t ids_nb0, int64_t ids_nb1) {
    const int r = (int) blockIdx.y;
    const int n0 = (int) blockIdx.x * 128;
    const int inst = ids_dst[r];
    const int t = inst / n_expert_used;
    const int s = inst - t * n_expert_used;
    const int32_t e = exl3_moe_id_at(ids, t, s, ids_nb0, ids_nb1);
    exl3_had_swiglu_pre_down(
        Cg + (size_t) r * size_n + n0, Cu + (size_t) r * size_n + n0,
        svh_g + (size_t) e * size_n + n0, svh_u + (size_t) e * size_n + n0,
        suh_d + (size_t) e * size_n + n0, A + (size_t) r * size_n + n0);
}

static void exl3_moe_swiglu_pre_down_cuda(
        const float * Cg, const float * Cu,
        const float * svh_g, const float * svh_u, const float * suh_d, half * A,
        const int32_t * ids, const int32_t * ids_dst,
        int n_inst, int size_n, int n_expert_used,
        int64_t ids_nb0, int64_t ids_nb1, cudaStream_t stream) {
    if (size_n % 128 != 0 || n_inst < 1) {
        return;
    }
    const dim3 grid((unsigned) (size_n / 128), (unsigned) n_inst);
    exl3_moe_swiglu_pre_down_kernel<<<grid, 32, 0, stream>>>(
        Cg, Cu, svh_g, svh_u, suh_d, A, ids, ids_dst, size_n, n_expert_used, ids_nb0, ids_nb1);
}
