#pragma once

// Fused EXL3 MoE FFN: pre-had + gate/up GEMM + swiglu + down GEMM + scatter.
// grid.x = 4 so every GEMM is no-split-K (gcd of N-tiles 4 and 16). Packed tiles stay packed.

#include "exl3_gemm.cuh"

#define EXL3_MOE_NX  4

static inline int exl3_moe_lock_n(int n_experts) {
    return 2 * n_experts;
}

template <int bits, int CB>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * 32 / 16)
void exl3_moe_ffn_kernel(
        const float * __restrict__ x,
        const float * __restrict__ suh_g,
        const float * __restrict__ suh_u,
        const float * __restrict__ suh_d,
        const float * __restrict__ svh_g,
        const float * __restrict__ svh_u,
        const float * __restrict__ svh_d,
        const uint16_t * __restrict__ Wg,
        const uint16_t * __restrict__ Wu,
        const uint16_t * __restrict__ Wd,
        half * __restrict__ A_g,
        half * __restrict__ A_u,
        half * __restrict__ A_d,
        float * __restrict__ C_g,
        float * __restrict__ C_u,
        float * __restrict__ C_d,
        float * __restrict__ dst,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ bounds,
        int * __restrict__ locks,
        int hidden_dim,
        int inter_dim,
        int n_experts,
        int n_expert_used,
        int x_ne1,
        int64_t ids_nb0,
        int64_t ids_nb1,
        size_t stride_g,
        size_t stride_u,
        size_t stride_d) {
    const int e = (int) blockIdx.y;
    if (e >= n_experts) {
        return;
    }
    const int lo = bounds[e];
    const int hi = bounds[e + 1];
    const int n = hi - lo;
    if (n <= 0) {
        return;
    }

    (void) ids;
    (void) ids_nb0;
    (void) ids_nb1;
    const int group_size = (int) gridDim.x;
    const int block_idx = (int) blockIdx.x;
    const int block_threads = (int) blockDim.x;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;
    const int warps_per_group = group_size * warps_per_block;

    const uint16_t * Bg = (const uint16_t *) ((const char *) Wg + (size_t) e * stride_g);
    const uint16_t * Bu = (const uint16_t *) ((const char *) Wu + (size_t) e * stride_u);
    const uint16_t * Bd = (const uint16_t *) ((const char *) Wd + (size_t) e * stride_d);
    const float * suh_ge = suh_g + (size_t) e * hidden_dim;
    const float * suh_ue = suh_u + (size_t) e * hidden_dim;
    const float * suh_de = suh_d + (size_t) e * inter_dim;
    const float * svh_ge = svh_g + (size_t) e * inter_dim;
    const float * svh_ue = svh_u + (size_t) e * inter_dim;
    const float * svh_de = svh_d + (size_t) e * hidden_dim;

    const int warps_k = hidden_dim / 128;
    const int total_pre = n * warps_k;
    for (int w = warp_idx0; w < total_pre; w += warps_per_group) {
        const int local = w / warps_k;
        const int k0 = (w - local * warps_k) * 128;
        const int inst = ids_dst[lo + local];
        const int tok = inst / n_expert_used;
        const int x_inst = (x_ne1 <= 1) ? tok : inst;
        const float * xr = x + (size_t) x_inst * hidden_dim;
        exl3_had_warp_row(xr, suh_ge, hidden_dim, 1, 0, k0, A_g + (size_t) (lo + local) * hidden_dim + k0);
        exl3_had_warp_row(xr, suh_ue, hidden_dim, 1, 0, k0, A_u + (size_t) (lo + local) * hidden_dim + k0);
    }
    group_barrier(e, group_size, locks);

    for (int m0 = 0; m0 < n; m0 += 16) {
        const int ms = n - m0 < 16 ? n - m0 : 16;
        exl3_gemm_inner<bits, EXL3_GEMM_SHAPE2, CB>(
            A_g + (size_t) (lo + m0) * hidden_dim, Bg, C_g + (size_t) (lo + m0) * inter_dim,
            ms, hidden_dim, inter_dim, nullptr, nullptr, nullptr, nullptr, 0);
    }
    group_barrier(e, group_size, locks);

    for (int m0 = 0; m0 < n; m0 += 16) {
        const int ms = n - m0 < 16 ? n - m0 : 16;
        exl3_gemm_inner<bits, EXL3_GEMM_SHAPE2, CB>(
            A_u + (size_t) (lo + m0) * hidden_dim, Bu, C_u + (size_t) (lo + m0) * inter_dim,
            ms, hidden_dim, inter_dim, nullptr, nullptr, nullptr, nullptr, 0);
    }
    group_barrier(e, group_size, locks);

    const int warps_n = inter_dim / 128;
    const int total_mid = n * warps_n;
    for (int w = warp_idx0; w < total_mid; w += warps_per_group) {
        const int local = w / warps_n;
        const int n0 = (w - local * warps_n) * 128;
        exl3_had_swiglu_pre_down(
            C_g + (size_t) (lo + local) * inter_dim + n0,
            C_u + (size_t) (lo + local) * inter_dim + n0,
            svh_ge + n0, svh_ue + n0, suh_de + n0,
            A_d + (size_t) (lo + local) * inter_dim + n0);
    }
    group_barrier(e, group_size, locks);

    for (int m0 = 0; m0 < n; m0 += 16) {
        const int ms = n - m0 < 16 ? n - m0 : 16;
        exl3_gemm_inner<bits, EXL3_GEMM_SHAPE2, CB>(
            A_d + (size_t) (lo + m0) * inter_dim, Bd, C_d + (size_t) (lo + m0) * hidden_dim,
            ms, inter_dim, hidden_dim, nullptr, nullptr, nullptr, nullptr, 0);
    }
    group_barrier(e, group_size, locks);

    const int total_out = n * warps_k;
    for (int w = warp_idx0; w < total_out; w += warps_per_group) {
        const int local = w / warps_k;
        const int n0 = (w - local * warps_k) * 128;
        const int inst = ids_dst[lo + local];
        exl3_had_out_f32(
            C_d + (size_t) (lo + local) * hidden_dim + n0,
            dst + (size_t) inst * hidden_dim + n0,
            svh_de + n0);
    }
}

template <int bits, int CB>
static void exl3_moe_ffn_launch(
        const float * x,
        const float * suh_g, const float * suh_u, const float * suh_d,
        const float * svh_g, const float * svh_u, const float * svh_d,
        const uint16_t * Wg, const uint16_t * Wu, const uint16_t * Wd,
        half * A_g, half * A_u, half * A_d,
        float * C_g, float * C_u, float * C_d, float * dst,
        const int32_t * ids, const int32_t * ids_dst, const int32_t * bounds,
        int * locks, int n_sms,
        int hidden_dim, int inter_dim, int n_experts, int n_expert_used, int x_ne1,
        int64_t ids_nb0, int64_t ids_nb1,
        size_t stride_g, size_t stride_u, size_t stride_d,
        cudaStream_t stream) {
    auto * kfn = exl3_moe_ffn_kernel<bits, CB>;
    const int smem = exl3_gemm_smem_bytes(bits, 32, 128);
    static bool smem_ok = false;
    if (!smem_ok) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        smem_ok = true;
    }
    (void) n_sms;
    const dim3 grid((unsigned) EXL3_MOE_NX, (unsigned) n_experts);
    const dim3 block((unsigned) (EXL3_GEMM_BASE_THREADS * 32 / 16));
    kfn<<<grid, block, (size_t) smem, stream>>>(
        x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd,
        A_g, A_u, A_d, C_g, C_u, C_d, dst,
        ids, ids_dst, bounds, locks,
        hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1,
        ids_nb0, ids_nb1, stride_g, stride_u, stride_d);
}

static bool exl3_moe_ffn_ok(int hidden_dim, int inter_dim, int bits_g, int bits_u, int bits_d) {
    if (bits_g != bits_u || bits_g != bits_d) {
        return false;
    }
    if (hidden_dim % 128 != 0 || inter_dim % 128 != 0) {
        return false;
    }
    return exl3_gemm_shape_ok(hidden_dim, inter_dim, bits_g) &&
           exl3_gemm_shape_ok(inter_dim, hidden_dim, bits_g);
}

static void exl3_moe_ffn_cuda(
        const float * x,
        const float * suh_g, const float * suh_u, const float * suh_d,
        const float * svh_g, const float * svh_u, const float * svh_d,
        const uint16_t * Wg, const uint16_t * Wu, const uint16_t * Wd,
        half * A_g, half * A_u, half * A_d,
        float * C_g, float * C_u, float * C_d, float * dst,
        const int32_t * ids, const int32_t * ids_dst, const int32_t * bounds,
        int * locks, int n_sms,
        int hidden_dim, int inter_dim, int bits, int n_experts, int n_expert_used, int x_ne1,
        int64_t ids_nb0, int64_t ids_nb1,
        size_t stride_g, size_t stride_u, size_t stride_d,
        cudaStream_t stream) {
    const int cb = ggml_exl3_get_codebook();
    if (cb == 1) {
        switch (bits) {
            case 1: exl3_moe_ffn_launch<1, 1>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
            case 2: exl3_moe_ffn_launch<2, 1>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
            case 3: exl3_moe_ffn_launch<3, 1>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
            case 4: exl3_moe_ffn_launch<4, 1>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 1: exl3_moe_ffn_launch<1, 2>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
        case 2: exl3_moe_ffn_launch<2, 2>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
        case 3: exl3_moe_ffn_launch<3, 2>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
        case 4: exl3_moe_ffn_launch<4, 2>(x, suh_g, suh_u, suh_d, svh_g, svh_u, svh_d, Wg, Wu, Wd, A_g, A_u, A_d, C_g, C_u, C_d, dst, ids, ids_dst, bounds, locks, n_sms, hidden_dim, inter_dim, n_experts, n_expert_used, x_ne1, ids_nb0, ids_nb1, stride_g, stride_u, stride_d, stream); break;
        default: break;
    }
}
