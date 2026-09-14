#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 reconstruct.cu / reconstruct_had_kernel
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// Fused reconstruct in the original basis:
//   W = diag(suh) . H128 . What . H128 . diag(svh)
// per 128x128 tile, 1/sqrt(128) per side. Dest F16 [K,N] ld=N (same as recon_full).
// suh/svh are F32 (GGUF). mul1 codebook (cb=2). Bits 1..8.
// cuBLAS: raw x, OP_N,OP_N, one GemmEx.

#include "exl3_dq.cuh"

#define EXL3_RECON_HAD_THREADS 256

// 4090: fused recon ~1.02x plain on large tiles; path wins by M=192 (ExL used 1024).
#ifndef EXL3_RECON_HAD_MIN_M
#define EXL3_RECON_HAD_MIN_M 192
#endif

template <int Kbits, int cb>
__global__ __launch_bounds__(EXL3_RECON_HAD_THREADS)
void exl3_reconstruct_had_kernel(
        half * __restrict__ g_unpacked,
        const char * __restrict__ packed,
        const float * __restrict__ suh,
        const float * __restrict__ svh,
        int packed_blocks_n) {

    constexpr int packed_size = 256 * Kbits / 16;
    constexpr float r_scale = 0.08838834764831845f;

    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int kb = blockIdx.y;
    const int nb = blockIdx.x;
    const int n = nb * 8;
    const int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&](int R, int q, int p) {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    const uint16_t * g_packed = (const uint16_t *) packed;
    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += EXL3_RECON_HAD_THREADS) {
        const int j = u / (8 * j_int4);
        const int r = u % (8 * j_int4);
        const uint16_t * gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + n)) * (size_t) packed_size;
        ((int4 *) s_packed[j])[r] = ((const int4 *) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (EXL3_RECON_HAD_THREADS / 32); ++jj) {
        const int j = (warp_id / 8) * (8 / (EXL3_RECON_HAD_THREADS / 256)) + jj;
        const int wn = warp_id % 8;
        FragB frag[2];
        dq_dispatch<Kbits, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4)) {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            const int r0 = j * 16 + (lane_id % 4) * 2;
            const int r1 = r0 + 1;
            const int r2 = r0 + 8;
            const int r3 = r0 + 9;
            const int c0 = lane_id / 8;
            const int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            const int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (EXL3_RECON_HAD_THREADS / 32);
#pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq) {
        const int q = warp_id * CHUNKS_PW + qq;
        const int qs = q ^ lane_id;
        half2 a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v = *((const half4 *) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
#pragma unroll
        for (int x = 0; x < 2; ++x) {
            half2 * v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[i] = shuffle_had_h2x32(v[i], lane_id);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4 *) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (EXL3_RECON_HAD_THREADS / 32);
    const float4 svf = ((const float4 *) svh)[nb * 32 + lane_id];
    half4 sv4;
    sv4.x = __floats2half2_rn(svf.x, svf.y);
    sv4.y = __floats2half2_rn(svf.z, svf.w);
#pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr) {
        const int R = warp_id * ROWS_PW + rr;
        const int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        const half2 v01 = stile[base];
        const half2 v23 = stile[base + 1];
        const float v0 = __low2float(v01), v1 = __high2float(v01);
        const float v2 = __low2float(v23), v3 = __high2float(v23);
        const float s0 = v0 + v1, d0 = v0 - v1;
        const float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        const half2 su2 = __float2half2_rn(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        *((half4 *) (g_unpacked + (size_t) (kb * 128 + R) * (size_t) row_len + nb * 128 + lane_id * 4)) = v;
    }
}

// packed_blocks_n stays N_full/16 so a prefix grid still indexes the full trellis.
static void exl3_reconstruct_had_prefix(
        const char * blocks, half * dst,
        const float * suh, const float * svh,
        int K, int N_full, int N_cols, int bits, cudaStream_t stream) {
    if (bits < 1 || bits > 8 || K <= 0 || N_full <= 0 || N_cols <= 0) {
        return;
    }
    dim3 block(EXL3_RECON_HAD_THREADS);
    dim3 grid((unsigned) (N_cols / 128), (unsigned) (K / 128));
    const int n_tiles = N_full / 16;

#define EXL3_RECON_HAD_CASE(b) \
    case b: \
        exl3_reconstruct_had_kernel<b, 2><<<grid, block, 0, stream>>>(dst, blocks, suh, svh, n_tiles); \
        break;

    switch (bits) {
        EXL3_RECON_HAD_CASE(1)
        EXL3_RECON_HAD_CASE(2)
        EXL3_RECON_HAD_CASE(3)
        EXL3_RECON_HAD_CASE(4)
        EXL3_RECON_HAD_CASE(5)
        EXL3_RECON_HAD_CASE(6)
        EXL3_RECON_HAD_CASE(7)
        EXL3_RECON_HAD_CASE(8)
        default: break;
    }
#undef EXL3_RECON_HAD_CASE
}

static void exl3_reconstruct_had_full(
        const char * blocks, half * dst,
        const float * suh, const float * svh,
        int K, int N, int bits, cudaStream_t stream) {
    exl3_reconstruct_had_prefix(blocks, dst, suh, svh, K, N, N, bits, stream);
}

// W already has diag(suh).H.What.H.diag(svh). Convert raw x only; do not scale by suh.
#define EXL3_CVT_THREADS 256

struct alignas(16) exl3_cvt8 {
    half2 a, b, c, d;
};

__global__ __launch_bounds__(EXL3_CVT_THREADS)
void exl3_convert_f32_to_f16(const float * __restrict__ x, half * __restrict__ y, int64_t n8) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) {
        return;
    }
    const float4 p = __ldg(((const float4 *) x) + 2 * i);
    const float4 q = __ldg(((const float4 *) x) + 2 * i + 1);
    exl3_cvt8 o;
    o.a = __floats2half2_rn(p.x, p.y);
    o.b = __floats2half2_rn(p.z, p.w);
    o.c = __floats2half2_rn(q.x, q.y);
    o.d = __floats2half2_rn(q.z, q.w);
    ((exl3_cvt8 *) y)[i] = o;
}

static void exl3_cvt_f32_f16(const float * x, half * y, int64_t n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    const int64_t n8 = n >> 3;
    const unsigned nb = (unsigned) ((n8 + EXL3_CVT_THREADS - 1) / EXL3_CVT_THREADS);
    exl3_convert_f32_to_f16<<<nb, EXL3_CVT_THREADS, 0, stream>>>(x, y, n8);
}

#ifndef EXL3_RECON_HAD_STANDALONE

#include "../common.cuh"

static void ggml_cuda_mul_mat_exl3_recon_had(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const void * x,
        ggml_type x_type,
        const float * suh,
        const float * svh) {
    const int64_t K = src0->ne[0];
    const int64_t packed_n = src0->ne[1];
    const int64_t N = dst->ne[0];
    const int64_t M = ggml_nrows(dst);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(N > 0 && N <= packed_n && ggml_nrows(dst) == M);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(K % 128 == 0 && packed_n % 128 == 0 && N % 128 == 0);
    GGML_ASSERT(x != nullptr && suh != nullptr && svh != nullptr);
    GGML_ASSERT(x_type == GGML_TYPE_F16 || x_type == GGML_TYPE_F32);
    (void) src1;

    const int bits = ggml_exl3_n_bits(src0->type);
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    ggml_cuda_pool_alloc<half> x_f16(ctx.pool());
    const half * xh = nullptr;
    if (x_type == GGML_TYPE_F16) {
        xh = (const half *) x;
    } else {
        const int64_t ne = M * K;
        GGML_ASSERT((ne & 7) == 0);
        x_f16.alloc((size_t) ne);
        exl3_cvt_f32_f16((const float *) x, x_f16.get(), ne, stream);
        xh = x_f16.get();
    }

    ggml_cuda_pool_alloc<half> w_f16(ctx.pool());
    w_f16.alloc((size_t) K * (size_t) N);
    exl3_reconstruct_had_prefix((const char *) src0->data, w_f16.get(), suh, svh,
            (int) K, (int) packed_n, (int) N, bits, stream);
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (int) N, (int) M, (int) K,
        &alpha,
        w_f16.get(), CUDA_R_16F, (int) N,
        xh,          CUDA_R_16F, (int) K,
        &beta,
        (float *) dst->data, CUDA_R_32F, (int) N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

#endif
