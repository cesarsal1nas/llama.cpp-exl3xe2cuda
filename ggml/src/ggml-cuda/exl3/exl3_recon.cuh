#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 reconstruct.cu / reconstruct_had_kernel
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// One-launch reconstruct. 8 tiles/block, coalesced int4 stores.
// exl3_reconstruct_full: dst F16 [K,N] ld=N (ExL layout; required to coalesce).
// FragB -> (k,n) matches the slice kernel. mul1 codebook (cb=2).
// cublas: OP_N,OP_N, N,M,K, w ld=N, x ld=K, dst ld=N.

#include "exl3_dq.cuh"

#define EXL3_RECON_THREADS 256

// ExLlamaV3 AUTO_RECONSTRUCT_THRESHOLD: gemm/gemv when M <= this, else reconstruct.
#ifndef EXL3_RECON_FULL_MIN_M
#define EXL3_RECON_FULL_MIN_M 192
#endif

template <int Kbits, int cb>
__global__ __launch_bounds__(EXL3_RECON_THREADS)
void exl3_reconstruct_full_kernel(
        half * __restrict__ g_unpacked,
        const char * __restrict__ packed,
        int packed_blocks_n) {

    constexpr int packed_size = 256 * Kbits / 16;

    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int k = blockIdx.y;
    const int n = blockIdx.x * 8;
    const int tiles_n = gridDim.x;
    const int out_blocks_n = tiles_n * 8;

    __shared__ uint32_t s_packed[8][packed_size / 2];

    const uint16_t * g_packed = (const uint16_t *) packed;
    g_packed += ((size_t) k * (size_t) packed_blocks_n + (size_t) n) * (size_t) packed_size;
    if (t < packed_size) {
        ((int4 *) s_packed)[t] = ((const int4 *) g_packed)[t];
    }
    __syncthreads();

    FragB frag[2];
    dq_dispatch<Kbits, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ half2 tile[16][8][8];

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
        const int r0 = (lane_id % 4) * 2;
        const int r1 = r0 + 1;
        const int r2 = r0 + 8;
        const int r3 = r0 + 9;
        const int c0 = lane_id / 8;
        const int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    const int r = t / 16;
    const int c = t % 16;
    int4 * tile_int4 = reinterpret_cast<int4 *>(tile);
    int4 * out_int4 = ((int4 *) g_unpacked) + ((size_t) (k * 16 + r) * 2 * (size_t) out_blocks_n + n * 2 + c);
    *out_int4 = tile_int4[t];
}

// Same decode/shuffle as full/slice, stores F16 [N,K] ld=K (old slice dest).
// Vectorized 16-byte stores along K. Warp stores are still strided by K.
template <int Kbits, int cb>
__global__ __launch_bounds__(EXL3_RECON_THREADS)
void exl3_reconstruct_full_nk_kernel(
        half * __restrict__ g_unpacked,
        const char * __restrict__ packed,
        int packed_blocks_n,
        int K) {

    constexpr int packed_size = 256 * Kbits / 16;

    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int k = blockIdx.y;
    const int n = blockIdx.x * 8;

    __shared__ uint32_t s_packed[8][packed_size / 2];

    const uint16_t * g_packed = (const uint16_t *) packed;
    g_packed += ((size_t) k * (size_t) packed_blocks_n + (size_t) n) * (size_t) packed_size;
    if (t < packed_size) {
        ((int4 *) s_packed)[t] = ((const int4 *) g_packed)[t];
    }
    __syncthreads();

    FragB frag[2];
    dq_dispatch<Kbits, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    __shared__ half2 tile[16][8][8];

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
        const int r0 = (lane_id % 4) * 2;
        const int r1 = r0 + 1;
        const int r2 = r0 + 8;
        const int r3 = r0 + 9;
        const int c0 = lane_id / 8;
        const int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    const int n_local = t / 2;
    const int k8 = t % 2;
    const int n_tile = n_local / 16;
    const int n_in = n_local % 16;
    const int c = n_in / 2;
    const int hi = n_in % 2;

    half tmp[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        half2 v = tile[k8 * 8 + i][n_tile][c];
        tmp[i] = hi ? __high2half(v) : __low2half(v);
    }
    const int n_abs = n * 16 + n_local;
    ((int4 *) (g_unpacked + (size_t) n_abs * (size_t) K + (size_t) k * 16))[k8] =
        *reinterpret_cast<int4 *>(tmp);
}

static void exl3_reconstruct_full_launch(
        const char * blocks, half * dst, int K, int N, int bits, cudaStream_t stream, bool dest_nk) {
    if (bits < 1 || bits > 8 || K <= 0 || N <= 0) {
        return;
    }
    const int n_tiles = N / 16;
    dim3 block(EXL3_RECON_THREADS);
    dim3 grid((unsigned) (N / 128), (unsigned) (K / 16));

#define EXL3_RECON_CASE(b) \
    case b: \
        if (dest_nk) { \
            exl3_reconstruct_full_nk_kernel<b, 2><<<grid, block, 0, stream>>>(dst, blocks, n_tiles, K); \
        } else { \
            exl3_reconstruct_full_kernel<b, 2><<<grid, block, 0, stream>>>(dst, blocks, n_tiles); \
        } \
        break;

    switch (bits) {
        EXL3_RECON_CASE(1)
        EXL3_RECON_CASE(2)
        EXL3_RECON_CASE(3)
        EXL3_RECON_CASE(4)
        EXL3_RECON_CASE(5)
        EXL3_RECON_CASE(6)
        EXL3_RECON_CASE(7)
        EXL3_RECON_CASE(8)
        default: break;
    }
#undef EXL3_RECON_CASE
}

// Reconstruct whole weight. dst is F16 [K, N], ld = N.
static void exl3_reconstruct_full(
        const char * blocks, half * dst, int K, int N, int bits, cudaStream_t stream) {
    exl3_reconstruct_full_launch(blocks, dst, K, N, bits, stream, false);
}

// Same values, dest F16 [N, K], ld = K (slice-kernel layout).
static void exl3_reconstruct_full_nk(
        const char * blocks, half * dst, int K, int N, int bits, cudaStream_t stream) {
    exl3_reconstruct_full_launch(blocks, dst, K, N, bits, stream, true);
}

#ifndef EXL3_RECON_STANDALONE

#include "../common.cuh"

static void ggml_cuda_mul_mat_exl3_recon_full(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        const half * x_f16) {
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = ggml_nrows(dst);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == N && ggml_nrows(dst) == M);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(K % 128 == 0 && N % 128 == 0);
    GGML_ASSERT(x_f16 != nullptr);
    (void) src1;

    const int bits = ggml_exl3_n_bits(src0->type);
    cudaStream_t stream = ctx.stream();
    cublasHandle_t handle = ctx.cublas_handle();

    ggml_cuda_pool_alloc<half> w_f16(ctx.pool());
    w_f16.alloc((size_t) K * (size_t) N);
    exl3_reconstruct_full((const char *) src0->data, w_f16.get(), (int) K, (int) N, bits, stream);
    CUDA_CHECK(cudaGetLastError());

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (int) N, (int) M, (int) K,
        &alpha,
        w_f16.get(), CUDA_R_16F, (int) N,
        x_f16,       CUDA_R_16F, (int) K,
        &beta,
        (float *) dst->data, CUDA_R_32F, (int) N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

#endif
