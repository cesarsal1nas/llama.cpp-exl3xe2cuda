#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 exl3_gemv_int8_sq_kernel
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// EXL3 int8-activation GEMV (ExLlamaV3 exl3_gemv_int8_sq_kernel, plain int8, F32 out).
// Input Hadamard + suh are optional (do_had). svh != null: output H128 * svh in the epilogue.

#include "exl3_dq.cuh"
#include "exl3_had.cuh"
#include "ptx.cuh"

#include <cstdlib>
#include <cuda_runtime.h>

#define EXL3_I8_THREADS 256
#define EXL3_I8_STAGE_D 4
#define EXL3_I8_SQ_KSPLIT_CAP 64
#define EXL3_I8_SQ_MINROWS 16
#define EXL3_I8_SQ_ROWS_MAX 512
#define EXL3_I8_SQ_COUNTERS_CAP 4096
#define EXL3_I8_SQ_WS_RESERVED (EXL3_I8_SQ_COUNTERS_CAP + 4 * EXL3_I8_SQ_KSPLIT_CAP * 8)

namespace exl3_i8 {

__device__ __forceinline__ int dp4a_us(uint32_t a, uint32_t b, int c) {
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

template <int bits>
__device__ __forceinline__ int wrap_idx(int i) {
    constexpr int words = bits * 256 / 32;
    return i >= words ? i - words : i;
}

template <int bits>
__device__ __forceinline__ void ext4w(const uint32_t * ptr, int t0,
                                     uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3) {
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + 3 * bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w3 = fshift(b, a, s2) & 0xffff;
    w2 = fshift(b, a, s2 + bits) & 0xffff;
    w1 = fshift(b, a, s2 + bits * 2) & 0xffff;
    w0 = fshift(b, a, s2 + bits * 3) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext2w(const uint32_t * ptr, int t0, uint32_t & w0, uint32_t & w1) {
    int b0 = (t0 + 257) * bits - 16;
    int b2 = b0 + bits + 16;
    int i0 = b0 / 32;
    int i2 = (b2 - 1) / 32;
    int s2 = (i2 + 1) * 32 - b2;
    uint32_t a = ptr[wrap_idx<bits>(i0)];
    uint32_t b = ptr[wrap_idx<bits>(i2)];
    w1 = fshift(b, a, s2) & 0xffff;
    w0 = fshift(b, a, s2 + bits) & 0xffff;
}

template <int bits>
__device__ __forceinline__ void ext8w(
        const uint32_t * ptr, int t0,
        uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3,
        uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    if constexpr (bits == 1) {
        uint32_t i1 = t0 >> 5;
        uint32_t i0 = (i1 + 7) & 7;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 24));
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 1);
        BFE16_IMM(w5, b, 2);
        BFE16_IMM(w4, b, 3);
        BFE16_IMM(w3, b, 4);
        BFE16_IMM(w2, b, 5);
        BFE16_IMM(w1, b, 6);
        BFE16_IMM(w0, b, 7);
    } else if constexpr (bits == 2) {
        uint32_t i1 = t0 >> 4;
        uint32_t i0 = (i1 + 15) & 15;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t0) & 8) << 1);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 2);
        BFE16_IMM(w5, b, 4);
        BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8);
        BFE16_IMM(w2, b, 10);
        BFE16_IMM(w1, b, 12);
        BFE16_IMM(w0, b, 14);
    } else if constexpr (bits == 3) {
        int b1 = (t0 + 257) * bits;
        int b0 = b1 - 16;
        int b2 = b1 + bits * 7;
        int i0 = b0 / 32;
        int i2 = (b2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = ptr[wrap_idx<bits>(i0)];
        uint32_t b = ptr[wrap_idx<bits>(i2)];
        w7 = fshift(b, a, s2);
        w6 = w7 >> bits;
        w5 = w6 >> bits;
        w4 = w5 >> bits;
        w3 = fshift(b, a, s2 + bits * 4);
        w2 = w3 >> bits;
        w1 = w2 >> bits;
        w0 = w1 >> bits;
        w7 &= 0xffff; w6 &= 0xffff; w5 &= 0xffff; w4 &= 0xffff;
        w3 &= 0xffff; w2 &= 0xffff; w1 &= 0xffff; w0 &= 0xffff;
    } else if constexpr (bits == 4) {
        uint32_t i1 = t0 >> 3;
        uint32_t i0 = (i1 + 31) & 31;
        uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        uint32_t s;
        FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 4);
        BFE16_IMM(w5, b, 8);
        BFE16_IMM(w4, b, 12);
        BFE16_IMM(w3, b, 16);
        w2 = s & 0xffff;
        BFE16_IMM(w1, s, 4);
        BFE16_IMM(w0, s, 8);
    } else if constexpr (bits == 7) {
        ext2w<bits>(ptr, t0,     w0, w1);
        ext2w<bits>(ptr, t0 + 2, w2, w3);
        ext2w<bits>(ptr, t0 + 4, w4, w5);
        ext2w<bits>(ptr, t0 + 6, w6, w7);
    } else {
        ext4w<bits>(ptr, t0,     w0, w1, w2, w3);
        ext4w<bits>(ptr, t0 + 4, w4, w5, w6, w7);
    }
}

__device__ __forceinline__ void extract8_4bits_words(uint32_t a, uint32_t b,
        uint32_t & w0, uint32_t & w1, uint32_t & w2, uint32_t & w3,
        uint32_t & w4, uint32_t & w5, uint32_t & w6, uint32_t & w7) {
    uint32_t s;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
}

__host__ __device__ constexpr bool stage_smem(int bits) {
    return bits == 3 || bits == 5 || bits == 7;
}

__host__ __device__ constexpr int sq_rows_max() {
    int cap = (80 * 1024) / (32 + 64);
    cap &= ~7;
    return cap < EXL3_I8_SQ_ROWS_MAX ? cap : EXL3_I8_SQ_ROWS_MAX;
}

__host__ __device__ constexpr int sq_rows_max_m2() {
    int cap = (80 * 1024) / (32 + 64 * 2);
    cap &= ~7;
    return cap < EXL3_I8_SQ_ROWS_MAX ? cap : EXL3_I8_SQ_ROWS_MAX;
}

__device__ __forceinline__ void stage_slice(
        const float * __restrict__ A,
        int size_k,
        const float * __restrict__ suh,
        bool do_had,
        float * __restrict__ qs,
        half * __restrict__ sh_ah,
        uint32_t * __restrict__ sh_as,
        float * __restrict__ sh_red,
        int kb0,
        int nrows) {
    const int t = threadIdx.x;
    const int nel = nrows * 16;
    __syncthreads();
    const int n128 = nel >> 7;
    for (int sp = t >> 5; sp < n128; sp += EXL3_I8_THREADS >> 5) {
        const float * xp = A + (kb0 << 4) + (sp << 7);
        const float * sptr = suh ? suh + (kb0 << 4) + (sp << 7) : nullptr;
        if (do_had) {
            exl3_had_warp_f32(xp, sptr, sh_ah + (sp << 7));
        } else {
            exl3_f32_to_f16_128(xp, sptr, sh_ah + (sp << 7));
        }
    }
    __syncthreads();

    float mx = 0.0f;
    for (int i = t; i < nel; i += EXL3_I8_THREADS) {
        mx = fmaxf(mx, fabsf(__half2float(sh_ah[i])));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    }
    if ((t & 31) == 0) {
        sh_red[t >> 5] = mx;
    }
    __syncthreads();
    if (t < 32) {
        float v = t < (EXL3_I8_THREADS >> 5) ? sh_red[t] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        }
        if (t == 0) {
            sh_red[32] = fmaxf(v, 1e-8f) / 127.0f;
        }
    }
    __syncthreads();
    const float q_s = sh_red[32];
    const float rq = 1.0f / q_s;
    int l1 = 0;
    for (int i = t; i < nel; i += EXL3_I8_THREADS) {
        float a = __half2float(sh_ah[i]);
        int v = __float2int_rn(a * rq);
        v = max(-127, min(127, v));
        sh_as[i] = ((uint32_t) (uint8_t) (int8_t) v) * 0x01010101u;
        l1 += v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        l1 += __shfl_xor_sync(0xffffffff, l1, o);
    }
    if ((t & 31) == 0) {
        ((int *) sh_red)[t >> 5] = l1;
    }
    __syncthreads();
    if (t < 32) {
        int v1 = t < (EXL3_I8_THREADS >> 5) ? ((int *) sh_red)[t] : 0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v1 += __shfl_xor_sync(0xffffffff, v1, o);
        }
        if (t == 0) {
            qs[0] = sh_red[32];
            ((int *) qs)[1] = v1;
            ((int *) qs)[2] = 0;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void unit_wide(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        int nb256, int kb0, int nrows, int packed_n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * 64 + 2 * lane;
    const int c2 = (lane & 1) ? 4 : 0;
    const int shfl_src = (lane & 16) | ((lane + 15) & 15);

    int iacc0 = 0, iacc1 = 0;
    uint2 r0 = *(const uint2 *) bp;
    uint2 r1 = {};
    if (nrows > 1) {
        r1 = *(const uint2 *) (bp + row_stride);
    }

    for (int kb = 0; kb < nrows; ++kb) {
        uint2 r2 = {};
        if (kb + 2 < nrows) {
            r2 = *(const uint2 *) (bp + (size_t) (kb + 2) * row_stride);
        }
        uint32_t prev = __shfl_sync(0xffffffff, r0.y, shfl_src);

        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
        extract8_4bits_words(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);
        extract8_4bits_words(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
        v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
        v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;

        const uint32_t * as = sh_as + (kb << 4);
        uint4 as0 = *(const uint4 *) (as + c2);
        uint4 as8 = *(const uint4 *) (as + c2 + 8);
        iacc0 = dp4a_us(w0, as0.x, iacc0);
        iacc0 = dp4a_us(w1, as0.y, iacc0);
        iacc0 = dp4a_us(w2, as8.x, iacc0);
        iacc0 = dp4a_us(w3, as8.y, iacc0);
        iacc1 = dp4a_us(w4, as0.x, iacc1);
        iacc1 = dp4a_us(w5, as0.y, iacc1);
        iacc1 = dp4a_us(w6, as8.x, iacc1);
        iacc1 = dp4a_us(w7, as8.y, iacc1);
        iacc0 = dp4a_us(v0, as0.z, iacc0);
        iacc0 = dp4a_us(v1, as0.w, iacc0);
        iacc0 = dp4a_us(v2, as8.z, iacc0);
        iacc0 = dp4a_us(v3, as8.w, iacc0);
        iacc1 = dp4a_us(v4, as0.z, iacc1);
        iacc1 = dp4a_us(v5, as0.w, iacc1);
        iacc1 = dp4a_us(v6, as8.z, iacc1);
        iacc1 = dp4a_us(v7, as8.w, iacc1);

        r0 = r1;
        r1 = r2;
    }

    iacc0 += __shfl_xor_sync(0xffffffff, iacc0, 1);
    iacc1 += __shfl_xor_sync(0xffffffff, iacc1, 1);
    if (!(lane & 1)) {
        const int nb = nbp * 2 + (lane >> 4);
        const int n0 = nb * 16 + ((lane & 15) >> 1);
        accs[n0] = iacc0;
        accs[n0 + 8] = iacc1;
    }
}

__device__ __forceinline__ void unit_wide_m2(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        int as_stride, int acc_stride,
        int nb256, int kb0, int nrows, int packed_n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * 64 + 2 * lane;
    const int c2 = (lane & 1) ? 4 : 0;
    const int shfl_src = (lane & 16) | ((lane + 15) & 15);

    int iacc0[2] = {}, iacc1[2] = {};
    uint2 r0 = *(const uint2 *) bp;
    uint2 r1 = {};
    if (nrows > 1) {
        r1 = *(const uint2 *) (bp + row_stride);
    }

    for (int kb = 0; kb < nrows; ++kb) {
        uint2 r2 = {};
        if (kb + 2 < nrows) {
            r2 = *(const uint2 *) (bp + (size_t) (kb + 2) * row_stride);
        }
        uint32_t prev = __shfl_sync(0xffffffff, r0.y, shfl_src);

        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
        extract8_4bits_words(prev, r0.x, w0, w1, w2, w3, w4, w5, w6, w7);
        extract8_4bits_words(r0.x, r0.y, v0, v1, v2, v3, v4, v5, v6, v7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
        v0 *= 0x83DCD12Du; v1 *= 0x83DCD12Du; v2 *= 0x83DCD12Du; v3 *= 0x83DCD12Du;
        v4 *= 0x83DCD12Du; v5 *= 0x83DCD12Du; v6 *= 0x83DCD12Du; v7 *= 0x83DCD12Du;

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t * as = sh_as + r * as_stride + (kb << 4);
            uint4 as0 = *(const uint4 *) (as + c2);
            uint4 as8 = *(const uint4 *) (as + c2 + 8);
            iacc0[r] = dp4a_us(w0, as0.x, iacc0[r]);
            iacc0[r] = dp4a_us(w1, as0.y, iacc0[r]);
            iacc0[r] = dp4a_us(w2, as8.x, iacc0[r]);
            iacc0[r] = dp4a_us(w3, as8.y, iacc0[r]);
            iacc1[r] = dp4a_us(w4, as0.x, iacc1[r]);
            iacc1[r] = dp4a_us(w5, as0.y, iacc1[r]);
            iacc1[r] = dp4a_us(w6, as8.x, iacc1[r]);
            iacc1[r] = dp4a_us(w7, as8.y, iacc1[r]);
            iacc0[r] = dp4a_us(v0, as0.z, iacc0[r]);
            iacc0[r] = dp4a_us(v1, as0.w, iacc0[r]);
            iacc0[r] = dp4a_us(v2, as8.z, iacc0[r]);
            iacc0[r] = dp4a_us(v3, as8.w, iacc0[r]);
            iacc1[r] = dp4a_us(v4, as0.z, iacc1[r]);
            iacc1[r] = dp4a_us(v5, as0.w, iacc1[r]);
            iacc1[r] = dp4a_us(v6, as8.z, iacc1[r]);
            iacc1[r] = dp4a_us(v7, as8.w, iacc1[r]);
        }

        r0 = r1;
        r1 = r2;
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        int a0 = iacc0[r] + __shfl_xor_sync(0xffffffff, iacc0[r], 1);
        int a1 = iacc1[r] + __shfl_xor_sync(0xffffffff, iacc1[r], 1);
        if (!(lane & 1)) {
            const int nb = nbp * 2 + (lane >> 4);
            const int n0 = nb * 16 + ((lane & 15) >> 1);
            accs[r * acc_stride + n0] = a0;
            accs[r * acc_stride + n0 + 8] = a1;
        }
    }
}

template <int bits>
__device__ __forceinline__ void pair_row(
        const uint32_t * blockA, const uint32_t * blockB,
        const uint32_t * as, int c2, int t0,
        int & ia0, int & ia1, int & ib0, int & ib1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    uint2 as01 = *(const uint2 *) (as + c2);
    uint2 as89 = *(const uint2 *) (as + c2 + 8);
    ia0 = dp4a_us(w0, as01.x, ia0);
    ia0 = dp4a_us(w1, as01.y, ia0);
    ia0 = dp4a_us(w2, as89.x, ia0);
    ia0 = dp4a_us(w3, as89.y, ia0);
    ia1 = dp4a_us(w4, as01.x, ia1);
    ia1 = dp4a_us(w5, as01.y, ia1);
    ia1 = dp4a_us(w6, as89.x, ia1);
    ia1 = dp4a_us(w7, as89.y, ia1);

    ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
    w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
    w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
    ib0 = dp4a_us(w0, as01.x, ib0);
    ib0 = dp4a_us(w1, as01.y, ib0);
    ib0 = dp4a_us(w2, as89.x, ib0);
    ib0 = dp4a_us(w3, as89.y, ib0);
    ib1 = dp4a_us(w4, as01.x, ib1);
    ib1 = dp4a_us(w5, as01.y, ib1);
    ib1 = dp4a_us(w6, as89.x, ib1);
    ib1 = dp4a_us(w7, as89.y, ib1);
}

__device__ __forceinline__ void pair_tail(
        int * __restrict__ accs, int nbp, int lane,
        int ia0, int ia1, int ib0, int ib1) {
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        ia0 += __shfl_xor_sync(0xffffffff, ia0, o);
        ia1 += __shfl_xor_sync(0xffffffff, ia1, o);
        ib0 += __shfl_xor_sync(0xffffffff, ib0, o);
        ib1 += __shfl_xor_sync(0xffffffff, ib1, o);
    }
    if ((lane & 3) == 0) {
        const int nA = (nbp * 2) * 16 + (lane >> 2);
        accs[nA] = ia0;
        accs[nA + 8] = ia1;
        accs[nA + 16] = ib0;
        accs[nA + 24] = ib1;
    }
}

template <int bits>
__device__ __forceinline__ void unit_narrow(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        int nb256, int kb0, int nrows, int packed_n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * bits / 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * (bits * 16);
    const int c2 = 2 * (lane & 3);
    int ia0 = 0, ia1 = 0, ib0 = 0, ib1 = 0;
    for (int kb = 0; kb < nrows; ++kb) {
        const uint32_t * blockA = bp + (size_t) kb * row_stride;
        pair_row<bits>(blockA, blockA + 8 * bits, sh_as + (kb << 4), c2, lane << 3,
                       ia0, ia1, ib0, ib1);
    }
    pair_tail(accs, nbp, lane, ia0, ia1, ib0, ib1);
}

template <int bits>
__device__ __forceinline__ void unit_narrow_m2(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        int as_stride, int acc_stride,
        int nb256, int kb0, int nrows, int packed_n) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * bits / 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * (bits * 16);
    const int c2 = 2 * (lane & 3);
    const int t0 = lane << 3;
    int ia0[2] = {}, ia1[2] = {}, ib0[2] = {}, ib1[2] = {};
    for (int kb = 0; kb < nrows; ++kb) {
        const uint32_t * blockA = bp + (size_t) kb * row_stride;
        const uint32_t * blockB = blockA + 8 * bits;
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t * as = sh_as + r * as_stride + (kb << 4);
            uint2 as01 = *(const uint2 *) (as + c2);
            uint2 as89 = *(const uint2 *) (as + c2 + 8);
            ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
            ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
            ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
            ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
            ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
            ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
            ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
            ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
        }
        ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t * as = sh_as + r * as_stride + (kb << 4);
            uint2 as01 = *(const uint2 *) (as + c2);
            uint2 as89 = *(const uint2 *) (as + c2 + 8);
            ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
            ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
            ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
            ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
            ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
            ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
            ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
            ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        pair_tail(accs + r * acc_stride, nbp, lane, ia0[r], ia1[r], ib0[r], ib1[r]);
    }
}

template <int bits>
__device__ __forceinline__ void unit_smem(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        uint32_t * __restrict__ sh_b,
        int nb256, int kb0, int nrows, int packed_n) {
    constexpr int D = EXL3_I8_STAGE_D;
    constexpr int pairwords = 16 * bits;
    constexpr int chunks = pairwords / 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * bits / 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t * sb = sh_b + warp * (D * pairwords);

    auto stage_row = [&](int kb) {
        if (kb < nrows && lane < chunks) {
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        }
        cp_async_fence();
    };
#pragma unroll
    for (int r = 0; r < D - 1; ++r) {
        stage_row(r);
    }

    const int c2 = 2 * (lane & 3);
    int ia0 = 0, ia1 = 0, ib0 = 0, ib1 = 0;
    for (int kb = 0; kb < nrows; ++kb) {
        cp_async_wait<D - 2>();
        __syncwarp();
        stage_row(kb + D - 1);
        const uint32_t * blockA = sb + (kb % D) * pairwords;
        pair_row<bits>(blockA, blockA + 8 * bits, sh_as + (kb << 4), c2, lane << 3,
                       ia0, ia1, ib0, ib1);
    }
    pair_tail(accs, nbp, lane, ia0, ia1, ib0, ib1);
}

template <int bits>
__device__ __forceinline__ void unit_smem_m2(
        const uint16_t * __restrict__ B,
        int * __restrict__ accs,
        const uint32_t * __restrict__ sh_as,
        int as_stride, int acc_stride,
        uint32_t * __restrict__ sh_b,
        int nb256, int kb0, int nrows, int packed_n) {
    constexpr int D = EXL3_I8_STAGE_D;
    constexpr int pairwords = 16 * bits;
    constexpr int chunks = pairwords / 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nbp = nb256 * 8 + warp;
    const int row_stride = packed_n * bits / 2;
    const uint32_t * bp = ((const uint32_t *) B) + (size_t) kb0 * row_stride + (size_t) nbp * pairwords;
    uint32_t * sb = sh_b + warp * (D * pairwords);
    auto stage_row = [&](int kb) {
        if (kb < nrows && lane < chunks) {
            cp_async(sb + (kb % D) * pairwords + lane * 4, bp + (size_t) kb * row_stride + lane * 4);
        }
        cp_async_fence();
    };
#pragma unroll
    for (int r = 0; r < D - 1; ++r) {
        stage_row(r);
    }
    const int c2 = 2 * (lane & 3);
    const int t0 = lane << 3;
    int ia0[2] = {}, ia1[2] = {}, ib0[2] = {}, ib1[2] = {};
    for (int kb = 0; kb < nrows; ++kb) {
        cp_async_wait<D - 2>();
        __syncwarp();
        stage_row(kb + D - 1);
        const uint32_t * blockA = sb + (kb % D) * pairwords;
        const uint32_t * blockB = blockA + 8 * bits;
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        ext8w<bits>(blockA, t0, w0, w1, w2, w3, w4, w5, w6, w7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t * as = sh_as + r * as_stride + (kb << 4);
            uint2 as01 = *(const uint2 *) (as + c2);
            uint2 as89 = *(const uint2 *) (as + c2 + 8);
            ia0[r] = dp4a_us(w0, as01.x, ia0[r]);
            ia0[r] = dp4a_us(w1, as01.y, ia0[r]);
            ia0[r] = dp4a_us(w2, as89.x, ia0[r]);
            ia0[r] = dp4a_us(w3, as89.y, ia0[r]);
            ia1[r] = dp4a_us(w4, as01.x, ia1[r]);
            ia1[r] = dp4a_us(w5, as01.y, ia1[r]);
            ia1[r] = dp4a_us(w6, as89.x, ia1[r]);
            ia1[r] = dp4a_us(w7, as89.y, ia1[r]);
        }
        ext8w<bits>(blockB, t0, w0, w1, w2, w3, w4, w5, w6, w7);
        w0 *= 0x83DCD12Du; w1 *= 0x83DCD12Du; w2 *= 0x83DCD12Du; w3 *= 0x83DCD12Du;
        w4 *= 0x83DCD12Du; w5 *= 0x83DCD12Du; w6 *= 0x83DCD12Du; w7 *= 0x83DCD12Du;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const uint32_t * as = sh_as + r * as_stride + (kb << 4);
            uint2 as01 = *(const uint2 *) (as + c2);
            uint2 as89 = *(const uint2 *) (as + c2 + 8);
            ib0[r] = dp4a_us(w0, as01.x, ib0[r]);
            ib0[r] = dp4a_us(w1, as01.y, ib0[r]);
            ib0[r] = dp4a_us(w2, as89.x, ib0[r]);
            ib0[r] = dp4a_us(w3, as89.y, ib0[r]);
            ib1[r] = dp4a_us(w4, as01.x, ib1[r]);
            ib1[r] = dp4a_us(w5, as01.y, ib1[r]);
            ib1[r] = dp4a_us(w6, as89.x, ib1[r]);
            ib1[r] = dp4a_us(w7, as89.y, ib1[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        pair_tail(accs + r * acc_stride, nbp, lane, ia0[r], ia1[r], ib0[r], ib1[r]);
    }
}

__device__ __forceinline__ void epilogue_group(
        const int * __restrict__ partials,
        const float * __restrict__ qsums,
        int pstride, int ksplit,
        float * __restrict__ C,
        const float * __restrict__ svh,
        float * __restrict__ sh_had,
        int nb256, int size_n, int qstride = 4) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= 2) {
        return;
    }
    const float k_inv  = __half2float(__ushort_as_half(0x1eee));
    const float k_bias = __half2float(__ushort_as_half(0xc931));
    const float aff = 1024.0f * k_inv + k_bias;
    const int base = nb256 * 256 + (warp & 1) * 128;
    float acc[4] = {};
    float corr = 0.0f;
    for (int sl = 0; sl < ksplit; ++sl) {
        const float q_s = qsums[qstride * sl];
        corr += aff * (q_s * (float) ((const int *) qsums)[qstride * sl + 1]);
        const int * p = partials + (size_t) sl * pstride + base;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i] += q_s * (float) __ldcg(p + lane * 4 + i);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc[i] = k_inv * acc[i] + corr;
    }
    if (!svh) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            C[base + lane * 4 + i] = acc[i];
        }
        return;
    }
    float * tmp = sh_had + warp * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        tmp[lane * 4 + i] = acc[i];
    }
    __syncwarp();
    exl3_had_out_f32(tmp, C + base, svh + base);
}

template <int bits, bool DO_HAD>
__global__ __launch_bounds__(EXL3_I8_THREADS)
void exl3_gemv_int8_sq_kernel(
        const float * __restrict__ A,
        const uint16_t * __restrict__ B,
        float * __restrict__ C,
        int size_k,
        int size_n,
        int packed_n,
        int * __restrict__ locks,
        const float * __restrict__ suh,
        const float * __restrict__ svh) {
    extern __shared__ uint32_t shmem[];

    const int rows_total = size_k >> 4;
    const int nb256_total = size_n / 256;
    int r = (rows_total * nb256_total + (int) gridDim.x - 1) / (int) gridDim.x;
    int rows_per = (max(r, min(2 * r, 32)) + 7) & ~7;
    rows_per = max(rows_per, EXL3_I8_SQ_MINROWS);
    rows_per = min(rows_per, sq_rows_max());
    rows_per = min(rows_per, (rows_total + 7) & ~7);
#ifdef EXL3_I8_ONESLICE
    rows_per = rows_total;
#endif
    const int ksplit = (rows_total + rows_per - 1) / rows_per;
    const int units = nb256_total * ksplit;
    const int pstride = size_n;

    int * counters = locks;
    float * qsums = (float *) (locks + EXL3_I8_SQ_COUNTERS_CAP);
    int * partials = locks + EXL3_I8_SQ_WS_RESERVED;

    half * sh_ah = (half *) shmem;
    uint32_t * sh_as = shmem + rows_per * 8;
    uint32_t * sh_b = sh_as + rows_per * 16;
    float * sh_had = (float *) (sh_b + (stage_smem(bits) ? 8 * EXL3_I8_STAGE_D * 16 * bits : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    const int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = (int) blockIdx.x; unit < units; unit += (int) gridDim.x) {
        const int slice = unit / nb256_total;
        const int nb256 = unit % nb256_total;
        const int kb0 = slice * rows_per;
        const int nrows = min(rows_per, rows_total - kb0);
        if (slice != prev_slice) {
            stage_slice(A, size_k, suh, DO_HAD, qsums + 4 * slice,
                        sh_ah, sh_as, sh_red, kb0, nrows);
            prev_slice = slice;
        }
        int * pacc = partials + (size_t) slice * pstride;
        if constexpr (bits == 4) {
            unit_wide(B, pacc, sh_as, nb256, kb0, nrows, packed_n);
        } else if constexpr (stage_smem(bits)) {
            unit_smem<bits>(B, pacc, sh_as, sh_b, nb256, kb0, nrows, packed_n);
        } else {
            unit_narrow<bits>(B, pacc, sh_as, nb256, kb0, nrows, packed_n);
        }

        __threadfence();
        __syncthreads();
        if (t == 0) {
            sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        }
        __syncthreads();
        if (sh_last) {
            epilogue_group(partials, qsums, pstride, ksplit, C, svh, sh_had, nb256, size_n);
            if (t == 0) {
                counters[nb256] = 0;
            }
        }
    }
}

static size_t smem_for(int bits, int rows_per) {
    size_t stage = stage_smem(bits) ? (size_t) 8 * EXL3_I8_STAGE_D * 16 * bits * 4 : 0;
    return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 + stage + (size_t) 2 * 128 * 4;
}

static int hmax(int a, int b) { return a > b ? a : b; }
static int hmin(int a, int b) { return a < b ? a : b; }

static void decomp(int grid, int size_k, int size_n, int & ksplit, int & rows_per) {
    const int rows_total = size_k / 16;
    const int nb256 = size_n / 256;
    int r = (rows_total * nb256 + grid - 1) / grid;
    rows_per = (hmax(r, hmin(2 * r, 32)) + 7) & ~7;
    rows_per = hmax(rows_per, EXL3_I8_SQ_MINROWS);
    rows_per = hmin(rows_per, sq_rows_max());
    rows_per = hmin(rows_per, (rows_total + 7) & ~7);
    ksplit = (rows_total + rows_per - 1) / rows_per;
#ifdef EXL3_I8_ONESLICE
    rows_per = rows_total;
    ksplit = 1;
#endif
}

template <int bits, bool DO_HAD>
static bool launch_sq(
        const float * A, const uint16_t * B, float * C,
        int size_k, int size_n, int packed_n, const float * suh, const float * svh,
        int * ws, int nsm, cudaStream_t stream) {
    auto * fn = (void *) exl3_gemv_int8_sq_kernel<bits, DO_HAD>;
    const int rows_max = sq_rows_max();
    static bool attr_set[9][2] = {};
    if (!attr_set[bits][DO_HAD ? 1 : 0]) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(bits, rows_max));
        cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        attr_set[bits][DO_HAD ? 1 : 0] = true;
    }

    int ksplit, rows_per;
    decomp(6 * nsm, size_k, size_n, ksplit, rows_per);
    size_t smem_guess = smem_for(bits, rows_per);
    int maxb = 1;
    struct occ_ent { void * fn; size_t smem; int maxb; };
    static occ_ent occ_cache[32];
    static int occ_n = 0;
    bool occ_hit = false;
    for (int i = 0; i < occ_n; ++i) {
        if (occ_cache[i].fn == fn && occ_cache[i].smem == smem_guess) {
            maxb = occ_cache[i].maxb;
            occ_hit = true;
            break;
        }
    }
    if (!occ_hit) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, EXL3_I8_THREADS, smem_guess);
        if (occ_n < 32) {
            occ_cache[occ_n++] = { fn, smem_guess, maxb };
        }
    }
    int grid = hmin(hmax(maxb, 1) * nsm, 1024);
    decomp(grid, size_k, size_n, ksplit, rows_per);
    size_t smem = smem_for(bits, rows_per);
    if (ksplit > EXL3_I8_SQ_KSPLIT_CAP) {
        return false;
    }
    if (size_n / 256 > EXL3_I8_SQ_COUNTERS_CAP) {
        return false;
    }

    void * args[] = {
        (void *) &A, (void *) &B, (void *) &C,
        (void *) &size_k, (void *) &size_n, (void *) &packed_n,
        (void *) &ws, (void *) &suh, (void *) &svh
    };
    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(EXL3_I8_THREADS), args, smem, stream);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

template <int bits, bool DO_HAD>
__global__ __launch_bounds__(EXL3_I8_THREADS)
void exl3_gemv_int8_sq_kernel_m2(
        const float * __restrict__ A,
        const uint16_t * __restrict__ B,
        float * __restrict__ C,
        int size_k,
        int size_n,
        int packed_n,
        int * __restrict__ locks,
        const float * __restrict__ suh,
        const float * __restrict__ svh) {
    extern __shared__ uint32_t shmem[];

    const int rows_total = size_k >> 4;
    const int nb256_total = size_n / 256;
    int rdiv = (rows_total * nb256_total + (int) gridDim.x - 1) / (int) gridDim.x;
    int rows_per = (max(rdiv, min(2 * rdiv, 32)) + 7) & ~7;
    rows_per = max(rows_per, EXL3_I8_SQ_MINROWS);
    rows_per = min(rows_per, sq_rows_max_m2());
    rows_per = min(rows_per, (rows_total + 7) & ~7);
    const int ksplit = (rows_total + rows_per - 1) / rows_per;
    const int units = nb256_total * ksplit;
    const int pstride = size_n;
    const int as_stride = rows_per * 16;

    int * counters = locks;
    float * qsums = (float *) (locks + EXL3_I8_SQ_COUNTERS_CAP);
    int * partials = locks + EXL3_I8_SQ_WS_RESERVED;

    half * sh_ah = (half *) shmem;
    uint32_t * sh_as = shmem + rows_per * 8;
    uint32_t * sh_b = sh_as + 2 * as_stride;
    float * sh_had = (float *) (sh_b + (stage_smem(bits) ? 8 * EXL3_I8_STAGE_D * 16 * bits : 0));
    __shared__ float sh_red[33];
    __shared__ int sh_last;

    const int t = threadIdx.x;
    int prev_slice = -1;
    for (int unit = (int) blockIdx.x; unit < units; unit += (int) gridDim.x) {
        const int slice = unit / nb256_total;
        const int nb256 = unit % nb256_total;
        const int kb0 = slice * rows_per;
        const int nrows = min(rows_per, rows_total - kb0);
        if (slice != prev_slice) {
            for (int row = 0; row < 2; ++row) {
                stage_slice(A + (size_t) row * size_k, size_k, suh, DO_HAD,
                            qsums + 4 * (slice * 2 + row),
                            sh_ah, sh_as + row * as_stride, sh_red, kb0, nrows);
            }
            prev_slice = slice;
        }
        int * pacc = partials + (size_t) slice * 2 * pstride;
        if constexpr (bits == 4) {
            unit_wide_m2(B, pacc, sh_as, as_stride, pstride, nb256, kb0, nrows, packed_n);
        } else if constexpr (stage_smem(bits)) {
            unit_smem_m2<bits>(B, pacc, sh_as, as_stride, pstride, sh_b, nb256, kb0, nrows, packed_n);
        } else {
            unit_narrow_m2<bits>(B, pacc, sh_as, as_stride, pstride, nb256, kb0, nrows, packed_n);
        }

        __threadfence();
        __syncthreads();
        if (t == 0) {
            sh_last = (atomicAdd(&counters[nb256], 1) == ksplit - 1) ? 1 : 0;
        }
        __syncthreads();
        if (sh_last) {
            for (int row = 0; row < 2; ++row) {
                epilogue_group(partials + row * pstride, qsums + 4 * row, 2 * pstride, ksplit,
                               C + (size_t) row * size_n, svh, sh_had, nb256, size_n, 8);
            }
            if (t == 0) {
                counters[nb256] = 0;
            }
        }
    }
}

static size_t smem_for_m2(int bits, int rows_per) {
    size_t stage = stage_smem(bits) ? (size_t) 8 * EXL3_I8_STAGE_D * 16 * bits * 4 : 0;
    return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * 2 + stage + (size_t) 4 * 128 * 4;
}

template <int bits, bool DO_HAD>
static bool launch_sq_m2(
        const float * A, const uint16_t * B, float * C,
        int size_k, int size_n, int packed_n, const float * suh, const float * svh,
        int * ws, int nsm, cudaStream_t stream) {
    auto * fn = (void *) exl3_gemv_int8_sq_kernel_m2<bits, DO_HAD>;
    const int rows_max = sq_rows_max_m2();
    static bool attr_set[9][2] = {};
    if (!attr_set[bits][DO_HAD ? 1 : 0]) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for_m2(bits, rows_max));
        cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        attr_set[bits][DO_HAD ? 1 : 0] = true;
    }

    int ksplit, rows_per;
    const int rows_total = size_k / 16;
    const int nb256 = size_n / 256;
    auto decomp_m2 = [&](int grid) {
        int rr = (rows_total * nb256 + grid - 1) / grid;
        rows_per = (hmax(rr, hmin(2 * rr, 32)) + 7) & ~7;
        rows_per = hmax(rows_per, EXL3_I8_SQ_MINROWS);
        rows_per = hmin(rows_per, rows_max);
        rows_per = hmin(rows_per, (rows_total + 7) & ~7);
        ksplit = (rows_total + rows_per - 1) / rows_per;
    };
    decomp_m2(6 * nsm);
    size_t smem_guess = smem_for_m2(bits, rows_per);
    int maxb = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, EXL3_I8_THREADS, smem_guess);
    int grid = hmin(hmax(maxb, 1) * nsm, 1024);
    decomp_m2(grid);
    size_t smem = smem_for_m2(bits, rows_per);
    if (ksplit > EXL3_I8_SQ_KSPLIT_CAP) {
        return false;
    }
    if (size_n / 256 > EXL3_I8_SQ_COUNTERS_CAP) {
        return false;
    }

    void * args[] = {
        (void *) &A, (void *) &B, (void *) &C,
        (void *) &size_k, (void *) &size_n, (void *) &packed_n,
        (void *) &ws, (void *) &suh, (void *) &svh
    };
    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(EXL3_I8_THREADS), args, smem, stream);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

// Fixed 16 MB workspace, zeroed once. Finishing blocks reset counters, so mixed
// shapes can share it without a per-launch memset (ExL gemv_int8_get_ws).
static int * persist_ws = nullptr;

static int * ensure_ws(int * ws, size_t need_ints) {
    (void) ws;
    const size_t cap = (size_t) (16 << 20) / sizeof(int);
    if (need_ints > cap) {
        return nullptr;
    }
    if (!persist_ws) {
        if (cudaMalloc(&persist_ws, (size_t) (16 << 20)) != cudaSuccess) {
            persist_ws = nullptr;
            return nullptr;
        }
        cudaMemset(persist_ws, 0, (size_t) (16 << 20));
    }
    return persist_ws;
}

} // namespace exl3_i8

static int exl3_gemv_int8_max_k() {
    static const int env = [] {
        const char * e = getenv("GGML_EXL3_INT8_MAX_K");
        return e ? atoi(e) : 0;
    }();
    if (env) {
        return env < 8 ? env : 8;
    }
    // 6-bit INT8 beats leftover fp16+post-had; ExL Ada keeps 5 vs their fp16 kernel
    return 6;
}

static bool exl3_gemv_int8_shape_ok(int size_k, int size_n, int bits) {
    if (size_n % 256 || size_k % 128) {
        return false;
    }
    if (bits < 1 || bits > exl3_gemv_int8_max_k()) {
        return false;
    }
    return true;
}

static bool exl3_gemv_int8_cuda(
        const float * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int bits,
        const float * suh, bool do_had, const float * svh,
        int * ws, int nsm, cudaStream_t stream,
        int packed_n = 0) {
    if (packed_n <= 0) {
        packed_n = size_n;
    }
    if (!exl3_gemv_int8_shape_ok(size_k, size_n, bits) || packed_n % 256) {
        return false;
    }
    int ksplit, rows_per;
    exl3_i8::decomp(nsm < 512 ? 2 * nsm : 1024, size_k, size_n, ksplit, rows_per);
    const size_t need = (size_t) EXL3_I8_SQ_WS_RESERVED + (size_t) ksplit * (size_m > 1 ? 2 : 1) * size_n;
    ws = exl3_i8::ensure_ws(ws, need);
    if (!ws) {
        return false;
    }

    if (size_m == 2) {
        if (do_had) {
            switch (bits) {
                case 1: return exl3_i8::launch_sq_m2<1, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 2: return exl3_i8::launch_sq_m2<2, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 3: return exl3_i8::launch_sq_m2<3, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 4: return exl3_i8::launch_sq_m2<4, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 5: return exl3_i8::launch_sq_m2<5, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 6: return exl3_i8::launch_sq_m2<6, true>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                default: return false;
            }
        }
        switch (bits) {
            case 1: return exl3_i8::launch_sq_m2<1, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 2: return exl3_i8::launch_sq_m2<2, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 3: return exl3_i8::launch_sq_m2<3, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 4: return exl3_i8::launch_sq_m2<4, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 5: return exl3_i8::launch_sq_m2<5, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 6: return exl3_i8::launch_sq_m2<6, false>(A, B, C, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            default: return false;
        }
    }

    auto one_row = [&](const float * Ar, float * Cr) -> bool {
        if (do_had) {
            switch (bits) {
                case 1: return exl3_i8::launch_sq<1, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 2: return exl3_i8::launch_sq<2, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 3: return exl3_i8::launch_sq<3, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 4: return exl3_i8::launch_sq<4, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 5: return exl3_i8::launch_sq<5, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                case 6: return exl3_i8::launch_sq<6, true>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
                default: return false;
            }
        }
        switch (bits) {
            case 1: return exl3_i8::launch_sq<1, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 2: return exl3_i8::launch_sq<2, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 3: return exl3_i8::launch_sq<3, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 4: return exl3_i8::launch_sq<4, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 5: return exl3_i8::launch_sq<5, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            case 6: return exl3_i8::launch_sq<6, false>(Ar, B, Cr, size_k, size_n, packed_n, suh, svh, ws, nsm, stream);
            default: return false;
        }
    };

    for (int r = 0; r < size_m; ++r) {
        if (!one_row(A + (size_t) r * size_k, C + (size_t) r * size_n)) {
            return false;
        }
    }
    return true;
}
