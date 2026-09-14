#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 exl3_gemm_inner.cuh
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// Tensor-core EXL3 GEMM from ExLlamaV3 exl3_gemm_inner.cuh.
// Skips Hadamard / suh / svh (graph applies those). Always writes float C.
// A is row-major [M,K] half (ggml x[K,T] F16). C is row-major [M,N] float (ggml dst[N,T]).

#include "exl3_dq.cuh"
#include "exl3_had.cuh"

#include <cstdint>
#include <cstdlib>

#define EXL3_GEMM_BASE_THREADS 256
#define EXL3_GEMM_SMEM_MAX     (90 * 1024)

#define EXL3_GEMM_MIN(a, b) ((a) < (b) ? (a) : (b))
#define EXL3_GEMM_CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ExLlamaV3 shapes used on Ada (sm_89) for this model:
//   shape 2: Ktile=32 Ntile=128 stages=4/3  block=512
//   shape 3: Ktile=32 Ntile=256 stages=4/3  block=512
#define EXL3_GEMM_SHAPE2  32, 128, 4, 3
#define EXL3_GEMM_SHAPE2_S3  32, 128, 3, 3
#define EXL3_GEMM_SHAPE3  32, 256, 4, 3

// One 16-row tile. A/B/C already point at this tile. gridDim.x is the K/N split.
template <int bits, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES, int CB>
__device__ void exl3_gemm_inner(
        const half * __restrict__ A,
        const uint16_t * __restrict__ B,
        float * __restrict__ C,
        const int size_m,
        const int size_k,
        const int size_n,
        int * __restrict__ locks,
        const int32_t * __restrict__ ids_dst,
        const float * __restrict__ svh,
        float * __restrict__ dst,
        const int row_base,
        const int sk0 = 0,
        const int sk1 = 0,
        const int sadd = 0) {
    constexpr int TILESIZE_M = 16;
    constexpr int TILEBLOCKS_M = TILESIZE_M / 16;
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    constexpr int sh_a_stage_size = TILESIZE_M * TILESIZE_K;
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    constexpr int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP;

    constexpr int A_COLS = TILESIZE_K / 8;
    constexpr int A_SWIZZLE_MASK = A_COLS - 1;
    constexpr int A_SWIZZLE_SHIFT = (A_COLS <= 2) ? 2 : 1;

    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_K % 16 == 0);
    static_assert(TILESIZE_N % 128 == 0);
    static_assert((SH_STAGES == 3 || SH_STAGES == 4) && FRAG_STAGES == 3);
    static_assert(TILEBLOCKS_K == 2);
    static_assert(EXL3_GEMM_SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size);

    extern __shared__ half shared[];
    half * sh_a = shared;
    uint16_t * sh_b = (uint16_t *) (sh_a + SH_STAGES * sh_a_stage_size);
    float * sh_c = (float *) (sh_b + sh_b_stage_size * SH_STAGES);

    __syncthreads();
    const int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    const int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_n = size_n / TILESIZE_N;
    const int blocks_n = tiles_n * TILEBLOCKS_N;

    const int num_slices = gridDim.x;
    const int slice_beg = tiles_k * tiles_n * (int) blockIdx.x / num_slices;
    const int slice_end = tiles_k * tiles_n * ((int) blockIdx.x + 1) / num_slices;
    const int slice_len = slice_end - slice_beg;
    if (slice_len < 1) {
        return;
    }

    const half * A_p = A;
    const uint16_t * B_p = B;
    float * C_p = C;
    const float * svh_e = svh;
    const int slice_m = 0;
    const int size_m_use = size_m;
    if (size_m_use <= 0) {
        return;
    }

    auto index_k = [&](int slice_i) { return slice_i % tiles_k; };
    auto index_n = [&](int slice_i) { return slice_i / tiles_k; };

    int slice0_k = index_k(slice_beg);
    int slice0_n = index_n(slice_beg);
    int slice0_iters = slice_len;

    const int gl_a_stride_k = TILESIZE_K;
    const int sh0_a_stride_m = TILESIZE_M * TILESIZE_K;
    const half * gl_a_ptr = A_p + slice_m * TILESIZE_M * size_k + slice0_k * gl_a_stride_k;
    half * sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    constexpr int load_a_iters = EXL3_GEMM_CEIL_DIV(sh0_a_stride_m / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int load_a_gl[load_a_iters];
    int load_a_sh[load_a_iters];
    for (int i = 0; i < load_a_iters; ++i) {
        const int k = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_a_stride_k / 8);
        const int m = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_a_stride_k / 8);
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
        pred_a_gl[i] = m < size_m_use;
    }

    const int gl_b_stride_k = blocks_n * TILEBLOCKS_K * 256 / 16 * bits;
    const int gl_b_stride_n = TILEBLOCKS_N * 256 / 16 * bits;
    const int sh0_b_stride_k = TILEBLOCKS_K * TILEBLOCKS_N * 256 / 16 * bits;
    const uint16_t * gl_b_ptr = B_p + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t * sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    constexpr int load_b_iters = EXL3_GEMM_CEIL_DIV(sh_b_stage_size / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int load_b_gl[load_b_iters];
    for (int i = 0; i < load_b_iters; ++i) {
        const int n = (i * EXL3_GEMM_BASE_THREADS + t) % (gl_b_stride_n / 8);
        const int k = (i * EXL3_GEMM_BASE_THREADS + t) / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n * 256 / 16 * bits / 8) + n;
        pred_b_gl[i] = i * EXL3_GEMM_BASE_THREADS + t < sh0_b_stride_k / 8;
    }

    auto advance0 = [&]() {
        slice0_k++;
        slice0_iters--;
        const int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice0_k >= tiles_k) {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A_p + slice_m * TILESIZE_M * size_k + slice0_k * gl_a_stride_k;
            gl_b_ptr = B_p + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
        } else {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    int slice1_k = slice0_k;
    int slice1_n = slice0_n;
    int slice1_iters = slice0_iters;
    half * sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t * sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&]() {
        slice1_k++;
        slice1_iters--;
        const int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;
        if (slice1_k >= tiles_k) {
            slice1_k = 0;
            slice1_n++;
        }
    };

    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int slice2_n = slice0_n;
    int slice2_iters = slice0_iters;
    float * gl_c_ptr = C_p + slice_m * TILESIZE_M * size_n + slice2_n * TILESIZE_N;

    FragA frag_a[FRAG_STAGES];
    FragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    FragC frag_c[FRAGS_N_PER_WARP];

    auto advance2 = [&]() {
        slice2_k++;
        slice2_iters--;
        if (slice2_k >= tiles_k) {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            gl_c_ptr += TILESIZE_N;
        }
    };

    auto async_load_gl = [&]() {
        if (sub_k) {
            cp_async_fence();
            return;
        }
        if (slice0_iters) {
            {
                const int4 * gl = (const int4 *) gl_a_ptr;
                int4 * sh = (int4 *) sh0_a_ptr;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i) {
                    if (pred_a_gl[i]) {
                        cp_async(sh + load_a_sh[i], gl + load_a_gl[i]);
                    }
                }
            }
            {
                const int4 * gl = (const int4 *) gl_b_ptr;
                int4 * sh = (int4 *) sh0_b_ptr;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i) {
                    if (pred_b_gl[i]) {
                        cp_async(sh + EXL3_GEMM_BASE_THREADS * i + t, gl + load_b_gl[i]);
                    }
                }
            }
            advance0();
        }
        cp_async_fence();
    };

    auto load_frags = [&](int buf) {
        if (!slice1_iters) {
            return;
        }
        {
            const int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
            const int base_c = lane_id / 16 + sub_k * 2;
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m) {
                const int R = r + m * 16;
                const int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                ldsm4(frag_a[buf], (int4 *) sh1_a_ptr + R * A_COLS + c_swizzled);
            }
        }
        #pragma unroll
        for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2) {
            const int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
            const uint32_t * shb = (const uint32_t *) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
            dq_dispatch<bits, CB>(shb, lane_id << 3, frag_b[buf][n2], frag_b[buf][n2 + 1]);
        }
        __syncthreads();
        advance1();
    };

    auto clear_frag_c = [&]() {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            frag_c[n] = {};
        }
    };

    auto threadblock_reduce = [&]() {
        auto store = [&](int i) {
            if (sub_k == i) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        *sh_red++ = frag_c[n][j];
                    }
                }
            }
            __syncthreads();
        };
        auto add = [&](int i) {
            if (sub_k == i) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        frag_c[n][j] += *sh_red++;
                    }
                }
            }
        };
        auto store_small = [&](int i) {
            if (sub_k == i && lane_id / 4 < size_m_use) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    *sh_red++ = frag_c[n][0];
                    *sh_red++ = frag_c[n][1];
                }
            }
            __syncthreads();
        };
        auto add_small = [&](int i) {
            if (sub_k == i && lane_id / 4 < size_m_use) {
                float * sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
                #pragma unroll
                for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
                    frag_c[n][0] += *sh_red++;
                    frag_c[n][1] += *sh_red++;
                }
            }
        };
        if (size_m_use <= 8) {
            store_small(1);
            add_small(0);
        } else {
            store(1);
            add(0);
        }
    };

    auto read_sum_gl = [&]() {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            const int r0 = lane_id / 4;
            const int r1 = r0 + 8;
            const int c = (lane_id % 4) * 2;
            if (r0 < size_m_use) {
                float * c_ptr = gl_c_ptr + r0 * size_n + (n0 + n) * 8 + c;
                frag_c[n][0] += *c_ptr++;
                frag_c[n][1] += *c_ptr++;
            }
            if (r1 < size_m_use) {
                float * c_ptr = gl_c_ptr + r1 * size_n + (n0 + n) * 8 + c;
                frag_c[n][2] += *c_ptr++;
                frag_c[n][3] += *c_ptr++;
            }
        }
    };

    auto write_sum_gl = [&]() {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            const int r0 = lane_id / 4;
            const int r1 = r0 + 8;
            const int c = (lane_id % 4) * 2;
            if (r0 < size_m_use) {
                float * c_ptr = gl_c_ptr + r0 * size_n + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][0];
                *c_ptr++ = frag_c[n][1];
            }
            if (r1 < size_m_use) {
                float * c_ptr = gl_c_ptr + r1 * size_n + (n0 + n) * 8 + c;
                *c_ptr++ = frag_c[n][2];
                *c_ptr++ = frag_c[n][3];
            }
        }
    };

    auto atomic_add_sum_gl = [&]() {
        const int n0 = warp_id * FRAGS_N_PER_WARP;
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            const int r0 = lane_id / 4;
            const int r1 = r0 + 8;
            const int c = (lane_id % 4) * 2;
            if (r0 < size_m_use) {
                float * c_ptr = gl_c_ptr + r0 * size_n + (n0 + n) * 8 + c;
                atomicAdd(c_ptr + 0, frag_c[n][0]);
                atomicAdd(c_ptr + 1, frag_c[n][1]);
            }
            if (r1 < size_m_use) {
                float * c_ptr = gl_c_ptr + r1 * size_n + (n0 + n) * 8 + c;
                atomicAdd(c_ptr + 0, frag_c[n][2]);
                atomicAdd(c_ptr + 1, frag_c[n][3]);
            }
        }
    };

    auto reduce = [&]() {
        threadblock_reduce();
        if (!locks) {
            if (!sub_k) {
                write_sum_gl();
            }
            if (dst && ids_dst && svh_e) {
                __syncthreads();
                const int row = (int) threadIdx.x / 32;
                if (row < size_m_use) {
                    const int inst = ids_dst[row_base + slice_m * TILESIZE_M + row];
                    const int n0 = slice2_n * TILESIZE_N;
                    for (int d = 0; d < TILESIZE_N; d += 128) {
                        exl3_had_out_f32(
                            C_p + (size_t) (slice_m * TILESIZE_M + row) * size_n + n0 + d,
                            dst + (size_t) inst * size_n + n0 + d,
                            svh_e + n0 + d);
                    }
                }
                __syncthreads();
            }
            clear_frag_c();
            return;
        }
        // ExL reverse-K lock: highest-K slice writes first, later slices read-add-write.
        const int lock_i = tiles_k - slice2_k - 1;
        const int lock_d = slice2_k - slice2_k0 + 1;
        int * lock = &locks[slice2_n];
        barrier_acquire(lock, lock_i);
        const bool first = lock_i == 0;
        const bool last = lock_i + lock_d == tiles_k;
        if (!sub_k && !first) {
            read_sum_gl();
        }
        if (!sub_k) {
            write_sum_gl();
        }
        barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    auto wait_stage = [&]() {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
    };

    auto matmul = [&](int buf) {
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n) {
            ptx_mma_m16n8k16(frag_a[buf], frag_b[buf][n], frag_c[n]);
        }
    };

    if (locks && (gridDim.z > 1 || sk1 > 0 || sadd)) {
        const int n = (sk1 > 0 || gridDim.z > 1) ? (int) blockIdx.x : index_n(slice_beg);
        int k_lo = sk1 > 0 ? sk0 : index_k(slice_beg);
        int k_hi = sk1 > 0 ? sk1 : k_lo + slice_len;
        if (gridDim.z > 1) {
            const int chunk = tiles_k / (int) gridDim.z;
            k_lo = (int) blockIdx.z * chunk;
            k_hi = k_lo + chunk;
        }
        const half * ga = A_p + (size_t) k_lo * TILESIZE_K;
        const uint16_t * gb = B_p + (size_t) n * gl_b_stride_n + (size_t) k_lo * gl_b_stride_k;
        gl_c_ptr = C_p + n * TILESIZE_N;
        clear_frag_c();
        for (int k = k_lo; k < k_hi; ++k) {
            if (!sub_k) {
                const int4 * gla = (const int4 *) ga;
                int4 * sha = (int4 *) sh_a;
                #pragma unroll
                for (int i = 0; i < load_a_iters; ++i) {
                    if (pred_a_gl[i]) {
                        cp_async(sha + load_a_sh[i], gla + load_a_gl[i]);
                    }
                }
                const int4 * glb = (const int4 *) gb;
                int4 * shb = (int4 *) sh_b;
                #pragma unroll
                for (int i = 0; i < load_b_iters; ++i) {
                    if (pred_b_gl[i]) {
                        cp_async(shb + EXL3_GEMM_BASE_THREADS * i + t, glb + load_b_gl[i]);
                    }
                }
            }
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();
            {
                const int r = (lane_id % 8) + 8 * ((lane_id / 8) % 2);
                const int base_c = lane_id / 16 + sub_k * 2;
                #pragma unroll
                for (int m = 0; m < TILEBLOCKS_M; ++m) {
                    const int R = r + m * 16;
                    const int c_swizzled = base_c ^ ((R >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK);
                    ldsm4(frag_a[0], (int4 *) sh_a + R * A_COLS + c_swizzled);
                }
            }
            #pragma unroll
            for (int n2 = 0; n2 < FRAGS_N_PER_WARP; n2 += 2) {
                const int sub_n2 = warp_id * FRAGS_N_PER_WARP / 2 + n2 / 2;
                const uint32_t * shb = (const uint32_t *) (sh_b + (sub_k * TILEBLOCKS_N + sub_n2) * 256 / 16 * bits);
                dq_dispatch<bits, CB>(shb, lane_id << 3, frag_b[0][n2], frag_b[0][n2 + 1]);
            }
            __syncthreads();
            matmul(0);
            ga += TILESIZE_K;
            gb += gl_b_stride_k;
        }
        threadblock_reduce();
        if (!sub_k) {
            if (sadd == 2) {
                atomic_add_sum_gl();
            } else if (sadd) {
                read_sum_gl();
                write_sum_gl();
            } else {
                write_sum_gl();
            }
        }
        return;
    }

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i) {
        async_load_gl();
    }
    wait_stage();
    clear_frag_c();
    load_frags(0);

    while (true) {
        async_load_gl();
        wait_stage();
        matmul(0);
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) {
            reduce();
            slice2_k0 = slice2_k + 1;
        }
        advance2();
        if (!slice2_iters) {
            break;
        }
        load_frags(1);

        async_load_gl();
        wait_stage();
        matmul(1);
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) {
            reduce();
            slice2_k0 = slice2_k + 1;
        }
        advance2();
        if (!slice2_iters) {
            break;
        }
        load_frags(2);

        async_load_gl();
        wait_stage();
        matmul(2);
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) {
            reduce();
            slice2_k0 = slice2_k + 1;
        }
        advance2();
        if (!slice2_iters) {
            break;
        }
        load_frags(0);
    }
}

template <int bits, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES, int CB>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(
        const half * __restrict__ A,
        const uint16_t * __restrict__ B,
        float * __restrict__ C,
        const int size_m,
        const int size_k,
        const int size_n,
        int * __restrict__ locks,
        const int32_t * __restrict__ bounds,
        size_t expert_stride,
        const int32_t * __restrict__ ids_dst,
        const float * __restrict__ svh,
        float * __restrict__ dst,
        const int sk0,
        const int sk1,
        const int sadd) {
    constexpr int TILESIZE_M = 16;
    const half * A_p = A;
    const uint16_t * B_p = B;
    float * C_p = C;
    int * locks_p = locks;
    int size_m_p = size_m;
    int row_base = 0;
    const float * svh_e = svh;
    const int e = (int) blockIdx.y;
    if (bounds) {
        const int lo = bounds[e];
        const int hi = bounds[e + 1];
        size_m_p = hi - lo;
        if (size_m_p <= 0) {
            return;
        }
        row_base = lo;
        A_p = A + (size_t) lo * size_k;
        B_p = (const uint16_t *) ((const char *) B + (size_t) e * expert_stride);
        C_p = C + (size_t) lo * size_n;
        if (svh) {
            svh_e = svh + (size_t) e * size_n;
        }
    }
    if (sk0 > 0 && gridDim.z > 1) {
        C_p += (size_t) blockIdx.z * (size_t) sk0;
    }

    // Dense path launches grid.y = 1 and walks M here so reverse-K blocks stay resident.
    // MoE (bounds) already loops M; grid.y is the expert.
    const int n_m_tiles = EXL3_GEMM_CEIL_DIV(size_m_p, TILESIZE_M);
    const int tiles_n = size_n / TILESIZE_N;
    for (int mt = 0; mt < n_m_tiles; ++mt) {
        const int slice_m = mt;
        const int size_m_use = EXL3_GEMM_MIN(size_m_p - slice_m * TILESIZE_M, TILESIZE_M);
        if (size_m_use <= 0) {
            continue;
        }
        // One lock per N-tile. Offset by expert and M-tile so reverse-K columns do not share slots.
        if (locks) {
            locks_p = bounds
                ? locks + ((size_t) e * (size_m / 16) + mt) * tiles_n
                : locks + (size_t) mt * tiles_n;
        }
        exl3_gemm_inner<bits, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, CB>(
            A_p + (size_t) slice_m * TILESIZE_M * size_k,
            B_p,
            C_p + (size_t) slice_m * TILESIZE_M * size_n,
            size_m_use, size_k, size_n, locks_p, ids_dst, svh_e, dst,
            row_base + slice_m * TILESIZE_M, sk0, sk1, sadd);
    }
}

inline int exl3_gemm_pick_shape(int size_k, int size_n, int bits) {
    // Ada / sm_89 path of ExLlamaV3 select_gemm_shape (non-multi).
    const bool mod_256 = (size_n % 256 == 0);
    if (mod_256 && bits <= 3) {
        if (size_k <= 2048) {
            return 2;
        }
        if (size_n < 4096 && size_k <= 12288) {
            return 2;
        }
        return 3;
    }
    if (size_n <= 16384) {
        return 2;
    }
    if (mod_256) {
        return 3;
    }
    return 2;
}

inline bool exl3_gemm_shape_ok(int size_k, int size_n, int bits) {
    if (bits < 1 || bits > 8) {
        return false;
    }
    if (size_k % 32 != 0 || size_n % 128 != 0 || size_k < 32 || size_n < 128) {
        return false;
    }
    return true;
}

inline int exl3_gemm_lock_elems(int size_m, int size_n) {
    const int n_m = EXL3_GEMM_CEIL_DIV(size_m, 16);
    return n_m * (size_n / 16);
}

inline int & exl3_gemm_id_nx_override() {
    static int v = -1;
    return v;
}

inline int & exl3_gemm_id_plane() {
    static int v = 0;
    return v;
}

inline int exl3_gemm_id_nx() {
    const int ov = exl3_gemm_id_nx_override();
    if (ov >= 0) {
        return ov;
    }
    static const int v = [] {
        const char * e = getenv("GGML_EXL3_ID_NX");
        return e ? atoi(e) : 0;
    }();
    return v;
}

inline int exl3_gemm_id_tile_n(int size_k, int size_n, int bits) {
    const int shape = exl3_gemm_pick_shape(size_k, size_n, bits);
    return (shape == 3 && (size_n % 256 == 0)) ? 256 : 128;
}

inline int exl3_gemm_id_lock_n(int n_experts, int n_m_max, int size_n, int tile_n) {
    return n_experts * n_m_max * (size_n / tile_n);
}

inline int exl3_gemm_smem_bytes(int bits, int tilesize_k, int tilesize_n) {
    const int sh_a = 16 * tilesize_k;
    const int sh_b = (tilesize_k / 16) * (tilesize_n / 16) * 16 * bits;
    const int frags_n = 2 * (tilesize_n / 16) / 8;
    const int sh_c = 4 * EXL3_GEMM_BASE_THREADS * frags_n;
    return 4 * (2 * sh_a + 2 * sh_b) + 4 * sh_c;
}

template <int bits, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES, int CB>
static void exl3_gemm_launch_shape(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n,
        int * locks, int n_sms, cudaStream_t stream) {
    auto * kfn = exl3_gemm_kernel<bits, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, CB>;
    const int smem = exl3_gemm_smem_bytes(bits, TILESIZE_K, TILESIZE_N);
    static bool smem_ok = false;
    if (!smem_ok) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        smem_ok = true;
    }
    const int tiles = (size_k / TILESIZE_K) * (size_n / TILESIZE_N);
    int n_blocks = tiles < n_sms ? tiles : n_sms;
    if (n_blocks < 1) {
        n_blocks = 1;
    }
    // grid.y is not M: extra M-tiles would occupy SMs and reverse-K would never run.
    const dim3 grid((unsigned) n_blocks, 1u);
    const dim3 block((unsigned) (EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16));
    kfn<<<grid, block, (size_t) smem, stream>>>(
        A, B, C, size_m, size_k, size_n, locks, nullptr, 0, nullptr, nullptr, nullptr, 0, 0, 0);
}

template <int bits, int CB>
static void exl3_gemm_launch_bits(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n,
        int * locks, int n_sms, int shape, cudaStream_t stream) {
    if (shape == 3 && (size_n % 256 == 0)) {
        exl3_gemm_launch_shape<bits, EXL3_GEMM_SHAPE3, CB>(A, B, C, size_m, size_k, size_n, locks, n_sms, stream);
    } else {
        exl3_gemm_launch_shape<bits, EXL3_GEMM_SHAPE2, CB>(A, B, C, size_m, size_k, size_n, locks, n_sms, stream);
    }
}

static void exl3_gemm_cuda(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int bits,
        int * locks, int n_sms, cudaStream_t stream) {
    const int shape = exl3_gemm_pick_shape(size_k, size_n, bits);
    const int cb = ggml_exl3_get_codebook();
    if (cb == 1) {
        switch (bits) {
            case 1: exl3_gemm_launch_bits<1, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 2: exl3_gemm_launch_bits<2, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 3: exl3_gemm_launch_bits<3, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 4: exl3_gemm_launch_bits<4, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 5: exl3_gemm_launch_bits<5, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 6: exl3_gemm_launch_bits<6, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 7: exl3_gemm_launch_bits<7, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            case 8: exl3_gemm_launch_bits<8, 1>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 1: exl3_gemm_launch_bits<1, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 2: exl3_gemm_launch_bits<2, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 3: exl3_gemm_launch_bits<3, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 4: exl3_gemm_launch_bits<4, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 5: exl3_gemm_launch_bits<5, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 6: exl3_gemm_launch_bits<6, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 7: exl3_gemm_launch_bits<7, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        case 8: exl3_gemm_launch_bits<8, 2>(A, B, C, size_m, size_k, size_n, locks, n_sms, shape, stream); break;
        default: break;
    }
}

template <int bits, int TILESIZE_K, int TILESIZE_N, int SH_STAGES, int FRAG_STAGES, int CB>
static void exl3_gemm_launch_shape_id(
        const half * A, const uint16_t * B, float * C,
        int size_k, int size_n, int n_m_max, int n_experts,
        int * locks, int n_sms, const int32_t * bounds, size_t expert_stride,
        const int32_t * ids_dst, const float * svh, float * dst, cudaStream_t stream) {
    auto * kfn = exl3_gemm_kernel<bits, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, CB>;
    const int smem = exl3_gemm_smem_bytes(bits, TILESIZE_K, TILESIZE_N);
    static bool smem_ok = false;
    if (!smem_ok) {
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        smem_ok = true;
    }
    (void) n_sms;
    const int tiles_n = size_n / TILESIZE_N;
    const int tiles_k = size_k / TILESIZE_K;
    const int want_nx = exl3_gemm_id_nx();
    const dim3 block((unsigned) (EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16));
    if (locks && want_nx > tiles_n && tiles_n > 0 && tiles_k % (want_nx / tiles_n) == 0) {
        const int n_split = want_nx / tiles_n;
        const dim3 grid((unsigned) (tiles_n * n_split), (unsigned) n_experts);
        kfn<<<grid, block, (size_t) smem, stream>>>(
            A, B, C, n_m_max * 16, size_k, size_n, locks, bounds, expert_stride, ids_dst, svh, dst,
            0, 0, 0);
        return;
    }
    if (tiles_n < 1) {
        return;
    }
    const dim3 grid((unsigned) tiles_n, (unsigned) n_experts);
    kfn<<<grid, block, (size_t) smem, stream>>>(
        A, B, C, n_m_max * 16, size_k, size_n, locks, bounds, expert_stride, ids_dst, svh, dst, 0, 0, 0);
}

template <int bits, int CB>
static void exl3_gemm_launch_bits_id(
        const half * A, const uint16_t * B, float * C,
        int size_k, int size_n, int n_m_max, int n_experts,
        int * locks, int n_sms, int shape, const int32_t * bounds, size_t expert_stride,
        const int32_t * ids_dst, const float * svh, float * dst, cudaStream_t stream) {
    if (locks) {
        exl3_gemm_launch_shape_id<bits, EXL3_GEMM_SHAPE2_S3, CB>(
            A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, bounds, expert_stride,
            ids_dst, svh, dst, stream);
    } else if (shape == 3 && (size_n % 256 == 0)) {
        exl3_gemm_launch_shape_id<bits, EXL3_GEMM_SHAPE3, CB>(
            A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, bounds, expert_stride,
            ids_dst, svh, dst, stream);
    } else {
        exl3_gemm_launch_shape_id<bits, EXL3_GEMM_SHAPE2, CB>(
            A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, bounds, expert_stride,
            ids_dst, svh, dst, stream);
    }
}

static void exl3_gemm_cuda_id(
        const half * A, const uint16_t * B, float * C,
        int size_k, int size_n, int bits, int n_m_max, int n_experts,
        int * locks, int n_sms, const int32_t * bounds, size_t expert_stride,
        const int32_t * ids_dst, const float * svh, float * dst, cudaStream_t stream) {
    const int shape = exl3_gemm_pick_shape(size_k, size_n, bits);
    const int cb = ggml_exl3_get_codebook();
    if (cb == 1) {
        switch (bits) {
            case 1: exl3_gemm_launch_bits_id<1, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 2: exl3_gemm_launch_bits_id<2, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 3: exl3_gemm_launch_bits_id<3, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 4: exl3_gemm_launch_bits_id<4, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 5: exl3_gemm_launch_bits_id<5, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 6: exl3_gemm_launch_bits_id<6, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 7: exl3_gemm_launch_bits_id<7, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            case 8: exl3_gemm_launch_bits_id<8, 1>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 1: exl3_gemm_launch_bits_id<1, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 2: exl3_gemm_launch_bits_id<2, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 3: exl3_gemm_launch_bits_id<3, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 4: exl3_gemm_launch_bits_id<4, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 5: exl3_gemm_launch_bits_id<5, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 6: exl3_gemm_launch_bits_id<6, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 7: exl3_gemm_launch_bits_id<7, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        case 8: exl3_gemm_launch_bits_id<8, 2>(A, B, C, size_k, size_n, n_m_max, n_experts, locks, n_sms, shape, bounds, expert_stride, ids_dst, svh, dst, stream); break;
        default: break;
    }
}
