#pragma once

// Copyright (c) 2025 Turboderp
// SPDX-License-Identifier: MIT
// Derived from ExLlamaV3 exl3_gemv_kernel.cuh
// https://github.com/turboderp-org/exllamav3  adapted for ggml.
// See licenses/LICENSE-exllamav3.

// Small-m EXL3 GEMV, from ExLlamaV3 exl3_gemv_kernel.cuh.
// FUSE=1: F32 x * suh, then H128, then the same MMA body.

#include "exl3_dq.cuh"
#include "exl3_had.cuh"

#include <cstdint>

#ifndef EXL3_GEMV_MAX_M
#define EXL3_GEMV_MAX_M 8
#endif

namespace exl3_gemv_ns {

__device__ __forceinline__ void mma_ab_f(const FragB & a01, const FragB & a23, const FragB & b, FragC & c) {
    const uint32_t * a0 = reinterpret_cast<const uint32_t *>(&a01);
    const uint32_t * a1 = reinterpret_cast<const uint32_t *>(&a23);
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    float * cc = reinterpret_cast<float *>(&c);
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(cc[0]), "+f"(cc[1]), "+f"(cc[2]), "+f"(cc[3])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1]));
}

template <int CB>
__device__ __forceinline__ void decode8(
        uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
        uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB & f0, FragB & f1) {
    f0[0] = decode_3inst_2<CB>(w0, w1);
    f0[1] = decode_3inst_2<CB>(w2, w3);
    f1[0] = decode_3inst_2<CB>(w4, w5);
    f1[1] = decode_3inst_2<CB>(w6, w7);
}

template <int CB>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB & f0, FragB & f1) {
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8<CB>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int CB>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    decode8<CB>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

template <int CB>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB & f0, FragB & f1) {
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    decode8<CB>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
            w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

} // namespace exl3_gemv_ns

struct exl3_gemv_id {
    const int32_t * ids;
    const int32_t * bounds;
    const int32_t * ids_src;
    const int32_t * ids_dst;
    const float * svh;
    int n_expert_used;
    int n_experts;
    int64_t ids_nb0;
    int64_t ids_nb1;
    size_t expert_stride;
    int x_ne1;
    int n_inst;
    int suh_2d;
};

// CFG 0: 512 thr, 2 n-tiles/warp, 16 k-splits. CFG 1: 256 thr, 4 n-tiles/warp, 8 k-splits.
// MMODE 0: m==1. MMODE 1: 2 <= m <= 8.
// FUSE 0: A is F16. FUSE 1: A_f32 * suh, H128 into warp smem, then F16 MMA.
// fp32 accumulate: fp16 MMA misses harness rel_|ref|>1 ~1e-3 vs reconstruct+cuBLAS.
template <int bits, int MMODE, int CFG, int FUSE, int CB>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void exl3_gemv_kernel(
        const half * __restrict__ A,
        const uint16_t * __restrict__ B,
        float * __restrict__ C,
        int size_m,
        int size_k,
        int size_n,
        int packed_n,
        const float * __restrict__ A_f32,
        const float * __restrict__ suh,
        exl3_gemv_id id) {
    constexpr int WK      = CFG == 0 ? 16 : 8;
    constexpr int WNT     = CFG == 0 ? 2 : 4;
    constexpr int PF      = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int ROWS    = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS    = WNT * 16;
    constexpr int TWORDS  = 8 * bits;
    constexpr bool SHUF   = (CB == 2) && (bits == 2 || bits == 3 || bits == 4);
    [[maybe_unused]] constexpr int LOADS   = bits == 2 ? WNT / 2 : WNT;
    [[maybe_unused]] constexpr int LSTRIDE = bits == 3 ? 24 : 32;
    constexpr int STAGE_W = SHUF ? 1 : WNT * TWORDS;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles     = packed_n / 16;
    const int kslices    = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = (kslices + WK - 1) / WK;
    const int ks0   = warp * chunk;
    const int myn   = max(0, min(chunk, kslices - ks0));

    const uint16_t * B_ptr = B;
    const float * suh_e = suh;
    int row_lo = 0;
    int n_rows = 1;
    int inst0 = -1;
    const bool grouped = id.bounds && id.ids_src && id.ids_dst && id.n_experts > 0;
    if (grouped) {
        const int e = (int) blockIdx.y;
        if (e >= id.n_experts) {
            return;
        }
        row_lo = id.bounds[e];
        const int row_hi = id.bounds[e + 1];
        n_rows = row_hi - row_lo;
        if (n_rows <= 0) {
            return;
        }
        B_ptr = (const uint16_t *) ((const char *) B + (size_t) e * id.expert_stride);
        if (suh && id.suh_2d) {
            suh_e = suh + (size_t) e * size_k;
        }
    } else if (id.ids && id.n_inst > 0) {
        inst0 = (int) blockIdx.y;
        if (inst0 >= id.n_inst) {
            return;
        }
        const int t = inst0 / id.n_expert_used;
        const int s = inst0 - t * id.n_expert_used;
        const int32_t e = *(const int32_t *) ((const char *) id.ids + t * id.ids_nb1 + s * id.ids_nb0);
        B_ptr = (const uint16_t *) ((const char *) B + (size_t) e * id.expert_stride);
        if (suh && id.suh_2d) {
            suh_e = suh + (size_t) e * size_k;
        }
    }

    const uint32_t * B32 = (const uint32_t *) B_ptr;
    const size_t slice_stride = (size_t) ntiles * TWORDS;
    const half * A_ptr = A;
    const float * Af_ptr = A_f32;
    const float * suh_ptr = suh_e;
    float * C_ptr = C;
    const half2 * A2 = (const half2 *) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));
    int row_src[ROWS];
    int row_dst[ROWS];
    int chunk_m = 1;
    const int row_step = grouped && MMODE == 1 ? ROWS : 1;

    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2) {
        const int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3) {
        const int t_offset = lane << 3;
        const int b1 = (t_offset + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    __shared__ uint32_t sh_stage[SHUF ? 1 : WK][SHUF ? 1 : 2][STAGE_W];
    extern __shared__ uint32_t exl3_gemv_dyn[];
    half * sh_had_dyn = reinterpret_cast<half *>(exl3_gemv_dyn);

    auto had_ptr = [&](int wr, int r) -> half * {
        return sh_had_dyn + ((size_t) wr * ROWS + r) * 128;
    };

    auto load_a = [&](int i, FragB & a01, FragB & a23) {
        if constexpr (FUSE) {
            const int local = ((ks0 + i) * 16) & 127;
            const int e0 = local + 2 * (lane & 3);
            const int sr = MMODE == 0 ? 0 : r0;
            a01[0] = r0_ok ? *reinterpret_cast<const half2 *>(had_ptr(warp, sr) + e0) : hzero;
            a23[0] = r0_ok ? *reinterpret_cast<const half2 *>(had_ptr(warp, sr) + e0 + 8) : hzero;
        } else {
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
        }
        a01[1] = hzero;
        a23[1] = hzero;
    };

    auto refresh_had = [&](int i, int & had_blk) {
        if constexpr (FUSE) {
            const int blk = ((ks0 + i) * 16) >> 7;
            if (blk != had_blk) {
                had_blk = blk;
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (grouped && r < chunk_m) {
                        exl3_had_warp_row(A_f32 + (size_t) row_src[r] * size_k, suh_e,
                            size_k, 1, 0, blk << 7, had_ptr(warp, r));
                    } else {
                        exl3_had_warp_row(Af_ptr, suh_ptr, size_k, chunk_m, r, blk << 7, had_ptr(warp, r));
                    }
                }
                __syncwarp();
            }
        }
    };

    for (int rr0 = 0; rr0 < n_rows; rr0 += row_step) {
        chunk_m = grouped ? min(row_step, n_rows - rr0)
                          : (MMODE == 0 ? 1 : min(size_m, ROWS));
        r0_ok = MMODE == 0 ? lane < 4 : r0 < chunk_m;
        if (grouped) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                if (r < chunk_m) {
                    row_src[r] = id.ids_src[row_lo + rr0 + r];
                    row_dst[r] = id.ids_dst[row_lo + rr0 + r];
                }
            }
            suh_ptr = suh_e;
        } else if (inst0 >= 0) {
            const int t = inst0 / id.n_expert_used;
            const int x_inst = (id.x_ne1 <= 1) ? t : inst0;
            if (A) {
                A_ptr = A + (size_t) x_inst * size_k;
                A2 = (const half2 *) A_ptr;
            }
            if (A_f32) {
                Af_ptr = A_f32 + (size_t) x_inst * size_k;
            }
            if (suh && !id.suh_2d) {
                suh_ptr = suh + (size_t) inst0 * size_k;
            } else {
                suh_ptr = suh_e;
            }
            C_ptr = C + (size_t) inst0 * size_n;
        }

    for (int group = (int) blockIdx.x; group < num_groups; group += (int) gridDim.x) {
        FragC acc[WNT][2] = {};
        int had_blk = -1;

        if constexpr (SHUF) {
            const uint32_t * bp = B32 + (size_t) ks0 * slice_stride + (size_t) group * WNT * TWORDS + lane;
            auto ld_b = [&](int i, int l) -> uint32_t {
                if constexpr (bits == 3) {
                    return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
                } else {
                    return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
                }
            };

            uint32_t pf[PF][LOADS];
            #pragma unroll
            for (int d = 0; d < PF; ++d) {
                if (d < myn) {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l) {
                        pf[d][l] = ld_b(d, l);
                    }
                }
            }

            for (int ib = 0; ib < myn; ib += PF) {
                #pragma unroll
                for (int d = 0; d < PF; ++d) {
                    const int i = ib + d;
                    if (i >= myn) {
                        break;
                    }

                    uint32_t bw[LOADS];
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l) {
                        bw[l] = pf[d][l];
                    }
                    if (i + PF < myn) {
                        #pragma unroll
                        for (int l = 0; l < LOADS; ++l) {
                            pf[d][l] = ld_b(i + PF, l);
                        }
                    }

                    refresh_had(i, had_blk);
                    FragB a01, a23;
                    load_a(i, a01, a23);

                    #pragma unroll
                    for (int t = 0; t < WNT; ++t) {
                        FragB f0, f1;
                        if constexpr (bits == 4) {
                            uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                            exl3_gemv_ns::dq8_regs_4bits<CB>(aw, bw[t], f0, f1);
                        } else if constexpr (bits == 2) {
                            const uint32_t w = bw[t >> 1];
                            const int base = (t & 1) << 4;
                            uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                            uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                            exl3_gemv_ns::dq8_regs_2bits<CB>(awv, bwv, lane << 3, f0, f1);
                        } else {
                            uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                            uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                            exl3_gemv_ns::dq8_regs_3bits<CB>(awv, bwv, x_s2, f0, f1);
                        }
                        exl3_gemv_ns::mma_ab_f(a01, a23, f0, acc[t][0]);
                        exl3_gemv_ns::mma_ab_f(a01, a23, f1, acc[t][1]);
                    }
                }
            }
        } else {
            auto load_tile = [&](int i, int buf) {
                const uint32_t * tp = B32 + (size_t) (ks0 + i) * slice_stride + (size_t) group * WNT * TWORDS;
                #pragma unroll
                for (int w = lane; w < WNT * TWORDS; w += 32) {
                    sh_stage[warp][buf][w] = __ldcs(tp + w);
                }
                __syncwarp();
            };

            if (myn > 0) {
                load_tile(0, 0);
            }
            for (int i = 0; i < myn; ++i) {
                const int buf = i & 1;
                if (i + 1 < myn) {
                    const uint32_t * tp = B32 + (size_t) (ks0 + i + 1) * slice_stride + (size_t) group * WNT * TWORDS;
                    #pragma unroll
                    for (int w = lane; w < WNT * TWORDS; w += 32) {
                        sh_stage[warp][buf ^ 1][w] = __ldcs(tp + w);
                    }
                }

                refresh_had(i, had_blk);
                FragB a01, a23;
                load_a(i, a01, a23);

                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    FragB f0, f1;
                    dq_dispatch<bits, CB>(&sh_stage[warp][buf][t * TWORDS], lane * 8, f0, f1);
                    exl3_gemv_ns::mma_ab_f(a01, a23, f0, acc[t][0]);
                    exl3_gemv_ns::mma_ab_f(a01, a23, f1, acc[t][1]);
                }
                if (i + 1 < myn) {
                    __syncwarp();
                }
            }
        }

        {
            const int c0 = 2 * (lane & 3);
            const bool store0 = MMODE == 0 ? lane < 4 : r0 < ROWS;
            const int sr0 = MMODE == 0 ? 0 : r0;
            if (store0) {
                #pragma unroll
                for (int t = 0; t < WNT; ++t) {
                    #pragma unroll
                    for (int f = 0; f < 2; ++f) {
                        const int col = t * 16 + f * 8 + c0;
                        sh_red[warp][sr0][col + 0] = acc[t][f][0];
                        sh_red[warp][sr0][col + 1] = acc[t][f][1];
                    }
                }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : chunk_m;
        for (int idx = (int) threadIdx.x; idx < COLS * rows_out; idx += THREADS) {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j) {
                sum += sh_red[j][r][c];
            }
            if (grouped) {
                C[(size_t) row_dst[r] * size_n + group * COLS + c] = sum;
            } else {
                C_ptr[(size_t) r * size_n + group * COLS + c] = sum;
            }
        }
        __syncthreads();
    }
    }
}

template <int bits, int MMODE, int CFG, int FUSE, int CB>
static void exl3_gemv_launch_cfg(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int packed_n, cudaStream_t stream,
        const float * A_f32, const float * suh, exl3_gemv_id id) {
    constexpr int block_dim = CFG == 0 ? 512 : 256;
    constexpr int cols = (CFG == 0 ? 2 : 4) * 16;
    constexpr int wk = CFG == 0 ? 16 : 8;
    constexpr int rows = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    const size_t dyn = FUSE ? (size_t) wk * rows * 128 * sizeof(half) : 0;
    if constexpr (FUSE) {
        static bool attr = false;
        if (!attr) {
            cudaFuncSetAttribute((void *) exl3_gemv_kernel<bits, MMODE, CFG, FUSE, CB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) (wk * EXL3_GEMV_MAX_M * 128 * sizeof(half)));
            attr = true;
        }
    }
    const unsigned gy = (id.bounds && id.n_experts > 0) ? (unsigned) id.n_experts :
            (id.n_inst > 0 ? (unsigned) id.n_inst : 1u);
    const dim3 grid((unsigned) (size_n / cols), gy);
    exl3_gemv_kernel<bits, MMODE, CFG, FUSE, CB><<<grid, block_dim, dyn, stream>>>(
        A, B, C, size_m, size_k, size_n, packed_n, A_f32, suh, id);
}

template <int bits, int FUSE, int CB>
static void exl3_gemv_launch_bits(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int packed_n, cudaStream_t stream,
        const float * A_f32, const float * suh, exl3_gemv_id id) {
    // CFG1 lost on sm_89 for N<=17408 K=5120; always narrow.
    if (size_m == 1) {
        exl3_gemv_launch_cfg<bits, 0, 0, FUSE, CB>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id);
    } else {
        exl3_gemv_launch_cfg<bits, 1, 0, FUSE, CB>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id);
    }
}

template <int FUSE>
static void exl3_gemv_launch_fuse(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int packed_n, int bits, cudaStream_t stream,
        const float * A_f32, const float * suh, exl3_gemv_id id = {}) {
    const int cb = ggml_exl3_get_codebook();
    if (cb == 1) {
        switch (bits) {
            case 1: exl3_gemv_launch_bits<1, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 2: exl3_gemv_launch_bits<2, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 3: exl3_gemv_launch_bits<3, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 4: exl3_gemv_launch_bits<4, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 5: exl3_gemv_launch_bits<5, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 6: exl3_gemv_launch_bits<6, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 7: exl3_gemv_launch_bits<7, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            case 8: exl3_gemv_launch_bits<8, FUSE, 1>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
            default: break;
        }
        return;
    }
    switch (bits) {
        case 1: exl3_gemv_launch_bits<1, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 2: exl3_gemv_launch_bits<2, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 3: exl3_gemv_launch_bits<3, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 4: exl3_gemv_launch_bits<4, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 5: exl3_gemv_launch_bits<5, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 6: exl3_gemv_launch_bits<6, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 7: exl3_gemv_launch_bits<7, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        case 8: exl3_gemv_launch_bits<8, FUSE, 2>(A, B, C, size_m, size_k, size_n, packed_n, stream, A_f32, suh, id); break;
        default: break;
    }
}

static void exl3_gemv_cuda(
        const half * A, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int bits, cudaStream_t stream,
        int packed_n = 0) {
    if (packed_n <= 0) {
        packed_n = size_n;
    }
    exl3_gemv_launch_fuse<0>(A, B, C, size_m, size_k, size_n, packed_n, bits, stream, nullptr, nullptr);
}

static void exl3_gemv_cuda_fused(
        const float * A_f32, const float * suh, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int bits, cudaStream_t stream,
        const float * svh = nullptr, int packed_n = 0) {
    if (packed_n <= 0) {
        packed_n = size_n;
    }
    exl3_gemv_launch_fuse<1>(nullptr, B, C, size_m, size_k, size_n, packed_n, bits, stream, A_f32, suh);
    if (svh) {
        exl3_post_had_svh_cuda(C, svh, size_m, size_n, stream);
    }
}

static void exl3_gemv_cuda_fused_id(
        const float * A_f32, const float * suh, const uint16_t * B, float * C,
        int size_m, int size_k, int size_n, int bits, cudaStream_t stream,
        const float * svh, int packed_n, exl3_gemv_id id) {
    if (packed_n <= 0) {
        packed_n = size_n;
    }
    if (size_m < 1) {
        size_m = 1;
    }
    if (size_m > EXL3_GEMV_MAX_M) {
        size_m = EXL3_GEMV_MAX_M;
    }
    exl3_gemv_launch_fuse<1>(nullptr, B, C, size_m, size_k, size_n, packed_n, bits, stream, A_f32, suh, id);
    if (svh && id.ids && id.n_inst > 0) {
        exl3_post_had_svh_id_cuda(
            C, svh, id.ids, id.n_inst, id.n_expert_used, size_n, id.ids_nb0, id.ids_nb1, stream);
    }
}
