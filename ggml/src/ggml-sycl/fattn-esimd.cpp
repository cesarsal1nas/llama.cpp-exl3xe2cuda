#include "fattn-esimd.hpp"
#include "fattn-common.hpp"

#include <cfloat>
#include <cstdio>
#include <cstring>

#if defined(__INTEL_LLVM_COMPILER)
    #include <sycl/ext/intel/esimd.hpp>
    #include <sycl/ext/intel/experimental/esimd/memory.hpp>
    #include <sycl/ext/intel/esimd/xmx/dpas.hpp>
    #include <sycl/ext/intel/experimental/grf_size_properties.hpp>
    #define GGML_SYCL_FATTN_HAS_ESIMD
#endif

extern int g_ggml_sycl_enable_esimd;

#ifdef GGML_SYCL_FATTN_HAS_ESIMD

namespace {

using namespace sycl::ext::intel::esimd;
namespace ex  = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

// Decode attention, q4_0 K and V, head size 256.
//
// Work-group: NTOK tokens x 2 D-halves x T key spans (one SIMD thread each). Thread (tok, hd, t)
// owns keys [key0, key0 + CH*16) of kv head h for the RQ query heads of token tok, and the 128
// output dims [128*hd, 128*hd + 128). Both D-halves compute the same scores (K read twice from
// L2, the dot products are cheap); V is only loaded for the owned half.
//
// Per 16-key chunk: K rows arrive through five LSC 2D transposed loads (row of 36 dwords, one
// vector per dword index with the 16 keys in lanes), so each q4_0 block's nibbles are 4 dword
// vectors (even blocks are 2 bytes misaligned: realigned with shift/or) and the dp4a runs against
// the q8_1-quantized Q row dword by dword. Scores get the mask, online softmax per query row,
// then the owned V half (two u8 2D loads, 64 + 32 bytes per key) is dequantized per key and
// accumulated into O[RQ][128] in fp32.
//
// Work-group reduce over the T key spans through SLM (log2 T rounds), then thread t == 0 writes
// the partial (O, m, l) in the flash_attn_combine_results layout.
// XV: timing hooks (never dispatched in production): 1 = loads only, 2 = all loads from the first 256 keys
// (L2-resident), 3 = no loads and no math (prologue + work-group reduce only).
template <int RQ, int T, int NTOK, bool XMX, int CH, int XV = 0, bool PRE = false, bool KWALK = false>
ESIMD_INLINE void fattn_dec_q4_0(const int8_t * Q8, const float * Qc, const uint32_t * K, const uint8_t * V,
                                 const char * mask, float * parts, sycl::float2 * meta,
                                 const int k_pitch, const int k_head_b, const int v_pitch,
                                 const int v_head_b, const int mask_stride, const int n_kv, const int n_kvh,
                                 const int n_head, const int nsplit, const int ntok_all, const sycl::nd_item<1> & it) {
    constexpr int KEYS_WG = T * CH * 16;
    constexpr int MR      = 8;  // dpas repeat count: query rows padded to 8
    constexpr int HALF    = 128;
    constexpr int SLOT_F  = RQ * HALF + 2 * RQ;  // floats per SLM slot: O half, m[RQ], l[RQ]
    constexpr int NSLOT   = (T / 2 > 0 ? T / 2 : 1) * 2 * NTOK;
    slm_init<NSLOT * SLOT_F * 4>();

    const int g     = it.get_group(0);
    const int n_groups_kv = nsplit * n_kvh;
    int tile = 0;
    int g2   = g;
    if constexpr (PRE) {
        tile = g / n_groups_kv;
        g2   = g % n_groups_kv;
    }
    const int h     = g2 % n_kvh;
    const int split = g2 / n_kvh;
    const int l     = it.get_local_id(0);
    const int t     = l % T;
    const int hd    = (l / T) & 1;
    const int tok   = tile * NTOK + l / (2 * T);
    const int tok_q = tok < ntok_all ? tok : ntok_all - 1;
    const int nwalk = KWALK ? ((n_kv + KEYS_WG - 1) / KEYS_WG) : 1;

    // ---- Q rows, pre-quantized to q8 by fattn_q8_prepass: Q8[tok][head][256], Qc[tok][head][16] = {d[8], -8*d*sum(q8)[8]}
    simd<int, 64>   Qd[RQ];
    simd<int8_t, MR * 32> Qa[8];  // XMX: per 32-dim block, A operand [8 rows][32 dims] int8 (rows >= RQ zero)
    simd<float, 8>  c0[RQ], c1[RQ];  // S += d_k * (sumi * c0 + c1)
    if constexpr (XMX && RQ < MR) {
#pragma unroll
        for (int b = 0; b < 8; ++b) Qa[b].template select<(MR - RQ) * 32, 1>(RQ * 32) = 0;
    }
#pragma unroll
    for (int r = 0; r < RQ; ++r) {
        const size_t      qrow = (size_t) tok_q * n_head + (size_t) (h * RQ + r);
        simd<int8_t, 256> q8   = block_load<int8_t, 256>(Q8 + qrow * 256);
        simd<float, 16>   cc   = block_load<float, 16>(Qc + qrow * 16);
        c0[r] = cc.template select<8, 1>(0);
        c1[r] = cc.template select<8, 1>(8);
        Qd[r] = q8.template bit_cast_view<int>();
        if constexpr (XMX) {
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                Qa[b].template select<32, 1>(32 * r) = q8.template select<32, 1>(32 * b);
            }
        }
    }

    simd<float, HALF>    O[RQ];       // dp4a path: O[r][dim]
    simd<float, MR * 16> Og[8];       // XMX path: Og[g][16*r + n] = O[r][16*g + n]
    float             m[RQ], lsum[RQ];
#pragma unroll
    for (int r = 0; r < RQ; ++r) {
        O[r]    = 0.0f;
        m[r]    = -FLT_MAX / 2.0f;
        lsum[r] = 0.0f;
    }
    if constexpr (XMX) {
#pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) Og[g2] = 0.0f;
    }

    struct blk {
        simd<uint32_t, 128> Tk[4];  // Tk[i][16*j + n]: dword 8i+j of key n
        simd<uint32_t, 64>  Tk4;    // Tk4[16*j + n]: dword 32+j of key n
        simd<uint8_t, 1024> V0;     // 64 bytes per key: half-row bytes [0, 64)
        simd<uint8_t, 512>  V1;     // 32 bytes per key: half-row bytes [52, 84)
        simd<uint32_t, 16>  Vd[4];  // XMX: dword holding the q4_0 scale of block bv for each key (lanes = keys)
        simd<uint16_t, 16>  mk;
    };
    const uint16_t * mrow = (const uint16_t *) (mask + (size_t) mask_stride * tok_q);
    // per-head 2D surfaces: 64 B aligned base + small x offset, width = this head's 144 B row.
    // Works for both the interleaved KV-cache layout (nb2 = 144) and head-major layouts (nb2 = 144 * n_kv).
    const size_t     khb  = (size_t) k_head_b * h;
    const size_t     vhb  = (size_t) v_head_b * h;
    const uint32_t * Kb   = (const uint32_t *) ((const char *) K + (khb & ~(size_t) 63));
    const uint8_t *  Vb   = V + (vhb & ~(size_t) 63);
    const int        kx0  = (int) (khb & 63) / 4;
    const int        kw   = (int) (khb & 63) + 144 - 1;
    const int        vx0  = (int) (vhb & 63) + 72 * hd;
    const int        vw   = (int) (vhb & 63) + 144 - 1;

    auto load = [&](blk & b, int key) {
        if constexpr (XV == 2) key = key & 255;  // timing only: everything from the first 256 keys (L2-resident)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            b.Tk[i] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(Kb, kw, n_kv - 1, k_pitch - 1, kx0 + 8 * i, key);
        }
        b.Tk4 = ex::lsc_load_2d<uint32_t, 4, 16, 1, true, false>(Kb, kw, n_kv - 1, k_pitch - 1, kx0 + 32, key);
        b.V0  = ex::lsc_load_2d<uint8_t, 64, 16, 1, false, false>(Vb, vw, n_kv - 1, v_pitch - 1, vx0, key);
        b.V1  = ex::lsc_load_2d<uint8_t, 32, 16, 1, false, false>(Vb, vw, n_kv - 1, v_pitch - 1, vx0 + 52, key);
        if constexpr (XMX) {
            // block bv scale at half-row byte 18*bv: dword (vx0 + 18*bv) / 4, low half for even bv, high half for odd
            const uint32_t * Vb32 = (const uint32_t *) Vb;
#pragma unroll
            for (int bv = 0; bv < 4; ++bv) {
                b.Vd[bv] = ex::lsc_load_2d<uint32_t, 1, 16, 1, true, false>(Vb32, vw, n_kv - 1, v_pitch - 1,
                                                                            (vx0 + 18 * bv) / 4, key);
            }
        }
        b.mk  = block_load<uint16_t, 16>(mrow + key);
    };

    auto Tsel = [&](blk & b, int i) -> simd<uint32_t, 16> {
        if (i < 32) {
            return b.Tk[i / 8].template select<16, 1>(16 * (i % 8));
        }
        return b.Tk4.template select<16, 1>(16 * (i - 32));
    };

    auto compute = [&](blk & b) {
        simd<float, 16> S[RQ];
#pragma unroll
        for (int r = 0; r < RQ; ++r) S[r] = 0.0f;

#pragma unroll
        for (int bq = 0; bq < 8; ++bq) {
            simd<uint32_t, 16> qd[4];
            simd<uint32_t, 16> dk;
            if (bq % 2 == 0) {
                // even block: starts at dword i0 (d = low half of dword i0, qs = bytes 2..17 -> straddle dwords)
                const int i0 = 9 * (bq / 2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qd[j] = (Tsel(b, i0 + j) >> 16u) | (Tsel(b, i0 + j + 1) << 16u);
                }
                dk = Tsel(b, i0) & 0xFFFFu;
            } else {
                // odd block: starts 2 bytes into dword i0-1 (d = high half of dword i0-1, qs = dwords i0..i0+3)
                const int i0 = 9 * (bq / 2) + 5;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qd[j] = Tsel(b, i0 + j);
                }
                dk = Tsel(b, i0 - 1) >> 16u;
            }
            simd<uint16_t, 16>   dk16 = convert<uint16_t>(dk);
            simd<sycl::half, 16> dkh  = dk16.template bit_cast_view<sycl::half>();
            simd<float, 16>      dkf  = convert<float>(dkh);
            simd<int, 16>      lo[4], hi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                simd<uint32_t, 16> lo_u = qd[j] & 0x0F0F0F0Fu;
                simd<uint32_t, 16> hi_u = (qd[j] >> 4u) & 0x0F0F0F0Fu;
                lo[j] = lo_u.template bit_cast_view<int>();
                hi[j] = hi_u.template bit_cast_view<int>();
            }
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                simd<int, 16> acc = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int qlo = Qd[r][8 * bq + j];
                    const int qhi = Qd[r][8 * bq + 4 + j];
                    acc = dp4a<int, int, int, int, 16>(acc, lo[j], simd<int, 16>(qlo));
                    acc = dp4a<int, int, int, int, 16>(acc, hi[j], simd<int, 16>(qhi));
                }
                const float cc0 = c0[r][bq];
                const float cc1 = c1[r][bq];
                S[r] += dkf * (convert<float>(acc) * cc0 + cc1);
            }
        }

        // mask (f16, -inf on masked keys), same for all heads of this token
        simd<sycl::half, 16> mh = b.mk.template bit_cast_view<sycl::half>();
        simd<float, 16>      mf = convert<float>(mh);

        simd<float, 16> P[RQ];
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            S[r] += mf;
            const float     smax = hmax<float>(S[r]);
            const float     mnew = m[r] > smax ? m[r] : smax;
            simd<float, 2>  ea;
            ea[0] = m[r] - mnew;
            ea[1] = 0.0f;
            simd<float, 2>  eav = exp(ea);
            const float     a   = eav[0];
            P[r]    = exp(S[r] - mnew);
            lsum[r] = lsum[r] * a + reduce<float>(P[r], std::plus<>());
            m[r]    = mnew;
            O[r] *= a;
        }

        // P * V over the owned 128 dims (4 q4_0 blocks per key)
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            simd<uint8_t, 64> row0 = b.V0.template select<64, 1>(64 * n);
            simd<uint8_t, 32> row1 = b.V1.template select<32, 1>(32 * n);
            simd<uint16_t, 32> row0h = row0.template bit_cast_view<uint16_t>();
            simd<uint16_t, 16> row1h = row1.template bit_cast_view<uint16_t>();
#pragma unroll
            for (int bv = 0; bv < 4; ++bv) {
                simd<uint8_t, 16>  qs;
                simd<uint16_t, 1>  dbits;
                if (bv < 3) {
                    qs    = row0.template select<16, 1>(2 + 18 * bv);
                    dbits = row0h.template select<1, 1>(9 * bv);
                } else {
                    qs    = row1.template select<16, 1>(4);
                    dbits = row1h.template select<1, 1>(1);
                }
                simd<sycl::half, 1> dvh = dbits.template bit_cast_view<sycl::half>();
                simd<float, 1>      dv1 = convert<float>(dvh);
                const float     dv  = dv1[0];
                const float     dv8 = -8.0f * dv;
                simd<float, 16> vlo = convert<float>(qs & (uint8_t) 0x0F) * dv + dv8;
                simd<float, 16> vhi = convert<float>(qs >> (uint8_t) 4) * dv + dv8;
#pragma unroll
                for (int r = 0; r < RQ; ++r) {
                    const float p = P[r][n];
                    O[r].template select<16, 1>(32 * bv)      += vlo * p;
                    O[r].template select<16, 1>(32 * bv + 16) += vhi * p;
                }
            }
        }
    };

    // XMX path: scores with one int8 dpas per q4_0 block (B = the 8 nibble dwords of 16 keys, already
    // in VNNI order after the transposed load; A = q8 Q rows), P*V with fp16 dpas (A = P scaled by the
    // per-key/per-block V scale, B = byte-interleaved key pairs -> nibbles -> half, minus 8).
    auto compute_xmx = [&](blk & b) {
        simd<float, 16> S[RQ];
#pragma unroll
        for (int r = 0; r < RQ; ++r) S[r] = 0.0f;

#pragma unroll
        for (int bq = 0; bq < 8; ++bq) {
            simd<uint32_t, 16> qd[4];
            simd<uint32_t, 16> dk;
            if (bq % 2 == 0) {
                const int i0 = 9 * (bq / 2);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qd[j] = (Tsel(b, i0 + j) >> 16u) | (Tsel(b, i0 + j + 1) << 16u);
                }
                dk = Tsel(b, i0) & 0xFFFFu;
            } else {
                const int i0 = 9 * (bq / 2) + 5;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qd[j] = Tsel(b, i0 + j);
                }
                dk = Tsel(b, i0 - 1) >> 16u;
            }
            simd<uint16_t, 16>   dk16 = convert<uint16_t>(dk);
            simd<sycl::half, 16> dkh  = dk16.template bit_cast_view<sycl::half>();
            simd<float, 16>      dkf  = convert<float>(dkh);
            simd<uint32_t, 128>  Bt;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                Bt.template select<16, 1>(16 * j)      = qd[j] & 0x0F0F0F0Fu;
                Bt.template select<16, 1>(64 + 16 * j) = (qd[j] >> 4u) & 0x0F0F0F0Fu;
            }
            simd<uint8_t, 512> B8 = Bt.template bit_cast_view<uint8_t>();
            simd<int, MR * 16> C  = xmx::dpas<8, MR, int, uint8_t, int8_t>(B8, Qa[bq]);
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                simd<int, 16> acc = C.template select<16, 1>(16 * r);
                const float   cc0 = c0[r][bq];
                const float   cc1 = c1[r][bq];
                S[r] += dkf * (convert<float>(acc) * cc0 + cc1);
            }
        }

        simd<sycl::half, 16> mh = b.mk.template bit_cast_view<sycl::half>();
        simd<float, 16>      mf = convert<float>(mh);

        simd<float, 16> P[RQ];
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            S[r] += mf;
            const float     smax = hmax<float>(S[r]);
            const float     mnew = m[r] > smax ? m[r] : smax;
            simd<float, 2>  ea;
            ea[0] = m[r] - mnew;
            ea[1] = 0.0f;
            simd<float, 2>  eav = exp(ea);
            const float     a   = eav[0];
            P[r]    = exp(S[r] - mnew);
            lsum[r] = lsum[r] * a + reduce<float>(P[r], std::plus<>());
            m[r]    = mnew;
#pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                Og[g2].template select<16, 1>(16 * r) *= a;
            }
        }

        // P*V: A = hf(P * dv_bv) [8 rows][16 keys]; B = key-pair interleaved nibbles as hf(1024 + nib) - 1032
        // (0x6400 | nib is the fp16 encoding of 1024 + nib, so no int->float conversion is needed).
        // A is split into hi + lo fp16 parts (x = hi + lo to ~22 bits) and fed through two dpas each:
        // a single fp16 A costs +0.8% PPL on the decode path, the split brings it back to the fp32 kernels.
        simd<sycl::half, MR * 16> A, Al;
        if constexpr (RQ < MR) {
            A.template select<(MR - RQ) * 16, 1>(RQ * 16)  = 0;
            Al.template select<(MR - RQ) * 16, 1>(RQ * 16) = 0;
        }
#pragma unroll
        for (int bv = 0; bv < 4; ++bv) {
            simd<uint32_t, 16>   dbits32 = (bv % 2 == 0) ? (b.Vd[bv] & 0xFFFFu) : (b.Vd[bv] >> 16u);
            simd<uint16_t, 16>   dbits   = convert<uint16_t>(dbits32);
            simd<sycl::half, 16> dvh     = dbits.template bit_cast_view<sycl::half>();
            simd<float, 16>      dvf     = convert<float>(dvh);
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                simd<float, 16>      x  = P[r] * dvf;
                simd<sycl::half, 16> xh = convert<sycl::half>(x);
                A.template select<16, 1>(16 * r)  = xh;
                Al.template select<16, 1>(16 * r) = convert<sycl::half>(x - convert<float>(xh));
            }
            simd<uint16_t, 256> Blo16, Bhi16;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                simd<uint16_t, 32> X;  // X[2n + j] = qs byte n of key 2p + j
                if (bv < 3) {
                    X.template select<16, 2>(0) = b.V0.template select<16, 1>(64 * (2 * p) + 2 + 18 * bv);
                    X.template select<16, 2>(1) = b.V0.template select<16, 1>(64 * (2 * p + 1) + 2 + 18 * bv);
                } else {
                    X.template select<16, 2>(0) = b.V1.template select<16, 1>(32 * (2 * p) + 4);
                    X.template select<16, 2>(1) = b.V1.template select<16, 1>(32 * (2 * p + 1) + 4);
                }
                Blo16.template select<32, 1>(32 * p) = (X & (uint16_t) 0x000F) | (uint16_t) 0x6400;
                Bhi16.template select<32, 1>(32 * p) = (X >> (uint16_t) 4) | (uint16_t) 0x6400;
            }
            simd<sycl::half, 256> Blo = Blo16.template bit_cast_view<sycl::half>();
            simd<sycl::half, 256> Bhi = Bhi16.template bit_cast_view<sycl::half>();
            Blo -= (sycl::half) 1032.0f;
            Bhi -= (sycl::half) 1032.0f;
            Og[2 * bv]     = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(Og[2 * bv], Blo, A);
            Og[2 * bv]     = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(Og[2 * bv], Blo, Al);
            Og[2 * bv + 1] = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(Og[2 * bv + 1], Bhi, A);
            Og[2 * bv + 1] = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(Og[2 * bv + 1], Bhi, Al);
        }
    };

    // ---- main loop: CH chunks in flight. KWALK: nsplit=1, walk every KEYS_WG slice (K once per Q-tile).
    for (int w = 0; w < nwalk; ++w) {
        const int key0 = (KWALK ? w : split) * KEYS_WG + t * CH * 16;
        blk B[CH];
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            if constexpr (XV == 3) break;  // timing only: no loads, no compute (prologue + reduce only)
            if (key0 + 16 * c < n_kv) load(B[c], key0 + 16 * c);
        }
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            if constexpr (XV == 3) break;
            if (key0 + 16 * c < n_kv) {
                if constexpr (XV == 1) {  // timing only: touch the loaded data, no math
                    simd<uint32_t, 16> acc = B[c].Tk[0].template select<16, 1>(0) ^ B[c].Tk4.template select<16, 1>(0) ^
                                             B[c].V0.template bit_cast_view<uint32_t>().template select<16, 1>(0) ^
                                             B[c].V1.template bit_cast_view<uint32_t>().template select<16, 1>(0);
                    Og[0].template select<16, 1>(0) += convert<float>(acc);
                } else if constexpr (XMX) {
                    compute_xmx(B[c]);
                } else {
                    compute(B[c]);
                }
            }
        }
    }
    if constexpr (XMX) {  // back to O[r][dim] for the reduce / write-out
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
#pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                O[r].template select<16, 1>(16 * g2) = Og[g2].template select<16, 1>(16 * r);
            }
        }
    }

    // ---- work-group reduce over the T key spans (per token, per D-half)
    const int slot_base = (tok * 2 + hd) * (T / 2);
#pragma unroll
    for (int s = T / 2; s >= 1; s >>= 1) {
        if (t >= s && t < 2 * s) {
            const uint32_t off = (uint32_t) (slot_base + (t - s)) * SLOT_F * 4;
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
#pragma unroll
                for (int q = 0; q < HALF / 32; ++q) {
                    slm_block_store<float, 32>(off + (r * HALF + 32 * q) * 4, O[r].template select<32, 1>(32 * q));
                }
            }
            simd<float, 2 * RQ> ml;
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                ml[r]      = m[r];
                ml[RQ + r] = lsum[r];
            }
            slm_scatter<float, 2 * RQ>(simd<uint32_t, 2 * RQ>(off + RQ * HALF * 4, 4), ml);
        }
        barrier();
        if (t < s) {
            const uint32_t     off = (uint32_t) (slot_base + t) * SLOT_F * 4;
            simd<float, 2 * RQ> ml  = slm_gather<float, 2 * RQ>(simd<uint32_t, 2 * RQ>(off + RQ * HALF * 4, 4));
#pragma unroll
            for (int r = 0; r < RQ; ++r) {
                const float    mp   = ml[r];
                const float    lp   = ml[RQ + r];
                const float    mnew = m[r] > mp ? m[r] : mp;
                simd<float, 2> ea;
                ea[0] = m[r] - mnew;
                ea[1] = mp - mnew;
                simd<float, 2> eav = exp(ea);
                const float    a   = eav[0];
                const float    ap  = eav[1];
                lsum[r] = lsum[r] * a + lp * ap;
                m[r]    = mnew;
                O[r] *= a;
#pragma unroll
                for (int q = 0; q < HALF / 32; ++q) {
                    simd<float, 32> op = slm_block_load<float, 32>(off + (r * HALF + 32 * q) * 4);
                    O[r].template select<32, 1>(32 * q) += op * ap;
                }
            }
        }
        barrier();
    }

    if (t == 0 && tok < ntok_all) {
#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            const size_t pidx = ((size_t) tok * n_head + (size_t) (h * RQ + r)) * nsplit + split;
            float *      po   = parts + pidx * 256 + HALF * hd;
            block_store<float, 64>(po, O[r].template select<64, 1>(0));
            block_store<float, 64>(po + 64, O[r].template select<64, 1>(64));
            if (hd == 0) {
                simd<float, 2> ml;
                ml[0] = m[r];
                ml[1] = lsum[r];
                scatter<float, 2>((float *) (meta + pidx), simd<uint32_t, 2>(0, 4), ml);
            }
        }
    }
}

// Q -> q8 per 32-dim block with the softmax scale folded in (one work-item per block).
// Q8[tok][head][256] int8; Qc[tok][head][0..7] = d, Qc[tok][head][8..15] = -8 * d * sum(q8).
static void fattn_q8_prepass(dpct::queue_ptr stream, const char * Q, int8_t * Q8, float * Qc, float scale,
                             int nb01, int nb02, int ntok, int n_head) {
    const size_t n = (size_t) ntok * n_head * 8;
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> wi) {
        const int    i    = (int) wi[0];
        const int    b    = i % 8;
        const int    head = (i / 8) % n_head;
        const int    tok  = i / (8 * n_head);
        const float * x   = (const float *) (Q + (size_t) nb02 * head + (size_t) nb01 * tok) + 32 * b;
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) {
            amax = sycl::fmax(amax, sycl::fabs(x[k] * scale));
        }
        const float d  = amax / 127.0f;
        const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
        int8_t * q     = Q8 + ((size_t) tok * n_head + head) * 256 + 32 * b;
        int      sum   = 0;
        for (int k = 0; k < 32; ++k) {
            const int v = (int) sycl::round(x[k] * scale * id);
            q[k] = (int8_t) v;
            sum += v;
        }
        float * c = Qc + ((size_t) tok * n_head + head) * 16;
        c[b]     = d;
        c[8 + b] = -8.0f * d * (float) sum;
    });
}

template <int RQ, int T, int NTOK, bool XMX, int CH, int XV = 0, bool PRE = false, bool KWALK = false>
void launch_fattn_dec_q4_0(dpct::queue_ptr stream, const int8_t * Q8, const float * Qc, const uint32_t * K,
                           const uint8_t * V, const char * mask, float * parts, sycl::float2 * meta,
                           int k_pitch, int k_head_b, int v_pitch, int v_head_b, int mask_stride, int n_kv,
                           int n_kvh, int n_head, int nsplit, int ntok_all) {
    const int    n_tiles = PRE ? (ntok_all + NTOK - 1) / NTOK : 1;
    const size_t wg      = (size_t) T * 2 * NTOK;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_tiles * nsplit * n_kvh * wg), sycl::range<1>(wg)),
        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            fattn_dec_q4_0<RQ, T, NTOK, XMX, CH, XV, PRE, KWALK>(Q8, Qc, K, V, mask, parts, meta, k_pitch, k_head_b, v_pitch,
                                                          v_head_b, mask_stride, n_kv, n_kvh, n_head, nsplit, ntok_all,
                                                          it);
        });
}

// key spans per (token, D-half): largest power of two <= 8 such that the SLM reduce buffer
// (T/2 * 2 * NTOK slots of RQ*130 floats) stays under 64 KB and the work-group has <= 32 threads
// (one Xe-core in 256-GRF mode).
constexpr int fattn_esimd_T(int rq, int ntok) {
    int t = 8;
    while (t > 1 && (t * ntok * rq > 126 || t * 2 * ntok > 32)) {
        t >>= 1;
    }
    return t;
}
constexpr int fattn_esimd_keys_per_wg(int rq, int ntok, int ch) { return fattn_esimd_T(rq, ntok) * ch * 16; }

// GGML_SYCL_FA_ESIMD: 0 = off, 1 = dp4a kernel [default: fp32 P*V, PPL-neutral], 2 = XMX (dpas) kernel
// (10% faster per op, fp16 hi+lo P*V: +0.35% PPL on the decode path)
static int fattn_esimd_mode() {
    static const int mode = ggml_sycl_get_env("GGML_SYCL_FA_ESIMD", -1);
    return mode;
}

static int fattn_esimd_prefill() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_FA_ESIMD_PREFILL", 0);
    return v;
}

static int fattn_esimd_prefill_dpas() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_FA_ESIMD_PREFILL_DPAS", 0);
    return v;
}

static int fattn_esimd_prefill_kvh() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_FA_ESIMD_PREFILL_KVH", 0);
    return v;
}

static int fattn_esimd_prefill_f16() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_FA_F16_DPAS", 0);
    return v;
}

}  // namespace

#endif  // GGML_SYCL_FATTN_HAS_ESIMD

bool ggml_sycl_flash_attn_ext_esimd_supported(const ggml_tensor * dst) {
#ifdef GGML_SYCL_FATTN_HAS_ESIMD
    const int mode = fattn_esimd_mode();
    if (mode == 0 || !g_ggml_sycl_enable_esimd) {
        return false;
    }
    if (mode < 0) {
        const int dev = ggml_sycl_get_device();
        if (!ggml_sycl_is_xe2_dgpu(ggml_sycl_info().devices[dev].hw_info.arch)) {
            return false;
        }
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (!Q || !K || !V || !mask || sinks) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0 ||
        mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    if (Q->ne[1] > 4 && (Q->ne[1] < 32 || (fattn_esimd_prefill() <= 0 && fattn_esimd_prefill_dpas() <= 0 &&
                                           fattn_esimd_prefill_kvh() <= 0 && fattn_esimd_prefill_f16() <= 0))) {
        return false;
    }
    if (K->ne[2] != V->ne[2] || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    if (gqa != 6 && gqa != 8) {
        return false;
    }
    if (K->ne[1] % 16 != 0 || V->ne[1] != K->ne[1] || mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] ||
        mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // LSC 2D block loads: 64 B aligned base, 16 B multiple pitch, 4 B aligned x offsets
    if (((uintptr_t) K->data & 63) || ((uintptr_t) V->data & 63) || K->nb[1] % 16 || V->nb[1] % 16 ||
        K->nb[2] % 4 || V->nb[2] % 4 || K->nb[1] > (1u << 24) || V->nb[1] > (1u << 24)) {
        return false;
    }
    if (((uintptr_t) mask->data & 31) || mask->nb[1] % 32 || Q->nb[1] % 16 || Q->nb[2] % 16 ||
        ((uintptr_t) Q->data & 15)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(dst);
    return false;
#endif
}

void ggml_sycl_flash_attn_ext_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifdef GGML_SYCL_FATTN_HAS_ESIMD
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int ntok   = (int) Q->ne[1];
    const int n_head = (int) Q->ne[2];
    const int n_kvh  = (int) K->ne[2];
    const int gqa    = n_head / n_kvh;
    const int n_kv   = (int) K->ne[1];
    const bool xmx   = fattn_esimd_mode() >= 2;
    const int ch     = 2;  // 16-key chunks per thread (both in flight); 4 spills, 1 leaves the fixed costs unamortized
    const int tile_n = ntok > 4 ? 4 : ntok;
    const int kpw    = fattn_esimd_keys_per_wg(gqa, tile_n, ch);
    const bool use_f16_prefill  = ntok > 4 && fattn_esimd_prefill_f16() > 0;
    const bool use_kvh_prefill  = ntok > 4 && fattn_esimd_prefill_kvh() > 0 && !use_f16_prefill;
    const bool use_dpas_prefill = ntok > 4 && fattn_esimd_prefill_dpas() > 0 && !use_kvh_prefill && !use_f16_prefill;
    const int nsplit = (use_dpas_prefill || use_kvh_prefill) ? 1 : (n_kv + kpw - 1) / kpw;

    if (use_f16_prefill) {
        ggml_sycl_flash_attn_ext_esimd_prefill_f16(ctx, dst);
        return;
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ggml_sycl_pool &                   pool = ctx.pool();
    dpct::queue_ptr                    stream = ctx.stream();
    ggml_sycl_pool_alloc<float>        parts(pool, (size_t) nsplit * ntok * n_head * 256);
    ggml_sycl_pool_alloc<sycl::float2> meta(pool, (size_t) nsplit * ntok * n_head);
    ggml_sycl_pool_alloc<int8_t>       q8(pool, (size_t) ntok * n_head * 256);
    ggml_sycl_pool_alloc<float>        qc(pool, (size_t) ntok * n_head * 16);

    const char *     Qp   = (const char *) Q->data;
    const uint32_t * Kp   = (const uint32_t *) K->data;
    const uint8_t *  Vp   = (const uint8_t *) V->data;
    const char *     Mp   = (const char *) mask->data;
    const int        nb01 = (int) Q->nb[1];
    const int        nb02 = (int) Q->nb[2];
    const int        kp   = (int) K->nb[1];
    const int        khd  = (int) K->nb[2];
    const int        vp   = (int) V->nb[1];
    const int        vhb  = (int) V->nb[2];
    const int        ms   = (int) mask->nb[1];

    fattn_q8_prepass(stream, Qp, q8.get(), qc.get(), scale, nb01, nb02, ntok, n_head);

    if (use_kvh_prefill) {
        if (gqa != 6) {
            GGML_ABORT("fattn esimd kvh prefill: unsupported gqa");
        }
        constexpr int NTOK_K = 8;
        constexpr int T_K    = fattn_esimd_T(6, NTOK_K);
        launch_fattn_dec_q4_0<6, T_K, NTOK_K, false, 2, 0, true, true>(
            stream, q8.get(), qc.get(), Kp, Vp, Mp, parts.get(), meta.get(), kp, khd, vp, vhb, ms, n_kv, n_kvh, n_head,
            1, ntok);
        SYCL_CHECK(0);
        {
            const int        nsplit_d = 1;
            const dpct::dim3 block_dim_combine(256, 1, 1);
            const dpct::dim3 blocks_num_combine(ntok, n_head, 1);
            const size_t     nbytes_shared_combine = (size_t) nsplit_d * sizeof(sycl::float2);
            float *          parts_ptr = parts.get();
            sycl::float2 *   meta_ptr  = meta.get();
            float *          dst_ptr   = (float *) dst->data;
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);
                cgh.parallel_for(sycl::nd_range<3>(blocks_num_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_combine_results<256>(
                                         parts_ptr, meta_ptr, dst_ptr, nsplit_d,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
        return;
    }

    if (use_dpas_prefill) {
        ggml_sycl_flash_attn_ext_esimd_prefill_dpas(ctx, dst, q8.get(), qc.get(), parts.get(), meta.get());
        SYCL_CHECK(0);
        {
            const int        nsplit_d = 1;
            const dpct::dim3 block_dim_combine(256, 1, 1);
            const dpct::dim3 blocks_num_combine(ntok, n_head, 1);
            const size_t     nbytes_shared_combine = (size_t) nsplit_d * sizeof(sycl::float2);
            float *          parts_ptr = parts.get();
            sycl::float2 *   meta_ptr  = meta.get();
            float *          dst_ptr   = (float *) dst->data;
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);
                cgh.parallel_for(sycl::nd_range<3>(blocks_num_combine * block_dim_combine, block_dim_combine),
                                 [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     GGML_UNUSED(item_ct1);
                                     flash_attn_combine_results<256>(
                                         parts_ptr, meta_ptr, dst_ptr, nsplit_d,
                                         dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        SYCL_CHECK(0);
        return;
    }

#define FA_ESIMD_CASE(RQ_, NTOK_)                                                                                    \
    if (gqa == RQ_ && ntok == NTOK_) {                                                                               \
        if (xmx) {                                                                                                   \
            launch_fattn_dec_q4_0<RQ_, fattn_esimd_T(RQ_, NTOK_), NTOK_, true, 2>(                                   \
                stream, q8.get(), qc.get(), Kp, Vp, Mp, parts.get(), meta.get(), kp, khd, vp, vhb, ms, n_kv,        \
                n_kvh, n_head, nsplit, ntok);                                                                        \
        } else {                                                                                                     \
            launch_fattn_dec_q4_0<RQ_, fattn_esimd_T(RQ_, NTOK_), NTOK_, false, 2>(                                  \
                stream, q8.get(), qc.get(), Kp, Vp, Mp, parts.get(), meta.get(), kp, khd, vp, vhb, ms, n_kv,        \
                n_kvh, n_head, nsplit, ntok);                                                                        \
        }                                                                                                            \
    } else
    if (ntok > 4) {
#define FA_ESIMD_PRE(RQ_)                                                                                            \
        if (gqa == RQ_) {                                                                                            \
            launch_fattn_dec_q4_0<RQ_, fattn_esimd_T(RQ_, 4), 4, false, 2, 0, true>(                                 \
                stream, q8.get(), qc.get(), Kp, Vp, Mp, parts.get(), meta.get(), kp, khd, vp, vhb, ms, n_kv,        \
                n_kvh, n_head, nsplit, ntok);                                                                        \
        } else
        FA_ESIMD_PRE(6) FA_ESIMD_PRE(8) { GGML_ABORT("fattn esimd prefill: unsupported gqa"); }
#undef FA_ESIMD_PRE
    } else
    FA_ESIMD_CASE(6, 1) FA_ESIMD_CASE(6, 2) FA_ESIMD_CASE(6, 3) FA_ESIMD_CASE(6, 4)
    FA_ESIMD_CASE(8, 1) FA_ESIMD_CASE(8, 2) FA_ESIMD_CASE(8, 3) FA_ESIMD_CASE(8, 4)
    {
        GGML_ABORT("fattn esimd: unsupported gqa/ntok");
    }
#undef FA_ESIMD_CASE
    SYCL_CHECK(0);

    // combine the key splits: same layout as launch_fattn's parallel_blocks path
    {
        const dpct::dim3 block_dim_combine(256, 1, 1);
        const dpct::dim3 blocks_num_combine(ntok, n_head, 1);
        const size_t     nbytes_shared_combine = (size_t) nsplit * sizeof(sycl::float2);
        float *          parts_ptr = parts.get();
        sycl::float2 *   meta_ptr  = meta.get();
        float *          dst_ptr   = (float *) dst->data;
        const int        parallel_blocks = nsplit;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(sycl::range<1>(nbytes_shared_combine), cgh);
            cgh.parallel_for(sycl::nd_range<3>(blocks_num_combine * block_dim_combine, block_dim_combine),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 GGML_UNUSED(item_ct1);
                                 flash_attn_combine_results<256>(
                                     parts_ptr, meta_ptr, dst_ptr, parallel_blocks,
                                     dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
    }
    SYCL_CHECK(0);
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn esimd: not compiled");
#endif
}
