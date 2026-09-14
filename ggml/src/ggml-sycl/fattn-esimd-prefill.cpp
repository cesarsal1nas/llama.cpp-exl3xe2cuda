// FA-2 prefill: Q-tile BM=8 as XMX dpas rows, K walked once per WG (no nsplit).
// Not fattn_dec_q4_0 PRE/NTOK grid (KERNEL_CUTS fa-esimd-prefill).
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
    #define GGML_SYCL_FATTN_PREFILL_DPAS
#endif

extern int g_ggml_sycl_enable_esimd;

#ifdef GGML_SYCL_FATTN_PREFILL_DPAS

namespace {

using namespace sycl::ext::intel::esimd;
namespace ex  = sycl::ext::intel::experimental::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

static int fattn_esimd_prefill_dpas_env() {
    static const int v = ggml_sycl_get_env("GGML_SYCL_FA_ESIMD_PREFILL_DPAS", 0);
    return v;
}

// BM query tokens of ONE query head = XMX MR=8 rows. T=8 K-spans, 2 D-halves.
// Each thread online-softmax over its strided keys, then SLM-reduces T. nsplit=1.
template <int BM, int T, int CH>
ESIMD_INLINE void fattn_prefill_q4_0_dpas(const int8_t * Q8, const float * Qc, const uint32_t * K, const uint8_t * V,
                                          const char * mask, float * parts, sycl::float2 * meta, const int k_pitch,
                                          const int k_head_b, const int v_pitch, const int v_head_b,
                                          const int mask_stride, const int n_kv, const int n_kvh, const int n_head,
                                          const int ntok, const int gqa, const sycl::nd_item<1> & it) {
    constexpr int MR     = 8;
    constexpr int HALF   = 128;
    constexpr int KEYS   = T * CH * 16;
    constexpr int SLOT_F = BM * HALF + 2 * BM;
    constexpr int NSLOT  = (T / 2 > 0 ? T / 2 : 1) * 2;
    static_assert(BM == MR, "XMX MR=8");
    slm_init<NSLOT * SLOT_F * 4>();

    const int g     = it.get_group(0);
    const int hq    = g % n_head;  // query head
    const int qtile = g / n_head;
    const int h     = hq / gqa;    // kv head (GQA share)
    const int l     = it.get_local_id(0);
    const int t     = l % T;
    const int hd    = (l / T) & 1;
    const int tok0  = qtile * BM;

    simd<int, 64>        Qd[BM];
    simd<int8_t, MR * 32> Qa[8];
    simd<float, 8>       c0[BM], c1[BM];
#pragma unroll
    for (int r = 0; r < BM; ++r) {
        const int tok = tok0 + r < ntok ? tok0 + r : ntok - 1;
        const size_t qrow = (size_t) tok * n_head + (size_t) hq;
        simd<int8_t, 256> q8 = block_load<int8_t, 256>(Q8 + qrow * 256);
        simd<float, 16>   cc = block_load<float, 16>(Qc + qrow * 16);
        c0[r] = cc.template select<8, 1>(0);
        c1[r] = cc.template select<8, 1>(8);
        Qd[r] = q8.template bit_cast_view<int>();
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            Qa[b].template select<32, 1>(32 * r) = q8.template select<32, 1>(32 * b);
        }
    }

    simd<float, HALF>    O[BM];
    simd<float, MR * 16> Og[8];
    float m[BM], lsum[BM];
#pragma unroll
    for (int r = 0; r < BM; ++r) {
        O[r]    = 0.0f;
        m[r]    = -FLT_MAX / 2.0f;
        lsum[r] = 0.0f;
    }
#pragma unroll
    for (int g2 = 0; g2 < 8; ++g2) {
        Og[g2] = 0.0f;
    }

    struct blk {
        simd<uint32_t, 128> Tk[4];
        simd<uint32_t, 64>  Tk4;
        simd<uint8_t, 1024> V0;
        simd<uint8_t, 512>  V1;
        simd<uint32_t, 16>  Vd[4];
        simd<uint16_t, 16>  mk[BM];
    };

    const size_t     khb = (size_t) k_head_b * h;
    const size_t     vhb = (size_t) v_head_b * h;
    const uint32_t * Kb  = (const uint32_t *) ((const char *) K + (khb & ~(size_t) 63));
    const uint8_t *  Vb  = V + (vhb & ~(size_t) 63);
    const int        kx0 = (int) (khb & 63) / 4;
    const int        kw  = (int) (khb & 63) + 144 - 1;
    const int        vx0 = (int) (vhb & 63) + 72 * hd;
    const int        vw  = (int) (vhb & 63) + 144 - 1;

    auto load = [&](blk & b, int key) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            b.Tk[i] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(Kb, kw, n_kv - 1, k_pitch - 1, kx0 + 8 * i, key);
        }
        b.Tk4 = ex::lsc_load_2d<uint32_t, 4, 16, 1, true, false>(Kb, kw, n_kv - 1, k_pitch - 1, kx0 + 32, key);
        b.V0  = ex::lsc_load_2d<uint8_t, 64, 16, 1, false, false>(Vb, vw, n_kv - 1, v_pitch - 1, vx0, key);
        b.V1  = ex::lsc_load_2d<uint8_t, 32, 16, 1, false, false>(Vb, vw, n_kv - 1, v_pitch - 1, vx0 + 52, key);
        const uint32_t * Vb32 = (const uint32_t *) Vb;
#pragma unroll
        for (int bv = 0; bv < 4; ++bv) {
            b.Vd[bv] = ex::lsc_load_2d<uint32_t, 1, 16, 1, true, false>(Vb32, vw, n_kv - 1, v_pitch - 1,
                                                                        (vx0 + 18 * bv) / 4, key);
        }
#pragma unroll
        for (int r = 0; r < BM; ++r) {
            const int tok = tok0 + r < ntok ? tok0 + r : ntok - 1;
            const uint16_t * mrow = (const uint16_t *) (mask + (size_t) mask_stride * tok);
            b.mk[r] = block_load<uint16_t, 16>(mrow + key);
        }
    };

    auto Tsel = [&](blk & b, int i) -> simd<uint32_t, 16> {
        if (i < 32) {
            return b.Tk[i / 8].template select<16, 1>(16 * (i % 8));
        }
        return b.Tk4.template select<16, 1>(16 * (i - 32));
    };

    auto compute_xmx = [&](blk & b) {
        simd<float, 16> S[BM];
#pragma unroll
        for (int r = 0; r < BM; ++r) {
            S[r] = 0.0f;
        }
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
            for (int r = 0; r < BM; ++r) {
                simd<int, 16> acc = C.template select<16, 1>(16 * r);
                const float   cc0 = c0[r][bq];
                const float   cc1 = c1[r][bq];
                S[r] += dkf * (convert<float>(acc) * cc0 + cc1);
            }
        }

        simd<float, 16> P[BM];
#pragma unroll
        for (int r = 0; r < BM; ++r) {
            simd<sycl::half, 16> mh = b.mk[r].template bit_cast_view<sycl::half>();
            S[r] += convert<float>(mh);
            const float smax = hmax<float>(S[r]);
            const float mnew = m[r] > smax ? m[r] : smax;
            simd<float, 2> ea;
            ea[0] = m[r] - mnew;
            ea[1] = 0.0f;
            simd<float, 2> eav = exp(ea);
            const float    a   = eav[0];
            P[r]    = exp(S[r] - mnew);
            lsum[r] = lsum[r] * a + reduce<float>(P[r], std::plus<>());
            m[r]    = mnew;
#pragma unroll
            for (int g2 = 0; g2 < 8; ++g2) {
                Og[g2].template select<16, 1>(16 * r) *= a;
            }
        }

        simd<sycl::half, MR * 16> A, Al;
#pragma unroll
        for (int bv = 0; bv < 4; ++bv) {
            simd<uint32_t, 16>   dbits32 = (bv % 2 == 0) ? (b.Vd[bv] & 0xFFFFu) : (b.Vd[bv] >> 16u);
            simd<uint16_t, 16>   dbits   = convert<uint16_t>(dbits32);
            simd<sycl::half, 16> dvh     = dbits.template bit_cast_view<sycl::half>();
            simd<float, 16>      dvf     = convert<float>(dvh);
#pragma unroll
            for (int r = 0; r < BM; ++r) {
                simd<float, 16>      x  = P[r] * dvf;
                simd<sycl::half, 16> xh = convert<sycl::half>(x);
                A.template select<16, 1>(16 * r)  = xh;
                Al.template select<16, 1>(16 * r) = convert<sycl::half>(x - convert<float>(xh));
            }
            simd<uint16_t, 256> Blo16, Bhi16;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                simd<uint16_t, 32> X;
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

    for (int kbase = 0; kbase < n_kv; kbase += KEYS) {
        const int key0 = kbase + t * CH * 16;
        blk B[CH];
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            if (key0 + 16 * c < n_kv) {
                load(B[c], key0 + 16 * c);
            }
        }
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            if (key0 + 16 * c < n_kv) {
                compute_xmx(B[c]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < BM; ++r) {
#pragma unroll
        for (int g2 = 0; g2 < 8; ++g2) {
            O[r].template select<16, 1>(16 * g2) = Og[g2].template select<16, 1>(16 * r);
        }
    }

    const int slot_base = hd * (T / 2);
#pragma unroll
    for (int s = T / 2; s >= 1; s >>= 1) {
        if (t >= s && t < 2 * s) {
            const uint32_t off = (uint32_t) (slot_base + (t - s)) * SLOT_F * 4;
#pragma unroll
            for (int r = 0; r < BM; ++r) {
#pragma unroll
                for (int q = 0; q < HALF / 32; ++q) {
                    slm_block_store<float, 32>(off + (r * HALF + 32 * q) * 4, O[r].template select<32, 1>(32 * q));
                }
            }
            simd<float, 2 * BM> ml;
#pragma unroll
            for (int r = 0; r < BM; ++r) {
                ml[r]       = m[r];
                ml[BM + r]  = lsum[r];
            }
            slm_scatter<float, 2 * BM>(simd<uint32_t, 2 * BM>(off + BM * HALF * 4, 4), ml);
        }
        barrier();
        if (t < s) {
            const uint32_t      off = (uint32_t) (slot_base + t) * SLOT_F * 4;
            simd<float, 2 * BM> ml  = slm_gather<float, 2 * BM>(simd<uint32_t, 2 * BM>(off + BM * HALF * 4, 4));
#pragma unroll
            for (int r = 0; r < BM; ++r) {
                const float    mp   = ml[r];
                const float    lp   = ml[BM + r];
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

    if (t == 0) {
#pragma unroll
        for (int r = 0; r < BM; ++r) {
            const int tok = tok0 + r;
            if (tok >= ntok) {
                continue;
            }
            const size_t pidx = (size_t) tok * n_head + (size_t) hq;
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

// f16×f16 XMX dpas on staged dense Q/K/V (same buffers oneDNN Graph SDPA consumes).
// kv-head grid, GQA RQ heads share each 16-key K tile. Not q4_0-in-kernel.
template <int BM, int RQ>
ESIMD_INLINE void fattn_prefill_f16_dpas(const sycl::half * Qf, const sycl::half * Kf, const sycl::half * Vf,
                                         const char * mask, float * dst, const int mask_stride, const int n_kv,
                                         const int n_kvh, const int n_head, const int ntok, const float scale,
                                         const sycl::nd_item<1> & it) {
    constexpr int MR   = 8;
    constexpr int D    = 256;
    constexpr int HALF = 128;
    static_assert(BM == MR, "XMX MR=8");
    slm_init<2 * RQ * BM * HALF * 4>();

    const int g     = it.get_group(0);
    const int h     = g % n_kvh;
    const int qtile = g / n_kvh;
    const int hd    = (int) it.get_local_id(0) & 1;
    const int tok0  = qtile * BM;
    const uint32_t o_base = (uint32_t) hd * RQ * BM * HALF * 4;

    simd<float, HALF> O[RQ][BM];
    float m[RQ][BM], lsum[RQ][BM];
#pragma unroll
    for (int r = 0; r < RQ; ++r) {
#pragma unroll
        for (int b = 0; b < BM; ++b) {
            O[r][b]    = 0.0f;
            m[r][b]    = -FLT_MAX / 2.0f;
            lsum[r][b] = 0.0f;
        }
    }

    auto load16 = [&](const sycl::half * p) { return block_load<sycl::half, 16>(p); };

    for (int key = 0; key < n_kv; key += 16) {
        simd<sycl::half, 16> Kd[16][HALF / 16];
        simd<sycl::half, 16> Vd[16][HALF / 16];
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            const int kk = key + n < n_kv ? key + n : n_kv - 1;
            const sycl::half * kp = Kf + ((size_t) h * n_kv + (size_t) kk) * D + hd * HALF;
            const sycl::half * vp = Vf + ((size_t) h * n_kv + (size_t) kk) * D + hd * HALF;
#pragma unroll
            for (int dd = 0; dd < HALF / 16; ++dd) {
                Kd[n][dd] = load16(kp + 16 * dd);
                Vd[n][dd] = load16(vp + 16 * dd);
            }
        }
        simd<sycl::half, 16> mk[BM];
#pragma unroll
        for (int b = 0; b < BM; ++b) {
            const int tok = tok0 + b < ntok ? tok0 + b : ntok - 1;
            const uint16_t * mrow = (const uint16_t *) (mask + (size_t) mask_stride * tok);
            simd<uint16_t, 16> mu = block_load<uint16_t, 16>(mrow + key);
            mk[b] = mu.template bit_cast_view<sycl::half>();
        }

#pragma unroll
        for (int r = 0; r < RQ; ++r) {
            const int hq = h * RQ + r;
            simd<sycl::half, 16> Qd[BM][HALF / 16];
#pragma unroll
            for (int b = 0; b < BM; ++b) {
                const int tok = tok0 + b < ntok ? tok0 + b : ntok - 1;
                const sycl::half * qp = Qf + ((size_t) hq * ntok + (size_t) tok) * D + hd * HALF;
#pragma unroll
                for (int dd = 0; dd < HALF / 16; ++dd) {
                    Qd[b][dd] = load16(qp + 16 * dd);
                }
            }

            simd<float, 16> S[BM];
#pragma unroll
            for (int b = 0; b < BM; ++b) {
                S[b] = 0.0f;
            }
#pragma unroll
            for (int dd = 0; dd < HALF / 16; ++dd) {
                simd<sycl::half, MR * 16> A;
#pragma unroll
                for (int b = 0; b < BM; ++b) {
                    A.template select<16, 1>(16 * b) = Qd[b][dd];
                }
                simd<sycl::half, 256> B;
#pragma unroll
                for (int n = 0; n < 16; ++n) {
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) {
                        simd<sycl::half, 2> p;
                        p[0] = Kd[n][dd][2 * kk];
                        p[1] = Kd[n][dd][2 * kk + 1];
                        B.template bit_cast_view<uint32_t>()[kk * 16 + n] = p.template bit_cast_view<uint32_t>()[0];
                    }
                }
                simd<float, MR * 16> C = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(
                    simd<float, MR * 16>(0), B, A);
#pragma unroll
                for (int b = 0; b < BM; ++b) {
                    S[b] += C.template select<16, 1>(16 * b);
                }
            }

            simd<float, 16> P[BM];
#pragma unroll
            for (int b = 0; b < BM; ++b) {
                S[b] = S[b] * scale + convert<float>(mk[b]);
                const float smax = hmax<float>(S[b]);
                const float mnew = m[r][b] > smax ? m[r][b] : smax;
                simd<float, 1> ea;
                ea[0] = m[r][b] - mnew;
                const float a = exp(ea)[0];
                P[b]        = exp(S[b] - mnew);
                lsum[r][b]  = lsum[r][b] * a + reduce<float>(P[b], std::plus<>());
                m[r][b]     = mnew;
                O[r][b] *= a;
            }

#pragma unroll
            for (int dd = 0; dd < HALF / 16; ++dd) {
                simd<sycl::half, MR * 16> A;
#pragma unroll
                for (int b = 0; b < BM; ++b) {
                    A.template select<16, 1>(16 * b) = convert<sycl::half>(P[b]);
                }
                simd<sycl::half, 256> B;
#pragma unroll
                for (int n = 0; n < 16; ++n) {
#pragma unroll
                    for (int kk = 0; kk < 8; ++kk) {
                        simd<sycl::half, 2> p;
                        p[0] = Vd[n][dd][2 * kk];
                        p[1] = Vd[n][dd][2 * kk + 1];
                        B.template bit_cast_view<uint32_t>()[kk * 16 + n] = p.template bit_cast_view<uint32_t>()[0];
                    }
                }
                simd<float, MR * 16> C = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(
                    simd<float, MR * 16>(0), B, A);
#pragma unroll
                for (int b = 0; b < BM; ++b) {
                    O[r][b].template select<16, 1>(16 * dd) += C.template select<16, 1>(16 * b);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RQ; ++r) {
#pragma unroll
        for (int b = 0; b < BM; ++b) {
            const int tok = tok0 + b;
            if (tok >= ntok) {
                continue;
            }
            const int hq = h * RQ + r;
            float * po = dst + ((size_t) tok * n_head + (size_t) hq) * D + hd * HALF;
            const float inv = lsum[r][b] > 0.0f ? 1.0f / lsum[r][b] : 0.0f;
            simd<float, HALF> out = O[r][b] * inv;
            block_store<float, 64>(po, out.template select<64, 1>(0));
            block_store<float, 64>(po + 64, out.template select<64, 1>(64));
        }
    }
    (void) o_base;
}

}  // namespace

#endif

bool ggml_sycl_flash_attn_ext_esimd_prefill_dpas_on() {
#ifdef GGML_SYCL_FATTN_PREFILL_DPAS
    return fattn_esimd_prefill_dpas_env() != 0 && g_ggml_sycl_enable_esimd;
#else
    return false;
#endif
}

void ggml_sycl_flash_attn_ext_esimd_prefill_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                                 const int8_t * q8, const float * qc, float * parts,
                                                 sycl::float2 * meta) {
#ifdef GGML_SYCL_FATTN_PREFILL_DPAS
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int ntok   = (int) Q->ne[1];
    const int n_head = (int) Q->ne[2];
    const int n_kvh  = (int) K->ne[2];
    const int gqa    = n_head / n_kvh;
    const int n_kv   = (int) K->ne[1];
    constexpr int BM = 8;
    constexpr int T  = 8;
    constexpr int CH = 2;
    const int n_tiles = (ntok + BM - 1) / BM;
    const size_t wg   = (size_t) T * 2;
    dpct::queue_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_head * n_tiles * wg), sycl::range<1>(wg)),
        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            fattn_prefill_q4_0_dpas<BM, T, CH>(q8, qc, (const uint32_t *) K->data, (const uint8_t *) V->data,
                                               (const char *) mask->data, parts, meta, (int) K->nb[1], (int) K->nb[2],
                                               (int) V->nb[1], (int) V->nb[2], (int) mask->nb[1], n_kv, n_kvh, n_head,
                                               ntok, gqa, it);
        });
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(q8);
    GGML_UNUSED(qc);
    GGML_UNUSED(parts);
    GGML_UNUSED(meta);
    GGML_ABORT("fattn prefill dpas: not compiled");
#endif
}

void ggml_sycl_flash_attn_ext_esimd_prefill_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
#ifdef GGML_SYCL_FATTN_PREFILL_DPAS
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int ntok   = (int) Q->ne[1];
    const int n_head = (int) Q->ne[2];
    const int n_kvh  = (int) K->ne[2];
    const int gqa    = n_head / n_kvh;
    const int n_kv   = (int) K->ne[1];
    constexpr int BM = 8;
    constexpr int RQ = 6;
    if (gqa != RQ) {
        GGML_ABORT("fattn f16 dpas: gqa!=6");
    }
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    ggml_sycl_pool & pool  = ctx.pool();
    ggml_sycl_pool_alloc<sycl::half> Qf(pool, (size_t) ntok * n_head * 256);
    ggml_sycl_pool_alloc<sycl::half> Kf(pool, (size_t) n_kv * n_kvh * 256);
    ggml_sycl_pool_alloc<sycl::half> Vf(pool, (size_t) n_kv * n_kvh * 256);

    {
        const int64_t d = 256, qn = ntok, H = n_head, mb = 1;
        const size_t  nb1 = Q->nb[1], nb2 = Q->nb[2], nb3 = Q->nb[3];
        const char *  qs  = (const char *) Q->data;
        sycl::half *  qd  = Qf.get();
        const int64_t n   = d * qn * H * mb;
        stream->parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> ix) {
            int64_t i = (int64_t) ix[0];
            const int64_t i0 = i % d;
            i /= d;
            const int64_t i1 = i % qn;
            i /= qn;
            const int64_t i2 = i % H;
            const int64_t i3 = i / H;
            const float * p = (const float *) (qs + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
            qd[ix[0]] = (sycl::half) (*p);
        });
    }
    auto stage_q4 = [&](const ggml_tensor * T, sycl::half * out) {
        const char * data = (const char *) T->data;
        const bool non_dense = ((int64_t) T->ne[1] * T->nb[1] != T->nb[2]) && T->ne[2] > 1;
        if (ggml_is_contiguously_allocated(T) && !non_dense) {
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(T->type, dst);
            to_fp16(data, out, ggml_nelements(T), stream);
        } else {
            const size_t ts = ggml_type_size(T->type);
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(T->type);
            const int64_t s01 = (int64_t) T->nb[1] / ts;
            const int64_t s02 = (int64_t) T->nb[2] / ts;
            const int64_t s03 = (int64_t) T->nb[3] / ts;
            to_fp16(data, out, T->ne[0], T->ne[1], T->ne[2], T->ne[3], s01, s02, s03, stream);
        }
    };
    stage_q4(K, Kf.get());
    stage_q4(V, Vf.get());

    const int n_tiles = (ntok + BM - 1) / BM;
    const size_t wg   = 2;
    const sycl::half * Qp = Qf.get();
    const sycl::half * Kp = Kf.get();
    const sycl::half * Vp = Vf.get();
    float *            Dp = (float *) dst->data;
    const char *       Mp = (const char *) mask->data;
    const int          ms = (int) mask->nb[1];
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_kvh * n_tiles * wg), sycl::range<1>(wg)),
        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            fattn_prefill_f16_dpas<BM, RQ>(Qp, Kp, Vp, Mp, Dp, ms, n_kv, n_kvh, n_head, ntok, scale, it);
        });
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn prefill f16 dpas: not compiled");
#endif
}
