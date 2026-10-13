#include "xe2mx-a8.hpp"

#include <cstdint>
#include <cstdio>

#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define GGML_SYCL_XE2MX_XMX_HAS_ESIMD
#endif

#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD

constexpr int XE2MX_WG  = 4;
constexpr int XE2MX_RPW = 16;
constexpr int XE2MX_KB  = 8;
constexpr int XE2MX_S8_KB = 4;

namespace {

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace ex  = sycl::ext::intel::experimental::esimd;

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_soa_core(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                                int K, int N, int Ndst, int M, int n_row_groups, uint32_t xorm, const int wg,
                                const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<SLM_SUMS + WG * RPW * NCM * 4>();

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qs32   = (const uint32_t *) vx;
    const uint16_t * d16    = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        tid    = it.get_local_id(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0   = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0   = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> acc[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct blk {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](blk & b) {
        simd<uint16_t, KB * 16> dblk = ex::lsc_load_2d<uint16_t, 16, KB>(
            d16, (unsigned) N * 2u - 1u, (unsigned) nb - 1u, (unsigned) N * 2u - 1u, row0, xib * KB);
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = dblk.template select<16, 1>(16 * s);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, xib * (KB * 4) + 8 * k, row0);
        }
        xib += WG;
    };
    auto load_a = [&](blk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };

    blk Ablk, Bblk;
    if (tid < nfat) {
        load_w(Ablk);
    }

    (void) xorm;
    auto math = [&](blk & b) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(b, t);
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
                B4 = B4 ^ simd<uint32_t, 64>(xorm);
                simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
                simd<float, 16> dw = convert<float>(dh);
                simd<int8_t, NC * 32> Am;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
                }
                simd<int, NC * 16> Ci =
                    xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                        simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                    simd<float, KB> dar8 = b.dar[c];
                    const float darc = dar8[sb];
                    acc[t * NC + c] += convert<float>(ci) * dw * darc;
                }
            }
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        load_w(Bblk);
        __asm__ volatile("");
        math(Ablk);
        if (ib + 2 * WG < nfat) {
            load_w(Ablk);
        }
        __asm__ volatile("");
        math(Bblk);
    }
    if (ib < nfat) {
        math(Ablk);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, acc[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) Ndst;
            scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) Ndst, ridx * 4u, u, msk);
        }
    }
}

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_soa(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                           int K, int N, int M, int n_row_groups, uint32_t xorm, const sycl::nd_item<1> & it) {
    gemm_soa_core<WG, KB, NC, NM>(vx, A, da, C, K, N, N, M, n_row_groups, xorm, (int) it.get_group(0), it);
}

// Multi-GEMV: up to 4 weights that share A (same K). Workgroups cover the weights' row groups back to back,
// so each weight runs the exact gemm_soa body with its own local workgroup index. One launch, one A quant.
struct xe2mx_mg_args {
    const uint8_t * vx[8];
    float *         C[8];
    int             N[8];       // weight rows, multiple of 16 (padded)
    int             Ndst[8];    // stored columns; dst row stride. Equals N when unpadded.
    int             wg_end[8];  // cumulative workgroup count, weight w owns [wg_end[w-1], wg_end[w])
    int             nw;
};

template <int WG, int KB, int NC, int NM>
static bool launch_soa_mg(queue_ptr stream, const xe2mx_mg_args & g, const int8_t * A, const float * da,
                          int64_t K, int64_t M) {
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const int total = g.wg_end[g.nw - 1];
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) total * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int gw = (int) it.get_group(0);
            int w = 0;
            int base = 0;
#pragma unroll
            for (int j = 0; j < 7; ++j) {
                if (j + 1 < g.nw && gw >= g.wg_end[j]) {
                    w = j + 1;
                    base = g.wg_end[j];
                }
            }
            gemm_soa_core<WG, KB, NC, NM>(g.vx[w], A, da, g.C[w], (int) K, g.N[w], g.Ndst[w], (int) M,
                                          g.N[w] / XE2MX_RPW, xorm, gw - base, it);
        });
    return true;
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                        int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa<WG, KB, NC, NM>(packed, A, da, C, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

// GGML_SYCL_XE2MX_BATCH: independent weights, each with its own A/da/K/N/M, one launch. Entry w owns
// workgroups [wg_end[w-1], wg_end[w]) and runs gemm_soa_core with its local index, Ndst == N and
// n_row_groups == N / RPW, which is exactly what launch_soa hands gemm_soa for that weight alone.
struct xe2mx_bt_args {
    const uint8_t * vx[XE2MX_BATCH_MAX];
    const int8_t *  A[XE2MX_BATCH_MAX];
    const float *   da[XE2MX_BATCH_MAX];
    float *         C[XE2MX_BATCH_MAX];
    int             K[XE2MX_BATCH_MAX];
    int             N[XE2MX_BATCH_MAX];
    int             M[XE2MX_BATCH_MAX];
    int             wg_end[XE2MX_BATCH_MAX];
    int             nw;
};

template <int WG, int KB, int NC, int NM>
static bool launch_soa_batch(queue_ptr stream, const xe2mx_bt_args & g) {
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const int total = g.wg_end[g.nw - 1];
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) total * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int gw = (int) it.get_group(0);
            int w = 0;
            int base = 0;
#pragma unroll
            for (int j = 0; j < XE2MX_BATCH_MAX - 1; ++j) {
                if (j + 1 < g.nw && gw >= g.wg_end[j]) {
                    w = j + 1;
                    base = g.wg_end[j];
                }
            }
            gemm_soa_core<WG, KB, NC, NM>(g.vx[w], g.A[w], g.da[w], g.C[w], g.K[w], g.N[w], g.N[w], g.M[w],
                                          g.N[w] / XE2MX_RPW, xorm, gw - base, it);
        });
    return true;
}

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_soa_glu(const uint8_t * vxg, const uint8_t * vxu, const int8_t * A, const float * da, float * C,
                                 float * C2, int K, int N, int M, int n_row_groups, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    constexpr uint32_t SLM_U = WG * RPW * NCM * 4;
    slm_init<SLM_SUMS + 2 * WG * RPW * NCM * 4>();

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qsg    = (const uint32_t *) vxg;
    const uint16_t * dsg    = (const uint16_t *) (vxg + (size_t) N * (size_t) pitch);
    const uint32_t * qsu    = (const uint32_t *) vxu;
    const uint16_t * dsu    = (const uint16_t *) (vxu + (size_t) N * (size_t) pitch);
    const int        tid    = it.get_local_id(0);
    const int        wg     = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0   = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0   = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> accg[NCM];
    simd<float, 16> accu[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        accg[c] = 0.0f;
        accu[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct wblk {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
    };
    struct ablk {
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](wblk & b, const uint32_t * qs32, const uint16_t * d16) {
        simd<uint16_t, KB * 16> dblk = ex::lsc_load_2d<uint16_t, 16, KB>(
            d16, (unsigned) N * 2u - 1u, (unsigned) nb - 1u, (unsigned) N * 2u - 1u, row0, xib * KB);
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = dblk.template select<16, 1>(16 * s);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, xib * (KB * 4) + 8 * k, row0);
        }
    };
    auto load_a = [&](ablk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };
    auto dpas_w = [&](wblk & b, ablk & a, simd<float, 16> * acc, int t) {
#pragma unroll
        for (int sb = 0; sb < KB; ++sb) {
            simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
            B4 = B4 ^ simd<uint32_t, 64>(xorm);
            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
            simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
            simd<float, 16> dw = convert<float>(dh);
            simd<int8_t, NC * 32> Am;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                Am.template select<32, 1>(32 * c) = a.y8[c].template select<32, 1>(32 * sb);
            }
            simd<int, NC * 16> Ci =
                xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                simd<float, KB> dar8 = a.dar[c];
                const float darc = dar8[sb];
                acc[t * NC + c] += convert<float>(ci) * dw * darc;
            }
        }
    };

    wblk Ag, Au, Bg, Bu;
    ablk Aa, Ba;
    if (tid < nfat) {
        load_w(Ag, qsg, dsg);
        load_w(Au, qsu, dsu);
        xib += WG;
    }

    auto math = [&](wblk & g, wblk & u, ablk & a) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(a, t);
            dpas_w(g, a, accg, t);
            dpas_w(u, a, accu, t);
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        load_w(Bg, qsg, dsg);
        load_w(Bu, qsu, dsu);
        xib += WG;
        __asm__ volatile("");
        math(Ag, Au, Aa);
        if (ib + 2 * WG < nfat) {
            load_w(Ag, qsg, dsg);
            load_w(Au, qsu, dsu);
            xib += WG;
        }
        __asm__ volatile("");
        math(Bg, Bu, Ba);
    }
    if (ib < nfat) {
        math(Ag, Au, Aa);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, accg[c]);
        slm_block_store<float, 16>(SLM_U + (tid * NCM + c) * 64, accu[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> ug = 0.0f;
            simd<float, RT> uu = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                ug += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
                uu += slm_block_load<float, RT>(SLM_U + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) N;
            if (C2) {
                scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, ug, msk);
                scatter<float, RT>(C2 + (size_t) (col0 + c) * (size_t) N, ridx * 4u, uu, msk);
            } else {
                simd<float, RT> out = ug / (simd<float, RT>(1.0f) + sycl::ext::intel::esimd::exp(-ug)) * uu;
                scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, out, msk);
            }
        }
    }
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa_glu(queue_ptr stream, const uint8_t * pg, const uint8_t * pu, const int8_t * A, const float * da,
                            float * C, int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto range = sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG));
    // 256 is the tuned GRF. 128 is an IGC 2.41.5 occupancy probe on the 12 ms glu kernel only.
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_GLU_GRF", 256) == 128) {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<128> };
        stream->parallel_for(range, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu<WG, KB, NC, NM>(pg, pu, A, da, C, nullptr, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    } else {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        stream->parallel_for(range, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu<WG, KB, NC, NM>(pg, pu, A, da, C, nullptr, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    }
    return true;
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa_pair(queue_ptr stream, const uint8_t * p0, const uint8_t * p1, const int8_t * A, const float * da,
                             float * C0, float * C1, int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu<WG, KB, NC, NM>(p0, p1, A, da, C0, C1, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

// Single-buffer gemm_soa_glu: same loads, dpas, accumulate order, SLM reduction and epilogue, but one
// gate wblk, one up wblk and one ablk (no Ag/Au/Bg/Bu, Aa/Ba ping-pong). Frees ~7 KB GRF so NM=2 fits.
template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_soa_glu1(const uint8_t * vxg, const uint8_t * vxu, const int8_t * A, const float * da, float * C,
                                  float * C2, int K, int N, int M, int n_row_groups, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    constexpr uint32_t SLM_U = WG * RPW * NCM * 4;
    slm_init<SLM_SUMS + 2 * WG * RPW * NCM * 4>();

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qsg    = (const uint32_t *) vxg;
    const uint16_t * dsg    = (const uint16_t *) (vxg + (size_t) N * (size_t) pitch);
    const uint32_t * qsu    = (const uint32_t *) vxu;
    const uint16_t * dsu    = (const uint16_t *) (vxu + (size_t) N * (size_t) pitch);
    const int        tid    = it.get_local_id(0);
    const int        wg     = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0   = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0   = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> accg[NCM];
    simd<float, 16> accu[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        accg[c] = 0.0f;
        accu[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct wblk {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
    };
    struct ablk {
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](wblk & b, const uint32_t * qs32, const uint16_t * d16) {
        simd<uint16_t, KB * 16> dblk = ex::lsc_load_2d<uint16_t, 16, KB>(
            d16, (unsigned) N * 2u - 1u, (unsigned) nb - 1u, (unsigned) N * 2u - 1u, row0, xib * KB);
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = dblk.template select<16, 1>(16 * s);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, xib * (KB * 4) + 8 * k, row0);
        }
    };
    auto load_a = [&](ablk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };
    auto dpas_w = [&](wblk & b, ablk & a, simd<float, 16> * acc, int t) {
#pragma unroll
        for (int sb = 0; sb < KB; ++sb) {
            simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
            B4 = B4 ^ simd<uint32_t, 64>(xorm);
            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
            simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
            simd<float, 16> dw = convert<float>(dh);
            simd<int8_t, NC * 32> Am;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                Am.template select<32, 1>(32 * c) = a.y8[c].template select<32, 1>(32 * sb);
            }
            simd<int, NC * 16> Ci =
                xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                simd<float, KB> dar8 = a.dar[c];
                const float darc = dar8[sb];
                acc[t * NC + c] += convert<float>(ci) * dw * darc;
            }
        }
    };

    wblk g, u;
    ablk a;
    for (int ib = tid; ib < nfat; ib += WG) {
        load_w(g, qsg, dsg);
        load_w(u, qsu, dsu);
        xib += WG;
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(a, t);
            dpas_w(g, a, accg, t);
            dpas_w(u, a, accu, t);
        }
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, accg[c]);
        slm_block_store<float, 16>(SLM_U + (tid * NCM + c) * 64, accu[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> ug = 0.0f;
            simd<float, RT> uu = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                ug += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
                uu += slm_block_load<float, RT>(SLM_U + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) N;
            if (C2) {
                scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, ug, msk);
                scatter<float, RT>(C2 + (size_t) (col0 + c) * (size_t) N, ridx * 4u, uu, msk);
            } else {
                simd<float, RT> out = ug / (simd<float, RT>(1.0f) + sycl::ext::intel::esimd::exp(-ug)) * uu;
                scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, out, msk);
            }
        }
    }
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa_glu1(queue_ptr stream, const uint8_t * pg, const uint8_t * pu, const int8_t * A, const float * da,
                             float * C, int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto range = sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG));
    if (ggml_sycl_get_env("GGML_SYCL_XE2MX_GLU_GRF", 256) == 128) {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<128> };
        stream->parallel_for(range, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu1<WG, KB, NC, NM>(pg, pu, A, da, C, nullptr, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    } else {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        stream->parallel_for(range, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu1<WG, KB, NC, NM>(pg, pu, A, da, C, nullptr, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    }
    return true;
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa_pair1(queue_ptr stream, const uint8_t * p0, const uint8_t * p1, const int8_t * A, const float * da,
                              float * C0, float * C1, int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_glu1<WG, KB, NC, NM>(p0, p1, A, da, C0, C1, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_soa_fn(const uint8_t * vx0, const uint8_t * vx1, const int8_t * A, const float * da,
                              float * C0, float * C1, int K, int N0, int N1, int M, int n_row_groups, uint32_t xorm,
                              const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    constexpr uint32_t SLM_U = WG * RPW * NCM * 4;
    slm_init<SLM_SUMS + 2 * WG * RPW * NCM * 4>();

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qs0    = (const uint32_t *) vx0;
    const uint16_t * ds0    = (const uint16_t *) (vx0 + (size_t) N0 * (size_t) pitch);
    const uint32_t * qs1    = (const uint32_t *) vx1;
    const uint16_t * ds1    = (const uint16_t *) (vx1 + (size_t) N1 * (size_t) pitch);
    const int        tid    = it.get_local_id(0);
    const int        wg     = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0   = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0   = row_gid * RPW;
    if (col0 >= M) {
        return;
    }
    const int do0 = row0 < N0 ? 1 : 0;
    const int do1 = row0 < N1 ? 1 : 0;
    if (!do0 && !do1) {
        return;
    }

    simd<float, 16> acc0[NCM];
    simd<float, 16> acc1[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc0[c] = 0.0f;
        acc1[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct wblk {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
    };
    struct ablk {
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](wblk & b, const uint32_t * qs32, const uint16_t * d16, int Nv) {
        simd<uint16_t, KB * 16> dblk = ex::lsc_load_2d<uint16_t, 16, KB>(
            d16, (unsigned) Nv * 2u - 1u, (unsigned) nb - 1u, (unsigned) Nv * 2u - 1u, row0, xib * KB);
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = dblk.template select<16, 1>(16 * s);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) Nv - 1u, pitch - 1u, xib * (KB * 4) + 8 * k, row0);
        }
    };
    auto load_a = [&](ablk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };
    auto dpas_w = [&](wblk & b, ablk & a, simd<float, 16> * acc, int t) {
#pragma unroll
        for (int sb = 0; sb < KB; ++sb) {
            simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
            B4 = B4 ^ simd<uint32_t, 64>(xorm);
            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
            simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
            simd<float, 16> dw = convert<float>(dh);
            simd<int8_t, NC * 32> Am;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                Am.template select<32, 1>(32 * c) = a.y8[c].template select<32, 1>(32 * sb);
            }
            simd<int, NC * 16> Ci =
                xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                simd<float, KB> dar8 = a.dar[c];
                const float darc = dar8[sb];
                acc[t * NC + c] += convert<float>(ci) * dw * darc;
            }
        }
    };

    wblk A0w, A1w, B0w, B1w;
    ablk Aa, Ba;
    if (tid < nfat) {
        if (do0) {
            load_w(A0w, qs0, ds0, N0);
        }
        if (do1) {
            load_w(A1w, qs1, ds1, N1);
        }
        xib += WG;
    }

    auto math = [&](wblk & w0, wblk & w1, ablk & a) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(a, t);
            if (do0) {
                dpas_w(w0, a, acc0, t);
            }
            if (do1) {
                dpas_w(w1, a, acc1, t);
            }
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        if (do0) {
            load_w(B0w, qs0, ds0, N0);
        }
        if (do1) {
            load_w(B1w, qs1, ds1, N1);
        }
        xib += WG;
        __asm__ volatile("");
        math(A0w, A1w, Aa);
        if (ib + 2 * WG < nfat) {
            if (do0) {
                load_w(A0w, qs0, ds0, N0);
            }
            if (do1) {
                load_w(A1w, qs1, ds1, N1);
            }
            xib += WG;
        }
        __asm__ volatile("");
        math(B0w, B1w, Ba);
    }
    if (ib < nfat) {
        math(A0w, A1w, Aa);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, acc0[c]);
        slm_block_store<float, 16>(SLM_U + (tid * NCM + c) * 64, acc1[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> u0 = 0.0f;
            simd<float, RT> u1 = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u0 += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
                u1 += slm_block_load<float, RT>(SLM_U + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            if (do0) {
                simd_mask<RT> m0 = ridx < (uint32_t) N0;
                scatter<float, RT>(C0 + (size_t) (col0 + c) * (size_t) N0, ridx * 4u, u0, m0);
            }
            if (do1) {
                simd_mask<RT> m1 = ridx < (uint32_t) N1;
                scatter<float, RT>(C1 + (size_t) (col0 + c) * (size_t) N1, ridx * 4u, u1, m1);
            }
        }
    }
}

template <int WG, int KB, int NC, int NM>
static bool launch_soa_fn(queue_ptr stream, const uint8_t * p0, const uint8_t * p1, const int8_t * A, const float * da,
                           float * C0, float * C1, int64_t K, int64_t N0, int64_t N1, int64_t M) {
    const int Nmax = N0 > N1 ? (int) N0 : (int) N1;
    const int n_row_groups = Nmax / XE2MX_RPW;
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_fn<WG, KB, NC, NM>(p0, p1, A, da, C0, C1, (int) K, (int) N0, (int) N1, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

template <int KB>
ESIMD_INLINE void gemv_soa(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                            int K, int N, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    const int        nb    = K / 32;
    const int        nfat  = nb / KB;
    const unsigned   pitch = (unsigned) nb * 16u;
    const uint32_t * qs32  = (const uint32_t *) vx;
    const uint16_t * d16   = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        row0  = (int) it.get_global_id(0) * RPW;
    if (row0 >= N) {
        return;
    }
    simd<float, 16> acc(0.0f);
    (void) xorm;
    for (int ib = 0; ib < nfat; ++ib) {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            db[s] = block_load<uint16_t, 16>(d16 + (size_t) (ib * KB + s) * (size_t) N + (size_t) row0);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, ib * (KB * 4) + 8 * k, row0);
        }
        simd<int8_t, KB * 32> y8 = block_load<int8_t, KB * 32>(A + (size_t) ib * KB * 32);
        simd<float, KB>      dar = block_load<float, KB>(da + (size_t) ib * KB);
#pragma unroll
        for (int sb = 0; sb < KB; ++sb) {
            simd<uint32_t, 64> B4 = T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
            B4 = B4 ^ simd<uint32_t, 64>(xorm);
            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
            simd<sycl::half, 16> dh = db[sb].template bit_cast_view<sycl::half>();
            simd<float, 16> dw = convert<float>(dh);
            simd<int8_t, 32> Am = y8.template select<32, 1>(32 * sb);
            simd<int, 16> Ci =
                xmx::dpas<8, 1, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    simd<int, 16>(0), Bw, Am);
            const float darc = dar[sb];
            acc += convert<float>(Ci) * dw * darc;
        }
    }
    acc.copy_to(C + (size_t) row0);
}

// Split-K decode GEMV. The old kernel gave one thread 16 rows and the whole K,
// so a 5120-wide matmul only launched a few hundred threads and sat on load latency.
// SK threads share those 16 rows, each walks a stride of the K blocks, then SLM-add.
template <int KB, int SK>
ESIMD_INLINE void gemv_soa_sk(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                               int K, int N, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    const int        nb    = K / 32;
    const int        nfat  = nb / KB;
    const unsigned   pitch = (unsigned) nb * 16u;
    const uint32_t * qs32  = (const uint32_t *) vx;
    const uint16_t * d16   = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        tid   = (int) it.get_local_id(0);
    const int        row0  = (int) it.get_group(0) * RPW;
    slm_init<SK * 64>();
    simd<float, 16> acc(0.0f);
    for (int ib = tid; ib < nfat; ib += SK) {
        const int ibn = ib + SK;
        if (ibn < nfat) {
#pragma unroll
            for (int k = 0; k < KB / 2; ++k) {
                ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                    qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, ibn * (KB * 4) + 8 * k, row0);
            }
        }
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            db[s] = block_load<uint16_t, 16>(d16 + (size_t) (ib * KB + s) * (size_t) N + (size_t) row0);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, ib * (KB * 4) + 8 * k, row0);
        }
        simd<int8_t, KB * 32> y8 = block_load<int8_t, KB * 32>(A + (size_t) ib * KB * 32);
        simd<float, KB>      dar = block_load<float, KB>(da + (size_t) ib * KB);
#pragma unroll
        for (int sb = 0; sb < KB; ++sb) {
            simd<uint32_t, 64> B4 = T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
            B4 = B4 ^ simd<uint32_t, 64>(xorm);
            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
            simd<sycl::half, 16> dh = db[sb].template bit_cast_view<sycl::half>();
            simd<float, 16> dw = convert<float>(dh);
            simd<int8_t, 32> Am = y8.template select<32, 1>(32 * sb);
            simd<int, 16> Ci =
                xmx::dpas<8, 1, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    simd<int, 16>(0), Bw, Am);
            const float darc = dar[sb];
            acc += convert<float>(Ci) * dw * darc;
        }
    }
    slm_block_store<float, 16>((uint32_t) (tid * 64), acc);
    barrier();
    if (tid == 0) {
        simd<float, 16> sum(0.0f);
#pragma unroll
        for (int s = 0; s < SK; ++s) {
            sum += slm_block_load<float, 16>((uint32_t) (s * 64));
        }
        sum.copy_to(C + (size_t) row0);
    }
}

template <int KB, int SK>
static bool launch_gemv_soa_sk(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da,
                                float * C, int64_t K, int64_t N) {
    const int n_tiles = (int) (N / XE2MX_RPW);
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_tiles * (size_t) SK), sycl::range<1>(SK)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemv_soa_sk<KB, SK>(packed, A, da, C, (int) K, (int) N, xorm, it);
        });
    return true;
}

template <int KB>
static bool launch_gemv_soa(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da,
                              float * C, int64_t K, int64_t N) {
    const int n_tiles = (int) (N / XE2MX_RPW);
    const int lgs = 16;
    const int n_groups = (n_tiles + lgs - 1) / lgs;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_groups * (size_t) lgs), sycl::range<1>(lgs)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemv_soa<KB>(packed, A, da, C, (int) K, (int) N, xorm, it);
        });
    return true;
}

template <int NC, int MAXN>
ESIMD_INLINE void gemm_ninner(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                              int K, int N, int M, const sycl::nd_item<1> & it) {
    constexpr int WG  = XE2MX_WG;
    constexpr int RPW = XE2MX_RPW;
    constexpr int KB  = XE2MX_KB;
    slm_init<WG * MAXN * RPW * 4>();

    const int        nb    = K / 32;
    const int        nfat  = nb / KB;
    const unsigned   pitch = (unsigned) nb * 16u;
    const uint32_t * qs32  = (const uint32_t *) vx;
    const uint16_t * d16   = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        tid   = it.get_local_id(0);
    const int        row0  = it.get_group(0) * RPW;

    for (int nbase = 0; nbase < M; nbase += MAXN) {
        const int nlim   = M - nbase < MAXN ? M - nbase : MAXN;
        const int ntiles = (nlim + NC - 1) / NC;
        for (int col = tid; col < nlim; col += WG) {
#pragma unroll
            for (int t = 0; t < WG; ++t) {
                slm_block_store<float, 16>((uint32_t) ((t * MAXN + col) * 64), simd<float, 16>(0.0f));
            }
        }
        barrier();

        for (int ib = tid; ib < nfat; ib += WG) {
            simd<uint32_t, 128> T[KB / 2];
            simd<uint16_t, 16>  db[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) {
                db[s] = block_load<uint16_t, 16>(d16 + (size_t) (ib * KB + s) * (size_t) N + (size_t) row0);
            }
#pragma unroll
            for (int k = 0; k < KB / 2; ++k) {
                T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                    qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, ib * (KB * 4) + 8 * k, row0);
            }
            simd<uint8_t, 256> Bw[KB];
            simd<float, 16>   dw[KB];
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<uint32_t, 64> B4 = T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
                B4 = B4 ^ simd<uint32_t, 64>(0x88888888u);
                Bw[sb] = B4.template bit_cast_view<uint8_t>();
                simd<sycl::half, 16> dh = db[sb].template bit_cast_view<sycl::half>();
                dw[sb] = convert<float>(dh);
            }
            for (int t = 0; t < ntiles; ++t) {
                const int col0 = nbase + t * NC;
                simd<int8_t, KB * 32> y8[NC];
                simd<float, KB>       dar[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int m = col0 + c;
                    if (m < M) {
                        y8[c]  = block_load<int8_t, KB * 32>(A + (size_t) m * (size_t) K + (size_t) ib * (KB * 32));
                        dar[c] = block_load<float, KB>(da + (size_t) m * (size_t) nb + (size_t) ib * KB);
                    } else {
                        y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                        dar[c] = simd<float, KB>(0.0f);
                    }
                }
                simd<float, 16> acc[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c] = 0.0f;
                }
#pragma unroll
                for (int sb = 0; sb < KB; ++sb) {
                    simd<int8_t, NC * 32> Am;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        Am.template select<32, 1>(32 * c) = y8[c].template select<32, 1>(32 * sb);
                    }
                    simd<int, NC * 16> Ci =
                        xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                            simd<int, NC * 16>(0), Bw[sb], Am);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                        simd<float, KB> dar8 = dar[c];
                        const float darc = dar8[sb];
                        acc[c] += convert<float>(ci) * dw[sb] * darc;
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int m = col0 + c;
                    if (m < M) {
                        const uint32_t off = (uint32_t) ((tid * MAXN + t * NC + c) * 64);
                        simd<float, 16> prev = slm_block_load<float, 16>(off);
                        slm_block_store<float, 16>(off, prev + acc[c]);
                    }
                }
            }
        }
        barrier();
        for (int col = tid; col < nlim; col += WG) {
            simd<float, 16> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, 16>((uint32_t) ((q * MAXN + col) * 64));
            }
            simd<uint32_t, 16> ridx((uint32_t) row0, 1);
            simd_mask<16> msk = ridx < (uint32_t) N;
            scatter<float, 16>(C + (size_t) (nbase + col) * (size_t) N, ridx * 4u, u, msk);
        }
        barrier();
    }
}

template <int NC, int MAXN>
static bool launch_ninner(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                            int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_row_groups * (size_t) XE2MX_WG), sycl::range<1>(XE2MX_WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_ninner<NC, MAXN>(packed, A, da, C, (int) K, (int) N, (int) M, it);
        });
    return true;
}

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_xmx_nt(const uint8_t * vx, const int8_t * A, const float * da, float * C,
                               int K, int N, int M, int n_row_groups, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr int NTILE = WG * RPW;

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qs32   = (const uint32_t *) vx;
    const uint16_t * d16    = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        tid     = it.get_local_id(0);
    const int        wg      = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0    = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0    = row_gid * NTILE + tid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> acc[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K;
        dap[c] = da + (size_t) mm * (size_t) nb;
    }

    struct blk {
        simd<uint32_t, 128> T[KB / 2];
        simd<uint16_t, 16>  db[KB];
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = 0;
    auto load_w = [&](blk & b) {
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = block_load<uint16_t, 16>(d16 + (size_t) (xib * KB + s) * (size_t) N + (size_t) row0);
        }
#pragma unroll
        for (int k = 0; k < KB / 2; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, xib * (KB * 4) + 8 * k, row0);
        }
        xib += 1;
    };
    auto load_a = [&](blk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += KB * 32;
                dap[t * NC + c] += KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };

    auto math = [&](blk & b) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(b, t);
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
                B4 = B4 ^ simd<uint32_t, 64>(xorm);
                simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
                simd<float, 16> dw = convert<float>(dh);
                simd<int8_t, NC * 32> Am;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
                }
                simd<int, NC * 16> Ci =
                    xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                        simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                    simd<float, KB> dar8 = b.dar[c];
                    const float darc = dar8[sb];
                    acc[t * NC + c] += convert<float>(ci) * dw * darc;
                }
            }
        }
    };

    blk Ablk, Bblk;
    if (nfat > 0) {
        load_w(Ablk);
    }

    int ib = 0;
    for (; ib + 1 < nfat; ib += 2) {
        load_w(Bblk);
        __asm__ volatile("");
        math(Ablk);
        if (ib + 2 < nfat) {
            load_w(Ablk);
        }
        __asm__ volatile("");
        math(Bblk);
    }
    if (ib < nfat) {
        math(Ablk);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            block_store<float, 16>(C + (size_t) (col0 + c) * (size_t) N + (size_t) row0, acc[c]);
        }
    }
}

template <int WG, int KB, int NC, int NM>
static bool launch_xmx_nt(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                            int64_t K, int64_t N, int64_t M) {
    constexpr int NTILE = WG * XE2MX_RPW;
    constexpr int NCM = NC * NM;
    if ((N % NTILE) != 0) {
        return false;
    }
    const int n_row_groups = (int) (N / NTILE);
    const int ncm = NCM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_xmx_nt<WG, KB, NC, NM>(packed, A, da, C, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

template <int NC, int NM>
ESIMD_INLINE void gemm_soa_f32(const uint8_t * vx, const float * x, float * C,
                                int K, int N, int M, int n_row_groups, uint32_t xorm, const sycl::nd_item<1> & it) {
    constexpr int WG  = XE2MX_WG;
    constexpr int RPW = XE2MX_RPW;
    constexpr int KB  = XE2MX_KB;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<SLM_SUMS + WG * RPW * NCM * 4>();

    const int        nb     = K / 32;
    const int        nfat   = nb / KB;
    const unsigned   pitch  = (unsigned) nb * 16u;
    const uint32_t * qs32   = (const uint32_t *) vx;
    const uint16_t * d16    = (const uint16_t *) (vx + (size_t) N * (size_t) pitch);
    const int        tid    = it.get_local_id(0);
    const int        wg     = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0   = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0   = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> acc[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc[c] = 0.0f;
    }

    const float * yp[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c] = x + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
    }

    struct blk {
        simd<uint32_t, 128> T[4];
        simd<uint16_t, 16>  db[KB];
        simd<int8_t, 256>   y8[NC];
        simd<float, 8>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](blk & b) {
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = block_load<uint16_t, 16>(d16 + (size_t) (xib * KB + s) * (size_t) N + (size_t) row0);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            b.T[k] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                qs32, pitch - 1u, (unsigned) N - 1u, pitch - 1u, xib * 32 + 8 * k, row0);
        }
        xib += WG;
    };
    auto load_a = [&](blk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                simd<float, 256> xf;
                xf.copy_from(yp[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
#pragma unroll
                for (int sb = 0; sb < KB; ++sb) {
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        const float v = xf[sb * 32 + i];
                        const float av = v < 0.0f ? -v : v;
                        if (av > amax) {
                            amax = av;
                        }
                    }
                    const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
                    b.dar[c].template select<1, 1>(sb) = s;
                    const float id = s > 0.0f ? 1.0f / s : 0.0f;
                    simd<int8_t, 32> yq;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float xq = xf[sb * 32 + i] * id;
                        int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
                        if (q < -127) {
                            q = -127;
                        }
                        if (q > 127) {
                            q = 127;
                        }
                        yq.template select<1, 1>(i) = (int8_t) q;
                    }
                    b.y8[c].template select<32, 1>(32 * sb) = yq;
                }
            } else {
                b.y8[c]  = simd<int8_t, 256>(int8_t(0));
                b.dar[c] = simd<float, 8>(0.0f);
            }
        }
    };

    blk Ablk, Bblk;
    if (tid < nfat) {
        load_w(Ablk);
    }

    auto math = [&](blk & b) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(b, t);
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<uint32_t, 64> B4 = b.T[sb / 2].template select<64, 1>(16 * ((sb % 2) * 4));
                B4 = B4 ^ simd<uint32_t, 64>(xorm);
                simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
                simd<float, 16> dw = convert<float>(dh);
                simd<int8_t, NC * 32> Am;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
                }
                simd<int, NC * 16> Ci =
                    xmx::dpas<8, NC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                        simd<int, NC * 16>(0), Bw, Am);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                    simd<float, KB> dar8 = b.dar[c];
                    const float darc = dar8[sb];
                    acc[t * NC + c] += convert<float>(ci) * dw * darc;
                }
            }
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        load_w(Bblk);
        __asm__ volatile("");
        math(Ablk);
        if (ib + 2 * WG < nfat) {
            load_w(Ablk);
        }
        __asm__ volatile("");
        math(Bblk);
    }
    if (ib < nfat) {
        math(Ablk);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, acc[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) N;
            scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, u, msk);
        }
    }
}

template <int NC, int NM>
static bool launch_soa_f32(queue_ptr stream, const uint8_t * packed, const float * x, float * C,
                             int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) XE2MX_WG), sycl::range<1>(XE2MX_WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_soa_f32<NC, NM>(packed, x, C, (int) K, (int) N, (int) M, n_row_groups, xorm, it);
        });
    return true;
}

constexpr int XE2MX_PF_RC  = 8;
constexpr int XE2MX_PF_LGS = 16;

template <int NT>
static bool launch_prefill_nt(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                               int64_t K, int64_t N, int64_t M) {
    constexpr int RC  = XE2MX_PF_RC;
    constexpr int LGS = XE2MX_PF_LGS;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int nstep = 16 * NT;
    if ((N % nstep) != 0) {
        return false;
    }
    const int m_span = RC * LGS;
    const int m_groups = (int) ((M + m_span - 1) / m_span);
    const int n_tiles = (int) (N / nstep);
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_groups * (size_t) LGS, (size_t) n_tiles),
                          sycl::range<2>((size_t) LGS, 1)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * RC;
            const int n0 = (int) it.get_global_id(1) * nstep;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = 0; ib < nb; ib += 2) {
                if (ib + 2 < nb) {
                    ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                        A, aw, ah, aw, (ib + 2) * 32, m0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qs32, sw, sh, sw, (ib + 2) * 4, n0 + t * 16);
                    }
                }
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am =
                        ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(A, aw, ah, aw, (ib + s) * 32, m0);
                    float dar[RC];
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + r;
                        dar[r] = 0.0f;
                        if (m < (int) M) {
                            dar[r] = da[(size_t) m * (size_t) nb + (size_t) (ib + s)];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int nt = n0 + t * 16;
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<int, RC * 16> Ci =
                            xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                simd<int, RC * 16>(0), Bw, Am);
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) nt);
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                            acc[t].template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar[r];
                        }
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<float, 16> row = acc[t].template select<16, 1>(r * 16);
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
        });
    return true;
}

template <int NT>
static bool launch_gemv_f32_nt(queue_ptr stream, const uint8_t * packed, const float * x, float * C,
                                int64_t K, int64_t N) {
    constexpr int nstep = 16 * NT;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0 || (N % nstep) != 0) {
        return false;
    }
    const int n_tiles = (int) (N / nstep);
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_tiles), sycl::range<1>(1)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int n0 = (int) it.get_global_id(0) * nstep;
            simd<float, 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = 0; ib < nb; ib += 2) {
                if (ib + 2 < nb) {
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qs32, sw, sh, sw, (ib + 2) * 4, n0 + t * 16);
                    }
                }
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<float, 32> xf;
                    xf.copy_from(x + (ib + s) * 32);
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        const float av = xf[i] < 0.0f ? -xf[i] : xf[i];
                        if (av > amax) {
                            amax = av;
                        }
                    }
                    const float dar = amax > 0.0f ? amax / 127.0f : 0.0f;
                    const float id = dar > 0.0f ? 1.0f / dar : 0.0f;
                    simd<int8_t, 32> Am;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        float xq = xf[i] * id;
                        int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
                        if (q < -127) {
                            q = -127;
                        }
                        if (q > 127) {
                            q = 127;
                        }
                        Am[i] = (int8_t) q;
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<int, 16> Ci =
                            xmx::dpas<8, 1, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                simd<int, 16>(0), Bw, Am);
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) (n0 + t * 16));
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
                        acc[t] += convert<float>(Ci) * dw * dar;
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t].copy_to(C + n0 + t * 16);
            }
        });
    return true;
}

static bool launch_prefill(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                             int64_t K, int64_t N, int64_t M) {
    if ((N % (16 * 4)) == 0) {
        return launch_prefill_nt<4>(stream, packed, A, da, C, K, N, M);
    }
    return launch_prefill_nt<1>(stream, packed, A, da, C, K, N, M);
}

template <int RM, int NT>
static bool launch_prefill_rm(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                               int64_t K, int64_t N, int64_t M) {
    constexpr int RC  = XE2MX_PF_RC;
    constexpr int LGS = XE2MX_PF_LGS;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int nstep = 16 * NT;
    if ((N % nstep) != 0) {
        return false;
    }
    const int m_span = RC * RM * LGS;
    const int m_groups = (int) ((M + m_span - 1) / m_span);
    const int n_tiles = (int) (N / nstep);
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_groups * (size_t) LGS, (size_t) n_tiles),
                          sycl::range<2>((size_t) LGS, 1)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * (RC * RM);
            const int n0 = (int) it.get_global_id(1) * nstep;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[RM][NT];
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[rmi][t] = 0.0f;
                }
            }
            for (int ib = 0; ib < nb; ib += 2) {
                if (ib + 2 < nb) {
#pragma unroll
                    for (int rmi = 0; rmi < RM; ++rmi) {
                        const int mr = m0 + rmi * RC;
                        if (mr < (int) M) {
                            ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                                A, aw, ah, aw, (ib + 2) * 32, mr);
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qs32, sw, sh, sw, (ib + 2) * 4, n0 + t * 16);
                    }
                }
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am[RM];
                    float dar[RM][RC];
#pragma unroll
                    for (int rmi = 0; rmi < RM; ++rmi) {
                        const int mr = m0 + rmi * RC;
                        if (mr < (int) M) {
                            Am[rmi] = ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                                A, aw, ah, aw, (ib + s) * 32, mr);
#pragma unroll
                            for (int r = 0; r < RC; ++r) {
                                const int m = mr + r;
                                dar[rmi][r] = 0.0f;
                                if (m < (int) M) {
                                    dar[rmi][r] = da[(size_t) m * (size_t) nb + (size_t) (ib + s)];
                                }
                            }
                        } else {
                            Am[rmi] = simd<int8_t, RC * 32>(int8_t(0));
#pragma unroll
                            for (int r = 0; r < RC; ++r) {
                                dar[rmi][r] = 0.0f;
                            }
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int nt = n0 + t * 16;
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) nt);
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
#pragma unroll
                        for (int rmi = 0; rmi < RM; ++rmi) {
                            simd<int, RC * 16> Ci =
                                xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                    simd<int, RC * 16>(0), Bw, Am[rmi]);
#pragma unroll
                            for (int r = 0; r < RC; ++r) {
                                simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                                acc[rmi][t].template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar[rmi][r];
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + rmi * RC + r;
                        if (m < (int) M) {
                            simd<float, 16> row = acc[rmi][t].template select<16, 1>(r * 16);
                            row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                        }
                    }
                }
            }
        });
    return true;
}

template <int RM, int NT>
static bool launch_prefill_isa(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                                 int64_t K, int64_t N, int64_t M) {
    constexpr int RC = XE2MX_PF_RC;
    constexpr int LM = 8;
    constexpr int LN = 4;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int BM = RC * RM;
    const int BN = 16 * NT;
    if ((N % (BN * LN)) != 0) {
        return false;
    }
    const int m_tiles = (int) ((M + BM - 1) / BM);
    const int n_tiles = (int) (N / BN);
    const int m_wg = (m_tiles + LM - 1) / LM;
    const int n_wg = n_tiles / LN;
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const unsigned dw = (unsigned) nb - 1u;
    const unsigned dh = (unsigned) M - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_wg * (size_t) LM, (size_t) n_wg * (size_t) LN),
                          sycl::range<2>((size_t) LM, (size_t) LN)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * BM;
            const int n0 = (int) it.get_global_id(1) * BN;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[RM][NT];
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[rmi][t] = 0.0f;
                }
            }
            for (int ib = 0; ib < nb; ib += 2) {
                if (ib + 2 < nb) {
#pragma unroll
                    for (int rmi = 0; rmi < RM; ++rmi) {
                        ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                            A, aw, ah, aw, (ib + 2) * 32, m0 + rmi * RC);
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qs32, sw, sh, sw, (ib + 2) * 4, n0 + t * 16);
                    }
                }
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am[RM];
                    simd<float, RC> dar[RM];
#pragma unroll
                    for (int rmi = 0; rmi < RM; ++rmi) {
                        const int mr = m0 + rmi * RC;
                        Am[rmi] = ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                            A, aw, ah, aw, (ib + s) * 32, mr);
                        dar[rmi] = ex::lsc_load_2d<float, 1, RC, 1, false, false>(
                            da, dw, dh, (unsigned) nb, (ib + s), mr);
                    }
                    simd<uint16_t, 16> db[NT];
                    simd<float, 16> dwv[NT];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        db[t] = block_load<uint16_t, 16>(d16 + (size_t) (ib + s) * (size_t) N + (size_t) (n0 + t * 16));
                        simd<sycl::half, 16> dh = db[t].template bit_cast_view<sycl::half>();
                        dwv[t] = convert<float>(dh);
#pragma unroll
                        for (int rmi = 0; rmi < RM; ++rmi) {
                            simd<int, RC * 16> Ci =
                                xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4,
                                          xmx::dpas_argument_type::s8>(simd<int, RC * 16>(0), Bw, Am[rmi]);
#pragma unroll
                            for (int r = 0; r < RC; ++r) {
                                simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                                const float ds = dar[rmi][r];
                                acc[rmi][t].template select<16, 1>(r * 16) += convert<float>(ci) * dwv[t] * ds;
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + rmi * RC + r;
                        if (m < (int) M) {
                            simd<float, 16> row = acc[rmi][t].template select<16, 1>(r * 16);
                            row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                        }
                    }
                }
            }
        });
    return true;
}

template <int RM, int NT>
static bool launch_prefill_isa_h(queue_ptr stream, const uint8_t * packed, const sycl::half * xh, float * C,
                                   int64_t K, int64_t N, int64_t M) {
    constexpr int RC = XE2MX_PF_RC;
    constexpr int LM = 8;
    constexpr int LN = 4;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int BM = RC * RM;
    const int BN = 16 * NT;
    if ((N % (BN * LN)) != 0) {
        return false;
    }
    const int m_tiles = (int) ((M + BM - 1) / BM);
    const int n_tiles = (int) (N / BN);
    const int m_wg = (m_tiles + LM - 1) / LM;
    const int n_wg = n_tiles / LN;
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_wg * (size_t) LM, (size_t) n_wg * (size_t) LN),
                          sycl::range<2>((size_t) LM, (size_t) LN)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * BM;
            const int n0 = (int) it.get_global_id(1) * BN;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[RM][NT];
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[rmi][t] = 0.0f;
                }
            }
            for (int ib = 0; ib < nb; ib += 2) {
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am[RM];
                    float dar[RM][RC];
#pragma unroll
                    for (int rmi = 0; rmi < RM; ++rmi) {
                        const int mr = m0 + rmi * RC;
                        simd<sycl::half, RC * 32> Ah =
                            ex::lsc_load_2d<sycl::half, 32, RC, 1, false, false>(
                                xh, aw, ah, aw, (ib + s) * 32, mr);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            simd<sycl::half, 32> hs = Ah.template select<32, 1>(r * 32);
                            simd<float, 32> xf = convert<float>(hs);
                            float amax = 0.0f;
#pragma unroll
                            for (int k = 0; k < 32; ++k) {
                                const float v = xf[k];
                                const float av = v < 0.0f ? -v : v;
                                if (av > amax) {
                                    amax = av;
                                }
                            }
                            const float ds = amax > 0.0f ? amax / 127.0f : 0.0f;
                            dar[rmi][r] = ds;
                            const float id = ds > 0.0f ? 1.0f / ds : 0.0f;
                            simd<int8_t, 32> qv;
#pragma unroll
                            for (int k = 0; k < 32; ++k) {
                                float xq = xf[k] * id;
                                int q = (int) (xq + (xq >= 0.0f ? 0.5f : -0.5f));
                                if (q < -127) {
                                    q = -127;
                                }
                                if (q > 127) {
                                    q = 127;
                                }
                                qv[k] = (int8_t) q;
                            }
                            Am[rmi].template select<32, 1>(r * 32) = qv;
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) (n0 + t * 16));
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dwv = convert<float>(dh);
#pragma unroll
                        for (int rmi = 0; rmi < RM; ++rmi) {
                            simd<int, RC * 16> Ci =
                                xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4,
                                          xmx::dpas_argument_type::s8>(simd<int, RC * 16>(0), Bw, Am[rmi]);
#pragma unroll
                            for (int r = 0; r < RC; ++r) {
                                simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                                acc[rmi][t].template select<16, 1>(r * 16) += convert<float>(ci) * dwv * dar[rmi][r];
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int rmi = 0; rmi < RM; ++rmi) {
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + rmi * RC + r;
                        if (m < (int) M) {
                            simd<float, 16> row = acc[rmi][t].template select<16, 1>(r * 16);
                            row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                        }
                    }
                }
            }
        });
    return true;
}

ESIMD_INLINE simd<sycl::half, 256> xe2mx_hf_vnni(const simd<sycl::half, 256> & hlo, const simd<sycl::half, 256> & hhi,
                                                 int off) {
    simd<sycl::half, 256> B;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            B[j * 32 + n * 2]     = hlo[n * 16 + off + j];
            B[j * 32 + n * 2 + 1] = hhi[n * 16 + off + j];
        }
    }
    return B;
}

template <int NT>
static bool launch_prefill_hf(queue_ptr stream, const uint8_t * packed, const sycl::half * xh, float * C,
                              int64_t K, int64_t N, int64_t M) {
    constexpr int RC = 8;
    constexpr int LM = 16;
    constexpr int LN = 4;
    const int nb = (int) (K / 32);
    const int BN = 16 * NT;
    if ((N % BN) != 0) {
        return false;
    }
    const int m_tiles = (int) ((M + RC - 1) / RC);
    const int n_tiles = (int) (N / BN);
    if ((n_tiles % LN) != 0) {
        return false;
    }
    const int m_wg = (m_tiles + LM - 1) / LM;
    const int n_wg = n_tiles / LN;
    const unsigned pitch = (unsigned) nb * 16u;
    const uint8_t * qs = packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const unsigned dw = (unsigned) N * 2u - 1u;
    const unsigned dh = (unsigned) nb - 1u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_wg * (size_t) LM, (size_t) n_wg * (size_t) LN),
                          sycl::range<2>((size_t) LM, (size_t) LN)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * RC;
            const int n0 = (int) it.get_global_id(1) * BN;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = 0; ib < nb; ++ib) {
                simd<sycl::half, RC * 32> Ah =
                    ex::lsc_load_2d<sycl::half, 32, RC, 1, false, false>(xh, aw, ah, aw, ib * 32, m0);
                simd<sycl::half, RC * 16> A0;
                simd<sycl::half, RC * 16> A1;
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    A0.template select<16, 1>(r * 16) = Ah.template select<16, 1>(r * 32);
                    A1.template select<16, 1>(r * 16) = Ah.template select<16, 1>(r * 32 + 16);
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int nt = n0 + t * 16;
                    simd<uint8_t, 256> q =
                        ex::lsc_load_2d<uint8_t, 16, 16, 1, false, false>(qs, sw, sh, sw, ib * 16, nt);
                    simd<uint16_t, 16> db = ex::lsc_load_2d<uint16_t, 16, 1>(d16, dw, dh, dw, nt, ib);
                    simd<sycl::half, 16> sc = db.template bit_cast_view<sycl::half>();
                    simd<uint8_t, 256> lo = q & simd<uint8_t, 256>(uint8_t(0x0f));
                    simd<uint8_t, 256> hi = q >> 4;
                    simd<sycl::half, 256> hlo = convert<sycl::half>(convert<int>(lo) - 8);
                    simd<sycl::half, 256> hhi = convert<sycl::half>(convert<int>(hi) - 8);
#pragma unroll
                    for (int n = 0; n < 16; ++n) {
                        const sycl::half s = sc[n];
                        hlo.template select<16, 1>(n * 16) = hlo.template select<16, 1>(n * 16) * s;
                        hhi.template select<16, 1>(n * 16) = hhi.template select<16, 1>(n * 16) * s;
                    }
                    simd<sycl::half, 256> B0 = xe2mx_hf_vnni(hlo, hhi, 0);
                    simd<sycl::half, 256> B1 = xe2mx_hf_vnni(hlo, hhi, 8);
                    acc[t] = xmx::dpas<8, RC, float, float, sycl::half, sycl::half>(acc[t], B0, A0);
                    acc[t] = xmx::dpas<8, RC, float, float, sycl::half, sycl::half>(acc[t], B1, A1);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<float, 16> row = acc[t].template select<16, 1>(r * 16);
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
        });
    return true;
}

template <int NT>
static bool launch_prefill_wg(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da, float * C,
                                int64_t K, int64_t N, int64_t M) {
    constexpr int RC  = XE2MX_PF_RC;
    constexpr int NM  = 4;
    constexpr int NN  = 4;
    constexpr int WG  = NM * NN;
    constexpr int BM  = RC * NM;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int nstep = 16 * NT * NN;
    if ((N % nstep) != 0) {
        return false;
    }
    const int m_tiles = (int) ((M + BM - 1) / BM);
    const int n_tiles = (int) (N / nstep);
    const int workgroups = m_tiles * n_tiles;
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            constexpr uint32_t SLM_A = 0;
            slm_init<BM * 32>();
            const int lid = (int) it.get_local_id(0);
            const int wg  = (int) it.get_group(0);
            const int tm  = lid / NN;
            const int tn  = lid % NN;
            const int m_tile = n_tiles > 0 ? wg / n_tiles : 0;
            const int n_tile = n_tiles > 0 ? wg % n_tiles : 0;
            const int mr = m_tile * BM + tm * RC;
            const int n0 = n_tile * nstep + tn * (16 * NT);
            simd<float, RC * 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = 0; ib < nb; ib += 2) {
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    if (tn == 0 && mr < (int) M) {
                        simd<int8_t, RC * 32> Am =
                            ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                                A, aw, ah, aw, (ib + s) * 32, mr);
                        slm_block_store<int8_t, RC * 32>(SLM_A + (uint32_t) tm * (RC * 32), Am);
                    }
                    barrier();
                    simd<int8_t, RC * 32> Am = slm_block_load<int8_t, RC * 32>(SLM_A + (uint32_t) tm * (RC * 32));
                    float dar[RC];
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = mr + r;
                        dar[r] = 0.0f;
                        if (m < (int) M) {
                            dar[r] = da[(size_t) m * (size_t) nb + (size_t) (ib + s)];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int nt = n0 + t * 16;
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) nt);
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
                        simd<int, RC * 16> Ci =
                            xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                simd<int, RC * 16>(0), Bw, Am);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                            acc[t].template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar[r];
                        }
                    }
                    barrier();
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = mr + r;
                    if (m < (int) M) {
                        simd<float, 16> row = acc[t].template select<16, 1>(r * 16);
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
        });
    return true;
}

template <int NT>
static bool launch_s8_pf_nt(queue_ptr stream, const int8_t * W8, const uint16_t * d16, const int8_t * A,
                              const float * da, float * C, int64_t K, int64_t N, int64_t M) {
    constexpr int RC  = XE2MX_PF_RC;
    constexpr int LGS = XE2MX_PF_LGS;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int nstep = 16 * NT;
    if ((N % nstep) != 0) {
        return false;
    }
    const int m_span = RC * LGS;
    const int m_groups = (int) ((M + m_span - 1) / m_span);
    const int n_tiles = (int) (N / nstep);
    const unsigned sw = (unsigned) N - 1u;
    const unsigned sh = (unsigned) K - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_groups * (size_t) LGS, (size_t) n_tiles),
                          sycl::range<2>((size_t) LGS, 1)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * RC;
            const int n0 = (int) it.get_global_id(1) * nstep;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = 0; ib < nb; ib += 2) {
                if (ib + 2 < nb) {
                    ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                        A, aw, ah, aw, (ib + 2) * 32, m0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<int8_t, 16, 32, 1, cache_hint::cached, cache_hint::cached>(
                            W8, sw, sh, sw, n0 + t * 16, (ib + 2) * 32);
                    }
                }
                simd<int8_t, 512> Bw[NT][2];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        Bw[t][s] = ex::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                            W8, sw, sh, sw, n0 + t * 16, (ib + s) * 32);
                    }
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am =
                        ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(A, aw, ah, aw, (ib + s) * 32, m0);
                    float dar[RC];
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + r;
                        dar[r] = 0.0f;
                        if (m < (int) M) {
                            dar[r] = da[(size_t) m * (size_t) nb + (size_t) (ib + s)];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        const int nt = n0 + t * 16;
                        simd<int, RC * 16> Ci =
                            xmx::dpas<8, RC, int, int, int8_t, int8_t, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(
                                simd<int, RC * 16>(0), Bw[t][s], Am);
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) nt);
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                            acc[t].template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar[r];
                        }
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<float, 16> row = acc[t].template select<16, 1>(r * 16);
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
        });
    return true;
}

constexpr int XE2MX_KS_WG = 4;
constexpr int XE2MX_KS_RC = 8;

template <int NT>
static bool launch_prefill_ksplit(queue_ptr stream, const uint8_t * packed, const int8_t * A, const float * da,
                                    float * C, int64_t K, int64_t N, int64_t M) {
    constexpr int RC = XE2MX_KS_RC;
    constexpr int WG = XE2MX_KS_WG;
    const int nb = (int) (K / 32);
    if ((nb % (2 * WG)) != 0) {
        return false;
    }
    const int nstep = 16 * NT;
    if ((N % nstep) != 0) {
        return false;
    }
    const int m_groups = (int) ((M + RC - 1) / RC);
    const int n_tiles = (int) (N / nstep);
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qs32 = (const uint32_t *) packed;
    const uint16_t * d16 = (const uint16_t *) (packed + (size_t) N * (size_t) pitch);
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    constexpr uint32_t SLM_SUMS = 0;
    constexpr uint32_t slm_bytes = WG * NT * RC * 16u * 4u;
    const uint32_t xorm = ggml_sycl_get_env("GGML_SYCL_XE2MX_PACKX", 0) ? 0u : 0x88888888u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_groups * (size_t) WG, (size_t) n_tiles),
                          sycl::range<2>((size_t) WG, 1)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            slm_init<slm_bytes>();
            const int tid = (int) it.get_local_id(0);
            const int m0 = (int) it.get_group(0) * RC;
            const int n0 = (int) it.get_group(1) * nstep;
            if (m0 >= (int) M) {
                return;
            }
            simd<float, RC * 16> acc[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[t] = 0.0f;
            }
            for (int ib = tid * 2; ib < nb; ib += 2 * WG) {
                if (ib + 2 * WG < nb) {
                    ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                        A, aw, ah, aw, (ib + 2 * WG) * 32, m0);
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qs32, sw, sh, sw, (ib + 2 * WG) * 4, n0 + t * 16);
                    }
                }
                simd<uint32_t, 128> T[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    T[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                        qs32, sw, sh, sw, ib * 4, n0 + t * 16);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    simd<int8_t, RC * 32> Am =
                        ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(A, aw, ah, aw, (ib + s) * 32, m0);
                    float dar[RC];
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + r;
                        dar[r] = 0.0f;
                        if (m < (int) M) {
                            dar[r] = da[(size_t) m * (size_t) nb + (size_t) (ib + s)];
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 64> B4 = T[t].template select<64, 1>(64 * s);
                        B4 = B4 ^ simd<uint32_t, 64>(xorm);
                        simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                        simd<int, RC * 16> Ci =
                            xmx::dpas<8, RC, int, int, uint8_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                simd<int, RC * 16>(0), Bw, Am);
                        simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                            d16 + (size_t) (ib + s) * (size_t) N + (size_t) (n0 + t * 16));
                        simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                        simd<float, 16> dw = convert<float>(dh);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                            acc[t].template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar[r];
                        }
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    slm_block_store<float, 16>(
                        SLM_SUMS + (((uint32_t) tid * NT + (uint32_t) t) * RC + (uint32_t) r) * 64u,
                        acc[t].template select<16, 1>(r * 16));
                }
            }
            barrier();
            if (tid == 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<float, 16> u = 0.0f;
#pragma unroll
                        for (int q = 0; q < WG; ++q) {
                            u += slm_block_load<float, 16>(
                                SLM_SUMS + (((uint32_t) q * NT + (uint32_t) t) * RC + (uint32_t) r) * 64u);
                        }
                        u.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
            }
        });
    return true;
}

constexpr int XE2MX_S8_RC  = 8;
constexpr int XE2MX_S8_LGS = 16;

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_s8_soa(const int8_t * W8, const uint16_t * d16, const int8_t * A, const float * da,
                              float * C, int K, int N, int M, int pitch, int n_row_groups, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<SLM_SUMS + WG * RPW * NCM * 4>();

    const int        nb    = K / 32;
    const int        nfat  = nb / KB;
    const unsigned   sw    = (unsigned) pitch - 1u;
    const unsigned   sh    = (unsigned) K - 1u;
    const int        tid   = it.get_local_id(0);
    const int        wg    = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0  = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0  = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> acc[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct blk {
        simd<int8_t, 512>  Bw[KB];
        simd<uint16_t, 16> db[KB];
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](blk & b) {
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            b.db[s] = block_load<uint16_t, 16>(d16 + (size_t) (xib * KB + s) * (size_t) N + (size_t) row0);
            b.Bw[s] = ex::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                W8, sw, sh, sw, row0, (xib * KB + s) * 32);
        }
        xib += WG;
    };
    auto load_a = [&](blk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };

    blk Ablk, Bblk;
    if (tid < nfat) {
        load_w(Ablk);
    }

    auto math = [&](blk & b) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(b, t);
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
                simd<float, 16> dw = convert<float>(dh);
                simd<int8_t, NC * 32> Am;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
                }
                simd<int, NC * 16> Ci =
                    xmx::dpas<8, NC, int, int, int8_t, int8_t>(simd<int, NC * 16>(0), b.Bw[sb], Am);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                    simd<float, KB> dar8 = b.dar[c];
                    const float darc = dar8[sb];
                    acc[t * NC + c] += convert<float>(ci) * dw * darc;
                }
            }
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        load_w(Bblk);
        __asm__ volatile("");
        math(Ablk);
        if (ib + 2 * WG < nfat) {
            load_w(Ablk);
        }
        __asm__ volatile("");
        math(Bblk);
    }
    if (ib < nfat) {
        math(Ablk);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, acc[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) N;
            scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, u, msk);
        }
    }
}

struct xe2mx_q8b {
    uint16_t d;
    int8_t qs[32];
} __attribute__((packed));
static_assert(sizeof(xe2mx_q8b) == 34, "q8b");

template <int WG, int KB, int NC, int NM>
ESIMD_INLINE void gemm_q8_aos(const int8_t * qs, const uint16_t * d16, const int8_t * A, const float * da,
                              float * C, int K, int N, int M, int n_row_groups, const sycl::nd_item<1> & it) {
    constexpr int RPW = XE2MX_RPW;
    constexpr int NCM = NC * NM;
    constexpr uint32_t SLM_SUMS = 0;
    slm_init<SLM_SUMS + WG * RPW * NCM * 4>();

    const int        nb    = K / 32;
    const int        nfat  = nb / KB;
    const unsigned   sw    = (unsigned) K - 1u;
    const unsigned   sh    = (unsigned) N - 1u;
    const int        tid   = it.get_local_id(0);
    const int        wg    = it.get_group(0);
    const int        row_gid = n_row_groups > 0 ? wg % n_row_groups : wg;
    const int        col0  = n_row_groups > 0 ? (wg / n_row_groups) * NCM : 0;
    const int        row0  = row_gid * RPW;
    if (col0 >= M) {
        return;
    }

    simd<float, 16> acc[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        acc[c] = 0.0f;
    }

    const int8_t * yp[NCM];
    const float *  dap[NCM];
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        const int m = col0 + c;
        const int mm = m < M ? m : 0;
        yp[c]  = A + (size_t) mm * (size_t) K + (size_t) tid * KB * 32;
        dap[c] = da + (size_t) mm * (size_t) nb + (size_t) tid * KB;
    }

    struct blk {
        simd<int8_t, 512>  Bw[KB];
        simd<uint16_t, 16> db[KB];
        simd<int8_t, KB * 32> y8[NC];
        simd<float, KB>      dar[NC];
    };
    int xib = tid;
    auto load_w = [&](blk & b) {
#pragma unroll
        for (int s = 0; s < KB; ++s) {
            const int ib = xib * KB + s;
            b.Bw[s] = ex::lsc_load_2d<int8_t, 32, 16, 1, false, false>(
                qs, sw, sh, sw, ib * 32, row0);
            simd<int8_t, 512> raw = b.Bw[s];
            simd<int8_t, 512> vn;
#pragma unroll
            for (int g = 0; g < 8; ++g) {
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    vn.template select<4, 1>(g * 64 + n * 4) =
                        raw.template select<4, 1>(n * 32 + g * 4);
                }
            }
            b.Bw[s] = vn;
            simd<uint16_t, 16> ds;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                ds[i] = d16[(size_t) (row0 + i) * (size_t) nb + (size_t) ib];
            }
            b.db[s] = ds;
        }
        xib += WG;
    };
    auto load_a = [&](blk & b, int t) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int m = col0 + t * NC + c;
            if (m < M) {
                b.y8[c]  = block_load<int8_t, KB * 32>(yp[t * NC + c]);
                b.dar[c] = block_load<float, KB>(dap[t * NC + c]);
                yp[t * NC + c] += WG * KB * 32;
                dap[t * NC + c] += WG * KB;
            } else {
                b.y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                b.dar[c] = simd<float, KB>(0.0f);
            }
        }
    };

    blk Ablk, Bblk;
    if (tid < nfat) {
        load_w(Ablk);
    }

    auto math = [&](blk & b) {
#pragma unroll
        for (int t = 0; t < NM; ++t) {
            load_a(b, t);
#pragma unroll
            for (int sb = 0; sb < KB; ++sb) {
                simd<sycl::half, 16> dh = b.db[sb].template bit_cast_view<sycl::half>();
                simd<float, 16> dw = convert<float>(dh);
                simd<int8_t, NC * 32> Am;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    Am.template select<32, 1>(32 * c) = b.y8[c].template select<32, 1>(32 * sb);
                }
                simd<int, NC * 16> Ci =
                    xmx::dpas<8, NC, int, int, int8_t, int8_t>(simd<int, NC * 16>(0), b.Bw[sb], Am);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                    simd<float, KB> dar8 = b.dar[c];
                    const float darc = dar8[sb];
                    acc[t * NC + c] += convert<float>(ci) * dw * darc;
                }
            }
        }
    };

    int ib = tid;
    for (; ib + WG < nfat; ib += 2 * WG) {
        load_w(Bblk);
        __asm__ volatile("");
        math(Ablk);
        if (ib + 2 * WG < nfat) {
            load_w(Ablk);
        }
        __asm__ volatile("");
        math(Bblk);
    }
    if (ib < nfat) {
        math(Ablk);
    }

#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        slm_block_store<float, 16>(SLM_SUMS + (tid * NCM + c) * 64, acc[c]);
    }
    barrier();
    constexpr int RT = RPW / WG;
#pragma unroll
    for (int c = 0; c < NCM; ++c) {
        if (col0 + c < M) {
            simd<float, RT> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, RT>(SLM_SUMS + (q * NCM + c) * 64 + tid * RT * 4);
            }
            simd<uint32_t, RT> ridx((uint32_t) (row0 + tid * RT), 1);
            simd_mask<RT> msk = ridx < (uint32_t) N;
            scatter<float, RT>(C + (size_t) (col0 + c) * (size_t) N, ridx * 4u, u, msk);
        }
    }
}

template <int WG, int KB, int NC, int NM>
static bool launch_q8_aos(queue_ptr stream, const int8_t * qs, const uint16_t * d16, const int8_t * A,
                           const float * da, float * C, int64_t K, int64_t N, int64_t M) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_q8_aos<WG, KB, NC, NM>(qs, d16, A, da, C, (int) K, (int) N, (int) M, n_row_groups, it);
        });
    return true;
}

template <int WG, int KB, int NC, int NM>
static bool launch_s8_soa(queue_ptr stream, const int8_t * W8, const uint16_t * d16, const int8_t * A,
                             const float * da, float * C, int64_t K, int64_t N, int64_t M, int64_t pitch) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const int ncm = NC * NM;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    const int workgroups = n_row_groups * m_tiles;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) WG), sycl::range<1>(WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_s8_soa<WG, KB, NC, NM>(W8, d16, A, da, C, (int) K, (int) N, (int) M, (int) pitch, n_row_groups, it);
        });
    return true;
}

template <int NC, int MAXN>
ESIMD_INLINE void gemm_s8_ninner(const int8_t * W8, const uint16_t * d16, const int8_t * A, const float * da,
                                 float * C, int K, int N, int M, int pitch, const sycl::nd_item<1> & it) {
    constexpr int WG  = XE2MX_WG;
    constexpr int RPW = XE2MX_RPW;
    constexpr int KB  = 8;
    slm_init<WG * MAXN * RPW * 4>();
    const int        nblocks = K / (KB * 32);
    const int        nb      = K / 32;
    const int        tid     = it.get_local_id(0);
    const int        row0    = it.get_group(0) * RPW;
    const unsigned   sw     = (unsigned) pitch - 1u;
    const unsigned   sh     = (unsigned) K - 1u;

    for (int nbase = 0; nbase < M; nbase += MAXN) {
        const int nlim   = M - nbase < MAXN ? M - nbase : MAXN;
        const int ntiles = (nlim + NC - 1) / NC;
        for (int col = tid; col < nlim; col += WG) {
#pragma unroll
            for (int t = 0; t < WG; ++t) {
                slm_block_store<float, 16>((uint32_t) ((t * MAXN + col) * 64), simd<float, 16>(0.0f));
            }
        }
        barrier();
        for (int ib = tid; ib < nblocks; ib += WG) {
            const int k0 = ib * KB * 32;
            simd<int8_t, 512>  Bw[KB];
            simd<float, 16>    dw[KB];
#pragma unroll
            for (int s = 0; s < KB; ++s) {
                Bw[s] = ex::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                    W8, sw, sh, sw, row0, k0 + s * 32);
                simd<uint16_t, 16> db = block_load<uint16_t, 16>(
                    d16 + (size_t) (ib * KB + s) * (size_t) N + (size_t) row0);
                simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                dw[s] = convert<float>(dh);
            }
            for (int t = 0; t < ntiles; ++t) {
                const int col0 = nbase + t * NC;
                simd<int8_t, KB * 32> y8[NC];
                simd<float, KB>       dar[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int m = col0 + c;
                    if (m < M) {
                        y8[c]  = block_load<int8_t, KB * 32>(A + (size_t) m * (size_t) K + (size_t) k0);
                        dar[c] = block_load<float, KB>(da + (size_t) m * (size_t) nb + (size_t) ib * KB);
                    } else {
                        y8[c]  = simd<int8_t, KB * 32>(int8_t(0));
                        dar[c] = simd<float, KB>(0.0f);
                    }
                }
                simd<float, 16> acc[NC];
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    acc[c] = 0.0f;
                }
#pragma unroll
                for (int sb = 0; sb < KB; ++sb) {
                    simd<int8_t, NC * 32> Am;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        Am.template select<32, 1>(32 * c) = y8[c].template select<32, 1>(32 * sb);
                    }
                    simd<int, NC * 16> Ci =
                        xmx::dpas<8, NC, int, int, int8_t, int8_t>(simd<int, NC * 16>(0), Bw[sb], Am);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<int, 16> ci = Ci.template select<16, 1>(16 * c);
                        const float darc = dar[c][sb];
                        acc[c] += convert<float>(ci) * dw[sb] * darc;
                    }
                }
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    const int m = col0 + c;
                    if (m < M) {
                        const uint32_t off = (uint32_t) ((tid * MAXN + t * NC + c) * 64);
                        simd<float, 16> prev = slm_block_load<float, 16>(off);
                        slm_block_store<float, 16>(off, prev + acc[c]);
                    }
                }
            }
        }
        barrier();
        for (int col = tid; col < nlim; col += WG) {
            simd<float, 16> u = 0.0f;
#pragma unroll
            for (int q = 0; q < WG; ++q) {
                u += slm_block_load<float, 16>((uint32_t) ((q * MAXN + col) * 64));
            }
            simd<uint32_t, 16> ridx((uint32_t) row0, 1);
            simd_mask<16> msk = ridx < (uint32_t) N;
            scatter<float, 16>(C + (size_t) (nbase + col) * (size_t) N, ridx * 4u, u, msk);
        }
        barrier();
    }
}

template <int NC, int MAXN>
static bool launch_s8_ninner(queue_ptr stream, const int8_t * W8, const uint16_t * d16, const int8_t * A,
                               const float * da, float * C, int64_t K, int64_t N, int64_t M, int64_t pitch) {
    const int n_row_groups = (int) (N / XE2MX_RPW);
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) n_row_groups * (size_t) XE2MX_WG), sycl::range<1>(XE2MX_WG)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            gemm_s8_ninner<NC, MAXN>(W8, d16, A, da, C, (int) K, (int) N, (int) M, (int) pitch, it);
        });
    return true;
}

template <int RC, int LGS>
static bool launch_s8(queue_ptr stream, const int8_t * W8, const uint16_t * d16, const int8_t * A,
                       const float * da, float * C, int64_t K, int64_t N, int64_t M, int64_t pitch) {
    const int n_tiles = (int) (N / 16);
    const int m_span  = RC * LGS;
    const int m_groups = (int) ((M + m_span - 1) / m_span);
    const int workgroups = m_groups * n_tiles;
    const unsigned sw = (unsigned) pitch - 1u;
    const unsigned sh = (unsigned) K - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const int nb = (int) (K / 32);
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * (size_t) LGS), sycl::range<1>(LGS)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            slm_init<512>();
            const int lid = (int) it.get_local_id(0);
            const int wg  = (int) it.get_group(0);
            const int m_group = wg / n_tiles;
            const int n0 = (wg - m_group * n_tiles) * 16;
            const int m0 = (m_group * LGS + lid) * RC;
            simd<float, RC * 16> acc(0.0f);
            for (int ib = 0; ib < nb; ++ib) {
                const int k0 = ib * 32;
                if (lid == 0) {
                    simd<int8_t, 512> Bw = ex::lsc_load_2d<int8_t, 16, 32, 1, false, true>(
                        W8, sw, sh, sw, n0, k0);
                    simd<uint32_t, 128> B32 = Bw.template bit_cast_view<uint32_t>();
                    slm_block_store<uint32_t, 128>(0, B32);
                }
                barrier();
                simd<uint32_t, 128> B32 = slm_block_load<uint32_t, 128>(0);
                simd<int8_t, 512> Bw = B32.template bit_cast_view<int8_t>();
                if (m0 < (int) M) {
                    simd<int8_t, RC * 32> Ar =
                        ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(A, aw, ah, aw, k0, m0);
                    simd<int, RC * 16> Ci =
                        xmx::dpas<8, RC, int, int, int8_t, int8_t>(simd<int, RC * 16>(0), Bw, Ar);
                    simd<uint16_t, 16> db = block_load<uint16_t, 16>(d16 + (size_t) ib * (size_t) N + (size_t) n0);
                    simd<sycl::half, 16> dh = db.template bit_cast_view<sycl::half>();
                    simd<float, 16> dw = convert<float>(dh);
#pragma unroll
                    for (int r = 0; r < RC; ++r) {
                        const int m = m0 + r;
                        float dar = 0.0f;
                        if (m < (int) M) {
                            dar = da[(size_t) m * (size_t) nb + (size_t) ib];
                        }
                        simd<int, 16> ci = Ci.template select<16, 1>(r * 16);
                        acc.template select<16, 1>(r * 16) += convert<float>(ci) * dw * dar;
                    }
                }
                barrier();
            }
#pragma unroll
            for (int r = 0; r < RC; ++r) {
                const int m = m0 + r;
                if (m < (int) M) {
                    simd<float, 16> row = acc.template select<16, 1>(r * 16);
                    row.copy_to(C + (size_t) m * (size_t) N + (size_t) n0);
                }
            }
        });
    return true;
}

template <int CHAIN>
ESIMD_INLINE simd<int, 128> s4t_mad(simd<int, 128> acc, simd<uint8_t, 256> Bw, simd<int8_t, 256> Am) {
    if constexpr (CHAIN != 0) {
        return xmx::dpas<8, 8, int, int, uint8_t, int8_t,
            xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(acc, Bw, Am);
    }
    simd<int, 128> Ci =
        xmx::dpas<8, 8, int, int, uint8_t, int8_t,
            xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(simd<int, 128>(0), Bw, Am);
    return acc + Ci;
}

template <int NT, int CHAIN, int DUAL>
static bool launch_s4t_nt(queue_ptr stream, const uint8_t * pg, const float * scg,
                           const uint8_t * pu, const float * scu,
                           const int8_t * A, const float * sa, float * C,
                           int64_t K, int64_t N, int64_t M) {
    constexpr int RC  = 8;
    constexpr int LM  = (DUAL == 0 && NT == 1) ? 8 : 4;
    constexpr int LN0 = (DUAL == 0 && NT == 1) ? 4 : 16;
    const int nb = (int) (K / 32);
    if ((nb & 1) != 0) {
        return false;
    }
    const int BN = 16 * NT;
    if ((N % BN) != 0) {
        return false;
    }
    if constexpr (DUAL != 0) {
        if (pu == nullptr || scu == nullptr) {
            return false;
        }
    }
    int LN = LN0;
    while (LN > 1 && (N % (BN * LN)) != 0) {
        LN /= 2;
    }
    if ((N % (BN * LN)) != 0) {
        return false;
    }
    const int m_tiles = (int) ((M + RC - 1) / RC);
    const int n_tiles = (int) (N / BN);
    const int m_wg = (m_tiles + LM - 1) / LM;
    const int n_wg = n_tiles / LN;
    const unsigned pitch = (unsigned) nb * 16u;
    const uint32_t * qg = (const uint32_t *) pg;
    const uint32_t * qu = (DUAL != 0) ? (const uint32_t *) pu : qg;
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = (unsigned) K - 1u;
    const unsigned ah = (unsigned) M - 1u;
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_wg * (size_t) LM, (size_t) n_wg * (size_t) LN),
                          sycl::range<2>((size_t) LM, (size_t) LN)),
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            constexpr uint32_t SLB = 512;
            const int lid0 = (int) it.get_local_id(0);
            const int lid1 = (int) it.get_local_id(1);
            const int m0 = (int) it.get_global_id(0) * RC;
            const int n0 = (int) it.get_global_id(1) * BN;
            simd<int, NT * RC * 16> gacc(0);
            simd<int, (DUAL != 0) ? (NT * RC * 16) : 1> uacc(0);
            if constexpr (NT == 1) {
                constexpr uint32_t B_BANK = LN0 * SLB * 4u;
                if constexpr (DUAL != 0) {
                    slm_init<B_BANK * 2u>();
                    const uint32_t slm_bg = (uint32_t) lid1 * SLB * 4u;
                    const uint32_t slm_bu = B_BANK + slm_bg;
                    for (int ib = 0; ib < nb; ib += 8) {
                        const int nblk = (ib + 8 <= nb) ? 4 : ((nb - ib) / 2);
                        if (ib + 8 < nb && m0 < (int) M) {
                            ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                                A, aw, ah, aw, (ib + 8) * 32, m0);
                        }
                        if (lid0 == 0) {
                            if (ib + 8 < nb) {
                                ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                    qg, sw, sh, sw, (ib + 8) * 4, n0);
                                ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                    qu, sw, sh, sw, (ib + 8) * 4, n0);
                            }
                            for (int u = 0; u < nblk; ++u) {
                                simd<uint32_t, 128> Tw =
                                    ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                                        qg, sw, sh, sw, (ib + u * 2) * 4, n0);
                                slm_block_store<uint32_t, 128>(slm_bg + (uint32_t) u * SLB, Tw);
                                Tw = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                                    qu, sw, sh, sw, (ib + u * 2) * 4, n0);
                                slm_block_store<uint32_t, 128>(slm_bu + (uint32_t) u * SLB, Tw);
                            }
                        }
                        barrier();
                        for (int u = 0; u < nblk; ++u) {
                            simd<uint32_t, 128> Tg = slm_block_load<uint32_t, 128>(slm_bg + (uint32_t) u * SLB);
                            simd<uint32_t, 128> Tu = slm_block_load<uint32_t, 128>(slm_bu + (uint32_t) u * SLB);
#pragma unroll
                            for (int s = 0; s < 2; ++s) {
                                simd<int8_t, RC * 32> Am;
                                if (m0 < (int) M) {
                                    Am = ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                                        A, aw, ah, aw, (ib + u * 2 + s) * 32, m0);
                                } else {
                                    Am = simd<int8_t, RC * 32>(int8_t(0));
                                }
                                simd<uint32_t, 64> B4 = Tg.template select<64, 1>(64 * s);
                                simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                                simd<int, RC * 16> ga = gacc.template select<RC * 16, 1>(0);
                                ga = s4t_mad<CHAIN>(ga, Bw, Am);
                                gacc.template select<RC * 16, 1>(0) = ga;
                                simd<uint32_t, 64> U4 = Tu.template select<64, 1>(64 * s);
                                simd<uint8_t, 256> Uw = U4.template bit_cast_view<uint8_t>();
                                simd<int, RC * 16> ua = uacc.template select<RC * 16, 1>(0);
                                ua = s4t_mad<CHAIN>(ua, Uw, Am);
                                uacc.template select<RC * 16, 1>(0) = ua;
                            }
                        }
                        barrier();
                    }
                } else {
                    for (int ib = 0; ib < nb; ib += 2) {
                        if (ib + 2 < nb && m0 < (int) M) {
                            ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                                A, aw, ah, aw, (ib + 2) * 32, m0);
                        }
                        if (ib + 2 < nb) {
                            ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                qg, sw, sh, sw, (ib + 2) * 4, n0);
                        }
                        simd<uint32_t, 128> T =
                            ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                                qg, sw, sh, sw, ib * 4, n0);
#pragma unroll
                        for (int s = 0; s < 2; ++s) {
                            simd<int8_t, RC * 32> Am;
                            if (m0 < (int) M) {
                                Am = ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                                    A, aw, ah, aw, (ib + s) * 32, m0);
                            } else {
                                Am = simd<int8_t, RC * 32>(int8_t(0));
                            }
                            simd<uint32_t, 64> B4 = T.template select<64, 1>(64 * s);
                            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                            simd<int, RC * 16> ga = gacc.template select<RC * 16, 1>(0);
                            ga = s4t_mad<CHAIN>(ga, Bw, Am);
                            gacc.template select<RC * 16, 1>(0) = ga;
                        }
                    }
                }
            } else {
            for (int ib = 0; ib < nb; ib += 8) {
                const int nblk = (ib + 8 <= nb) ? 4 : ((nb - ib) / 2);
                if (ib + 8 < nb && m0 < (int) M) {
                    ex::lsc_prefetch_2d<int8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                        A, aw, ah, aw, (ib + 8) * 32, m0);
                }
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (ib + 8 < nb) {
                        ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                            qg, sw, sh, sw, (ib + 8) * 4, n0 + t * 16);
                        if constexpr (DUAL != 0) {
                            ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                qu, sw, sh, sw, (ib + 8) * 4, n0 + t * 16);
                        }
                    }
                }
                for (int u = 0; u < nblk; ++u) {
                    simd<uint32_t, 128> Tg[NT];
                    simd<uint32_t, (DUAL != 0) ? 128 : 1> Tu[NT];
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        Tg[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                            qg, sw, sh, sw, (ib + u * 2) * 4, n0 + t * 16);
                        if constexpr (DUAL != 0) {
                            Tu[t] = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                                qu, sw, sh, sw, (ib + u * 2) * 4, n0 + t * 16);
                        }
                    }
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        simd<int8_t, RC * 32> Am;
                        if (m0 < (int) M) {
                            Am = ex::lsc_load_2d<int8_t, 32, RC, 1, false, false>(
                                A, aw, ah, aw, (ib + u * 2 + s) * 32, m0);
                        } else {
                            Am = simd<int8_t, RC * 32>(int8_t(0));
                        }
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            simd<uint32_t, 64> B4 = Tg[t].template select<64, 1>(64 * s);
                            simd<uint8_t, 256> Bw = B4.template bit_cast_view<uint8_t>();
                            simd<int, RC * 16> ga = gacc.template select<RC * 16, 1>(t * RC * 16);
                            if constexpr (CHAIN != 0) {
                                ga = xmx::dpas<8, RC, int, int, uint8_t, int8_t,
                                    xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(ga, Bw, Am);
                            } else {
                                simd<int, RC * 16> Ci =
                                    xmx::dpas<8, RC, int, int, uint8_t, int8_t,
                                        xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                        simd<int, RC * 16>(0), Bw, Am);
                                ga = ga + Ci;
                            }
                            gacc.template select<RC * 16, 1>(t * RC * 16) = ga;
                            if constexpr (DUAL != 0) {
                                simd<uint32_t, 64> U4 = Tu[t].template select<64, 1>(64 * s);
                                simd<uint8_t, 256> Uw = U4.template bit_cast_view<uint8_t>();
                                simd<int, RC * 16> ua = uacc.template select<RC * 16, 1>(t * RC * 16);
                                if constexpr (CHAIN != 0) {
                                    ua = xmx::dpas<8, RC, int, int, uint8_t, int8_t,
                                        xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(ua, Uw, Am);
                                } else {
                                    simd<int, RC * 16> Ci =
                                        xmx::dpas<8, RC, int, int, uint8_t, int8_t,
                                            xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                                            simd<int, RC * 16>(0), Uw, Am);
                                    ua = ua + Ci;
                                }
                                uacc.template select<RC * 16, 1>(t * RC * 16) = ua;
                            }
                        }
                    }
                }
            }
            }
            if (m0 >= (int) M) {
                return;
            }
            const float sa0 = sa[0];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                simd<float, 16> sg;
                sg.copy_from(scg + (size_t) (n0 + t * 16));
                simd<float, 16> su;
                if constexpr (DUAL != 0) {
                    su.copy_from(scu + (size_t) (n0 + t * 16));
                }
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<int, 16> cg = gacc.template select<16, 1>(t * RC * 16 + r * 16);
                        simd<float, 16> row = convert<float>(cg) * sg * sa0;
                        if constexpr (DUAL != 0) {
                            simd<int, 16> cu = uacc.template select<16, 1>(t * RC * 16 + r * 16);
                            simd<float, 16> uu = convert<float>(cu) * su * sa0;
                            row = row / (simd<float, 16>(1.0f) + sycl::ext::intel::esimd::exp(-row)) * uu;
                        }
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * 16));
                    }
                }
            }
        });
    return true;
}

} // namespace

#endif

bool ggml_sycl_xe2mx_gemv_f32(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * x,
                               float * C,
                               int64_t K,
                               int64_t N) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || x == nullptr || C == nullptr) {
        return false;
    }
    if ((K % 32) != 0 || (N % 16) != 0 || N <= 0 || K < 64) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        if ((N % 64) == 0) {
            return launch_gemv_f32_nt<4>(ctx.stream(), packed, x, C, K, N);
        }
        return launch_gemv_f32_nt<1>(ctx.stream(), packed, x, C, K, N);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx gemv: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(x);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    return false;
#endif
}

bool ggml_sycl_xe2mx_gemm_s8(ggml_backend_sycl_context & ctx,
                              const int8_t * W8,
                              const uint16_t * d16,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M,
                              int64_t pitch) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (W8 == nullptr || d16 == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0 || pitch < N || (pitch % 16) != 0 || pitch < 64) {
        return false;
    }
    if (K < 64) {
        return false;
    }
    try {
        if (M >= 8 && ((K / 32) % XE2MX_S8_KB) == 0) {
            launch_s8_soa<XE2MX_WG, XE2MX_S8_KB, 8, 4>(ctx.stream(), W8, d16, A, da, C, K, N, M, pitch);
        } else if (M >= 4 && ((K / 32) % XE2MX_S8_KB) == 0) {
            launch_s8_soa<XE2MX_WG, XE2MX_S8_KB, 4, 1>(ctx.stream(), W8, d16, A, da, C, K, N, M, pitch);
        } else if (M >= 2 && ((K / 32) % XE2MX_S8_KB) == 0) {
            launch_s8_soa<XE2MX_WG, XE2MX_S8_KB, 2, 1>(ctx.stream(), W8, d16, A, da, C, K, N, M, pitch);
        } else if (((K / 32) % XE2MX_S8_KB) == 0) {
            launch_s8_soa<XE2MX_WG, XE2MX_S8_KB, 1, 1>(ctx.stream(), W8, d16, A, da, C, K, N, M, pitch);
        } else {
            launch_s8<8, 16>(ctx.stream(), W8, d16, A, da, C, K, N, M, pitch);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx s8 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(W8);
    GGML_UNUSED(d16);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    GGML_UNUSED(pitch);
    return false;
#endif
}

bool ggml_sycl_xe2mx_gemm_s4t(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * scw,
                               const int8_t * A,
                               const float * sa,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M,
                               const float * glu_up) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    GGML_UNUSED(glu_up);
    if (packed == nullptr || scw == nullptr || A == nullptr || sa == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    static int chain_dead = 0;
    static int nt2_dead = 0;
    const int want_chain = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TCHAIN", 1);
    const int want_nt2 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TNT", 1) >= 2;
    auto run = [&](int chain) -> bool {
        if (want_nt2 && !nt2_dead && (N % 512) == 0) {
            try {
                if (chain) {
                    if (launch_s4t_nt<2, 1, 0>(ctx.stream(), packed, scw, nullptr, nullptr, A, sa, C, K, N, M)) {
                        return true;
                    }
                } else if (launch_s4t_nt<2, 0, 0>(ctx.stream(), packed, scw, nullptr, nullptr, A, sa, C, K, N, M)) {
                    return true;
                }
            } catch (const sycl::exception & e) {
                GGML_SYCL_XE2_LOG( "xe2mx s4t nt2: %s\n", e.what());
                fflush(stderr);
                nt2_dead = 1;
            }
        }
        if (chain) {
            return launch_s4t_nt<1, 1, 0>(ctx.stream(), packed, scw, nullptr, nullptr, A, sa, C, K, N, M);
        }
        return launch_s4t_nt<1, 0, 0>(ctx.stream(), packed, scw, nullptr, nullptr, A, sa, C, K, N, M);
    };
    try {
        if (want_chain && !chain_dead) {
            if (run(1)) {
                return true;
            }
        }
        return run(0);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx s4t dpas: %s\n", e.what());
        fflush(stderr);
        if (!chain_dead) {
            chain_dead = 1;
            try {
                return run(0);
            } catch (const sycl::exception & e2) {
                GGML_SYCL_XE2_LOG( "xe2mx s4t add: %s\n", e2.what());
                fflush(stderr);
                return false;
            }
        }
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(scw);
    GGML_UNUSED(A);
    GGML_UNUSED(sa);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    GGML_UNUSED(glu_up);
    return false;
#endif
}

bool ggml_sycl_xe2mx_gemm_s4t_glu(ggml_backend_sycl_context & ctx,
                                   const uint8_t * pg,
                                   const float * scg,
                                   const uint8_t * pu,
                                   const float * scu,
                                   const int8_t * A,
                                   const float * sa,
                                   float * C,
                                   int64_t K,
                                   int64_t N,
                                   int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (pg == nullptr || scg == nullptr || pu == nullptr || scu == nullptr
        || A == nullptr || sa == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    static int chain_dead = 0;
    static int nt2_dead = 0;
    static int glu_dead = 0;
    if (glu_dead) {
        return false;
    }
    const int want_chain = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TCHAIN", 1);
    const int want_nt2 = ggml_sycl_get_env("GGML_SYCL_XE2MX_S4TNT", 1) >= 2;
    auto run = [&](int chain) -> bool {
        if (want_nt2 && !nt2_dead && (N % 512) == 0) {
            try {
                if (chain) {
                    if (launch_s4t_nt<2, 1, 1>(ctx.stream(), pg, scg, pu, scu, A, sa, C, K, N, M)) {
                        return true;
                    }
                } else if (launch_s4t_nt<2, 0, 1>(ctx.stream(), pg, scg, pu, scu, A, sa, C, K, N, M)) {
                    return true;
                }
            } catch (const sycl::exception & e) {
                GGML_SYCL_XE2_LOG( "xe2mx s4t glu nt2: %s\n", e.what());
                fflush(stderr);
                nt2_dead = 1;
            }
        }
        if (chain) {
            return launch_s4t_nt<1, 1, 1>(ctx.stream(), pg, scg, pu, scu, A, sa, C, K, N, M);
        }
        return launch_s4t_nt<1, 0, 1>(ctx.stream(), pg, scg, pu, scu, A, sa, C, K, N, M);
    };
    try {
        if (want_chain && !chain_dead) {
            if (run(1)) {
                return true;
            }
        }
        return run(0);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx s4t glu: %s\n", e.what());
        fflush(stderr);
        if (!chain_dead) {
            chain_dead = 1;
            try {
                return run(0);
            } catch (const sycl::exception & e2) {
                GGML_SYCL_XE2_LOG( "xe2mx s4t glu add: %s\n", e2.what());
                fflush(stderr);
                glu_dead = 1;
                return false;
            }
        }
        glu_dead = 1;
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(pg);
    GGML_UNUSED(scg);
    GGML_UNUSED(pu);
    GGML_UNUSED(scu);
    GGML_UNUSED(A);
    GGML_UNUSED(sa);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_gemm_s8_vnni(ggml_backend_sycl_context & ctx,
                                   const int8_t * W8,
                                   const uint16_t * d16,
                                   const int8_t * A,
                                   const float * da,
                                   float * C,
                                   int64_t K,
                                   int64_t N,
                                   int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (W8 == nullptr || d16 == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    try {
        if ((N % 64) == 0) {
            return launch_s8_pf_nt<4>(ctx.stream(), W8, d16, A, da, C, K, N, M);
        }
        return launch_s8_pf_nt<1>(ctx.stream(), W8, d16, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx s8x dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(W8);
    GGML_UNUSED(d16);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_gemm_q8(ggml_backend_sycl_context & ctx,
                               const void * q8,
                               const int8_t * A,
                               const float * da,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (q8 == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (K < 64 || ((K / 32) % XE2MX_S8_KB) != 0) {
        return false;
    }
    const int8_t * qs = (const int8_t *) q8;
    const uint16_t * d16 = (const uint16_t *) (qs + (size_t) K * (size_t) N);
    try {
        if (M >= 8) {
            launch_q8_aos<XE2MX_WG, XE2MX_S8_KB, 8, 4>(ctx.stream(), qs, d16, A, da, C, K, N, M);
        } else if (M >= 4) {
            launch_q8_aos<XE2MX_WG, XE2MX_S8_KB, 4, 1>(ctx.stream(), qs, d16, A, da, C, K, N, M);
        } else if (M >= 2) {
            launch_q8_aos<XE2MX_WG, XE2MX_S8_KB, 2, 1>(ctx.stream(), qs, d16, A, da, C, K, N, M);
        } else {
            launch_q8_aos<1, XE2MX_S8_KB, 1, 1>(ctx.stream(), qs, d16, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx q8 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(q8);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_prefill(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx pf dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_prefill_n8(ggml_backend_sycl_context & ctx,
                                 const uint8_t * packed,
                                 const int8_t * A,
                                 const float * da,
                                 float * C,
                                 int64_t K,
                                 int64_t N,
                                 int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1024 || (K % 32) != 0 || (N % 128) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill_nt<8>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx pf n8 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_ksplit(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % (2 * XE2MX_KS_WG) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        if ((N % 64) == 0) {
            return launch_prefill_ksplit<4>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        if ((N % 32) == 0) {
            return launch_prefill_ksplit<2>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        return launch_prefill_ksplit<1>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx ks dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

// Short-K M=2 mats (qkv, Q, linear-attn) sit at ~half bus: 4 threads do not
// keep the load pipe full. WG=8 walks the same 16 DPAS rows with more loads
// in flight. Fat N (FFN 17408) stays on WG=4. Default off.
static bool xe2mx_m2_short(int64_t K, int64_t N) {
    static const int wg = ggml_sycl_get_env("GGML_SYCL_XE2MX_M2WG", 4);
    return wg == 8 && K <= 6144 && N > 0 && N <= 12288;
}

// M=5..7 still takes two 4-row weight loads. MID16=1 uses the one-load 16-row tile there.
// M=2 and M=4 stay on their current tiles. Default off.
static bool xe2mx_mid16(int64_t M) {
    static const int on = ggml_sycl_get_env("GGML_SYCL_XE2MX_MID16", 0);
    return on == 1 && M >= 5 && M < 8;
}

bool ggml_sycl_xe2mx_xmx(ggml_backend_sycl_context & ctx,
                          const uint8_t * packed,
                          const int8_t * A,
                          const float * da,
                          float * C,
                          int64_t K,
                          int64_t N,
                          int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M == 1 && ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32V", 0)) {
            launch_gemv_soa_sk<XE2MX_KB, 8>(stream, packed, A, da, C, K, N);
        } else if (M >= 32 && ggml_sycl_get_env("GGML_SYCL_XE2MX_NI", 0)) {
            launch_ninner<4, 256>(stream, packed, A, da, C, K, N, M);
        } else if (M < 32 && ggml_sycl_get_env("GGML_SYCL_XE2MX_Q32KB", 8) == 16 && ((K / 32) % 16) == 0) {
            if (M >= 8) {
                launch_soa<XE2MX_WG, 16, 8, 4>(stream, packed, A, da, C, K, N, M);
            } else if (M >= 4) {
                launch_soa<XE2MX_WG, 16, 4, 1>(stream, packed, A, da, C, K, N, M);
            } else if (M >= 2) {
                launch_soa<XE2MX_WG, 16, 2, 1>(stream, packed, A, da, C, K, N, M);
            } else {
                launch_soa<XE2MX_WG, 16, 1, 1>(stream, packed, A, da, C, K, N, M);
            }
        } else if (M >= 8) {
            // GGML_SYCL_XE2MX_XMX_TILE (M>=8 only): 0 = <8,4>, 41 = <4,1>, 81 = <8,1>, 82 = <8,2>. Other -> <8,4>.
            static const int xmx_tile = ggml_sycl_get_env("GGML_SYCL_XE2MX_XMX_TILE", 0);
            switch (xmx_tile) {
                case 41:
                    launch_soa<XE2MX_WG, XE2MX_KB, 4, 1>(stream, packed, A, da, C, K, N, M);
                    break;
                case 81:
                    launch_soa<XE2MX_WG, XE2MX_KB, 8, 1>(stream, packed, A, da, C, K, N, M);
                    break;
                case 82:
                    launch_soa<XE2MX_WG, XE2MX_KB, 8, 2>(stream, packed, A, da, C, K, N, M);
                    break;
                default:
                    launch_soa<XE2MX_WG, XE2MX_KB, 8, 4>(stream, packed, A, da, C, K, N, M);
                    break;
            }
        } else if (M >= 4) {
            if (xe2mx_mid16(M)) {
                launch_soa<XE2MX_WG, XE2MX_KB, 8, 2>(stream, packed, A, da, C, K, N, M);
            } else {
                launch_soa<XE2MX_WG, XE2MX_KB, 4, 1>(stream, packed, A, da, C, K, N, M);
            }
        } else if (M >= 2) {
            if (xe2mx_m2_short(K, N)) {
                launch_soa<8, XE2MX_KB, 2, 1>(stream, packed, A, da, C, K, N, M);
            } else {
                launch_soa<XE2MX_WG, XE2MX_KB, 2, 1>(stream, packed, A, da, C, K, N, M);
            }
        } else {
            launch_soa<XE2MX_WG, XE2MX_KB, 1, 1>(stream, packed, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_glu(ggml_backend_sycl_context & ctx,
                               const uint8_t * pg,
                               const uint8_t * pu,
                               const int8_t * A,
                               const float * da,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (pg == nullptr || pu == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            // GGML_SYCL_XE2MX_GLU_TILE (M>=8 only): 0 = <8,4>, 41 = <4,1>, 81 = <8,1>, 82 = <8,2>,
            // 91 = glu1 <8,1>, 92 = glu1 <8,2>. Other -> <8,4>.
            static const int glu_tile = ggml_sycl_get_env("GGML_SYCL_XE2MX_GLU_TILE", 0);
            switch (glu_tile) {
                case 41:
                    launch_soa_glu<XE2MX_WG, XE2MX_KB, 4, 1>(stream, pg, pu, A, da, C, K, N, M);
                    break;
                case 81:
                    launch_soa_glu<XE2MX_WG, XE2MX_KB, 8, 1>(stream, pg, pu, A, da, C, K, N, M);
                    break;
                case 82:
                    launch_soa_glu<XE2MX_WG, XE2MX_KB, 8, 2>(stream, pg, pu, A, da, C, K, N, M);
                    break;
                case 91:
                    launch_soa_glu1<XE2MX_WG, XE2MX_KB, 8, 1>(stream, pg, pu, A, da, C, K, N, M);
                    break;
                case 92:
                    launch_soa_glu1<XE2MX_WG, XE2MX_KB, 8, 2>(stream, pg, pu, A, da, C, K, N, M);
                    break;
                default:
                    launch_soa_glu<XE2MX_WG, XE2MX_KB, 8, 4>(stream, pg, pu, A, da, C, K, N, M);
                    break;
            }
        } else if (M >= 4) {
            if (xe2mx_mid16(M)) {
                launch_soa_glu1<XE2MX_WG, XE2MX_KB, 8, 2>(stream, pg, pu, A, da, C, K, N, M);
            } else {
                launch_soa_glu<XE2MX_WG, XE2MX_KB, 4, 1>(stream, pg, pu, A, da, C, K, N, M);
            }
        } else if (M >= 2) {
            launch_soa_glu<XE2MX_WG, XE2MX_KB, 2, 1>(stream, pg, pu, A, da, C, K, N, M);
        } else {
            launch_soa_glu<XE2MX_WG, XE2MX_KB, 1, 1>(stream, pg, pu, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx fglu dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(pg);
    GGML_UNUSED(pu);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_pair(ggml_backend_sycl_context & ctx,
                                const uint8_t * p0,
                                const uint8_t * p1,
                                const int8_t * A,
                                const float * da,
                                float * C0,
                                float * C1,
                                int64_t K,
                                int64_t N,
                                int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (p0 == nullptr || p1 == nullptr || A == nullptr || da == nullptr || C0 == nullptr || C1 == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            // Same GGML_SYCL_XE2MX_GLU_TILE values as ggml_sycl_xe2mx_xmx_glu; 91/92 use gemm_soa_glu1.
            static const int pair_tile = ggml_sycl_get_env("GGML_SYCL_XE2MX_GLU_TILE", 0);
            switch (pair_tile) {
                case 41:
                    launch_soa_pair<XE2MX_WG, XE2MX_KB, 4, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
                case 81:
                    launch_soa_pair<XE2MX_WG, XE2MX_KB, 8, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
                case 82:
                    launch_soa_pair<XE2MX_WG, XE2MX_KB, 8, 2>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
                case 91:
                    launch_soa_pair1<XE2MX_WG, XE2MX_KB, 8, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
                case 92:
                    launch_soa_pair1<XE2MX_WG, XE2MX_KB, 8, 2>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
                default:
                    launch_soa_pair<XE2MX_WG, XE2MX_KB, 8, 4>(stream, p0, p1, A, da, C0, C1, K, N, M);
                    break;
            }
        } else if (M >= 4) {
            if (xe2mx_mid16(M)) {
                launch_soa_pair1<XE2MX_WG, XE2MX_KB, 8, 2>(stream, p0, p1, A, da, C0, C1, K, N, M);
            } else {
                launch_soa_pair<XE2MX_WG, XE2MX_KB, 4, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
            }
        } else if (M >= 2) {
            if (xe2mx_m2_short(K, N)) {
                launch_soa_pair<8, XE2MX_KB, 2, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
            } else {
                launch_soa_pair<XE2MX_WG, XE2MX_KB, 2, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
            }
        } else {
            launch_soa_pair<XE2MX_WG, XE2MX_KB, 1, 1>(stream, p0, p1, A, da, C0, C1, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx f2 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C0);
    GGML_UNUSED(C1);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_fn(ggml_backend_sycl_context & ctx,
                             const uint8_t * p0,
                             const uint8_t * p1,
                             const int8_t * A,
                             const float * da,
                             float * C0,
                             float * C1,
                             int64_t K,
                             int64_t N0,
                             int64_t N1,
                             int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (p0 == nullptr || p1 == nullptr || A == nullptr || da == nullptr || C0 == nullptr || C1 == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N0 % 16) != 0 || (N1 % 16) != 0 || N0 <= 0 || N1 <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa_fn<XE2MX_WG, XE2MX_KB, 8, 4>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
        } else if (M >= 4) {
            if (xe2mx_mid16(M)) {
                launch_soa_fn<XE2MX_WG, XE2MX_KB, 8, 2>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
            } else {
                launch_soa_fn<XE2MX_WG, XE2MX_KB, 4, 1>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
            }
        } else if (M >= 2) {
            if (xe2mx_m2_short(K, N0) && xe2mx_m2_short(K, N1)) {
                launch_soa_fn<8, XE2MX_KB, 2, 1>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
            } else {
                launch_soa_fn<XE2MX_WG, XE2MX_KB, 2, 1>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
            }
        } else {
            launch_soa_fn<XE2MX_WG, XE2MX_KB, 1, 1>(stream, p0, p1, A, da, C0, C1, K, N0, N1, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx fn dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C0);
    GGML_UNUSED(C1);
    GGML_UNUSED(K);
    GGML_UNUSED(N0);
    GGML_UNUSED(N1);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_batch(ggml_backend_sycl_context & ctx, const xe2mx_batch_ent * e, int nw) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (e == nullptr || nw < 1 || nw > XE2MX_BATCH_MAX) {
        return false;
    }
    const int64_t M = e[0].M;
    // M == 1 keeps its own path (Q32V GEMV); M >= 32 is prefill.
    if (M < 2 || M >= 32) {
        return false;
    }
    const int ncm = M >= 8 ? 32 : M >= 4 ? 4 : 2;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    xe2mx_bt_args g = {};
    g.nw = nw;
    int acc = 0;
    for (int w = 0; w < nw; ++w) {
        const xe2mx_batch_ent & x = e[w];
        if (x.vx == nullptr || x.A == nullptr || x.da == nullptr || x.C == nullptr || x.M != M) {
            return false;
        }
        if ((x.K % 32) != 0 || (x.N % 16) != 0 || x.N <= 0 || x.K > INT32_MAX || x.N > INT32_MAX) {
            return false;
        }
        if ((x.K / 32) % XE2MX_KB != 0 || (unsigned) (x.K / 32) * 16u < 64u) {
            return false;
        }
        g.vx[w] = x.vx;
        g.A[w] = x.A;
        g.da[w] = x.da;
        g.C[w] = x.C;
        g.K[w] = (int) x.K;
        g.N[w] = (int) x.N;
        g.M[w] = (int) x.M;
        acc += (int) (x.N / XE2MX_RPW) * m_tiles;
        g.wg_end[w] = acc;
    }
    for (int w = nw; w < XE2MX_BATCH_MAX; ++w) {
        g.vx[w] = g.vx[0];
        g.A[w] = g.A[0];
        g.da[w] = g.da[0];
        g.C[w] = g.C[0];
        g.K[w] = g.K[0];
        g.N[w] = g.N[0];
        g.M[w] = g.M[0];
        g.wg_end[w] = acc;
    }
    queue_ptr stream = ctx.stream();
    try {
        // Same tile pick as ggml_sycl_xe2mx_xmx for 2 <= M < 32 with Q32KB at its default.
        if (M >= 8) {
            launch_soa_batch<XE2MX_WG, XE2MX_KB, 8, 4>(stream, g);
        } else if (M >= 4) {
            launch_soa_batch<XE2MX_WG, XE2MX_KB, 4, 1>(stream, g);
        } else {
            launch_soa_batch<XE2MX_WG, XE2MX_KB, 2, 1>(stream, g);
        }
    } catch (const sycl::exception & ex) {
        GGML_SYCL_XE2_LOG( "xe2mx batch dpas: %s\n", ex.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(e);
    GGML_UNUSED(nw);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_mg(ggml_backend_sycl_context & ctx,
                            const uint8_t * const * vx,
                            float * const * C,
                            const int64_t * N,
                            const int64_t * Ndst,
                            int nw,
                            const int8_t * A,
                            const float * da,
                            int64_t K,
                            int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (nw < 1 || nw > 8 || A == nullptr || da == nullptr || M < 1 || (K % 32) != 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0 || (unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    const int ncm = M >= 8 ? 32 : M >= 4 ? 4 : M >= 2 ? 2 : 1;
    const int m_tiles = (int) ((M + ncm - 1) / ncm);
    xe2mx_mg_args g = {};
    g.nw = nw;
    int acc = 0;
    for (int w = 0; w < nw; ++w) {
        if (vx[w] == nullptr || C[w] == nullptr || N[w] <= 0 || (N[w] % 16) != 0) {
            return false;
        }
        const int64_t nd = Ndst ? Ndst[w] : N[w];
        if (nd <= 0 || nd > N[w]) {
            return false;
        }
        g.vx[w] = vx[w];
        g.C[w] = C[w];
        g.N[w] = (int) N[w];
        g.Ndst[w] = (int) nd;
        acc += (int) (N[w] / XE2MX_RPW) * m_tiles;
        g.wg_end[w] = acc;
    }
    for (int w = nw; w < 8; ++w) {
        g.wg_end[w] = acc;
        g.vx[w] = vx[0];
        g.C[w] = C[0];
        g.N[w] = (int) N[0];
        g.Ndst[w] = Ndst ? (int) Ndst[0] : (int) N[0];
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa_mg<XE2MX_WG, XE2MX_KB, 8, 4>(stream, g, A, da, K, M);
        } else if (M >= 4) {
            if (xe2mx_mid16(M)) {
                launch_soa_mg<XE2MX_WG, XE2MX_KB, 8, 2>(stream, g, A, da, K, M);
            } else {
                launch_soa_mg<XE2MX_WG, XE2MX_KB, 4, 1>(stream, g, A, da, K, M);
            }
        } else if (M >= 2) {
            bool short_mg = xe2mx_m2_short(K, 16);
            for (int w = 0; short_mg && w < nw; ++w) {
                short_mg = xe2mx_m2_short(K, N[w]);
            }
            if (short_mg) {
                launch_soa_mg<8, XE2MX_KB, 2, 1>(stream, g, A, da, K, M);
            } else {
                launch_soa_mg<XE2MX_WG, XE2MX_KB, 2, 1>(stream, g, A, da, K, M);
            }
        } else {
            launch_soa_mg<XE2MX_WG, XE2MX_KB, 1, 1>(stream, g, A, da, K, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx mg dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(vx);
    GGML_UNUSED(C);
    GGML_UNUSED(N);
    GGML_UNUSED(Ndst);
    GGML_UNUSED(nw);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(K);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx2(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    constexpr int WG2 = 2;
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa<WG2, XE2MX_KB, 8, 4>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 4) {
            launch_soa<WG2, XE2MX_KB, 4, 1>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 2) {
            launch_soa<WG2, XE2MX_KB, 2, 1>(stream, packed, A, da, C, K, N, M);
        } else {
            launch_soa<WG2, XE2MX_KB, 1, 1>(stream, packed, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx2 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx1(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    constexpr int WG1 = 1;
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa<WG1, XE2MX_KB, 8, 4>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 4) {
            launch_soa<WG1, XE2MX_KB, 4, 1>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 2) {
            launch_soa<WG1, XE2MX_KB, 2, 1>(stream, packed, A, da, C, K, N, M);
        } else {
            launch_soa<WG1, XE2MX_KB, 1, 1>(stream, packed, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx1 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx8(ggml_backend_sycl_context & ctx,
                             const uint8_t * packed,
                             const int8_t * A,
                             const float * da,
                             float * C,
                             int64_t K,
                             int64_t N,
                             int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    constexpr int WG8 = 8;
    constexpr int KB4 = 4;
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % (KB4 * WG8) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa<WG8, KB4, 8, 4>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 4) {
            launch_soa<WG8, KB4, 4, 1>(stream, packed, A, da, C, K, N, M);
        } else if (M >= 2) {
            launch_soa<WG8, KB4, 2, 1>(stream, packed, A, da, C, K, N, M);
        } else {
            launch_soa<WG8, KB4, 1, 1>(stream, packed, A, da, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx8 dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_nt(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const int8_t * A,
                               const float * da,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    constexpr int WG8 = 8;
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % (WG8 * XE2MX_RPW)) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        if (M >= 8) {
            return launch_xmx_nt<WG8, XE2MX_KB, 8, 4>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        if (M >= 4) {
            return launch_xmx_nt<WG8, XE2MX_KB, 4, 1>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        if (M >= 2) {
            return launch_xmx_nt<WG8, XE2MX_KB, 2, 1>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        return launch_xmx_nt<WG8, XE2MX_KB, 1, 1>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx nt dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_blk(ggml_backend_sycl_context & ctx,
                            const uint8_t * packed,
                            const int8_t * A,
                            const float * da,
                            float * C,
                            int64_t K,
                            int64_t N,
                            int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    const int rm = ggml_sycl_get_env("GGML_SYCL_XE2MX_RM", 2);
    try {
        if ((N % 64) != 0) {
            return false;
        }
        if (rm >= 4) {
            return launch_prefill_rm<4, 4>(ctx.stream(), packed, A, da, C, K, N, M);
        }
        return launch_prefill_rm<2, 4>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx blk dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_isa(ggml_backend_sycl_context & ctx,
                          const uint8_t * packed,
                          const int8_t * A,
                          const float * da,
                          float * C,
                          int64_t K,
                          int64_t N,
                          int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1024 || (K % 32) != 0 || (N % 128) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill_isa<6, 2>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx isa dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_isa_h(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const sycl::half * xh,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || xh == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1024 || (K % 32) != 0 || (N % 128) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill_isa_h<6, 2>(ctx.stream(), packed, xh, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx isa_h dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(xh);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_isa_hf(ggml_backend_sycl_context & ctx,
                             const uint8_t * packed,
                             const sycl::half * xh,
                             float * C,
                             int64_t K,
                             int64_t N,
                             int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || xh == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1024 || (K % 32) != 0 || (N % 256) != 0 || N <= 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill_hf<4>(ctx.stream(), packed, xh, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx isa_hf dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(xh);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_wg(ggml_backend_sycl_context & ctx,
                          const uint8_t * packed,
                          const int8_t * A,
                          const float * da,
                          float * C,
                          int64_t K,
                          int64_t N,
                          int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || A == nullptr || da == nullptr || C == nullptr) {
        return false;
    }
    if (M < 32 || (K % 32) != 0 || (N % 128) != 0 || N <= 0) {
        return false;
    }
    if (((K / 32) & 1) != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    try {
        return launch_prefill_wg<2>(ctx.stream(), packed, A, da, C, K, N, M);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx wg dpas: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(A);
    GGML_UNUSED(da);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

bool ggml_sycl_xe2mx_xmx_f32(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * x,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (packed == nullptr || x == nullptr || C == nullptr) {
        return false;
    }
    if (M < 1 || (K % 32) != 0 || (N % 16) != 0 || N <= 0) {
        return false;
    }
    if ((K / 32) % XE2MX_KB != 0) {
        return false;
    }
    if ((unsigned) (K / 32) * 16u < 64u) {
        return false;
    }
    queue_ptr stream = ctx.stream();
    try {
        if (M >= 8) {
            launch_soa_f32<8, 4>(stream, packed, x, C, K, N, M);
        } else if (M >= 4) {
            launch_soa_f32<4, 1>(stream, packed, x, C, K, N, M);
        } else if (M >= 2) {
            launch_soa_f32<2, 1>(stream, packed, x, C, K, N, M);
        } else {
            launch_soa_f32<1, 1>(stream, packed, x, C, K, N, M);
        }
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx xmx f32: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(packed);
    GGML_UNUSED(x);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    return false;
#endif
}

#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
namespace ex  = sycl::ext::intel::experimental::esimd;

template <int DUAL, int NT, int HF = 0>
static bool launch_lo_s44_nt(queue_ptr stream, const uint8_t * a4, const float * sa,
                              const uint8_t * wg, const float * scg,
                              const uint8_t * wu, const float * scu,
                              float * C, int64_t M, int64_t N, int64_t K, int64_t g) {
    constexpr int RC = 8;
    constexpr int LM = 8;
    constexpr int LN0 = 4;
    constexpr int BN = 16;
    constexpr int nstep = BN * NT;
    if (!stream || !a4 || !sa || !wg || !scg || !C) {
        return false;
    }
    if constexpr (DUAL != 0) {
        if (!wu || !scu) {
            return false;
        }
    }
    if (M < 1 || (K % 64) != 0 || (N % nstep) != 0 || (g % 64) != 0 || g < 64 || (K % g) != 0) {
        return false;
    }
    int LN = LN0;
    while (LN > 1 && (N % (nstep * LN)) != 0) {
        LN /= 2;
    }
    if ((N % (nstep * LN)) != 0) {
        return false;
    }
    const int nbg = (int) (K / g);
    const int ibg = (int) (g / 32);
    const int m_tiles = (int) ((M + RC - 1) / RC);
    const int n_tiles = (int) (N / nstep);
    const int m_wg = (m_tiles + LM - 1) / LM;
    const int n_wg = n_tiles / LN;
    const unsigned pitch = (unsigned) (K / 2);
    const uint32_t * qg = (const uint32_t *) wg;
    const uint32_t * qu = (DUAL != 0) ? (const uint32_t *) wu : qg;
    const unsigned sw = pitch - 1u;
    const unsigned sh = (unsigned) N - 1u;
    const unsigned aw = HF ? ((unsigned) K * 2u - 1u) : (pitch - 1u);
    const unsigned ah = (unsigned) M - 1u;
    const sycl::half * xh = HF ? reinterpret_cast<const sycl::half *>(a4) : nullptr;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<2>(sycl::range<2>((size_t) m_wg * (size_t) LM, (size_t) n_wg * (size_t) LN),
                          sycl::range<2>((size_t) LM, (size_t) LN)),
        props,
        [=](sycl::nd_item<2> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * RC;
            const int n0 = (int) it.get_global_id(1) * nstep;
            simd<float, RC * 16> accf[NT];
            simd<float, RC * 16> accu[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                accf[t] = 0.0f;
                if constexpr (DUAL != 0) {
                    accu[t] = 0.0f;
                }
            }
            for (int ig = 0; ig < nbg; ++ig) {
                simd<int, RC * 16> acc[NT];
                simd<int, RC * 16> acc2[NT];
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    acc[t] = 0;
                    if constexpr (DUAL != 0) {
                        acc2[t] = 0;
                    }
                }
                for (int tt = 0; tt < ibg; tt += 2) {
                    const int ib = ig * ibg + tt;
                    if (tt + 2 < ibg) {
                        const int ibp = ib + 2;
                        if (m0 < (int) M) {
                            if constexpr (HF != 0) {
                                ex::lsc_prefetch_2d<sycl::half, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                                    xh, aw, ah, aw, ibp * 32, m0);
                            } else {
                                ex::lsc_prefetch_2d<uint8_t, 32, RC, 1, cache_hint::cached, cache_hint::cached>(
                                    a4, aw, ah, aw, ibp * 16, m0);
                            }
                        }
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                qg, sw, sh, sw, ibp * 4, n0 + t * BN);
                            if constexpr (DUAL != 0) {
                                ex::lsc_prefetch_2d<uint32_t, 8, 16, 1, cache_hint::cached, cache_hint::cached>(
                                    qu, sw, sh, sw, ibp * 4, n0 + t * BN);
                            }
                        }
                    }
                    simd<uint8_t, RC * 32> Am;
                    if (m0 < (int) M) {
                        if constexpr (HF != 0) {
#pragma unroll
                            for (int p = 0; p < 2; ++p) {
                                simd<sycl::half, RC * 32> hh =
                                    ex::lsc_load_2d<sycl::half, 32, RC, 1, false, false>(
                                        xh, aw, ah, aw, (ib + p) * 32, m0);
#pragma unroll
                                for (int r = 0; r < RC; ++r) {
                                    const int m = m0 + r;
                                    const float scv = (m < (int) M)
                                        ? sa[(size_t) m * (size_t) nbg + (size_t) ig] : 0.f;
                                    const float id = scv > 0.f ? 1.f / scv : 0.f;
                                    simd<sycl::half, 32> hs = hh.template select<32, 1>(r * 32);
                                    simd<float, 32> xf = convert<float>(hs) * id;
                                    simd<int, 32> qi = convert<int>(rnde<float>(xf));
                                    qi.merge(simd<int, 32>(7), qi > 7);
                                    qi.merge(simd<int, 32>(-8), qi < -8);
                                    simd<int, 16> qe = qi.template select<16, 2>(0);
                                    simd<int, 16> qo = qi.template select<16, 2>(1);
                                    simd<uint8_t, 16> pk = convert<uint8_t>((qe & 15) | ((qo & 15) << 4));
                                    Am.template select<16, 1>(r * 32 + p * 16) = pk;
                                }
                            }
                        } else {
                            Am = ex::lsc_load_2d<uint8_t, 32, RC, 1, false, false>(
                                a4, aw, ah, aw, ib * 16, m0);
                        }
                    } else {
                        Am = simd<uint8_t, RC * 32>(uint8_t(0));
                    }
#pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        simd<uint32_t, 128> Tg = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                            qg, sw, sh, sw, ib * 4, n0 + t * BN);
                        simd<uint8_t, 512> Bw = Tg.template bit_cast_view<uint8_t>();
                        acc[t] = xmx::dpas<8, RC, int, int, uint8_t, uint8_t,
                            xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s4>(acc[t], Bw, Am);
                        if constexpr (DUAL != 0) {
                            simd<uint32_t, 128> Tu = ex::lsc_load_2d<uint32_t, 8, 16, 1, true, false>(
                                qu, sw, sh, sw, ib * 4, n0 + t * BN);
                            simd<uint8_t, 512> Uw = Tu.template bit_cast_view<uint8_t>();
                            acc2[t] = xmx::dpas<8, RC, int, int, uint8_t, uint8_t,
                                xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s4>(acc2[t], Uw, Am);
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        const float s = sa[(size_t) m * (size_t) nbg + (size_t) ig];
#pragma unroll
                        for (int t = 0; t < NT; ++t) {
                            simd<int, 16> ci = acc[t].template select<16, 1>(r * 16);
                            accf[t].template select<16, 1>(r * 16) += convert<float>(ci) * s;
                            if constexpr (DUAL != 0) {
                                simd<int, 16> ci2 = acc2[t].template select<16, 1>(r * 16);
                                accu[t].template select<16, 1>(r * 16) += convert<float>(ci2) * s;
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                simd<float, 16> sn = block_load<float, 16>(scg + n0 + t * BN);
                simd<float, 16> su(0.0f);
                if constexpr (DUAL != 0) {
                    su = block_load<float, 16>(scu + n0 + t * BN);
                }
#pragma unroll
                for (int r = 0; r < RC; ++r) {
                    const int m = m0 + r;
                    if (m < (int) M) {
                        simd<float, 16> row = accf[t].template select<16, 1>(r * 16) * sn;
                        if constexpr (DUAL != 0) {
                            simd<float, 16> urow = accu[t].template select<16, 1>(r * 16) * su;
                            row = row / (simd<float, 16>(1.0f) + sycl::ext::intel::esimd::exp(-row)) * urow;
                        }
                        row.copy_to(C + (size_t) m * (size_t) N + (size_t) (n0 + t * BN));
                    }
                }
            }
        });
    return true;
}

template <int DUAL>
static bool launch_lo_s44(queue_ptr stream, const uint8_t * a4, const float * sa,
                           const uint8_t * wg, const float * scg,
                           const uint8_t * wu, const float * scu,
                           float * C, int64_t M, int64_t N, int64_t K, int64_t g) {
    const int nt = ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPASNT", 4);
    if (nt >= 8 && launch_lo_s44_nt<DUAL, 8>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    if (nt >= 4 && launch_lo_s44_nt<DUAL, 4>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    if (launch_lo_s44_nt<DUAL, 2>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    return launch_lo_s44_nt<DUAL, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g);
}

template <int DUAL>
static bool launch_lo_hf_s44(queue_ptr stream, const uint8_t * a4, const float * sa,
                             const uint8_t * wg, const float * scg,
                             const uint8_t * wu, const float * scu,
                             float * C, int64_t M, int64_t N, int64_t K, int64_t g) {
    const int nt = ggml_sycl_get_env("GGML_SYCL_XE2MX_LODPASNT", 16);
    if (nt >= 16 && launch_lo_s44_nt<DUAL, 16, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    if (nt >= 8 && launch_lo_s44_nt<DUAL, 8, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    if (nt >= 4 && launch_lo_s44_nt<DUAL, 4, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    if (launch_lo_s44_nt<DUAL, 2, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g)) {
        return true;
    }
    return launch_lo_s44_nt<DUAL, 1, 1>(stream, a4, sa, wg, scg, wu, scu, C, M, N, K, g);
}

static bool launch_pack_hf_s4(queue_ptr stream, const sycl::half * xh, const float * sa,
                              uint8_t * A4, int64_t M, int64_t K, int64_t g) {
    constexpr int RC = 8;
    constexpr int LM = 16;
    if (!stream || !xh || !sa || !A4) {
        return false;
    }
    if (M < 1 || (K % 64) != 0 || (g % 64) != 0 || g < 64 || (K % g) != 0) {
        return false;
    }
    const int nbg = (int) (K / g);
    const int ibg = (int) (g / 32);
    const int m_tiles = (int) ((M + RC - 1) / RC);
    const int m_wg = (m_tiles + LM - 1) / LM;
    const unsigned aw = (unsigned) K * 2u - 1u;
    const unsigned ah = (unsigned) M - 1u;
    const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) m_wg * (size_t) LM), sycl::range<1>((size_t) LM)),
        props,
        [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int m0 = (int) it.get_global_id(0) * RC;
            if (m0 >= (int) M) {
                return;
            }
            for (int ig = 0; ig < nbg; ++ig) {
                for (int tt = 0; tt < ibg; tt += 2) {
                    const int ib = ig * ibg + tt;
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        simd<sycl::half, RC * 32> hh =
                            ex::lsc_load_2d<sycl::half, 32, RC, 1, false, false>(
                                xh, aw, ah, aw, (ib + p) * 32, m0);
#pragma unroll
                        for (int r = 0; r < RC; ++r) {
                            const int m = m0 + r;
                            if (m < (int) M) {
                            const float scv = sa[(size_t) m * (size_t) nbg + (size_t) ig];
                            const float id = scv > 0.f ? 1.f / scv : 0.f;
                            simd<sycl::half, 32> hs = hh.template select<32, 1>(r * 32);
                            simd<float, 32> xf = convert<float>(hs) * id;
                            simd<int, 32> qi = convert<int>(rnde<float>(xf));
                            qi.merge(simd<int, 32>(7), qi > 7);
                            qi.merge(simd<int, 32>(-8), qi < -8);
                            simd<int, 16> qe = qi.template select<16, 2>(0);
                            simd<int, 16> qo = qi.template select<16, 2>(1);
                            simd<uint8_t, 16> pk = convert<uint8_t>((qe & 15) | ((qo & 15) << 4));
                            pk.copy_to(A4 + (size_t) m * (size_t) (K / 2) + (size_t) (ib * 16 + p * 16));
                            }
                        }
                    }
                }
            }
        });
    return true;
}

#endif

bool ggml_sycl_xe2mx_lo_s44(ggml_backend_sycl_context & ctx,
                             const uint8_t * a4,
                             const float * sa,
                             const uint8_t * w4,
                             const float * scw,
                             float * C,
                             int64_t K,
                             int64_t N,
                             int64_t M,
                             int64_t g) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (a4 == nullptr || sa == nullptr || w4 == nullptr || scw == nullptr || C == nullptr) {
        return false;
    }
    try {
        return launch_lo_s44<0>(ctx.stream(), a4, sa, w4, scw, nullptr, nullptr, C, M, N, K, g);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx lo s44: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(a4);
    GGML_UNUSED(sa);
    GGML_UNUSED(w4);
    GGML_UNUSED(scw);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    GGML_UNUSED(g);
    return false;
#endif
}

bool ggml_sycl_xe2mx_lo_hf_s44(ggml_backend_sycl_context & ctx,
                                const sycl::half * xh,
                                const float * sa,
                                const uint8_t * w4,
                                const float * scw,
                                float * C,
                                int64_t K,
                                int64_t N,
                                int64_t M,
                                int64_t g) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (xh == nullptr || sa == nullptr || w4 == nullptr || scw == nullptr || C == nullptr) {
        return false;
    }
    try {
        return launch_lo_hf_s44<0>(ctx.stream(), reinterpret_cast<const uint8_t *>(xh), sa,
            w4, scw, nullptr, nullptr, C, M, N, K, g);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx lo hf s44: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(xh);
    GGML_UNUSED(sa);
    GGML_UNUSED(w4);
    GGML_UNUSED(scw);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    GGML_UNUSED(g);
    return false;
#endif
}

bool xe2mx_pack_hf_s4(queue_ptr stream, const sycl::half * xh, const float * sa, uint8_t * A4,
                      int64_t M, int64_t K, int64_t g) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (xh == nullptr || sa == nullptr || A4 == nullptr) {
        return false;
    }
    try {
        return launch_pack_hf_s4(stream, xh, sa, A4, M, K, g);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx pack hf s4: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(stream);
    GGML_UNUSED(xh);
    GGML_UNUSED(sa);
    GGML_UNUSED(A4);
    GGML_UNUSED(M);
    GGML_UNUSED(K);
    GGML_UNUSED(g);
    return false;
#endif
}

bool ggml_sycl_xe2mx_lo_s44_glu(ggml_backend_sycl_context & ctx,
                                 const uint8_t * a4,
                                 const float * sa,
                                 const uint8_t * wg,
                                 const float * scg,
                                 const uint8_t * wu,
                                 const float * scu,
                                 float * C,
                                 int64_t K,
                                 int64_t N,
                                 int64_t M,
                                 int64_t g) {
#ifdef GGML_SYCL_XE2MX_XMX_HAS_ESIMD
    if (a4 == nullptr || sa == nullptr || wg == nullptr || scg == nullptr || wu == nullptr || scu == nullptr || C == nullptr) {
        return false;
    }
    try {
        return launch_lo_s44<1>(ctx.stream(), a4, sa, wg, scg, wu, scu, C, M, N, K, g);
    } catch (const sycl::exception & e) {
        GGML_SYCL_XE2_LOG( "xe2mx lo s44 glu: %s\n", e.what());
        fflush(stderr);
        return false;
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(a4);
    GGML_UNUSED(sa);
    GGML_UNUSED(wg);
    GGML_UNUSED(scg);
    GGML_UNUSED(wu);
    GGML_UNUSED(scu);
    GGML_UNUSED(C);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(M);
    GGML_UNUSED(g);
    return false;
#endif
}
