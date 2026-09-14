#pragma once

#if defined(__INTEL_LLVM_COMPILER)
#include "exl3.hpp"
#include <cstdint>
#include <type_traits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// EXL3 GEMV on Xe2. Opt-in: GGML_SYCL_EXL3_XMX / _ESIMD / _I8.
// I8 uses unsigned hash byte-sum (CUDA dp4a.u32.s32). Signed dpas is wrong.
// Cooperative sequential-K hung on B65; do not use tid==0 + barrier.

namespace ggml_sycl_esimd {
namespace exl3_xmx {

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

inline uint32_t fshift(uint32_t b, uint32_t a, int shift) {
    const uint64_t merged = (uint64_t(a) << 32) | uint64_t(b);
    return uint32_t(merged >> shift);
}

template <int N>
ESIMD_INLINE simd<sycl::half, N> decode_mul1_v(simd<uint32_t, N> x) {
    x = x * simd<uint32_t, N>(0x83DCD12Du);
    simd<uint32_t, N> sum = dp4a<uint32_t, uint32_t, uint32_t, uint32_t, N>(
            simd<uint32_t, N>(0x6400u), x, simd<uint32_t, N>(0x01010101u));
    simd<uint16_t, N> hu = convert<uint16_t>(sum);
    simd<sycl::half, N> h = hu.template bit_cast_view<sycl::half>();
    simd<float, N> hf = convert<float>(h);
    const float k_inv  = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float k_bias = float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    return convert<sycl::half>(hf * k_inv + k_bias);
}

ESIMD_INLINE void scatter_vnni_j(simd<sycl::half, 256> & Bvnni, int j, simd<sycl::half, 32> h) {
#pragma unroll
    for (int lane = 0; lane < 32; ++lane) {
        const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
        const int col = (lane / 8) * 2 + ((lane & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
        Bvnni[(row / 2) * 32 + col * 2 + (row & 1)] = h[lane];
    }
}

template <int bits>
ESIMD_INLINE simd<sycl::half, 256> decode_pack_vnni(const uint32_t * pack) {
    simd<sycl::half, 256> Bvnni(sycl::half(0));
    if constexpr (bits == 4) {
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 32> a;
        a.template select<31, 1>(1) = p.template select<31, 1>(0);
        a[0] = p[31];
        const simd<uint32_t, 32> s = (a << 12) | (p >> 20);
        scatter_vnni_j(Bvnni, 7, decode_mul1_v(p & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 6, decode_mul1_v((p >> 4) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 5, decode_mul1_v((p >> 8) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 4, decode_mul1_v((p >> 12) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 3, decode_mul1_v((p >> 16) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 2, decode_mul1_v(s & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 1, decode_mul1_v((s >> 4) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 0, decode_mul1_v((s >> 8) & simd<uint32_t, 32>(0xffffu)));
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> p;
        p.copy_from(pack);
        simd<uint32_t, 32> val;
#pragma unroll
        for (int lane = 0; lane < 32; ++lane) {
            const int i1 = lane >> 1;
            const int i0 = (i1 + 15) & 15;
            const uint32_t b = p[i1];
            const uint32_t a = p[i0];
            val[lane] = (lane & 1) ? b : ((a << 16) | (b >> 16));
        }
        scatter_vnni_j(Bvnni, 7, decode_mul1_v(val & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 6, decode_mul1_v((val >> 2) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 5, decode_mul1_v((val >> 4) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 4, decode_mul1_v((val >> 6) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 3, decode_mul1_v((val >> 8) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 2, decode_mul1_v((val >> 10) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 1, decode_mul1_v((val >> 12) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 0, decode_mul1_v((val >> 14) & simd<uint32_t, 32>(0xffffu)));
    } else if constexpr (bits == 1) {
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 32> val;
#pragma unroll
        for (int lane = 0; lane < 32; ++lane) {
            const int i1 = lane >> 2;
            const int i0 = (i1 + 7) & 7;
            const uint32_t b = p[i1];
            const uint32_t a = p[i0];
            const int sh = (~(lane * 8)) & 24;
            val[lane] = uint32_t(((uint64_t(a) << 32) | uint64_t(b)) >> sh);
        }
        scatter_vnni_j(Bvnni, 7, decode_mul1_v(val & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 6, decode_mul1_v((val >> 1) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 5, decode_mul1_v((val >> 2) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 4, decode_mul1_v((val >> 3) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 3, decode_mul1_v((val >> 4) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 2, decode_mul1_v((val >> 5) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 1, decode_mul1_v((val >> 6) & simd<uint32_t, 32>(0xffffu)));
        scatter_vnni_j(Bvnni, 0, decode_mul1_v((val >> 7) & simd<uint32_t, 32>(0xffffu)));
    } else {
        constexpr int pack_u32 = bits * 8;
#pragma unroll
        for (int lane = 0; lane < 32; ++lane) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int t_offset = lane * 8 + j;
                const int b0 = t_offset * bits + bits - 16 + 256 * bits;
                const int b1 = b0 + 16;
                const int i0 = b0 / 32;
                const int i1 = (b1 - 1) / 32;
                const int s0 = (i1 + 1) * 32 - b1;
                const uint32_t a = pack[i0 % pack_u32];
                const uint32_t b = pack[i1 % pack_u32];
                const uint32_t w = fshift(b, a, s0) & 0xffffu;
                simd<uint32_t, 1> wv(w);
                const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
                const int col = (lane / 8) * 2 + ((lane & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
                Bvnni[(row / 2) * 32 + col * 2 + (row & 1)] = decode_mul1_v(wv)[0];
            }
        }
    }
    return Bvnni;
}

template <int bits>
ESIMD_INLINE void extract8_s(const uint32_t * ptr, int t_offset, uint32_t w[8]) {
    if constexpr (bits == 4) {
        const int i1 = t_offset >> 3;
        const int i0 = (i1 + 31) & 31;
        const uint32_t a = ptr[i0];
        const uint32_t b = ptr[i1];
        const uint32_t s = fshift(b, a, 20);
        w[7] = b & 0xffffu;
        w[6] = (b >> 4) & 0xffffu;
        w[5] = (b >> 8) & 0xffffu;
        w[4] = (b >> 12) & 0xffffu;
        w[3] = (b >> 16) & 0xffffu;
        w[2] = s & 0xffffu;
        w[1] = (s >> 4) & 0xffffu;
        w[0] = (s >> 8) & 0xffffu;
    } else if constexpr (bits == 2) {
        const int i1 = t_offset >> 4;
        const int i0 = (i1 + 15) & 15;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t_offset) & 8) << 1);
        w[7] = b & 0xffffu;
        w[6] = (b >> 2) & 0xffffu;
        w[5] = (b >> 4) & 0xffffu;
        w[4] = (b >> 6) & 0xffffu;
        w[3] = (b >> 8) & 0xffffu;
        w[2] = (b >> 10) & 0xffffu;
        w[1] = (b >> 12) & 0xffffu;
        w[0] = (b >> 14) & 0xffffu;
    } else if constexpr (bits == 1) {
        const int i1 = t_offset >> 5;
        const int i0 = (i1 + 7) & 7;
        const uint32_t a = ptr[i0];
        uint32_t b = ptr[i1];
        b = fshift(b, a, ((~t_offset) & 24));
        w[7] = b & 0xffffu;
        w[6] = (b >> 1) & 0xffffu;
        w[5] = (b >> 2) & 0xffffu;
        w[4] = (b >> 3) & 0xffffu;
        w[3] = (b >> 4) & 0xffffu;
        w[2] = (b >> 5) & 0xffffu;
        w[1] = (b >> 6) & 0xffffu;
        w[0] = (b >> 7) & 0xffffu;
    } else if constexpr (bits == 3) {
        const int b1 = (t_offset + 257) * bits;
        const int b0 = b1 - 16;
        const int b2 = b1 + bits * 7;
        const int mod = bits * 8;
        const uint32_t a = ptr[(b0 / 32) % mod];
        const uint32_t b = ptr[((b2 - 1) / 32) % mod];
        const int s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[7 - j] = fshift(b, a, s2 + bits * j) & 0xffffu;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int t = t_offset + j;
            const int b0 = t * bits + bits - 16 + 256 * bits;
            const int b1 = b0 + 16;
            const int i0 = b0 / 32;
            const int i1 = (b1 - 1) / 32;
            const int s0 = (i1 + 1) * 32 - b1;
            const int mod = bits * 8;
            w[j] = fshift(ptr[i1 % mod], ptr[i0 % mod], s0) & 0xffffu;
        }
    }
}

ESIMD_INLINE sycl::half decode_mul1_s(uint32_t x) {
    simd<uint32_t, 1> v(x);
    return decode_mul1_v(v)[0];
}

template <int bits, int MR>
ESIMD_INLINE void gemv_n_tile(const uint32_t * B, const float * x, float * C,
                              int M, int K, int N, int packed_n, int n_tile, int tid) {
    constexpr int WG = 16;
    constexpr int RN = 16;
    constexpr int RK = 16;
    constexpr int pack_u32 = bits * 8;
    slm_init<WG * MR * RN * 4>();

    const int ntiles  = packed_n / 16;
    const int kslices = K / 16;
    simd<float, MR * RN> acc(0.f);

    for (int kt = tid; kt < kslices; kt += WG) {
        const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
        simd<sycl::half, RK * RN> Bvnni = decode_pack_vnni<bits>(pack);
        simd<sycl::half, MR * RK> A(sycl::half(0));
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                simd<float, RK> xr;
                xr.copy_from(x + size_t(r) * size_t(K) + size_t(kt) * RK);
                A.template select<RK, 1>(r * RK) = convert<sycl::half>(xr);
            }
        }
        acc = xmx::dpas<8, MR, float, float, sycl::half, sycl::half>(acc, Bvnni, A);
    }

    slm_block_store<float, MR * RN>(uint32_t(tid * MR * RN * 4), acc);
    barrier();
    if (tid == 0) {
        simd<float, MR * RN> sum(0.f);
#pragma unroll
        for (int q = 0; q < WG; ++q) {
            sum += slm_block_load<float, MR * RN>(uint32_t(q * MR * RN * 4));
        }
        const int n0 = n_tile * RN;
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                simd<float, RN> row = sum.template select<RN, 1>(r * RN);
                row.copy_to(C + size_t(r) * size_t(N) + size_t(n0));
            }
        }
    }
}

} // namespace exl3_xmx

// Register GEMV. WG=16, two MMA lanes / thread, SIMD-8 decode. No full-tile walk.
// No VNNI scatter, no dpas (loses at M=1/2), no tid==0 sequential-K.
namespace exl3_reg {

using namespace sycl::ext::intel::esimd;

constexpr int WG = 16;

template <int j>
ESIMD_INLINE simd<float, 32> x_rows(simd<float, 16> x16) {
    constexpr int base = (j % 2) + ((j % 4 >= 2) ? 8 : 0);
    simd<float, 4> g;
    g[0] = x16[base];
    g[1] = x16[base + 2];
    g[2] = x16[base + 4];
    g[3] = x16[base + 6];
    simd<float, 32> o;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        o.template select<4, 1>(t * 4) = g;
    }
    return o;
}

template <int j>
ESIMD_INLINE void accum16(simd<float, 16> & acc, simd<float, 32> prod) {
    constexpr int col0 = (j >= 4) ? 8 : 0;
    simd<float, 8> s = prod.template select<8, 4>(0) + prod.template select<8, 4>(1)
            + prod.template select<8, 4>(2) + prod.template select<8, 4>(3);
    acc.template select<8, 1>(col0) += s;
}

ESIMD_INLINE simd<uint32_t, 32> lane32() {
    simd<uint32_t, 32> l(0);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        l[i] = uint32_t(i);
    }
    return l;
}

ESIMD_INLINE simd<uint32_t, 32> fshift_v(simd<uint32_t, 32> b, simd<uint32_t, 32> a, simd<uint32_t, 32> s) {
    simd<uint32_t, 32> r = (a << (simd<uint32_t, 32>(32) - s)) | (b >> s);
    r.merge(b, s == 0);
    r.merge(a, s == 32);
    r.merge(a >> (s - simd<uint32_t, 32>(32)), s > 32);
    return r;
}

ESIMD_INLINE simd<uint32_t, 32> gather_u32(const uint32_t * p, simd<uint32_t, 32> idx) {
    return gather<uint32_t, 32>(p, idx * simd<uint32_t, 32>(4u));
}

template <int bits>
ESIMD_INLINE void extract_all_j(const uint32_t * pack, simd<uint32_t, 32> w[8]) {
    if constexpr (bits == 4) {
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 32> a;
        a.template select<31, 1>(1) = p.template select<31, 1>(0);
        a[0] = p[31];
        const simd<uint32_t, 32> s = (a << 12) | (p >> 20);
        w[7] = p & simd<uint32_t, 32>(0xffffu);
        w[6] = (p >> 4) & simd<uint32_t, 32>(0xffffu);
        w[5] = (p >> 8) & simd<uint32_t, 32>(0xffffu);
        w[4] = (p >> 12) & simd<uint32_t, 32>(0xffffu);
        w[3] = (p >> 16) & simd<uint32_t, 32>(0xffffu);
        w[2] = s & simd<uint32_t, 32>(0xffffu);
        w[1] = (s >> 4) & simd<uint32_t, 32>(0xffffu);
        w[0] = (s >> 8) & simd<uint32_t, 32>(0xffffu);
    } else if constexpr (bits == 2) {
        const simd<uint32_t, 32> lane = lane32();
        const simd<uint32_t, 32> i1 = lane >> 1;
        const simd<uint32_t, 32> i0 = (i1 + 15) & 15;
        const simd<uint32_t, 32> a = gather_u32(pack, i0);
        const simd<uint32_t, 32> b = gather_u32(pack, i1);
        simd<uint32_t, 32> val = (a << 16) | (b >> 16);
        val.merge(b, (lane & 1) != 0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[j] = (val >> (14 - 2 * j)) & simd<uint32_t, 32>(0xffffu);
        }
    } else if constexpr (bits == 1) {
        const simd<uint32_t, 32> lane = lane32();
        const simd<uint32_t, 32> i1 = lane >> 2;
        const simd<uint32_t, 32> i0 = (i1 + 7) & 7;
        const simd<uint32_t, 32> a = gather_u32(pack, i0);
        const simd<uint32_t, 32> b = gather_u32(pack, i1);
        const simd<uint32_t, 32> sh = (~(lane * 8)) & 24;
        const simd<uint32_t, 32> val = fshift_v(b, a, sh);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[j] = (val >> (7 - j)) & simd<uint32_t, 32>(0xffffu);
        }
    } else if constexpr (bits == 3) {
        constexpr int pack_u32 = bits * 8;
        const simd<uint32_t, 32> t = lane32() * 8;
        const simd<uint32_t, 32> b1 = (t + 257) * uint32_t(bits);
        const simd<uint32_t, 32> b0 = b1 - 16;
        const simd<uint32_t, 32> b2 = b1 + uint32_t(bits * 7);
        const simd<uint32_t, 32> i0 = b0 / 32;
        const simd<uint32_t, 32> i2 = (b2 - 1) / 32;
        const simd<uint32_t, 32> s2 = (i2 + 1) * 32 - b2;
        const simd<uint32_t, 32> a = gather_u32(pack, i0 % uint32_t(pack_u32));
        const simd<uint32_t, 32> b = gather_u32(pack, i2 % uint32_t(pack_u32));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[7 - j] = fshift_v(b, a, s2 + uint32_t(bits * j)) & simd<uint32_t, 32>(0xffffu);
        }
    } else {
        constexpr int pack_u32 = bits * 8;
        const simd<uint32_t, 32> lane = lane32();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const simd<uint32_t, 32> t = lane * 8 + uint32_t(j);
            const simd<uint32_t, 32> b0 = t * uint32_t(bits) + uint32_t(bits - 16 + 256 * bits);
            const simd<uint32_t, 32> b1 = b0 + 16;
            const simd<uint32_t, 32> i0 = b0 / 32;
            const simd<uint32_t, 32> i1 = (b1 - 1) / 32;
            const simd<uint32_t, 32> s0 = (i1 + 1) * 32 - b1;
            const simd<uint32_t, 32> a = gather_u32(pack, i0 % uint32_t(pack_u32));
            const simd<uint32_t, 32> b = gather_u32(pack, i1 % uint32_t(pack_u32));
            w[j] = fshift_v(b, a, s0) & simd<uint32_t, 32>(0xffffu);
        }
    }
}

template <int bits, int MR>
ESIMD_INLINE void gemv_n_tile(const uint32_t * B, const float * x, float * C,
                              int M, int K, int N, int packed_n, int n_tile, int tid) {
    constexpr int RN = 16;
    constexpr int pack_u32 = bits * 8;
    slm_init<WG * 4 * MR * 4>();

    const int ntiles  = packed_n / 16;
    const int kslices = K / 16;
    float a0[MR], a1[MR], b0[MR], b1[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        a0[r] = 0.f;
        a1[r] = 0.f;
        b0[r] = 0.f;
        b1[r] = 0.f;
    }

    for (int kt = 0; kt < kslices; ++kt) {
        const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
        simd<float, 16> x16[MR];
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                x16[r].copy_from(x + size_t(r) * size_t(K) + size_t(kt) * 16);
            } else {
                x16[r] = 0.f;
            }
        }
        uint32_t wA[8], wB[8];
        exl3_xmx::extract8_s<bits>(pack, tid * 8, wA);
        exl3_xmx::extract8_s<bits>(pack, (tid + 16) * 8, wB);
        simd<uint32_t, 8> va(0), vb(0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            va[j] = wA[j];
            vb[j] = wB[j];
        }
        simd<float, 8> ha = convert<float>(exl3_xmx::decode_mul1_v(va));
        simd<float, 8> hb = convert<float>(exl3_xmx::decode_mul1_v(vb));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (tid % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4 >= 2) ? 8 : 0);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                const float xv = float(x16[r][row]);
                const float va = float(ha[j]);
                const float vb = float(hb[j]);
                if (j >= 4) {
                    a1[r] += va * xv;
                    b1[r] += vb * xv;
                } else {
                    a0[r] += va * xv;
                    b0[r] += vb * xv;
                }
            }
        }
    }

    simd<float, 4 * MR> mine(0.f);
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        mine[r]              = a0[r];
        mine[MR + r]         = a1[r];
        mine[2 * MR + r]     = b0[r];
        mine[3 * MR + r]     = b1[r];
    }
    slm_block_store<float, 4 * MR>(uint32_t(tid * 4 * MR * 4), mine);
    barrier();
    const int n0 = n_tile * RN;
    if ((tid & 3) == 0) {
        const int col_a = (tid / 8) * 2 + ((tid & 4) ? 1 : 0);
        const int col_b = (((tid + 16) / 8) * 2) + (((tid + 16) & 4) ? 1 : 0);
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r >= M) {
                continue;
            }
            float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                simd<float, 4 * MR> o = slm_block_load<float, 4 * MR>(uint32_t((tid + u) * 4 * MR * 4));
                s0 += o[r];
                s1 += o[MR + r];
                s2 += o[2 * MR + r];
                s3 += o[3 * MR + r];
            }
            float * rowp = C + size_t(r) * size_t(N) + size_t(n0);
            rowp[col_a]     = s0;
            rowp[col_a + 8] = s1;
            rowp[col_b]     = s2;
            rowp[col_b + 8] = s3;
        }
    }
    (void) N;
}

} // namespace exl3_reg

// Vector extract + F32 FMA. WG=4 splits K. No per-tile i8 quant.
namespace exl3_vec {

using namespace sycl::ext::intel::esimd;

constexpr int WG = 4;

template <int bits, int MR>
ESIMD_INLINE void gemv_n_tile(const uint32_t * B, const float * x, float * C,
                              int M, int K, int N, int packed_n,
                              int n0, int n_here, int kt0, int kt1, int tid) {
    constexpr int RN = 16;
    constexpr int NTMAX = 4;
    constexpr int pack_u32 = bits * 8;
    slm_init<WG * NTMAX * MR * RN * 4>();

    const int ntiles = packed_n / 16;
    simd<float, RN> acc[NTMAX][MR];
#pragma unroll
    for (int t = 0; t < NTMAX; ++t) {
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            acc[t][r] = 0.f;
        }
    }

    for (int kt = kt0 + tid; kt < kt1; kt += WG) {
        simd<float, 16> x16[MR];
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                x16[r].copy_from(x + size_t(r) * size_t(K) + size_t(kt) * 16);
            } else {
                x16[r] = 0.f;
            }
        }
#pragma unroll
        for (int t = 0; t < NTMAX; ++t) {
            if (t >= n_here) {
                continue;
            }
            const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
            simd<uint32_t, 32> wj[8];
            exl3_reg::extract_all_j<bits>(pack, wj);
            auto fma_j = [&](auto jtag) {
                constexpr int J = decltype(jtag)::value;
                simd<float, 32> h = convert<float>(exl3_xmx::decode_mul1_v(wj[J]));
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    simd<float, 32> prod = h * exl3_reg::x_rows<J>(x16[r]);
                    exl3_reg::accum16<J>(acc[t][r], prod);
                }
            };
            fma_j(std::integral_constant<int, 0>{});
            fma_j(std::integral_constant<int, 1>{});
            fma_j(std::integral_constant<int, 2>{});
            fma_j(std::integral_constant<int, 3>{});
            fma_j(std::integral_constant<int, 4>{});
            fma_j(std::integral_constant<int, 5>{});
            fma_j(std::integral_constant<int, 6>{});
            fma_j(std::integral_constant<int, 7>{});
        }
    }

    for (int t = 0; t < n_here; ++t) {
        simd<float, MR * RN> flat;
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            flat.template select<RN, 1>(r * RN) = acc[t][r];
        }
        slm_block_store<float, MR * RN>(uint32_t(tid * MR * RN * 4), flat);
        barrier();
        simd<float, MR * RN> sum(0.f);
#pragma unroll
        for (int q = 0; q < WG; ++q) {
            sum += slm_block_load<float, MR * RN>(uint32_t(q * MR * RN * 4));
        }
        const int nbase = (n0 + t) * RN;
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                simd<float, 4> part = sum.template select<4, 1>(r * RN + tid * 4);
                part.copy_to(C + size_t(r) * size_t(N) + size_t(nbase + tid * 4));
            }
        }
        barrier();
    }
}

} // namespace exl3_vec

template <int bits, int MR>
inline bool launch_exl3_gemv_xmx_mr(const float * x, const uint32_t * B, float * C,
                                    int M, int K, int N, int packed_n, queue_ptr stream) {
    constexpr int WG = 16;
    const int n_tiles = N / 16;
    if (n_tiles <= 0 || (N % 16) || (K % 16) || M <= 0 || M > EXL3_GEMV_MAX_M) {
        return false;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        const sycl::nd_range<1> nd(sycl::range<1>(size_t(n_tiles) * WG), sycl::range<1>(WG));
        stream->parallel_for(nd, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            exl3_xmx::gemv_n_tile<bits, MR>(B, x, C, M, K, N, packed_n,
                    int(it.get_group(0)), int(it.get_local_id(0)));
        });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL xmx sycl: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

template <int bits>
inline bool launch_exl3_gemv_xmx(const float * x, const uint32_t * B, float * C,
                                 int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 1) {
        return launch_exl3_gemv_xmx_mr<bits, 1>(x, B, C, M, K, N, packed_n, stream);
    }
    if (M <= 2) {
        return launch_exl3_gemv_xmx_mr<bits, 2>(x, B, C, M, K, N, packed_n, stream);
    }
    return launch_exl3_gemv_xmx_mr<bits, EXL3_GEMV_MAX_M>(x, B, C, M, K, N, packed_n, stream);
}

inline bool launch_exl3_gemv_xmx_bits(int bits, const float * x, const uint32_t * B, float * C,
                                      int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemv_xmx<1>(x, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemv_xmx<2>(x, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemv_xmx<3>(x, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemv_xmx<4>(x, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemv_xmx<5>(x, B, C, M, K, N, packed_n, stream);
        case 6: return launch_exl3_gemv_xmx<6>(x, B, C, M, K, N, packed_n, stream);
        case 7: return launch_exl3_gemv_xmx<7>(x, B, C, M, K, N, packed_n, stream);
        case 8: return launch_exl3_gemv_xmx<8>(x, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

template <int bits, int MR>
inline bool launch_exl3_gemv_esimd_mr(const float * x, const uint32_t * B, float * C,
                                      int M, int K, int N, int packed_n, queue_ptr stream) {
    constexpr int WG = exl3_reg::WG;
    const int n_tiles = N / 16;
    if (n_tiles <= 0 || (N % 16) || (K % 16) || M <= 0 || M > EXL3_GEMV_MAX_M) {
        return false;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        const sycl::nd_range<1> nd(sycl::range<1>(size_t(n_tiles) * WG), sycl::range<1>(WG));
        stream->parallel_for(nd, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            exl3_reg::gemv_n_tile<bits, MR>(B, x, C, M, K, N, packed_n,
                    int(it.get_group(0)), int(it.get_local_id(0)));
        });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL esimd sycl: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

template <int bits>
inline bool launch_exl3_gemv_esimd(const float * x, const uint32_t * B, float * C,
                                   int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 1) {
        return launch_exl3_gemv_esimd_mr<bits, 1>(x, B, C, M, K, N, packed_n, stream);
    }
    if (M <= 2) {
        return launch_exl3_gemv_esimd_mr<bits, 2>(x, B, C, M, K, N, packed_n, stream);
    }
    return launch_exl3_gemv_esimd_mr<bits, EXL3_GEMV_MAX_M>(x, B, C, M, K, N, packed_n, stream);
}

inline bool launch_exl3_gemv_esimd_bits(int bits, const float * x, const uint32_t * B, float * C,
                                        int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemv_esimd<1>(x, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemv_esimd<2>(x, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemv_esimd<3>(x, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemv_esimd<4>(x, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemv_esimd<5>(x, B, C, M, K, N, packed_n, stream);
        case 6: return launch_exl3_gemv_esimd<6>(x, B, C, M, K, N, packed_n, stream);
        case 7: return launch_exl3_gemv_esimd<7>(x, B, C, M, K, N, packed_n, stream);
        case 8: return launch_exl3_gemv_esimd<8>(x, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

// Hash bytes x int8 x, same affine as CUDA exl3_gemv_int8. Xe2 dp4a at M=1/2.
namespace exl3_i8 {

using namespace sycl::ext::intel::esimd;

constexpr int WG = 4;

ESIMD_INLINE float hbits(uint16_t u) {
    return float(sycl::bit_cast<sycl::half>(u));
}

template <int bits, int MR>
ESIMD_INLINE void gemv_n_tile(const uint32_t * B, const float * x, float * C,
                              int M, int K, int N, int packed_n, int n_tile, int tid) {
    constexpr int RN = 16;
    constexpr int pack_u32 = bits * 8;
    slm_init<WG * MR * RN * 4>();

    const int ntiles  = packed_n / 16;
    const int kslices = K / 16;
    const float kinv  = hbits(0x1eee);
    const float kbias = hbits(0xc931);
    const float aff   = 1024.f * kinv + kbias;

    simd<float, RN> acc[MR];
    float corr[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = 0.f;
        corr[r] = 0.f;
    }

    for (int kt = tid; kt < kslices; kt += WG) {
        const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
        simd<uint32_t, 32> wj[8];
        exl3_reg::extract_all_j<bits>(pack, wj);
        simd<float, 16> x16[MR];
        float mx = 0.f;
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            if (r < M) {
                x16[r].copy_from(x + size_t(r) * size_t(K) + size_t(kt) * 16);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float a = x16[r][i];
                    const float aa = a < 0.f ? -a : a;
                    mx = mx < aa ? aa : mx;
                }
            } else {
                x16[r] = 0.f;
            }
        }
        const float q_s = (mx < 1e-8f ? 1e-8f : mx) / 127.f;
        const float rq  = 1.f / q_s;
        simd<int, 16> xi[MR];
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            simd<int, 16> v = convert<int>(rnde<float>(x16[r] * rq));
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                int t = v[i];
                t = t > 127 ? 127 : (t < -127 ? -127 : t);
                v[i] = t;
            }
            xi[r] = v;
            int s = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += v[i];
            }
            corr[r] += q_s * float(s);
        }

        auto fma_j = [&](auto jtag) {
            constexpr int J = decltype(jtag)::value;
            simd<uint32_t, 32> h = wj[J] * simd<uint32_t, 32>(0x83DCD12Du);
            simd<uint32_t, 32> bs = (h & simd<uint32_t, 32>(0xffu))
                    + ((h >> 8) & simd<uint32_t, 32>(0xffu))
                    + ((h >> 16) & simd<uint32_t, 32>(0xffu))
                    + ((h >> 24) & simd<uint32_t, 32>(0xffu));
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                simd<int, 32> xv = convert<int>(exl3_reg::x_rows<J>(convert<float>(xi[r])));
                simd<int, 32> prod = convert<int>(bs) * xv;
                constexpr int col0 = (J >= 4) ? 8 : 0;
                simd<int, 8> s = prod.template select<8, 4>(0) + prod.template select<8, 4>(1)
                        + prod.template select<8, 4>(2) + prod.template select<8, 4>(3);
                acc[r].template select<8, 1>(col0) += convert<float>(s) * q_s;
            }
        };
        fma_j(std::integral_constant<int, 0>{});
        fma_j(std::integral_constant<int, 1>{});
        fma_j(std::integral_constant<int, 2>{});
        fma_j(std::integral_constant<int, 3>{});
        fma_j(std::integral_constant<int, 4>{});
        fma_j(std::integral_constant<int, 5>{});
        fma_j(std::integral_constant<int, 6>{});
        fma_j(std::integral_constant<int, 7>{});
    }

    simd<float, MR * RN> flat;
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        acc[r] = acc[r] * kinv + aff * corr[r];
        flat.template select<RN, 1>(r * RN) = acc[r];
    }
    slm_block_store<float, MR * RN>(uint32_t(tid * MR * RN * 4), flat);
    barrier();
    simd<float, MR * RN> sum(0.f);
#pragma unroll
    for (int q = 0; q < WG; ++q) {
        sum += slm_block_load<float, MR * RN>(uint32_t(q * MR * RN * 4));
    }
    const int n0 = n_tile * RN;
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        if (r < M) {
            simd<float, 4> part = sum.template select<4, 1>(r * RN + tid * 4);
            part.copy_to(C + size_t(r) * size_t(N) + size_t(n0 + tid * 4));
        }
    }
}

} // namespace exl3_i8

template <int bits, int MR>
inline bool launch_exl3_gemv_i8_mr(const float * x, const uint32_t * B, float * C,
                                   int M, int K, int N, int packed_n, queue_ptr stream) {
    constexpr int WG = exl3_i8::WG;
    const int n_tiles = N / 16;
    if (n_tiles <= 0 || (N % 16) || (K % 16) || M <= 0 || M > EXL3_GEMV_MAX_M) {
        return false;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        const sycl::nd_range<1> nd(sycl::range<1>(size_t(n_tiles) * WG), sycl::range<1>(WG));
        stream->parallel_for(nd, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            exl3_i8::gemv_n_tile<bits, MR>(B, x, C, M, K, N, packed_n,
                    int(it.get_group(0)), int(it.get_local_id(0)));
        });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL i8 sycl: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

template <int bits>
inline bool launch_exl3_gemv_i8(const float * x, const uint32_t * B, float * C,
                                int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 1) {
        return launch_exl3_gemv_i8_mr<bits, 1>(x, B, C, M, K, N, packed_n, stream);
    }
    if (M <= 2) {
        return launch_exl3_gemv_i8_mr<bits, 2>(x, B, C, M, K, N, packed_n, stream);
    }
    return launch_exl3_gemv_i8_mr<bits, EXL3_GEMV_MAX_M>(x, B, C, M, K, N, packed_n, stream);
}

inline bool launch_exl3_gemv_i8_bits(int bits, const float * x, const uint32_t * B, float * C,
                                     int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemv_i8<1>(x, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemv_i8<2>(x, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemv_i8<3>(x, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemv_i8<4>(x, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemv_i8<5>(x, B, C, M, K, N, packed_n, stream);
        case 6: return launch_exl3_gemv_i8<6>(x, B, C, M, K, N, packed_n, stream);
        case 7: return launch_exl3_gemv_i8<7>(x, B, C, M, K, N, packed_n, stream);
        case 8: return launch_exl3_gemv_i8<8>(x, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

template <int bits, int MR>
inline bool launch_exl3_gemv_vec_mr(const float * x, const uint32_t * B, float * C,
                                    int M, int K, int N, int packed_n, queue_ptr stream) {
    constexpr int WG = exl3_vec::WG;
    const int n_tiles = N / 16;
    if (n_tiles <= 0 || (N % 16) || (K % 16) || M <= 0 || M > EXL3_GEMV_MAX_M) {
        return false;
    }
    int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_GLB_NT", 4);
    if (nt < 1) {
        nt = 1;
    }
    if (nt > 4) {
        nt = 4;
    }
    int n_kparts = ggml_sycl_get_env("GGML_SYCL_EXL3_KSPLIT", 8);
    if (n_kparts < 1) {
        n_kparts = 1;
    }
    if (n_kparts > 32) {
        n_kparts = 32;
    }
    const int ks = K / 16;
    if (n_kparts > ks) {
        n_kparts = ks > 0 ? ks : 1;
    }
    const int n_groups_n = (n_tiles + nt - 1) / nt;
    float * out = C;
    if (n_kparts > 1) {
        static thread_local float * parts = nullptr;
        static thread_local size_t cap = 0;
        static thread_local sycl::queue * last = nullptr;
        const size_t need = size_t(n_kparts) * size_t(M) * size_t(N);
        if (!parts || need > cap || last != stream) {
            if (parts && last) {
                sycl::free(parts, *last);
            }
            parts = sycl::malloc_device<float>(need, *stream);
            cap = need;
            last = stream;
        }
        out = parts;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        const sycl::nd_range<1> nd(sycl::range<1>(size_t(n_groups_n) * size_t(n_kparts) * WG), sycl::range<1>(WG));
        const int nt_use = nt;
        const int np = n_kparts;
        stream->parallel_for(nd, props, [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
            const int gid = int(it.get_group(0));
            const int n0 = (gid % n_groups_n) * nt_use;
            const int k_part = gid / n_groups_n;
            int n_here = n_tiles - n0;
            if (n_here > nt_use) {
                n_here = nt_use;
            }
            if (n_here < 0) {
                n_here = 0;
            }
            const int chunk = (ks + np - 1) / np;
            const int kt0 = k_part * chunk;
            int kt1 = kt0 + chunk;
            if (kt1 > ks) {
                kt1 = ks;
            }
            float * base = out + (np > 1 ? size_t(k_part) * size_t(M) * size_t(N) : 0);
            exl3_vec::gemv_n_tile<bits, MR>(B, x, base, M, K, N, packed_n,
                    n0, n_here, kt0, kt1, int(it.get_local_id(0)));
        });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL vec sycl: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    if (n_kparts > 1) {
        const int np = n_kparts;
        const float * parts = out;
        const size_t mn = size_t(M) * size_t(N);
        stream->parallel_for(sycl::range<1>(mn), [=](sycl::id<1> i) {
            float s = 0.f;
            for (int p = 0; p < np; ++p) {
                s += parts[size_t(p) * mn + i];
            }
            C[i] = s;
        });
    }
    return true;
}

template <int bits>
inline bool launch_exl3_gemv_vec(const float * x, const uint32_t * B, float * C,
                                 int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 1) {
        return launch_exl3_gemv_vec_mr<bits, 1>(x, B, C, M, K, N, packed_n, stream);
    }
    if (M <= 2) {
        return launch_exl3_gemv_vec_mr<bits, 2>(x, B, C, M, K, N, packed_n, stream);
    }
    return launch_exl3_gemv_vec_mr<bits, EXL3_GEMV_MAX_M>(x, B, C, M, K, N, packed_n, stream);
}

inline bool launch_exl3_gemv_vec_bits(int bits, const float * x, const uint32_t * B, float * C,
                                      int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemv_vec<1>(x, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemv_vec<2>(x, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemv_vec<3>(x, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemv_vec<4>(x, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemv_vec<5>(x, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

// Reconstruct What [N,K] ld=K. One ESIMD thread per 16x16. SIMD extract, 16 contig K stores.
namespace exl3_recon {

using namespace sycl::ext::intel::esimd;

template <int bits>
ESIMD_INLINE void recon_tile(const uint32_t * B, sycl::half * W,
                             int dest_ld, int packed_n, int n_tile, int k_tile, int k_store,
                             int wkn, int dest_n) {
    constexpr int pack_u32 = bits * 8;
    const int ntiles = packed_n / 16;
    const uint32_t * pack = B + (size_t(k_tile) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
    simd<uint32_t, 32> wj[8];
    exl3_reg::extract_all_j<bits>(pack, wj);
    simd<sycl::half, 256> kn(sycl::half(0));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        simd<sycl::half, 32> h = exl3_xmx::decode_mul1_v(wj[j]);
#pragma unroll
        for (int lane = 0; lane < 32; ++lane) {
            const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4 >= 2) ? 8 : 0);
            const int col = (lane / 8) * 2 + ((lane & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
            kn[row * 16 + col] = h[lane];
        }
    }
    const int n0 = n_tile * 16;
    if (wkn) {
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            simd<sycl::half, 16> row;
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                row[n] = kn[k * 16 + n];
            }
            row.copy_to(W + size_t(k_store + k) * size_t(dest_n) + size_t(n0));
        }
    } else {
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            simd<sycl::half, 16> col;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                col[k] = kn[k * 16 + n];
            }
            col.copy_to(W + size_t(n0 + n) * size_t(dest_ld) + size_t(k_store));
        }
    }
}

} // namespace exl3_recon

template <int bits>
inline bool launch_exl3_recon_esimd(const uint32_t * B, sycl::half * W,
                                    int K, int N, int packed_n, queue_ptr stream,
                                    int k0 = 0, int k_len = 0, int dest_ld = 0) {
    if ((K % 16) || (N % 16) || packed_n < N) {
        return false;
    }
    if (k_len <= 0) {
        k_len = K;
    }
    if (dest_ld <= 0) {
        dest_ld = K;
    }
    if (k0 < 0 || (k0 % 16) || (k_len % 16) || k0 + k_len > K || dest_ld < k_len) {
        return false;
    }
    const int n_tiles = N / 16;
    const int k_tiles = k_len / 16;
    const int k0t = k0 / 16;
    constexpr int KT = 8;
    const int k_groups = (k_tiles + KT - 1) / KT;
    const int tiles = n_tiles * k_groups;
    constexpr int WG = 16;
    const int gsz = ((tiles + WG - 1) / WG) * WG;
    const int wkn = ggml_sycl_get_env("GGML_SYCL_EXL3_WKN", 1);
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(gsz)), sycl::range<1>(WG)),
                props,
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    int tile = int(it.get_global_id(0));
                    tile = tile < tiles ? tile : (tiles - 1);
                    const int n_tile = tile % n_tiles;
                    const int kg = tile / n_tiles;
                    for (int tk = 0; tk < KT; ++tk) {
                        const int k_local = kg * KT + tk;
                        if (k_local < k_tiles) {
                            exl3_recon::recon_tile<bits>(B, W, dest_ld, packed_n, n_tile,
                                    k0t + k_local, k_local * 16, wkn, N);
                        }
                    }
                });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL recon esimd: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

// Prefill: one n-tile / WG. Cooperative decode to SLM, each tid keeps 4 m-tiles in GRF.
// Decode once per K-tile. M multiple of 8, M<=512. C[m*N+n].
namespace exl3_gemm {

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

constexpr int WG = 16;
constexpr int RM = 8;
constexpr int RN = 16;
constexpr int NM = 4;

ESIMD_INLINE void slm_vnni_put(int k, int n, sycl::half v) {
    slm_scalar_store<sycl::half>(uint32_t(((k / 2) * 32 + n * 2 + (k & 1)) * 2), v);
}

template <int bits>
ESIMD_INLINE void gemm_n_tile(const uint32_t * B, const sycl::half * X, float * C,
                              int M, int K, int N, int packed_n, int n_tile, int tid) {
    slm_init<512>();
    const int n_mt = M / RM;
    const int ntiles = packed_n / 16;
    const int n0 = n_tile * RN;
    simd<float, RM * RN> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);

    auto load_A = [&](int tm, int kt) {
        simd<sycl::half, RM * 16> A(sycl::half(0));
        if (tm < n_mt) {
            const int m0 = tm * RM;
#pragma unroll
            for (int r = 0; r < RM; ++r) {
                simd<sycl::half, 16> row;
                row.copy_from(X + size_t(m0 + r) * size_t(K) + size_t(kt) * 16);
                A.template select<16, 1>(r * 16) = row;
            }
        }
        return A;
    };

    for (int kt = 0; kt < K / 16; ++kt) {
        const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(bits * 8);
        uint32_t wA[8], wB[8];
        exl3_xmx::extract8_s<bits>(pack, tid * 8, wA);
        exl3_xmx::extract8_s<bits>(pack, (tid + 16) * 8, wB);
        simd<uint32_t, 8> va(0), vb(0);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            va[j] = wA[j];
            vb[j] = wB[j];
        }
        simd<sycl::half, 8> ha = exl3_xmx::decode_mul1_v(va);
        simd<sycl::half, 8> hb = exl3_xmx::decode_mul1_v(vb);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ka = (tid % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4 >= 2) ? 8 : 0);
            const int na = (tid / 8) * 2 + ((tid & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
            const int kb = ((tid + 16) % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4 >= 2) ? 8 : 0);
            const int nb = (((tid + 16) / 8) * 2) + (((tid + 16) & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
            slm_vnni_put(ka, na, ha[j]);
            slm_vnni_put(kb, nb, hb[j]);
        }
        barrier();
        simd<sycl::half, 256> Bvnni = slm_block_load<sycl::half, 256>(0);
        acc0 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc0, Bvnni, load_A(tid + 0 * WG, kt));
        acc1 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc1, Bvnni, load_A(tid + 1 * WG, kt));
        acc2 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc2, Bvnni, load_A(tid + 2 * WG, kt));
        acc3 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc3, Bvnni, load_A(tid + 3 * WG, kt));
        barrier();
    }

    auto store_acc = [&](int tm, simd<float, RM * RN> acc) {
        if (tm >= n_mt) {
            return;
        }
        const int m0 = tm * RM;
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            simd<float, RN> row = acc.template select<RN, 1>(r * RN);
            row.copy_to(C + size_t(m0 + r) * size_t(N) + size_t(n0));
        }
    };
    store_acc(tid + 0 * WG, acc0);
    store_acc(tid + 1 * WG, acc1);
    store_acc(tid + 2 * WG, acc2);
    store_acc(tid + 3 * WG, acc3);
}

} // namespace exl3_gemm

template <int bits>
inline bool launch_exl3_gemm_packed_xmx(const sycl::half * X, const uint32_t * B, float * C,
                                        int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 0 || (M % 8) || M > 512 || (N % 16) || (K % 16)) {
        return false;
    }
    const int n_tiles = N / 16;
    if (n_tiles <= 0) {
        return false;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(n_tiles) * exl3_gemm::WG), sycl::range<1>(exl3_gemm::WG)),
                props,
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    exl3_gemm::gemm_n_tile<bits>(B, X, C, M, K, N, packed_n,
                            int(it.get_group(0)), int(it.get_local_id(0)));
                });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL packed xmx: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

// xe2-16 GEMV. WG=16, one N-col / thread. No full-pack copy. Same NT/KSPLIT as SIMT.
namespace exl3_xe2 {

using namespace sycl::ext::intel::esimd;

constexpr int WG = 16;

template <int bits>
ESIMD_INLINE simd<uint32_t, 16> extract16_kcol(const uint32_t * pack, int n) {
    uint32_t w0[8], w1[8];
    if constexpr (bits == 4) {
        const int mid = (n * 2) & 31;
        const uint32_t a = pack[(mid + 31) & 31];
        const uint32_t b = pack[mid];
        const uint32_t c = pack[(mid + 1) & 31];
        const uint32_t s0 = exl3_xmx::fshift(b, a, 20);
        const uint32_t s1 = exl3_xmx::fshift(c, b, 20);
        w0[7] = b & 0xffffu; w0[6] = (b >> 4) & 0xffffu; w0[5] = (b >> 8) & 0xffffu;
        w0[4] = (b >> 12) & 0xffffu; w0[3] = (b >> 16) & 0xffffu;
        w0[2] = s0 & 0xffffu; w0[1] = (s0 >> 4) & 0xffffu; w0[0] = (s0 >> 8) & 0xffffu;
        w1[7] = c & 0xffffu; w1[6] = (c >> 4) & 0xffffu; w1[5] = (c >> 8) & 0xffffu;
        w1[4] = (c >> 12) & 0xffffu; w1[3] = (c >> 16) & 0xffffu;
        w1[2] = s1 & 0xffffu; w1[1] = (s1 >> 4) & 0xffffu; w1[0] = (s1 >> 8) & 0xffffu;
    } else {
        exl3_xmx::extract8_s<bits>(pack, n * 16, w0);
        exl3_xmx::extract8_s<bits>(pack, n * 16 + 8, w1);
    }
    simd<uint32_t, 16> w(0);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        w[k]     = w0[k];
        w[k + 8] = w1[k];
    }
    return w;
}

template <int N>
ESIMD_INLINE simd<float, N> hash_bs(simd<uint32_t, N> w) {
    w = w * simd<uint32_t, N>(0x83DCD12Du);
    simd<uint32_t, N> bs = dp4a<uint32_t, uint32_t, uint32_t, uint32_t, N>(
            simd<uint32_t, N>(0), w, simd<uint32_t, N>(0x01010101u));
    return convert<float>(bs);
}

ESIMD_INLINE simd<float, 16> hash_bs16(simd<uint32_t, 16> w) {
    return hash_bs<16>(w);
}

template <int SH, int N = 16>
ESIMD_INLINE simd<uint32_t, N> fsh_v(simd<uint32_t, N> b, simd<uint32_t, N> a) {
    if constexpr (SH == 0) {
        return b;
    } else if constexpr (SH >= 32) {
        return a >> (SH - 32);
    } else {
        return (b >> SH) | (a << (32 - SH));
    }
}

ESIMD_INLINE simd<uint32_t, 16> blend_odd(simd<uint32_t, 16> even_v, simd<uint32_t, 16> odd_v) {
    even_v.select<8, 2>(1) = odd_v.select<8, 2>(1);
    return even_v;
}

ESIMD_INLINE void unpack8_b4(simd<uint32_t, 16> a, simd<uint32_t, 16> b, simd<uint32_t, 16> w[8]) {
    const simd<uint32_t, 16> s = (a << 12) | (b >> 20);
    w[7] = b & simd<uint32_t, 16>(0xffffu);
    w[6] = (b >> 4) & simd<uint32_t, 16>(0xffffu);
    w[5] = (b >> 8) & simd<uint32_t, 16>(0xffffu);
    w[4] = (b >> 12) & simd<uint32_t, 16>(0xffffu);
    w[3] = (b >> 16) & simd<uint32_t, 16>(0xffffu);
    w[2] = s & simd<uint32_t, 16>(0xffffu);
    w[1] = (s >> 4) & simd<uint32_t, 16>(0xffffu);
    w[0] = (s >> 8) & simd<uint32_t, 16>(0xffffu);
}

ESIMD_INLINE void unpack8_b2(simd<uint32_t, 16> val, simd<uint32_t, 16> w[8]) {
    w[7] = val & simd<uint32_t, 16>(0xffffu);
    w[6] = (val >> 2) & simd<uint32_t, 16>(0xffffu);
    w[5] = (val >> 4) & simd<uint32_t, 16>(0xffffu);
    w[4] = (val >> 6) & simd<uint32_t, 16>(0xffffu);
    w[3] = (val >> 8) & simd<uint32_t, 16>(0xffffu);
    w[2] = (val >> 10) & simd<uint32_t, 16>(0xffffu);
    w[1] = (val >> 12) & simd<uint32_t, 16>(0xffffu);
    w[0] = (val >> 14) & simd<uint32_t, 16>(0xffffu);
}

ESIMD_INLINE void unpack8_b1(simd<uint32_t, 16> val, simd<uint32_t, 16> w[8]) {
    w[7] = val & simd<uint32_t, 16>(0xffffu);
    w[6] = (val >> 1) & simd<uint32_t, 16>(0xffffu);
    w[5] = (val >> 2) & simd<uint32_t, 16>(0xffffu);
    w[4] = (val >> 3) & simd<uint32_t, 16>(0xffffu);
    w[3] = (val >> 4) & simd<uint32_t, 16>(0xffffu);
    w[2] = (val >> 5) & simd<uint32_t, 16>(0xffffu);
    w[1] = (val >> 6) & simd<uint32_t, 16>(0xffffu);
    w[0] = (val >> 7) & simd<uint32_t, 16>(0xffffu);
}

template <int N>
ESIMD_INLINE void unpack8_b3n(simd<uint32_t, N> t0, simd<uint32_t, N> t1, simd<uint32_t, N> w[8]) {
    const simd<uint32_t, N> m(0xffffu);
    w[7] = t0 & m;
    w[6] = (t0 >> 3) & m;
    w[5] = (t0 >> 6) & m;
    w[4] = (t0 >> 9) & m;
    w[3] = t1 & m;
    w[2] = (t1 >> 3) & m;
    w[1] = (t1 >> 6) & m;
    w[0] = (t1 >> 9) & m;
}

ESIMD_INLINE void unpack8_b3(simd<uint32_t, 16> t0, simd<uint32_t, 16> t1, simd<uint32_t, 16> w[8]) {
    unpack8_b3n<16>(t0, t1, w);
}

ESIMD_INLINE void mac8(simd<float, 16> & acc, simd<uint32_t, 16> w[8], const simd<float, 16> & xv, int j0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        acc += hash_bs16(w[j]) * simd<float, 16>(float(xv[j0 + j]));
    }
}

ESIMD_INLINE void mac8_2(simd<float, 16> & a0, simd<float, 16> & a1, simd<uint32_t, 16> w[8],
                         const simd<float, 16> & x0, const simd<float, 16> & x1, int j0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const simd<float, 16> h = hash_bs16(w[j]);
        a0 += h * simd<float, 16>(float(x0[j0 + j]));
        a1 += h * simd<float, 16>(float(x1[j0 + j]));
    }
}

ESIMD_INLINE void mac8_4(simd<float, 16> & a0, simd<float, 16> & a1,
                         simd<float, 16> & a2, simd<float, 16> & a3,
                         simd<uint32_t, 16> w[8],
                         const simd<float, 16> & x0, const simd<float, 16> & x1,
                         const simd<float, 16> & x2, const simd<float, 16> & x3, int j0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const simd<float, 16> h = hash_bs16(w[j]);
        a0 += h * simd<float, 16>(float(x0[j0 + j]));
        a1 += h * simd<float, 16>(float(x1[j0 + j]));
        a2 += h * simd<float, 16>(float(x2[j0 + j]));
        a3 += h * simd<float, 16>(float(x3[j0 + j]));
    }
}

ESIMD_INLINE void mac8_8(simd<float, 16> & a0, simd<float, 16> & a1,
                         simd<float, 16> & a2, simd<float, 16> & a3,
                         simd<float, 16> & a4, simd<float, 16> & a5,
                         simd<float, 16> & a6, simd<float, 16> & a7,
                         simd<uint32_t, 16> w[8],
                         const simd<float, 16> & x0, const simd<float, 16> & x1,
                         const simd<float, 16> & x2, const simd<float, 16> & x3,
                         const simd<float, 16> & x4, const simd<float, 16> & x5,
                         const simd<float, 16> & x6, const simd<float, 16> & x7, int j0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const simd<float, 16> h = hash_bs16(w[j]);
        a0 += h * simd<float, 16>(float(x0[j0 + j]));
        a1 += h * simd<float, 16>(float(x1[j0 + j]));
        a2 += h * simd<float, 16>(float(x2[j0 + j]));
        a3 += h * simd<float, 16>(float(x3[j0 + j]));
        a4 += h * simd<float, 16>(float(x4[j0 + j]));
        a5 += h * simd<float, 16>(float(x5[j0 + j]));
        a6 += h * simd<float, 16>(float(x6[j0 + j]));
        a7 += h * simd<float, 16>(float(x7[j0 + j]));
    }
}

template <int bits>
ESIMD_INLINE void mac_wins(const uint32_t * pack, const simd<float, 16> & xv, simd<float, 16> & acc) {
    if constexpr (bits == 4) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 16> b = p.select<16, 2>(0);
        simd<uint32_t, 16> c = p.select<16, 2>(1);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = p.select<1, 1>(31);
        a.select<15, 1>(1) = c.select<15, 1>(0);
        unpack8_b4(a, b, w);
        mac8(acc, w, xv, 0);
        unpack8_b4(b, c, w);
        mac8(acc, w, xv, 8);
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 16> b;
        b.copy_from(pack);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = b.select<1, 1>(15);
        a.select<15, 1>(1) = b.select<15, 1>(0);
        unpack8_b2((a << 16) | (b >> 16), w);
        mac8(acc, w, xv, 0);
        unpack8_b2(b, w);
        mac8(acc, w, xv, 8);
    } else if constexpr (bits == 3) {
        simd<uint32_t, 32> p(0);
        simd<uint32_t, 16> lo;
        lo.copy_from(pack);
        p.select<16, 1>(0) = lo;
        simd<uint32_t, 8> hi8;
        hi8.copy_from(pack + 16);
        p.select<8, 1>(16) = hi8;
        simd<uint32_t, 16> a, b;
        simd<uint32_t, 8> ae;
        ae.select<1, 1>(0) = p.select<1, 1>(23);
        ae.select<7, 1>(1) = p.select<7, 3>(2);
        a.select<8, 2>(0) = ae;
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(0);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        simd<uint32_t, 16> w[8];
        unpack8_b3(blend_odd(fsh_v<8>(b, a), fsh_v<24>(b, a)),
                   blend_odd(fsh_v<20>(b, a), fsh_v<36>(b, a)), w);
        mac8(acc, w, xv, 0);
        a.select<8, 2>(0) = p.select<8, 3>(0);
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(1);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        unpack8_b3(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)),
                   blend_odd(fsh_v<28>(b, a), fsh_v<12>(b, a)), w);
        mac8(acc, w, xv, 8);
    } else if constexpr (bits == 1) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 8> ar;
        ar.select<1, 1>(0) = p.select<1, 1>(7);
        ar.select<7, 1>(1) = p.select<7, 1>(0);
        simd<uint32_t, 16> a, b;
        a.select<8, 2>(0) = ar;
        a.select<8, 2>(1) = ar;
        b.select<8, 2>(0) = p;
        b.select<8, 2>(1) = p;
        unpack8_b1(blend_odd(fsh_v<24>(b, a), fsh_v<8>(b, a)), w);
        mac8(acc, w, xv, 0);
        unpack8_b1(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)), w);
        mac8(acc, w, xv, 8);
    } else {
        simd<uint32_t, 16> wj[16];
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            simd<uint32_t, 16> col = extract16_kcol<bits>(pack, n);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                wj[k][n] = uint32_t(col[k]);
            }
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            acc += hash_bs16(wj[j]) * simd<float, 16>(float(xv[j]));
        }
    }
}

template <int bits>
ESIMD_INLINE void mac_wins2(const uint32_t * pack,
                            const simd<float, 16> & x0, const simd<float, 16> & x1,
                            simd<float, 16> & a0, simd<float, 16> & a1) {
    if constexpr (bits == 4) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 16> b = p.select<16, 2>(0);
        simd<uint32_t, 16> c = p.select<16, 2>(1);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = p.select<1, 1>(31);
        a.select<15, 1>(1) = c.select<15, 1>(0);
        unpack8_b4(a, b, w);
        mac8_2(a0, a1, w, x0, x1, 0);
        unpack8_b4(b, c, w);
        mac8_2(a0, a1, w, x0, x1, 8);
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 16> b;
        b.copy_from(pack);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = b.select<1, 1>(15);
        a.select<15, 1>(1) = b.select<15, 1>(0);
        unpack8_b2((a << 16) | (b >> 16), w);
        mac8_2(a0, a1, w, x0, x1, 0);
        unpack8_b2(b, w);
        mac8_2(a0, a1, w, x0, x1, 8);
    } else if constexpr (bits == 3) {
        simd<uint32_t, 32> p(0);
        simd<uint32_t, 16> lo;
        lo.copy_from(pack);
        p.select<16, 1>(0) = lo;
        simd<uint32_t, 8> hi8;
        hi8.copy_from(pack + 16);
        p.select<8, 1>(16) = hi8;
        simd<uint32_t, 16> a, b;
        simd<uint32_t, 8> ae;
        ae.select<1, 1>(0) = p.select<1, 1>(23);
        ae.select<7, 1>(1) = p.select<7, 3>(2);
        a.select<8, 2>(0) = ae;
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(0);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        simd<uint32_t, 16> w[8];
        unpack8_b3(blend_odd(fsh_v<8>(b, a), fsh_v<24>(b, a)),
                   blend_odd(fsh_v<20>(b, a), fsh_v<36>(b, a)), w);
        mac8_2(a0, a1, w, x0, x1, 0);
        a.select<8, 2>(0) = p.select<8, 3>(0);
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(1);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        unpack8_b3(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)),
                   blend_odd(fsh_v<28>(b, a), fsh_v<12>(b, a)), w);
        mac8_2(a0, a1, w, x0, x1, 8);
    } else if constexpr (bits == 1) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 8> ar;
        ar.select<1, 1>(0) = p.select<1, 1>(7);
        ar.select<7, 1>(1) = p.select<7, 1>(0);
        simd<uint32_t, 16> a, b;
        a.select<8, 2>(0) = ar;
        a.select<8, 2>(1) = ar;
        b.select<8, 2>(0) = p;
        b.select<8, 2>(1) = p;
        unpack8_b1(blend_odd(fsh_v<24>(b, a), fsh_v<8>(b, a)), w);
        mac8_2(a0, a1, w, x0, x1, 0);
        unpack8_b1(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)), w);
        mac8_2(a0, a1, w, x0, x1, 8);
    } else {
        mac_wins<bits>(pack, x0, a0);
        mac_wins<bits>(pack, x1, a1);
    }
}

template <int bits>
ESIMD_INLINE void mac_wins4(const uint32_t * pack,
                            const simd<float, 16> & x0, const simd<float, 16> & x1,
                            const simd<float, 16> & x2, const simd<float, 16> & x3,
                            simd<float, 16> & a0, simd<float, 16> & a1,
                            simd<float, 16> & a2, simd<float, 16> & a3) {
    if constexpr (bits == 4) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 16> b = p.select<16, 2>(0);
        simd<uint32_t, 16> c = p.select<16, 2>(1);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = p.select<1, 1>(31);
        a.select<15, 1>(1) = c.select<15, 1>(0);
        unpack8_b4(a, b, w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 0);
        unpack8_b4(b, c, w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 8);
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 16> b;
        b.copy_from(pack);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = b.select<1, 1>(15);
        a.select<15, 1>(1) = b.select<15, 1>(0);
        unpack8_b2((a << 16) | (b >> 16), w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 0);
        unpack8_b2(b, w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 8);
    } else if constexpr (bits == 3) {
        simd<uint32_t, 32> p(0);
        simd<uint32_t, 16> lo;
        lo.copy_from(pack);
        p.select<16, 1>(0) = lo;
        simd<uint32_t, 8> hi8;
        hi8.copy_from(pack + 16);
        p.select<8, 1>(16) = hi8;
        simd<uint32_t, 16> a, b;
        simd<uint32_t, 8> ae;
        ae.select<1, 1>(0) = p.select<1, 1>(23);
        ae.select<7, 1>(1) = p.select<7, 3>(2);
        a.select<8, 2>(0) = ae;
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(0);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        simd<uint32_t, 16> w[8];
        unpack8_b3(blend_odd(fsh_v<8>(b, a), fsh_v<24>(b, a)),
                   blend_odd(fsh_v<20>(b, a), fsh_v<36>(b, a)), w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 0);
        a.select<8, 2>(0) = p.select<8, 3>(0);
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(1);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        unpack8_b3(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)),
                   blend_odd(fsh_v<28>(b, a), fsh_v<12>(b, a)), w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 8);
    } else if constexpr (bits == 1) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 8> ar;
        ar.select<1, 1>(0) = p.select<1, 1>(7);
        ar.select<7, 1>(1) = p.select<7, 1>(0);
        simd<uint32_t, 16> a, b;
        a.select<8, 2>(0) = ar;
        a.select<8, 2>(1) = ar;
        b.select<8, 2>(0) = p;
        b.select<8, 2>(1) = p;
        unpack8_b1(blend_odd(fsh_v<24>(b, a), fsh_v<8>(b, a)), w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 0);
        unpack8_b1(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)), w);
        mac8_4(a0, a1, a2, a3, w, x0, x1, x2, x3, 8);
    } else {
        mac_wins<bits>(pack, x0, a0);
        mac_wins<bits>(pack, x1, a1);
        mac_wins<bits>(pack, x2, a2);
        mac_wins<bits>(pack, x3, a3);
    }
}

template <int bits>
ESIMD_INLINE void mac_wins8(const uint32_t * pack,
                            const simd<float, 16> & x0, const simd<float, 16> & x1,
                            const simd<float, 16> & x2, const simd<float, 16> & x3,
                            const simd<float, 16> & x4, const simd<float, 16> & x5,
                            const simd<float, 16> & x6, const simd<float, 16> & x7,
                            simd<float, 16> & a0, simd<float, 16> & a1,
                            simd<float, 16> & a2, simd<float, 16> & a3,
                            simd<float, 16> & a4, simd<float, 16> & a5,
                            simd<float, 16> & a6, simd<float, 16> & a7) {
    if constexpr (bits == 4) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 16> b = p.select<16, 2>(0);
        simd<uint32_t, 16> c = p.select<16, 2>(1);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = p.select<1, 1>(31);
        a.select<15, 1>(1) = c.select<15, 1>(0);
        unpack8_b4(a, b, w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 0);
        unpack8_b4(b, c, w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 8);
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 16> b;
        b.copy_from(pack);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = b.select<1, 1>(15);
        a.select<15, 1>(1) = b.select<15, 1>(0);
        unpack8_b2((a << 16) | (b >> 16), w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 0);
        unpack8_b2(b, w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 8);
    } else if constexpr (bits == 3) {
        simd<uint32_t, 32> p(0);
        simd<uint32_t, 16> lo;
        lo.copy_from(pack);
        p.select<16, 1>(0) = lo;
        simd<uint32_t, 8> hi8;
        hi8.copy_from(pack + 16);
        p.select<8, 1>(16) = hi8;
        simd<uint32_t, 16> a, b;
        simd<uint32_t, 8> ae;
        ae.select<1, 1>(0) = p.select<1, 1>(23);
        ae.select<7, 1>(1) = p.select<7, 3>(2);
        a.select<8, 2>(0) = ae;
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(0);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        simd<uint32_t, 16> w[8];
        unpack8_b3(blend_odd(fsh_v<8>(b, a), fsh_v<24>(b, a)),
                   blend_odd(fsh_v<20>(b, a), fsh_v<36>(b, a)), w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 0);
        a.select<8, 2>(0) = p.select<8, 3>(0);
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(1);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        unpack8_b3(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)),
                   blend_odd(fsh_v<28>(b, a), fsh_v<12>(b, a)), w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 8);
    } else if constexpr (bits == 1) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 8> ar;
        ar.select<1, 1>(0) = p.select<1, 1>(7);
        ar.select<7, 1>(1) = p.select<7, 1>(0);
        simd<uint32_t, 16> a, b;
        a.select<8, 2>(0) = ar;
        a.select<8, 2>(1) = ar;
        b.select<8, 2>(0) = p;
        b.select<8, 2>(1) = p;
        unpack8_b1(blend_odd(fsh_v<24>(b, a), fsh_v<8>(b, a)), w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 0);
        unpack8_b1(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)), w);
        mac8_8(a0, a1, a2, a3, a4, a5, a6, a7, w, x0, x1, x2, x3, x4, x5, x6, x7, 8);
    } else {
        mac_wins4<bits>(pack, x0, x1, x2, x3, a0, a1, a2, a3);
        mac_wins4<bits>(pack, x4, x5, x6, x7, a4, a5, a6, a7);
    }
}

template <int bits>
ESIMD_INLINE void store_k16(const simd<uint32_t, 16> w[8], sycl::half * W, int k0, int N, int n0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        exl3_xmx::decode_mul1_v(w[j]).copy_to(W + size_t(k0 + j) * size_t(N) + size_t(n0) * 16);
    }
}

template <int bits>
ESIMD_INLINE void decode_store(const uint32_t * pack, sycl::half * W, int k0, int N, int n0) {
    if constexpr (bits == 4) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 32> p;
        p.copy_from(pack);
        simd<uint32_t, 16> b = p.select<16, 2>(0);
        simd<uint32_t, 16> c = p.select<16, 2>(1);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = p.select<1, 1>(31);
        a.select<15, 1>(1) = c.select<15, 1>(0);
        unpack8_b4(a, b, w);
        store_k16<bits>(w, W, k0, N, n0);
        unpack8_b4(b, c, w);
        store_k16<bits>(w, W, k0 + 8, N, n0);
    } else if constexpr (bits == 2) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 16> b;
        b.copy_from(pack);
        simd<uint32_t, 16> a;
        a.select<1, 1>(0) = b.select<1, 1>(15);
        a.select<15, 1>(1) = b.select<15, 1>(0);
        unpack8_b2((a << 16) | (b >> 16), w);
        store_k16<bits>(w, W, k0, N, n0);
        unpack8_b2(b, w);
        store_k16<bits>(w, W, k0 + 8, N, n0);
    } else if constexpr (bits == 3) {
        simd<uint32_t, 32> p(0);
        simd<uint32_t, 16> lo;
        lo.copy_from(pack);
        p.select<16, 1>(0) = lo;
        simd<uint32_t, 8> hi8;
        hi8.copy_from(pack + 16);
        p.select<8, 1>(16) = hi8;
        simd<uint32_t, 16> a, b;
        simd<uint32_t, 8> ae;
        ae.select<1, 1>(0) = p.select<1, 1>(23);
        ae.select<7, 1>(1) = p.select<7, 3>(2);
        a.select<8, 2>(0) = ae;
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(0);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        simd<uint32_t, 16> w[8];
        unpack8_b3(blend_odd(fsh_v<8>(b, a), fsh_v<24>(b, a)),
                   blend_odd(fsh_v<20>(b, a), fsh_v<36>(b, a)), w);
        store_k16<bits>(w, W, k0, N, n0);
        a.select<8, 2>(0) = p.select<8, 3>(0);
        a.select<8, 2>(1) = p.select<8, 3>(1);
        b.select<8, 2>(0) = p.select<8, 3>(1);
        b.select<8, 2>(1) = p.select<8, 3>(2);
        unpack8_b3(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)),
                   blend_odd(fsh_v<28>(b, a), fsh_v<12>(b, a)), w);
        store_k16<bits>(w, W, k0 + 8, N, n0);
    } else if constexpr (bits == 1) {
        simd<uint32_t, 16> w[8];
        simd<uint32_t, 8> p;
        p.copy_from(pack);
        simd<uint32_t, 8> ar;
        ar.select<1, 1>(0) = p.select<1, 1>(7);
        ar.select<7, 1>(1) = p.select<7, 1>(0);
        simd<uint32_t, 16> a, b;
        a.select<8, 2>(0) = ar;
        a.select<8, 2>(1) = ar;
        b.select<8, 2>(0) = p;
        b.select<8, 2>(1) = p;
        unpack8_b1(blend_odd(fsh_v<24>(b, a), fsh_v<8>(b, a)), w);
        store_k16<bits>(w, W, k0, N, n0);
        unpack8_b1(blend_odd(fsh_v<16>(b, a), fsh_v<0>(b, a)), w);
        store_k16<bits>(w, W, k0 + 8, N, n0);
    } else {
#pragma unroll
        for (int n = 0; n < 16; ++n) {
            simd<uint32_t, 16> col = extract16_kcol<bits>(pack, n);
            simd<sycl::half, 16> h = exl3_xmx::decode_mul1_v(col);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                W[size_t(k0 + k) * size_t(N) + size_t(n0) * 16 + size_t(n)] = h[k];
            }
        }
    }
}

template <int bits>
ESIMD_INLINE void gemm_n_tile(const uint32_t * B, const sycl::half * X, float * C,
                              int M, int K, int N, int packed_n, int n_tile, int tid, int m_base) {
    slm_init<512>();
    constexpr int RM = 8;
    constexpr int RN = 16;
    constexpr int WG = 16;
    namespace xmx = sycl::ext::intel::esimd::xmx;
    const int n_mt = (M - m_base + RM - 1) / RM;
    const int ntiles = packed_n / 16;
    const int n0 = n_tile * RN;
    simd<float, RM * RN> acc0(0.f), acc1(0.f), acc2(0.f), acc3(0.f);

    for (int kt = 0; kt < K / 16; ++kt) {
        const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(bits * 8);
        simd<sycl::half, 16> h = exl3_xmx::decode_mul1_v(extract16_kcol<bits>(pack, tid));
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            exl3_gemm::slm_vnni_put(k, tid, h[k]);
        }
        barrier();
        simd<sycl::half, 256> Bvnni = slm_block_load<sycl::half, 256>(0);
        auto load_A = [&](int tm) {
            simd<sycl::half, RM * 16> A(sycl::half(0));
            if (tm < n_mt) {
                const int m0 = m_base + tm * RM;
#pragma unroll
                for (int r = 0; r < RM; ++r) {
                    if (m0 + r < M) {
                        simd<sycl::half, 16> row;
                        row.copy_from(X + size_t(m0 + r) * size_t(K) + size_t(kt) * 16);
                        A.template select<16, 1>(r * 16) = row;
                    }
                }
            }
            return A;
        };
        acc0 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc0, Bvnni, load_A(tid + 0 * WG));
        acc1 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc1, Bvnni, load_A(tid + 1 * WG));
        acc2 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc2, Bvnni, load_A(tid + 2 * WG));
        acc3 = xmx::dpas<8, RM, float, float, sycl::half, sycl::half>(acc3, Bvnni, load_A(tid + 3 * WG));
        barrier();
    }

    auto store_acc = [&](int tm, simd<float, RM * RN> acc) {
        if (tm >= n_mt) {
            return;
        }
        const int m0 = m_base + tm * RM;
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            if (m0 + r < M) {
                simd<float, RN> row = acc.template select<RN, 1>(r * RN);
                row.copy_to(C + size_t(m0 + r) * size_t(N) + size_t(n0));
            }
        }
    };
    store_acc(tid + 0 * WG, acc0);
    store_acc(tid + 1 * WG, acc1);
    store_acc(tid + 2 * WG, acc2);
    store_acc(tid + 3 * WG, acc3);
}

template <int bits, int NT>
ESIMD_INLINE void gemv_vn(const uint32_t * B, const float * x, float * C,
                          int K, int N, int packed_n,
                          int n0, int n_here, int kt0, int kt1) {
    constexpr int pack_u32 = bits * 8;
    const int ntiles = packed_n / 16;
    simd<float, 16> acc0(0.f);
    simd<float, 16> acc1(0.f);
    simd<float, 16> acc2(0.f);
    simd<float, 16> acc3(0.f);
    float sum_x = 0.f;
    int kt = kt0;
    while (kt < kt1) {
        simd<float, 16> xv0;
        xv0.copy_from(x + size_t(kt) * 16);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum_x += float(xv0[k]);
        }
        const uint32_t * p0 = B + (size_t(kt) * size_t(ntiles) + size_t(n0)) * size_t(pack_u32);
        if (kt + 1 < kt1) {
            simd<float, 16> xv1;
            xv1.copy_from(x + size_t(kt + 1) * 16);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum_x += float(xv1[k]);
            }
            const uint32_t * p1 = p0 + size_t(ntiles) * size_t(pack_u32);
            mac_wins<bits>(p0, xv0, acc0);
            mac_wins<bits>(p1, xv1, acc0);
            if (n_here > 1) {
                mac_wins<bits>(p0 + pack_u32, xv0, acc1);
                mac_wins<bits>(p1 + pack_u32, xv1, acc1);
            }
            if constexpr (NT >= 4) {
                if (n_here > 2) {
                    mac_wins<bits>(p0 + 2 * pack_u32, xv0, acc2);
                    mac_wins<bits>(p1 + 2 * pack_u32, xv1, acc2);
                }
                if (n_here > 3) {
                    mac_wins<bits>(p0 + 3 * pack_u32, xv0, acc3);
                    mac_wins<bits>(p1 + 3 * pack_u32, xv1, acc3);
                }
            }
            kt += 2;
        } else {
            mac_wins<bits>(p0, xv0, acc0);
            if (n_here > 1) {
                mac_wins<bits>(p0 + pack_u32, xv0, acc1);
            }
            if constexpr (NT >= 4) {
                if (n_here > 2) {
                    mac_wins<bits>(p0 + 2 * pack_u32, xv0, acc2);
                }
                if (n_here > 3) {
                    mac_wins<bits>(p0 + 3 * pack_u32, xv0, acc3);
                }
            }
            kt += 1;
        }
    }
    const float k_inv = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float aff = 1024.f * k_inv + float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    {
        simd<float, 16> y = acc0 * k_inv + simd<float, 16>(aff * sum_x);
        y.copy_to(C + size_t(n0) * 16);
    }
    if (n_here > 1) {
        simd<float, 16> y = acc1 * k_inv + simd<float, 16>(aff * sum_x);
        y.copy_to(C + size_t(n0 + 1) * 16);
    }
    if constexpr (NT >= 4) {
        if (n_here > 2) {
            simd<float, 16> y = acc2 * k_inv + simd<float, 16>(aff * sum_x);
            y.copy_to(C + size_t(n0 + 2) * 16);
        }
        if (n_here > 3) {
            simd<float, 16> y = acc3 * k_inv + simd<float, 16>(aff * sum_x);
            y.copy_to(C + size_t(n0 + 3) * 16);
        }
    }
}

// spec verify M=2. One decode, two x rows. NT=2.
template <int bits>
ESIMD_INLINE void gemv_vn2(const uint32_t * B, const float * x, float * C,
                           int K, int N, int packed_n,
                           int n0, int n_here, int kt0, int kt1) {
    constexpr int pack_u32 = bits * 8;
    const int ntiles = packed_n / 16;
    simd<float, 16> a00(0.f), a01(0.f), a10(0.f), a11(0.f);
    float sum0 = 0.f;
    float sum1 = 0.f;
    int kt = kt0;
    while (kt < kt1) {
        simd<float, 16> x00, x10;
        x00.copy_from(x + size_t(kt) * 16);
        x10.copy_from(x + size_t(K) + size_t(kt) * 16);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            sum0 += float(x00[k]);
            sum1 += float(x10[k]);
        }
        const uint32_t * p0 = B + (size_t(kt) * size_t(ntiles) + size_t(n0)) * size_t(pack_u32);
        if (kt + 1 < kt1) {
            simd<float, 16> x01, x11;
            x01.copy_from(x + size_t(kt + 1) * 16);
            x11.copy_from(x + size_t(K) + size_t(kt + 1) * 16);
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                sum0 += float(x01[k]);
                sum1 += float(x11[k]);
            }
            const uint32_t * p1 = p0 + size_t(ntiles) * size_t(pack_u32);
            mac_wins2<bits>(p0, x00, x10, a00, a10);
            mac_wins2<bits>(p1, x01, x11, a00, a10);
            if (n_here > 1) {
                mac_wins2<bits>(p0 + pack_u32, x00, x10, a01, a11);
                mac_wins2<bits>(p1 + pack_u32, x01, x11, a01, a11);
            }
            kt += 2;
        } else {
            mac_wins2<bits>(p0, x00, x10, a00, a10);
            if (n_here > 1) {
                mac_wins2<bits>(p0 + pack_u32, x00, x10, a01, a11);
            }
            kt += 1;
        }
    }
    const float k_inv = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float aff = 1024.f * k_inv + float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    {
        simd<float, 16> y0 = a00 * k_inv + simd<float, 16>(aff * sum0);
        simd<float, 16> y1 = a10 * k_inv + simd<float, 16>(aff * sum1);
        y0.copy_to(C + size_t(n0) * 16);
        y1.copy_to(C + size_t(N) + size_t(n0) * 16);
    }
    if (n_here > 1) {
        simd<float, 16> y0 = a01 * k_inv + simd<float, 16>(aff * sum0);
        simd<float, 16> y1 = a11 * k_inv + simd<float, 16>(aff * sum1);
        y0.copy_to(C + size_t(n0 + 1) * 16);
        y1.copy_to(C + size_t(N) + size_t(n0 + 1) * 16);
    }
}

// spec verify M=3/4. NT=1 so GRF matches gemv_vn2 (4 accs). Pair k-tiles like vn2.
template <int bits>
ESIMD_INLINE void gemv_vn4(const uint32_t * B, const float * x, float * C,
                           int M, int K, int N, int packed_n,
                           int n0, int kt0, int kt1) {
    constexpr int pack_u32 = bits * 8;
    const int ntiles = packed_n / 16;
    simd<float, 16> a0(0.f), a1(0.f), a2(0.f), a3(0.f);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int kt = kt0;
    while (kt < kt1) {
        simd<float, 16> x0, x1, x2(0.f), x3(0.f);
        x0.copy_from(x + size_t(kt) * 16);
        x1.copy_from(x + size_t(K) + size_t(kt) * 16);
        if (M > 2) {
            x2.copy_from(x + size_t(2) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 3) {
            x3.copy_from(x + size_t(3) * size_t(K) + size_t(kt) * 16);
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            s0 += float(x0[k]);
            s1 += float(x1[k]);
            s2 += float(x2[k]);
            s3 += float(x3[k]);
        }
        const uint32_t * p0 = B + (size_t(kt) * size_t(ntiles) + size_t(n0)) * size_t(pack_u32);
        if (kt + 1 < kt1) {
            simd<float, 16> y0, y1, y2(0.f), y3(0.f);
            y0.copy_from(x + size_t(kt + 1) * 16);
            y1.copy_from(x + size_t(K) + size_t(kt + 1) * 16);
            if (M > 2) {
                y2.copy_from(x + size_t(2) * size_t(K) + size_t(kt + 1) * 16);
            }
            if (M > 3) {
                y3.copy_from(x + size_t(3) * size_t(K) + size_t(kt + 1) * 16);
            }
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                s0 += float(y0[k]);
                s1 += float(y1[k]);
                s2 += float(y2[k]);
                s3 += float(y3[k]);
            }
            const uint32_t * p1 = p0 + size_t(ntiles) * size_t(pack_u32);
            mac_wins4<bits>(p0, x0, x1, x2, x3, a0, a1, a2, a3);
            mac_wins4<bits>(p1, y0, y1, y2, y3, a0, a1, a2, a3);
            kt += 2;
        } else {
            mac_wins4<bits>(p0, x0, x1, x2, x3, a0, a1, a2, a3);
            kt += 1;
        }
    }
    const float k_inv = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float aff = 1024.f * k_inv + float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    {
        simd<float, 16> y = a0 * k_inv + simd<float, 16>(aff * s0);
        y.copy_to(C + size_t(n0) * 16);
        y = a1 * k_inv + simd<float, 16>(aff * s1);
        y.copy_to(C + size_t(N) + size_t(n0) * 16);
        if (M > 2) {
            y = a2 * k_inv + simd<float, 16>(aff * s2);
            y.copy_to(C + size_t(2) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 3) {
            y = a3 * k_inv + simd<float, 16>(aff * s3);
            y.copy_to(C + size_t(3) * size_t(N) + size_t(n0) * 16);
        }
    }
}

// spec verify M=5..8. NT=1, 8 accs, shared decode.
template <int bits>
ESIMD_INLINE void gemv_vn8(const uint32_t * B, const float * x, float * C,
                           int M, int K, int N, int packed_n,
                           int n0, int kt0, int kt1) {
    constexpr int pack_u32 = bits * 8;
    const int ntiles = packed_n / 16;
    simd<float, 16> a0(0.f), a1(0.f), a2(0.f), a3(0.f);
    simd<float, 16> a4(0.f), a5(0.f), a6(0.f), a7(0.f);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
    int kt = kt0;
    while (kt < kt1) {
        simd<float, 16> x0, x1, x2(0.f), x3(0.f), x4(0.f), x5(0.f), x6(0.f), x7(0.f);
        x0.copy_from(x + size_t(kt) * 16);
        x1.copy_from(x + size_t(K) + size_t(kt) * 16);
        if (M > 2) {
            x2.copy_from(x + size_t(2) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 3) {
            x3.copy_from(x + size_t(3) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 4) {
            x4.copy_from(x + size_t(4) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 5) {
            x5.copy_from(x + size_t(5) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 6) {
            x6.copy_from(x + size_t(6) * size_t(K) + size_t(kt) * 16);
        }
        if (M > 7) {
            x7.copy_from(x + size_t(7) * size_t(K) + size_t(kt) * 16);
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            s0 += float(x0[k]);
            s1 += float(x1[k]);
            s2 += float(x2[k]);
            s3 += float(x3[k]);
            s4 += float(x4[k]);
            s5 += float(x5[k]);
            s6 += float(x6[k]);
            s7 += float(x7[k]);
        }
        const uint32_t * p = B + (size_t(kt) * size_t(ntiles) + size_t(n0)) * size_t(pack_u32);
        mac_wins8<bits>(p, x0, x1, x2, x3, x4, x5, x6, x7, a0, a1, a2, a3, a4, a5, a6, a7);
        kt += 1;
    }
    const float k_inv = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float aff = 1024.f * k_inv + float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    {
        simd<float, 16> y = a0 * k_inv + simd<float, 16>(aff * s0);
        y.copy_to(C + size_t(n0) * 16);
        y = a1 * k_inv + simd<float, 16>(aff * s1);
        y.copy_to(C + size_t(N) + size_t(n0) * 16);
        if (M > 2) {
            y = a2 * k_inv + simd<float, 16>(aff * s2);
            y.copy_to(C + size_t(2) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 3) {
            y = a3 * k_inv + simd<float, 16>(aff * s3);
            y.copy_to(C + size_t(3) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 4) {
            y = a4 * k_inv + simd<float, 16>(aff * s4);
            y.copy_to(C + size_t(4) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 5) {
            y = a5 * k_inv + simd<float, 16>(aff * s5);
            y.copy_to(C + size_t(5) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 6) {
            y = a6 * k_inv + simd<float, 16>(aff * s6);
            y.copy_to(C + size_t(6) * size_t(N) + size_t(n0) * 16);
        }
        if (M > 7) {
            y = a7 * k_inv + simd<float, 16>(aff * s7);
            y.copy_to(C + size_t(7) * size_t(N) + size_t(n0) * 16);
        }
    }
}

} // namespace exl3_xe2

template <int bits, int MR>
inline bool launch_exl3_gemv_xe2_esimd_mr(const float * x, const uint32_t * B, float * C,
                                          int M, int K, int N, int packed_n, queue_ptr stream) {
    const int n_tiles = N / 16;
    const int ks = K / 16;
    if (n_tiles <= 0 || (N % 16) || (K % 16) || M < 1 || M > MR) {
        return false;
    }
    if constexpr (bits >= 5) {
        return false;
    }
    int n_kparts = ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_KSPLIT", 16);
    if (n_kparts < 1) {
        n_kparts = 1;
    }
    if (n_kparts > ks) {
        n_kparts = ks > 0 ? ks : 1;
    }
    if (N >= 65536 && n_kparts > 4) {
        n_kparts = 4;
    }
    int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_NT", 2);
    if (nt != 2 && nt != 4) {
        return false;
    }
    // M=2: NT=2. M>=3: NT=1 (same 4-acc GRF as gemv_vn2).
    if (M >= 3) {
        nt = 1;
    } else if (M > 1) {
        nt = 2;
    }
    const int n_groups_n = (n_tiles + nt - 1) / nt;
    const int g_glb = n_groups_n * n_kparts;
    float * out = C;
    if (n_kparts > 1) {
        static thread_local float * parts = nullptr;
        static thread_local size_t cap = 0;
        static thread_local sycl::queue * last = nullptr;
        const size_t need = size_t(n_kparts) * size_t(M) * size_t(N);
        if (!parts || need > cap || last != stream) {
            if (parts && last) {
                sycl::free(parts, *last);
            }
            parts = sycl::malloc_device<float>(need, *stream);
            cap = need;
            last = stream;
        }
        out = parts;
    }
    try {
        auto submit = [&](auto nt_tag) {
            constexpr int NT = decltype(nt_tag)::value;
            stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(size_t(g_glb)), sycl::range<1>(1)),
                    [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                        const int gid = int(it.get_group(0));
                        const int n0 = (gid % n_groups_n) * NT;
                        const int k_part = gid / n_groups_n;
                        int n_here = n_tiles - n0;
                        if (n_here > NT) {
                            n_here = NT;
                        }
                        const int chunk = (ks + n_kparts - 1) / n_kparts;
                        const int kt0 = k_part * chunk;
                        int kt1 = kt0 + chunk;
                        if (kt1 > ks) {
                            kt1 = ks;
                        }
                        float * dst = out + size_t(k_part) * size_t(M) * size_t(N);
                        if constexpr (MR == 1) {
                            exl3_xe2::gemv_vn<bits, NT>(B, x, dst, K, N, packed_n,
                                    n0, n_here, kt0, kt1);
                        } else if constexpr (MR == 2) {
                            exl3_xe2::gemv_vn2<bits>(B, x, dst, K, N, packed_n,
                                    n0, n_here, kt0, kt1);
                        } else if constexpr (MR == 4) {
                            exl3_xe2::gemv_vn4<bits>(B, x, dst, M, K, N, packed_n,
                                    n0, kt0, kt1);
                        } else {
                            exl3_xe2::gemv_vn8<bits>(B, x, dst, M, K, N, packed_n,
                                    n0, kt0, kt1);
                        }
                    });
        };
        if constexpr (MR == 1) {
            if (nt == 4) {
                submit(std::integral_constant<int, 4>{});
            } else {
                submit(std::integral_constant<int, 2>{});
            }
        } else if constexpr (MR == 2) {
            submit(std::integral_constant<int, 2>{});
        } else {
            submit(std::integral_constant<int, 1>{});
        }
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL xe2 esimd: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    if (n_kparts > 1) {
        const int np = n_kparts;
        const float * parts = out;
        const size_t mn = size_t(M) * size_t(N);
        stream->parallel_for(sycl::range<1>(mn), [=](sycl::id<1> i) {
            float s = 0.f;
            for (int p = 0; p < np; ++p) {
                s += parts[size_t(p) * mn + i];
            }
            C[i] = s;
        });
    }
    return true;
}

inline bool launch_exl3_gemv_xe2_esimd_bits(int bits, const float * x, const uint32_t * B, float * C,
                                            int M, int K, int N, int packed_n, queue_ptr stream) {
    auto go = [&](auto mr) {
        constexpr int MR = decltype(mr)::value;
        switch (bits) {
            case 1: return launch_exl3_gemv_xe2_esimd_mr<1, MR>(x, B, C, M, K, N, packed_n, stream);
            case 2: return launch_exl3_gemv_xe2_esimd_mr<2, MR>(x, B, C, M, K, N, packed_n, stream);
            case 3: return launch_exl3_gemv_xe2_esimd_mr<3, MR>(x, B, C, M, K, N, packed_n, stream);
            case 4: return launch_exl3_gemv_xe2_esimd_mr<4, MR>(x, B, C, M, K, N, packed_n, stream);
            default: return false;
        }
    };
    if (M == 1) {
        return go(std::integral_constant<int, 1>{});
    }
    if (M == 2) {
        return go(std::integral_constant<int, 2>{});
    }
    if (M <= 4) {
        return go(std::integral_constant<int, 4>{});
    }
    if (M <= 8) {
        return go(std::integral_constant<int, 8>{});
    }
    return false;
}

template <int bits>
inline bool launch_exl3_recon_xe2_vn(const uint32_t * B, sycl::half * W,
                                     int K, int N, int packed_n, queue_ptr stream) {
    if ((K % 16) || (N % 16) || packed_n < N) {
        return false;
    }
    constexpr int KT = 8;
    constexpr int pack_u32 = bits * 8;
    const int n_tiles = N / 16;
    const int k_tiles = K / 16;
    const int k_groups = (k_tiles + KT - 1) / KT;
    const int groups = n_tiles * k_groups;
    try {
        stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(groups)), sycl::range<1>(1)),
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    const int gid = int(it.get_group(0));
                    const int n_tile = gid % n_tiles;
                    const int kg = gid / n_tiles;
                    const int ntiles = packed_n / 16;
                    for (int tk = 0; tk < KT; ++tk) {
                        const int k_tile = kg * KT + tk;
                        if (k_tile < k_tiles) {
                            const uint32_t * pack = B + (size_t(k_tile) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
                            exl3_xe2::decode_store<bits>(pack, W, k_tile * 16, N, n_tile);
                        }
                    }
                });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL recon xe2 vn: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

inline bool launch_exl3_recon_xe2_vn_bits(int bits, const uint32_t * B, sycl::half * W,
                                          int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_recon_xe2_vn<1>(B, W, K, N, packed_n, stream);
        case 2: return launch_exl3_recon_xe2_vn<2>(B, W, K, N, packed_n, stream);
        case 3: return launch_exl3_recon_xe2_vn<3>(B, W, K, N, packed_n, stream);
        case 4: return launch_exl3_recon_xe2_vn<4>(B, W, K, N, packed_n, stream);
        default: return false;
    }
}

template <int bits>
inline bool launch_exl3_recon_xe2_esimd(const uint32_t * B, sycl::half * W,
                                        int K, int N, int packed_n, queue_ptr stream) {
    if ((K % 16) || (N % 16) || packed_n < N) {
        return false;
    }
    constexpr int KT = 8;
    constexpr int WG = 16;
    constexpr int pack_u32 = bits * 8;
    const int n_tiles = N / 16;
    const int k_tiles = K / 16;
    const int k_groups = (k_tiles + KT - 1) / KT;
    const int groups = n_tiles * k_groups;
    const int wkn = ggml_sycl_get_env("GGML_SYCL_EXL3_WKN", 1);
    try {
        stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(groups) * WG), sycl::range<1>(WG)),
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    const int tid = int(it.get_local_id(0));
                    const int gid = int(it.get_group(0));
                    const int n_tile = gid % n_tiles;
                    const int kg = gid / n_tiles;
                    const int n = n_tile * 16 + tid;
                    const int ntiles = packed_n / 16;
                    for (int tk = 0; tk < KT; ++tk) {
                        const int k_tile = kg * KT + tk;
                        if (k_tile < k_tiles) {
                            const uint32_t * pack = B + (size_t(k_tile) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32);
                            sycl::ext::intel::esimd::simd<sycl::half, 16> h =
                                    exl3_xmx::decode_mul1_v(exl3_xe2::extract16_kcol<bits>(pack, tid));
                            if (wkn) {
#pragma unroll
                                for (int k = 0; k < 16; ++k) {
                                    W[size_t(k_tile * 16 + k) * size_t(N) + size_t(n)] = h[k];
                                }
                            } else {
                                h.copy_to(W + size_t(n) * size_t(K) + size_t(k_tile) * 16);
                            }
                        }
                    }
                });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL recon xe2 esimd: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

inline bool launch_exl3_recon_xe2_esimd_bits(int bits, const uint32_t * B, sycl::half * W,
                                             int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_recon_xe2_esimd<1>(B, W, K, N, packed_n, stream);
        case 2: return launch_exl3_recon_xe2_esimd<2>(B, W, K, N, packed_n, stream);
        case 3: return launch_exl3_recon_xe2_esimd<3>(B, W, K, N, packed_n, stream);
        case 4: return launch_exl3_recon_xe2_esimd<4>(B, W, K, N, packed_n, stream);
        case 5: return launch_exl3_recon_xe2_esimd<5>(B, W, K, N, packed_n, stream);
        case 6: return launch_exl3_recon_xe2_esimd<6>(B, W, K, N, packed_n, stream);
        case 7: return launch_exl3_recon_xe2_esimd<7>(B, W, K, N, packed_n, stream);
        case 8: return launch_exl3_recon_xe2_esimd<8>(B, W, K, N, packed_n, stream);
        default: return false;
    }
}

template <int bits>
inline bool launch_exl3_gemm_xe2_xmx(const sycl::half * X, const uint32_t * B, float * C,
                                     int M, int K, int N, int packed_n, queue_ptr stream) {
    if (M <= 0 || (N % 16) || (K % 16) || packed_n < N) {
        return false;
    }
    constexpr int MCHUNK = 512;
    const int n_tiles = N / 16;
    const int n_mg = (M + MCHUNK - 1) / MCHUNK;
    if (n_tiles <= 0 || n_mg <= 0) {
        return false;
    }
    try {
        const auto props = sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> };
        stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(n_tiles) * size_t(n_mg) * exl3_gemm::WG),
                                  sycl::range<1>(exl3_gemm::WG)),
                props,
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    const int gid = int(it.get_group(0));
                    const int n_tile = gid % n_tiles;
                    const int m_base = (gid / n_tiles) * MCHUNK;
                    exl3_xe2::gemm_n_tile<bits>(B, X, C, M, K, N, packed_n,
                            n_tile, int(it.get_local_id(0)), m_base);
                });
    } catch (const sycl::exception & e) {
        fprintf(stderr, "EXL3_SYCL xe2 xmx: %s\n", e.what());
        fflush(stderr);
        return false;
    }
    return true;
}

inline bool launch_exl3_gemm_xe2_xmx_bits(int bits, const sycl::half * X, const uint32_t * B, float * C,
                                          int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemm_xe2_xmx<1>(X, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemm_xe2_xmx<2>(X, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemm_xe2_xmx<3>(X, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemm_xe2_xmx<4>(X, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemm_xe2_xmx<5>(X, B, C, M, K, N, packed_n, stream);
        case 6: return launch_exl3_gemm_xe2_xmx<6>(X, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

inline bool launch_exl3_gemm_packed_xmx_bits(int bits, const sycl::half * X, const uint32_t * B, float * C,
                                             int M, int K, int N, int packed_n, queue_ptr stream) {
    switch (bits) {
        case 1: return launch_exl3_gemm_packed_xmx<1>(X, B, C, M, K, N, packed_n, stream);
        case 2: return launch_exl3_gemm_packed_xmx<2>(X, B, C, M, K, N, packed_n, stream);
        case 3: return launch_exl3_gemm_packed_xmx<3>(X, B, C, M, K, N, packed_n, stream);
        case 4: return launch_exl3_gemm_packed_xmx<4>(X, B, C, M, K, N, packed_n, stream);
        case 5: return launch_exl3_gemm_packed_xmx<5>(X, B, C, M, K, N, packed_n, stream);
        case 6: return launch_exl3_gemm_packed_xmx<6>(X, B, C, M, K, N, packed_n, stream);
        case 7: return launch_exl3_gemm_packed_xmx<7>(X, B, C, M, K, N, packed_n, stream);
        case 8: return launch_exl3_gemm_packed_xmx<8>(X, B, C, M, K, N, packed_n, stream);
        default: return false;
    }
}

inline bool launch_exl3_recon_esimd_bits(int bits, const uint32_t * B, sycl::half * W,
                                         int K, int N, int packed_n, queue_ptr stream,
                                         int k0 = 0, int k_len = 0, int dest_ld = 0) {
    switch (bits) {
        case 1: return launch_exl3_recon_esimd<1>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 2: return launch_exl3_recon_esimd<2>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 3: return launch_exl3_recon_esimd<3>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 4: return launch_exl3_recon_esimd<4>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 5: return launch_exl3_recon_esimd<5>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 6: return launch_exl3_recon_esimd<6>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 7: return launch_exl3_recon_esimd<7>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        case 8: return launch_exl3_recon_esimd<8>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld);
        default: return false;
    }
}

} // namespace ggml_sycl_esimd

#endif
