#include "exl3.hpp"

#include "convert.hpp"
#include "fwht.hpp"
#include "gemm.hpp"
#if defined(__INTEL_LLVM_COMPILER)
#include "exl3-esimd.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

// Trellis decode (cb=2) + 16x16 map from harness decode_tile_python.

namespace {

inline uint32_t exl3_fshift(uint32_t b, uint32_t a, int shift) {
    const uint64_t merged = (uint64_t(a) << 32) | uint64_t(b);
    return uint32_t(merged >> shift);
}

inline sycl::half decode_mul1(uint32_t x) {
    x *= 0x83DCD12Du;
    const uint32_t sum = 0x6400u
            + (x & 0xffu) + ((x >> 8) & 0xffu)
            + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu);
    const sycl::half h      = sycl::bit_cast<sycl::half>(uint16_t(sum));
    const sycl::half k_inv  = sycl::bit_cast<sycl::half>(uint16_t(0x1eee));
    const sycl::half k_bias = sycl::bit_cast<sycl::half>(uint16_t(0xc931));
    return sycl::fma(h, k_inv, k_bias);
}

inline int hash_byte_sum(uint32_t x) {
    x *= 0x83DCD12Du;
    return int((x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu));
}

// One 16-bit trellis word. Same window as CUDA dq() / harness decode_tile_python.
template <int bits>
inline uint32_t window16(const uint32_t * ptr, int t_offset) {
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int s0 = (i1 + 1) * 32 - b1;
    const int mod = bits * 8;
    const uint32_t a = ptr[i0 % mod];
    const uint32_t b = ptr[i1 % mod];
    return exl3_fshift(b, a, s0) & 0xffffu;
}

template <int bits>
inline void extract8_ab(uint32_t a, uint32_t b, int t_offset, uint32_t w[8]) {
    if constexpr (bits == 4) {
        const uint32_t s = exl3_fshift(b, a, 20);
        w[7] = b & 0xffffu;
        w[6] = (b >> 4) & 0xffffu;
        w[5] = (b >> 8) & 0xffffu;
        w[4] = (b >> 12) & 0xffffu;
        w[3] = (b >> 16) & 0xffffu;
        w[2] = s & 0xffffu;
        w[1] = (s >> 4) & 0xffffu;
        w[0] = (s >> 8) & 0xffffu;
    } else if constexpr (bits == 2) {
        b = exl3_fshift(b, a, ((~t_offset) & 8) << 1);
        w[7] = b & 0xffffu;
        w[6] = (b >> 2) & 0xffffu;
        w[5] = (b >> 4) & 0xffffu;
        w[4] = (b >> 6) & 0xffffu;
        w[3] = (b >> 8) & 0xffffu;
        w[2] = (b >> 10) & 0xffffu;
        w[1] = (b >> 12) & 0xffffu;
        w[0] = (b >> 14) & 0xffffu;
    } else if constexpr (bits == 1) {
        b = exl3_fshift(b, a, ((~t_offset) & 24));
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
        const int b2 = b1 + bits * 7;
        const int s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
        uint32_t t = exl3_fshift(b, a, s2);
        w[7] = t & 0xffffu;
        t >>= 3; w[6] = t & 0xffffu;
        t >>= 3; w[5] = t & 0xffffu;
        t >>= 3; w[4] = t & 0xffffu;
        t = exl3_fshift(b, a, s2 + 12);
        w[3] = t & 0xffffu;
        t >>= 3; w[2] = t & 0xffffu;
        t >>= 3; w[1] = t & 0xffffu;
        t >>= 3; w[0] = t & 0xffffu;
    }
}

template <int bits>
inline void extract8_idx(int t_offset, int & i0, int & i1) {
    if constexpr (bits == 4) {
        i1 = t_offset >> 3;
        i0 = (i1 + 31) & 31;
    } else if constexpr (bits == 2) {
        i1 = t_offset >> 4;
        i0 = (i1 + 15) & 15;
    } else if constexpr (bits == 1) {
        i1 = t_offset >> 5;
        i0 = (i1 + 7) & 7;
    } else if constexpr (bits == 3) {
        const int b1 = (t_offset + 257) * bits;
        const int b0 = b1 - 16;
        const int b2 = b1 + bits * 7;
        const int mod = bits * 8;
        i0 = (b0 / 32) % mod;
        i1 = ((b2 - 1) / 32) % mod;
    } else {
        i0 = 0;
        i1 = 0;
    }
}

template <int bits>
inline void extract8(const uint32_t * ptr, int t_offset, uint32_t w[8]) {
    if constexpr (bits <= 4) {
        int i0, i1;
        extract8_idx<bits>(t_offset, i0, i1);
        extract8_ab<bits>(ptr[i0], ptr[i1], t_offset, w);
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[j] = window16<bits>(ptr, t_offset + j);
        }
    }
}

// K-col: 16 windows for column n. bits=4 is 3 words; bits=1/2 share one pair.
template <int bits>
inline void extract16_kcol(const uint32_t * ptr, int n, uint32_t w[16]) {
    if constexpr (bits == 4) {
        const int mid = (n * 2) & 31;
        const uint32_t a = ptr[(mid + 31) & 31];
        const uint32_t b = ptr[mid];
        const uint32_t c = ptr[(mid + 1) & 31];
        extract8_ab<4>(a, b, n * 16, w);
        extract8_ab<4>(b, c, n * 16 + 8, w + 8);
    } else if constexpr (bits == 2) {
        const int t = n * 16;
        const int i1 = t >> 4;
        const int i0 = (i1 + 15) & 15;
        const uint32_t a = ptr[i0];
        const uint32_t b = ptr[i1];
        extract8_ab<2>(a, b, t, w);
        extract8_ab<2>(a, b, t + 8, w + 8);
    } else if constexpr (bits == 1) {
        const int t = n * 16;
        const int i1 = t >> 5;
        const int i0 = (i1 + 7) & 7;
        const uint32_t a = ptr[i0];
        const uint32_t b = ptr[i1];
        extract8_ab<1>(a, b, t, w);
        extract8_ab<1>(a, b, t + 8, w + 8);
    } else {
        extract8<bits>(ptr, n * 16, w);
        extract8<bits>(ptr, n * 16 + 8, w + 8);
    }
}

// Row-major 16x16. Same (lane, j) map as harness/verify/run_check.py decode_tile_python.
template <int bits, typename Acc>
inline void decode_tile(const uint32_t * ptr, int lane, Acc tile) {
    uint32_t w[8];
    extract8<bits>(ptr, lane * 8, w);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float val = float(decode_mul1(w[j]));
        const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
        const int col = (lane / 8) * 2 + ((lane & 4) ? 1 : 0) + (j >= 4 ? 8 : 0);
        tile[row * 16 + col] = val;
    }
}

inline bool exl3_pack_xe2() {
    return ggml_exl3_get_pack() != 0;
}

inline int exl3_wkn() {
    return ggml_sycl_get_env("GGML_SYCL_EXL3_WKN", 1);
}

// xe2-16: lane n owns What[:,n]. Overlap along K. Shuffle x, no full-pack copy.
// AFFINE: acc += bs*x, then y = k_inv*acc + (1024*k_inv+k_bias)*sum(x).
// Same closed form as CUDA int8 gemv; skips per-weight half fma. Not bit-identical.
template <int bits, int ROWS, int NT, int AFFINE>
void gemv_kernel_xe2(const float * x, const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n, int n_kparts,
                     sycl::nd_item<1> it) {
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const auto sg = it.get_sub_group();
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int k_part = gid / n_groups_n;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + n_kparts - 1) / n_kparts;
    const int kt0 = k_part * chunk;
    const int kt1 = sycl::min(kslices, kt0 + chunk);

    float acc[NT][ROWS];
    float sum_x[ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[t][r] = 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sum_x[r] = 0.0f;
    }

    float x0[ROWS][16];
    float x1[ROWS][16];
    auto fill_x = [&](int kt, float dest[ROWS][16]) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float my = 0.0f;
            if (r < size_m) {
                my = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
            }
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                dest[r][k] = sycl::select_from_group(sg, my, k);
                s += dest[r][k];
            }
            if constexpr (AFFINE) {
                sum_x[r] += s;
            }
        }
    };
    auto mac_kt = [&](int kt, float xr[ROWS][16]) {
        for (int t = 0; t < n_here; ++t) {
            const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
            uint32_t w[16];
            extract16_kcol<bits>(pack, lane, w);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v0, v1;
                if constexpr (AFFINE) {
                    v0 = float(hash_byte_sum(w[k]));
                    v1 = float(hash_byte_sum(w[k + 8]));
                } else {
                    v0 = float(decode_mul1(w[k]));
                    v1 = float(decode_mul1(w[k + 8]));
                }
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    acc[t][r] += v0 * xr[r][k];
                    acc[t][r] += v1 * xr[r][k + 8];
                }
            }
        }
    };
    int kt = kt0;
    if (kt < kt1) {
        fill_x(kt, x0);
    }
    while (kt < kt1) {
        if (kt + 1 < kt1) {
            fill_x(kt + 1, x1);
            mac_kt(kt, x0);
            mac_kt(kt + 1, x1);
            kt += 2;
            if (kt < kt1) {
                fill_x(kt, x0);
            }
        } else {
            mac_kt(kt, x0);
            kt += 1;
        }
    }

    float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
    const float k_inv = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float aff = 1024.f * k_inv + float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (t < n_here && r < size_m) {
                float y = acc[t][r];
                if constexpr (AFFINE) {
                    y = k_inv * y + aff * sum_x[r];
                }
                base[size_t(r) * size_t(size_n) + size_t(n0 + t) * 16 + lane] = y;
            }
        }
    }
}

template <int bits>
void recon_xe2_kernel(const uint32_t * B, sycl::half * W, int K, int dest_n, int packed_n,
                      int k0, int k_len, int dest_ld, int wkn, sycl::nd_item<1> it) {
    constexpr int KT = 8;
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int dest_ntiles = dest_n / 16;
    const int packed_ntiles = packed_n / 16;
    const int k0t = k0 / 16;
    const int k_tiles = k_len / 16;
    const int n_tile = gid % dest_ntiles;
    const int kg = gid / dest_ntiles;
    const int n = n_tile * 16 + lane;
    for (int tk = 0; tk < KT; ++tk) {
        const int k_tile = k0t + kg * KT + tk;
        if (k_tile >= k0t + k_tiles) {
            break;
        }
        const uint32_t * pack = B + (size_t(k_tile) * size_t(packed_ntiles) + size_t(n_tile)) * size_t(pack_u32);
        uint32_t w[16];
        extract16_kcol<bits>(pack, lane, w);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int kk_full = k_tile * 16 + k;
            const int kk_loc = (k_tile - k0t) * 16 + k;
            const size_t idx = wkn ? size_t(kk_full) * size_t(dest_n) + size_t(n)
                                   : size_t(n) * size_t(dest_ld) + size_t(kk_loc);
            W[idx] = decode_mul1(w[k]);
        }
    }
}

template <int bits, int ROWS>
void launch_gemv_xe2(const float * x, const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n,
                     queue_ptr stream) {
    const int n_tiles = size_n / 16;
    int n_kparts = ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_KSPLIT", 16);
    if (n_kparts < 1) {
        n_kparts = 1;
    }
    if (n_kparts > 32) {
        n_kparts = 32;
    }
    const int ks = size_k / 16;
    if (n_kparts > ks) {
        n_kparts = ks > 0 ? ks : 1;
    }
    if (size_n >= 65536 && n_kparts > 4) {
        n_kparts = 4;
    }
    int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_NT", 2);
    if (nt < 1) {
        nt = 1;
    }
    if (nt > 4) {
        nt = 4;
    }
    const int n_groups_n = (n_tiles + nt - 1) / nt;
    const int g_glb = n_groups_n * n_kparts;
    float * out = C;
    if (n_kparts > 1) {
        static thread_local float * parts = nullptr;
        static thread_local size_t cap = 0;
        static thread_local sycl::queue * last = nullptr;
        const size_t need = size_t(n_kparts) * size_t(size_m) * size_t(size_n);
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
    const int affine = ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_AFFINE", 1);
    auto submit = [&](auto nt_tag, auto aff_tag) {
        constexpr int NT = decltype(nt_tag)::value;
        constexpr int AFF = decltype(aff_tag)::value;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(size_t(g_glb) * 16), sycl::range<1>(16)),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        gemv_kernel_xe2<bits, ROWS, NT, AFF>(x, B, out, size_m, size_k, size_n, packed_n,
                                                             n_kparts, it);
                    });
        });
    };
    auto submit_nt = [&](auto nt_tag) {
        if (affine) {
            submit(nt_tag, std::integral_constant<int, 1>{});
        } else {
            submit(nt_tag, std::integral_constant<int, 0>{});
        }
    };
    if (nt >= 4) {
        submit_nt(std::integral_constant<int, 4>{});
    } else if (nt <= 1) {
        submit_nt(std::integral_constant<int, 1>{});
    } else {
        submit_nt(std::integral_constant<int, 2>{});
    }
    if (n_kparts > 1) {
        const int np = n_kparts;
        const float * parts = out;
        const size_t mn = size_t(size_m) * size_t(size_n);
        stream->parallel_for(sycl::range<1>(mn), [=](sycl::id<1> i) {
            float s = 0.f;
            for (int p = 0; p < np; ++p) {
                s += parts[size_t(p) * mn + i];
            }
            C[i] = s;
        });
    }
}

void launch_gemv_xe2_bits(int bits, const float * x, const uint32_t * B, float * C,
                          int size_m, int size_k, int size_n, int packed_n,
                          queue_ptr stream) {
#define EXL3_SYCL_GEMV_XE2_CASE(b) \
    case b: \
        if (size_m <= 1) { \
            launch_gemv_xe2<b, 1>(x, B, C, size_m, size_k, size_n, packed_n, stream); \
        } else if (size_m <= 2) { \
            launch_gemv_xe2<b, 2>(x, B, C, size_m, size_k, size_n, packed_n, stream); \
        } else { \
            launch_gemv_xe2<b, EXL3_GEMV_MAX_M>(x, B, C, size_m, size_k, size_n, packed_n, stream); \
        } \
        break;
    switch (bits) {
        EXL3_SYCL_GEMV_XE2_CASE(1)
        EXL3_SYCL_GEMV_XE2_CASE(2)
        EXL3_SYCL_GEMV_XE2_CASE(3)
        EXL3_SYCL_GEMV_XE2_CASE(4)
        EXL3_SYCL_GEMV_XE2_CASE(5)
        EXL3_SYCL_GEMV_XE2_CASE(6)
        EXL3_SYCL_GEMV_XE2_CASE(7)
        EXL3_SYCL_GEMV_XE2_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_GEMV_XE2_CASE
}

template <int bits>
void launch_recon_xe2(const uint32_t * B, sycl::half * W, int K, int N, int packed_n, queue_ptr stream,
                      int k0 = 0, int k_len = 0, int dest_ld = 0, int wkn_force = -1) {
    if (k_len <= 0) {
        k_len = K;
    }
    if (dest_ld <= 0) {
        dest_ld = K;
    }
    constexpr int KT = 8;
    const int n_tiles = N / 16;
    const int k_tiles = k_len / 16;
    const int k_groups = (k_tiles + KT - 1) / KT;
    const size_t groups = size_t(n_tiles) * size_t(k_groups);
    const int wkn = wkn_force >= 0 ? wkn_force : exl3_wkn();
    stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 16), sycl::range<1>(16)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                recon_xe2_kernel<bits>(B, W, K, N, packed_n, k0, k_len, dest_ld, wkn, it);
            });
}

void launch_recon_xe2_bits(int bits, const uint32_t * B, sycl::half * W,
                           int K, int N, int packed_n, queue_ptr stream,
                           int k0 = 0, int k_len = 0, int dest_ld = 0, int wkn_force = -1) {
#define EXL3_SYCL_RECON_XE2_CASE(b) \
    case b: launch_recon_xe2<b>(B, W, K, N, packed_n, stream, k0, k_len, dest_ld, wkn_force); break;
    switch (bits) {
        EXL3_SYCL_RECON_XE2_CASE(1)
        EXL3_SYCL_RECON_XE2_CASE(2)
        EXL3_SYCL_RECON_XE2_CASE(3)
        EXL3_SYCL_RECON_XE2_CASE(4)
        EXL3_SYCL_RECON_XE2_CASE(5)
        EXL3_SYCL_RECON_XE2_CASE(6)
        EXL3_SYCL_RECON_XE2_CASE(7)
        EXL3_SYCL_RECON_XE2_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_RECON_XE2_CASE
}

#define EXL3_HAD_ISQ 0.08838834764831843f

// 128-pt Sylvester FWHT. WG=16 is Xe2-native; WG=32 is the old fill.
template <int WG>
inline void fwht128_slm(float * slm, int lane, sycl::nd_item<1> it) {
    static_assert(WG == 16 || WG == 32, "fwht WG");
    constexpr int n = 128 / WG;
    for (int h = 1; h < 128; h *= 2) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            const int p = i * WG + lane;
            const int q = p ^ h;
            if (p < q) {
                const float a = slm[p];
                const float b = slm[q];
                slm[p] = a + b;
                slm[q] = a - b;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

template <int WG>
inline void had_in_row(const float * x, const float * suh, float * slm,
                       int size_k, int size_m, int row, int k0,
                       int lane, sycl::nd_item<1> it) {
    static_assert(WG == 16 || WG == 32, "had WG");
    constexpr int n = 128 / WG;
#pragma unroll
    for (int i = 0; i < n; ++i) {
        const int col = i * WG + lane;
        float v = 0.0f;
        if (row < size_m) {
            v = x[size_t(row) * size_t(size_k) + size_t(k0) + col];
            if (suh) {
                v *= suh[k0 + col];
            }
        }
        slm[col] = v * EXL3_HAD_ISQ;
    }
    it.barrier(sycl::access::fence_space::local_space);
    fwht128_slm<WG>(slm, lane, it);
}

void post_had_svh_kernel(float * C, const float * svh, int size_n, int size_m,
                         sycl::nd_item<1> it, float * slm) {
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int nblk = size_n / 128;
    const int row = gid / nblk;
    const int nb  = gid % nblk;
    if (row >= size_m) {
        return;
    }
    float * dst = C + size_t(row) * size_t(size_n) + size_t(nb) * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        slm[i * 32 + lane] = dst[i * 32 + lane] * EXL3_HAD_ISQ;
    }
    it.barrier(sycl::access::fence_space::local_space);
    fwht128_slm<32>(slm, lane, it);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int col = i * 32 + lane;
        float v = slm[col];
        if (svh) {
            v *= svh[nb * 128 + col];
        }
        dst[col] = v;
    }
}

void launch_post_had_svh(float * C, const float * svh, int size_n, int size_m, queue_ptr stream) {
    const size_t groups = size_t(size_n / 128) * size_t(size_m);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(128), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(groups * 32), sycl::range<1>(32)),
                [=](sycl::nd_item<1> it) {
                    post_had_svh_kernel(C, svh, size_n, size_m, it, get_pointer(slm));
                });
    });
}

void had_in_x_kernel(const float * x, const float * suh, float * dst,
                     int size_k, int size_m, sycl::nd_item<1> it, float * slm) {
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int nblk = size_k / 128;
    const int row = gid / nblk;
    const int k0  = (gid % nblk) * 128;
    if (row >= size_m) {
        return;
    }
    had_in_row<32>(x, suh, slm, size_k, size_m, row, k0, lane, it);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[size_t(row) * size_t(size_k) + size_t(k0) + i * 32 + lane] = slm[i * 32 + lane];
    }
}

void launch_had_in_x(const float * x, const float * suh, float * dst,
                     int size_k, int size_m, queue_ptr stream) {
    const size_t groups = size_t(size_m) * size_t(size_k / 128);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(128), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(groups * 32), sycl::range<1>(32)),
                [=](sycl::nd_item<1> it) {
                    had_in_x_kernel(x, suh, dst, size_k, size_m, it, get_pointer(slm));
                });
    });
}

inline void exl3_fwht8(float a[8], const sycl::nd_item<2> & item) {
    constexpr int SG = 16;
    constexpr int EL = 8;
    const sycl::sub_group sg = item.get_sub_group();
    const int lane = int(sg.get_local_linear_id());
#pragma unroll
    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float x = a[j + k];
                const float y = a[j + k + h];
                a[j + k] = x + y;
                a[j + k + h] = x - y;
            }
        }
    }
#pragma unroll
    for (int h = 1; h < SG; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; ++j) {
            const float a2 = dpct::permute_sub_group_by_xor(sg, a[j], h, SG);
            a[j] = (lane & h) == 0 ? a[j] + a2 : a2 - a[j];
        }
    }
}

inline void exl3_fwht8x2(float a[8], float b[8], const sycl::nd_item<2> & item) {
    constexpr int SG = 16;
    constexpr int EL = 8;
    const sycl::sub_group sg = item.get_sub_group();
    const int lane = int(sg.get_local_linear_id());
#pragma unroll
    for (int h = 1; h < EL; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; j += 2 * h) {
#pragma unroll
            for (int k = 0; k < h; ++k) {
                const float ax = a[j + k];
                const float ay = a[j + k + h];
                a[j + k] = ax + ay;
                a[j + k + h] = ax - ay;
                const float bx = b[j + k];
                const float by = b[j + k + h];
                b[j + k] = bx + by;
                b[j + k + h] = bx - by;
            }
        }
    }
#pragma unroll
    for (int h = 1; h < SG; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; ++j) {
            const float a2 = dpct::permute_sub_group_by_xor(sg, a[j], h, SG);
            const float b2 = dpct::permute_sub_group_by_xor(sg, b[j], h, SG);
            a[j] = (lane & h) == 0 ? a[j] + a2 : a2 - a[j];
            b[j] = (lane & h) == 0 ? b[j] + b2 : b2 - b[j];
        }
    }
}

inline void exl3_store_half8(sycl::half * d, const float a[8]) {
    sycl::vec<sycl::half, 8> hv;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        hv[i] = sycl::half(a[i]);
    }
    *reinterpret_cast<sycl::vec<sycl::half, 8> *>(d) = hv;
}

inline void exl3_ffn_mid_body(float g[8], float u[8], const float sg8[8], const float su8[8],
                              const float sd8[8], const sycl::nd_item<2> & item) {
    constexpr int EL = 8;
    exl3_fwht8x2(g, u, item);
#pragma unroll
    for (int i = 0; i < EL; ++i) {
        const float gv = g[i] * sg8[i];
        const float uv = u[i] * su8[i];
        const float silu = gv / (1.f + sycl::exp(-gv));
        g[i] = silu * uv * sd8[i] * EXL3_HAD_ISQ;
    }
    exl3_fwht8(g, item);
}

inline void exl3_load8(const float * p, float a[8], float scale) {
    const sycl::vec<float, 4> lo = *reinterpret_cast<const sycl::vec<float, 4> *>(p);
    const sycl::vec<float, 4> hi = *reinterpret_cast<const sycl::vec<float, 4> *>(p + 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = lo[i] * scale;
        a[i + 4] = hi[i] * scale;
    }
}

inline void exl3_load8_scale(const float * p, float a[8]) {
    const sycl::vec<float, 4> lo = *reinterpret_cast<const sycl::vec<float, 4> *>(p);
    const sycl::vec<float, 4> hi = *reinterpret_cast<const sycl::vec<float, 4> *>(p + 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = lo[i];
        a[i + 4] = hi[i];
    }
}

template <typename Dst>
void exl3_fwht128_vec8(const float * src, Dst * dst, const int64_t row,
                       const float scale, const float * sin8, const float * sout8,
                       const sycl::nd_item<2> & item) {
    constexpr int EL = 8;
    const int lane = int(item.get_sub_group().get_local_linear_id());
    const int base = lane * EL;
    float a[EL];
    exl3_load8(src + row * 128 + base, a, scale);
    if (sin8) {
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            a[i] *= sin8[i];
        }
    }
    exl3_fwht8(a, item);
    if (sout8) {
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            a[i] *= sout8[i];
        }
    }
    Dst * d = dst + row * 128 + base;
    if constexpr (std::is_same<Dst, float>::value) {
        sycl::vec<float, 4> olo, ohi;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            olo[i] = a[i];
            ohi[i] = a[i + 4];
        }
        *reinterpret_cast<sycl::vec<float, 4> *>(d) = olo;
        *reinterpret_cast<sycl::vec<float, 4> *>(d + 4) = ohi;
    } else {
        sycl::vec<sycl::half, 8> hv;
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            hv[i] = sycl::half(a[i]);
        }
        *reinterpret_cast<sycl::vec<sycl::half, 8> *>(d) = hv;
    }
}

void launch_exl3_had_to_f16(const float * src, sycl::half * dst, const float * suh,
                            int64_t n_rows, int64_t kin, queue_ptr stream) {
    constexpr int SG = 16;
    constexpr int RPB = 16;
    const int64_t num_blocks = (n_rows + RPB - 1) / RPB;
    stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_blocks * RPB, SG), sycl::range<2>(RPB, SG)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                const int lane = int(item.get_sub_group().get_local_linear_id());
                const int64_t row = int64_t(item.get_global_id(0));
                if (row >= n_rows) {
                    return;
                }
                float sin_reg[8];
                const float * sin8 = nullptr;
                if (suh && kin > 0) {
                    const int64_t off = (row * 128) % kin;
                    const float * sp = suh + off + lane * 8;
                    const sycl::vec<float, 4> slo = *reinterpret_cast<const sycl::vec<float, 4> *>(sp);
                    const sycl::vec<float, 4> shi = *reinterpret_cast<const sycl::vec<float, 4> *>(sp + 4);
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sin_reg[i] = slo[i];
                        sin_reg[i + 4] = shi[i];
                    }
                    sin8 = sin_reg;
                }
                exl3_fwht128_vec8<sycl::half>(src, dst, row, EXL3_HAD_ISQ, sin8, nullptr, item);
            });
}

void launch_exl3_had_svh(float * io, const float * svh, int64_t n_rows, int64_t kout,
                         queue_ptr stream, const float * add_src, float * add_dst) {
    constexpr int SG = 16;
    constexpr int RPB = 16;
    constexpr int EL = 8;
    const int64_t num_blocks = (n_rows + RPB - 1) / RPB;
    const int64_t n_blk = kout > 0 ? kout / 128 : 1;
    stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_blocks * RPB, SG), sycl::range<2>(RPB, SG)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                const int lane = int(item.get_sub_group().get_local_linear_id());
                const int64_t row = int64_t(item.get_global_id(0));
                if (row >= n_rows) {
                    return;
                }
                const int base = lane * EL;
                const int64_t off = (row % n_blk) * 128;
                float sout_reg[8];
                const float * sout8 = nullptr;
                if (svh && kout > 0) {
                    exl3_load8_scale(svh + off + base, sout_reg);
                    sout8 = sout_reg;
                }
                if (!add_src) {
                    exl3_fwht128_vec8<float>(io, io, row, EXL3_HAD_ISQ, nullptr, sout8, item);
                    return;
                }
                float a[8];
                exl3_load8(io + row * 128 + base, a, EXL3_HAD_ISQ);
                exl3_fwht8(a, item);
                float r[8];
                exl3_load8_scale(add_src + row * 128 + base, r);
                {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        const float s = a[i] * (sout8 ? sout8[i] : 1.f);
                        a[i] = s + r[i];
                    }
                }
                float * d = (add_dst ? add_dst : io) + row * 128 + base;
                sycl::vec<float, 4> olo, ohi;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    olo[i] = a[i];
                    ohi[i] = a[i + 4];
                }
                *reinterpret_cast<sycl::vec<float, 4> *>(d) = olo;
                *reinterpret_cast<sycl::vec<float, 4> *>(d + 4) = ohi;
            });
}

void launch_exl3_ffn_mid(const float * gate, const float * up,
                         const float * svh_g, const float * svh_u, const float * suh_d,
                         sycl::half * dst, int64_t n_rows, int64_t kn, queue_ptr stream) {
    constexpr int SG = 16;
    constexpr int EL = 8;
    const int v2 = ggml_sycl_get_env("GGML_SYCL_EXL3_MID_V2", 1);
    const int64_t n_blk = kn / 128;
    const int64_t n_tok = n_blk > 0 ? n_rows / n_blk : n_rows;

    if (v2 == 0) {
        constexpr int RPB = 16;
        const int64_t num_blocks = (n_rows + RPB - 1) / RPB;
        stream->parallel_for(
                sycl::nd_range<2>(sycl::range<2>(num_blocks * RPB, SG), sycl::range<2>(RPB, SG)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                    const int lane = int(item.get_sub_group().get_local_linear_id());
                    const int64_t row = int64_t(item.get_global_id(0));
                    if (row >= n_rows) {
                        return;
                    }
                    const int base = lane * EL;
                    const int64_t off = (row * 128) % kn;
                    float sg8[8];
                    float su8[8];
                    float sd8[8];
                    exl3_load8_scale(svh_g + off + base, sg8);
                    exl3_load8_scale(svh_u + off + base, su8);
                    exl3_load8_scale(suh_d + off + base, sd8);
                    float g[8];
                    float u[8];
                    exl3_load8(gate + row * 128 + base, g, EXL3_HAD_ISQ);
                    exl3_load8(up + row * 128 + base, u, EXL3_HAD_ISQ);
                    exl3_fwht8(g, item);
                    exl3_fwht8(u, item);
#pragma unroll
                    for (int i = 0; i < EL; ++i) {
                        const float gv = g[i] * sg8[i];
                        const float uv = u[i] * su8[i];
                        const float silu = gv / (1.f + sycl::exp(-gv));
                        g[i] = silu * uv * sd8[i] * EXL3_HAD_ISQ;
                    }
                    exl3_fwht8(g, item);
                    exl3_store_half8(dst + row * 128 + base, g);
                });
        return;
    }

    if (v2 == 2) {
        const int64_t tok_pad = (n_tok + 7) / 8 * 8;
        stream->parallel_for(
                sycl::nd_range<2>(sycl::range<2>(tok_pad, n_blk * SG), sycl::range<2>(8, SG)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                    const int lane = int(item.get_sub_group().get_local_linear_id());
                    const int64_t tok = int64_t(item.get_global_id(0));
                    const int64_t blk = int64_t(item.get_group(1));
                    if (tok >= n_tok) {
                        return;
                    }
                    const int base = lane * EL;
                    const int64_t off = blk * 128;
                    float sg8[8];
                    float su8[8];
                    float sd8[8];
                    exl3_load8_scale(svh_g + off + base, sg8);
                    exl3_load8_scale(svh_u + off + base, su8);
                    exl3_load8_scale(suh_d + off + base, sd8);
                    const int64_t e = tok * kn + off + base;
                    float g[8];
                    float u[8];
                    exl3_load8(gate + e, g, EXL3_HAD_ISQ);
                    exl3_load8(up + e, u, EXL3_HAD_ISQ);
                    exl3_ffn_mid_body(g, u, sg8, su8, sd8, item);
                    exl3_store_half8(dst + e, g);
                });
        return;
    }

    if (v2 == 3) {
        stream->parallel_for(
                sycl::nd_range<2>(sycl::range<2>(n_tok, n_blk * SG), sycl::range<2>(1, SG)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                    const int lane = int(item.get_sub_group().get_local_linear_id());
                    const int64_t tok = int64_t(item.get_global_id(0));
                    const int64_t blk = int64_t(item.get_group(1));
                    const int base = lane * EL;
                    const int64_t off = blk * 128;
                    float sg8[8];
                    float su8[8];
                    float sd8[8];
                    exl3_load8_scale(svh_g + off + base, sg8);
                    exl3_load8_scale(svh_u + off + base, su8);
                    exl3_load8_scale(suh_d + off + base, sd8);
                    const int64_t e = tok * kn + off + base;
                    float g[8];
                    float u[8];
                    exl3_load8(gate + e, g, EXL3_HAD_ISQ);
                    exl3_load8(up + e, u, EXL3_HAD_ISQ);
                    exl3_ffn_mid_body(g, u, sg8, su8, sd8, item);
                    exl3_store_half8(dst + e, g);
                });
        return;
    }

    constexpr int RPB = 8;
    const int64_t num_blocks = (n_rows + RPB - 1) / RPB;
    stream->parallel_for(
            sycl::nd_range<2>(sycl::range<2>(num_blocks * RPB, SG), sycl::range<2>(RPB, SG)),
            [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                const int lane = int(item.get_sub_group().get_local_linear_id());
                const int64_t row = int64_t(item.get_global_id(0));
                if (row >= n_rows) {
                    return;
                }
                const int base = lane * EL;
                const int64_t off = (row % n_blk) * 128;
                float sg8[8];
                float su8[8];
                float sd8[8];
                exl3_load8_scale(svh_g + off + base, sg8);
                exl3_load8_scale(svh_u + off + base, su8);
                exl3_load8_scale(suh_d + off + base, sd8);
                float g[8];
                float u[8];
                exl3_load8(gate + row * 128 + base, g, EXL3_HAD_ISQ);
                exl3_load8(up + row * 128 + base, u, EXL3_HAD_ISQ);
                exl3_ffn_mid_body(g, u, sg8, su8, sd8, item);
                exl3_store_half8(dst + row * 128 + base, g);
            });
}

// One q_s / xsum per (row, 16-wide K). WG=16 matches Xe2 subgroup.
void quant_x_i8_kernel(const float * x, int8_t * x_i8, float * qs, int * xsum,
                       int size_k, int size_m, sycl::nd_item<1> it) {
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int nkt = size_k / 16;
    const int row = gid / nkt;
    const int kt  = gid % nkt;
    if (row >= size_m) {
        return;
    }
    const float v  = x[size_t(row) * size_t(size_k) + size_t(kt) * 16 + lane];
    const float mx = sycl::reduce_over_group(it.get_group(), sycl::fabs(v), sycl::maximum<float>());
    const float q_s = (mx < 1e-8f ? 1e-8f : mx) / 127.0f;
    int iv = int(sycl::rint(v / q_s));
    iv = iv > 127 ? 127 : (iv < -127 ? -127 : iv);
    x_i8[size_t(row) * size_t(size_k) + size_t(kt) * 16 + lane] = int8_t(iv);
    const int s = sycl::reduce_over_group(it.get_group(), iv, sycl::plus<int>());
    if (lane == 0) {
        qs[row * nkt + kt]   = q_s;
        xsum[row * nkt + kt] = s;
    }
}

void launch_quant_x_i8(const float * x, int8_t * x_i8, float * qs, int * xsum,
                       int size_k, int size_m, queue_ptr stream) {
    const int nkt = size_k / 16;
    const size_t groups = size_t(size_m) * size_t(nkt);
    stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(groups * 16), sycl::range<1>(16)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                quant_x_i8_kernel(x, x_i8, qs, xsum, size_k, size_m, it);
            });
}

// Integer MAC, x reused across NT tiles.
template <int bits, int ROWS, int NT>
void gemv_kernel_i8n(const int8_t * x_i8, const float * qs, const int * xsum,
                     const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n,
                     sycl::nd_item<1> it,
                     uint32_t * sh_pack, int * sh_x, float * sh_tile) {
    constexpr int WG = 32;
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;

    int   ia0[NT][ROWS];
    int   ia1[NT][ROWS];
    float fa0[NT][ROWS];
    float fa1[NT][ROWS];
    float corr[ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            fa0[t][r] = 0.0f;
            fa1[t][r] = 0.0f;
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        corr[r] = 0.0f;
    }

    auto mac_lane = [&](const uint32_t * pack, int * a0, int * a1) {
        uint32_t w[8];
        extract8<bits>(pack, lane * 8, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
            int * dest = (j >= 4) ? a1 : a0;
            const int bs = hash_byte_sum(w[j]);
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                dest[r] += bs * sh_x[r * 16 + row];
            }
        }
    };

    float qsl[ROWS];
    for (int kt = 0; kt < kslices; ++kt) {
        if (lane < 16) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int v = 0;
                if (r < size_m) {
                    v = int(x_i8[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane]);
                }
                sh_x[r * 16 + lane] = v;
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            qsl[r] = 0.0f;
            if (r < size_m) {
                qsl[r] = qs[r * kslices + kt];
                corr[r] += qsl[r] * float(xsum[r * kslices + kt]);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                ia0[t][r] = 0;
                ia1[t][r] = 0;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int t = 0; t < n_here; ++t) {
            for (int i = lane; i < pack_u32; i += WG) {
                sh_pack[i] = B[(size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32) + i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            mac_lane(sh_pack, ia0[t], ia1[t]);
            it.barrier(sycl::access::fence_space::local_space);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                fa0[t][r] += qsl[r] * float(ia0[t][r]);
                fa1[t][r] += qsl[r] * float(ia1[t][r]);
            }
        }
    }

    const float kinv  = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float kbias = float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    const float aff   = 1024.0f * kinv + kbias;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sh_tile[lane] = fa0[t][r];
            sh_tile[32 + lane] = fa1[t][r];
            it.barrier(sycl::access::fence_space::local_space);
            if (lane < 8 && r < size_m && t < n_here) {
                const int g = lane * 4;
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    s0 += sh_tile[g + u];
                    s1 += sh_tile[32 + g + u];
                }
                const int nbase = (n0 + t) * 16;
                float * rowp = C + size_t(r) * size_t(size_n);
                rowp[nbase + lane]     = kinv * s0 + aff * corr[r];
                rowp[nbase + 8 + lane] = kinv * s1 + aff * corr[r];
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
    }
}

template <int bits, int ROWS, int NT>
void launch_gemv_i8n(const int8_t * x_i8, const float * qs, const int * xsum,
                     const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n, queue_ptr stream) {
    constexpr int pack_u32 = bits * 8;
    const int n_tiles = size_n / 16;
    const int n_groups_n = (n_tiles + NT - 1) / NT;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(pack_u32), cgh);
        sycl::local_accessor<int, 1>      sh_x(sycl::range<1>(ROWS * 16), cgh);
        sycl::local_accessor<float, 1>    sh_tile(sycl::range<1>(64), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(n_groups_n) * 32), sycl::range<1>(32)),
                [=](sycl::nd_item<1> it) {
                    gemv_kernel_i8n<bits, ROWS, NT>(x_i8, qs, xsum, B, C,
                            size_m, size_k, size_n, packed_n, it,
                            get_pointer(sh_pack), get_pointer(sh_x), get_pointer(sh_tile));
                });
    });
}

void launch_gemv_i8n_bits(int bits, const int8_t * x_i8, const float * qs, const int * xsum,
                          const uint32_t * B, float * C,
                          int size_m, int size_k, int size_n, int packed_n, queue_ptr stream) {
    const int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_NT", 4);
#define EXL3_SYCL_I8N_CASE(b) \
    case b: \
        if (size_m <= 1) { \
            if (nt >= 8) { \
                launch_gemv_i8n<b, 1, 8>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
            } else { \
                launch_gemv_i8n<b, 1, 4>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
            } \
        } else if (size_m <= 2) { \
            if (nt >= 8) { \
                launch_gemv_i8n<b, 2, 8>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
            } else { \
                launch_gemv_i8n<b, 2, 4>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
            } \
        } else if (nt >= 4) { \
            launch_gemv_i8n<b, EXL3_GEMV_MAX_M, 4>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
        } else { \
            launch_gemv_i8n<b, EXL3_GEMV_MAX_M, 2>(x_i8, qs, xsum, B, C, size_m, size_k, size_n, packed_n, stream); \
        } \
        break;
    switch (bits) {
        EXL3_SYCL_I8N_CASE(1)
        EXL3_SYCL_I8N_CASE(2)
        EXL3_SYCL_I8N_CASE(3)
        EXL3_SYCL_I8N_CASE(4)
        EXL3_SYCL_I8N_CASE(5)
        EXL3_SYCL_I8N_CASE(6)
        EXL3_SYCL_I8N_CASE(7)
        EXL3_SYCL_I8N_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_I8N_CASE
}

bool exl3_i8n_want() {
    return ggml_sycl_get_env("GGML_SYCL_EXL3_I8N", 0) != 0;
}

bool exl3_xmx_want() {
#if defined(__INTEL_LLVM_COMPILER)
    if (!g_ggml_sycl_enable_esimd) {
        return false;
    }
    return ggml_sycl_get_env("GGML_SYCL_EXL3_XMX", 0) != 0;
#else
    return false;
#endif
}

bool exl3_i8_want() {
#if defined(__INTEL_LLVM_COMPILER)
    if (!g_ggml_sycl_enable_esimd) {
        return false;
    }
    if (exl3_xmx_want()) {
        return false;
    }
    return ggml_sycl_get_env("GGML_SYCL_EXL3_I8", 0) != 0;
#else
    return false;
#endif
}

bool exl3_esimd_want() {
#if defined(__INTEL_LLVM_COMPILER)
    if (!g_ggml_sycl_enable_esimd) {
        return false;
    }
    if (exl3_xmx_want() || exl3_i8_want()) {
        return false;
    }
    return ggml_sycl_get_env("GGML_SYCL_EXL3_ESIMD", 0) != 0;
#else
    return false;
#endif
}

bool exl3_vec_want() {
#if defined(__INTEL_LLVM_COMPILER)
    if (!g_ggml_sycl_enable_esimd) {
        return false;
    }
    if (exl3_xmx_want() || exl3_i8_want() || exl3_esimd_want()) {
        return false;
    }
    return ggml_sycl_get_env("GGML_SYCL_EXL3_VEC", 0) != 0;
#else
    return false;
#endif
}

template <int bits, int WG>
inline void decode_tile_wg(const uint32_t * pack, int lane, float * tile) {
    static_assert(WG == 16 || WG == 32, "decode WG");
    constexpr int reps = 32 / WG;
#pragma unroll
    for (int t = 0; t < reps; ++t) {
        decode_tile<bits>(pack, lane + t * WG, tile);
    }
}

template <int bits, int ROWS, int WG, bool USE_I8>
void gemv_kernel(const float * x, const uint32_t * B, float * C,
                 int size_m, int size_k, int size_n, int packed_n,
                 const float * suh, int do_had, int n_kparts,
                 sycl::nd_item<1> it,
                 uint32_t * sh_pack, float * sh_tile, float * sh_x, float * sh_had) {
    static_assert(WG == 16 || WG == 32, "gemv WG");
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int gid = int(it.get_group(0));
    const int n_tile = gid % n_dest_tiles;
    const int k_part = gid / n_dest_tiles;
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;

    float acc0[ROWS];
    float acc1[ROWS];
    float corr[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc0[r] = 0.0f;
        acc1[r] = 0.0f;
        corr[r] = 0.0f;
    }

    auto fma_lane = [&](int L, const uint32_t * pack, const float * x16, float q_s) {
        uint32_t w[8];
        extract8<bits>(pack, L * 8, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (L % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
            float * dest = (j >= 4) ? acc1 : acc0;
            if constexpr (USE_I8) {
                const float bs = float(hash_byte_sum(w[j]));
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    dest[r] += bs * x16[r * 16 + row] * q_s;
                }
            } else {
                const float val = float(decode_mul1(w[j]));
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    dest[r] += val * x16[r * 16 + row];
                }
            }
        }
    };

    auto consume_tile = [&](int kt, const float * x16) {
        for (int i = lane; i < pack_u32; i += WG) {
            sh_pack[i] = B[(size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32) + i];
        }
        float q_s = 1.0f;
        if constexpr (USE_I8) {
            float mx = 0.0f;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    const float a = x16[r * 16 + k];
                    const float aa = a < 0.0f ? -a : a;
                    mx = mx < aa ? aa : mx;
                }
            }
            q_s = (mx < 1e-8f ? 1e-8f : mx) / 127.0f;
            const float rq = 1.0f / q_s;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    int v = int(sycl::rint(x16[r * 16 + k] * rq));
                    v = v > 127 ? 127 : (v < -127 ? -127 : v);
                    s += float(v);
                    if (lane < 16 && k == lane) {
                        sh_x[r * 16 + k] = float(v);
                    }
                }
                corr[r] += q_s * s;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        const float * xw = USE_I8 ? sh_x : x16;
        fma_lane(lane, sh_pack, xw, q_s);
        if constexpr (WG == 16) {
            fma_lane(lane + 16, sh_pack, xw, q_s);
        }
        it.barrier(sycl::access::fence_space::local_space);
    };

    if (do_had) {
        const int nblk = size_k / 128;
        const int chunk = (nblk + n_kparts - 1) / n_kparts;
        const int b0 = k_part * chunk;
        const int b1 = sycl::min(nblk, b0 + chunk);
        for (int kb = b0; kb < b1; ++kb) {
            const int k0 = kb * 128;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                had_in_row<WG>(x, suh, sh_had + r * 128, size_k, size_m, r, k0, lane, it);
            }
            for (int t = 0; t < 8; ++t) {
                if (lane < 16) {
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        sh_x[r * 16 + lane] = sh_had[r * 128 + t * 16 + lane];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                consume_tile(k0 / 16 + t, sh_x);
            }
        }
    } else if constexpr (USE_I8) {
        const int chunk = (kslices + n_kparts - 1) / n_kparts;
        const int kt0 = k_part * chunk;
        const int kt1 = sycl::min(kslices, kt0 + chunk);
        for (int kt = kt0; kt < kt1; ++kt) {
            if (lane < 16) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float v = 0.0f;
                    if (r < size_m) {
                        v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
                    }
                    sh_x[r * 16 + lane] = v;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            consume_tile(kt, sh_x);
        }
    } else {
        const int chunk = (kslices + n_kparts - 1) / n_kparts;
        const int kt0 = k_part * chunk;
        const int kt1 = sycl::min(kslices, kt0 + chunk);
        auto load_kx = [&](int kt, int slot) {
            uint32_t * pk = sh_pack + slot * pack_u32;
            float * xsl = sh_x + slot * ROWS * 16;
            for (int i = lane; i < pack_u32; i += WG) {
                pk[i] = B[(size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32) + i];
            }
            if (lane < 16) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float v = 0.0f;
                    if (r < size_m) {
                        v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
                    }
                    xsl[r * 16 + lane] = v;
                }
            }
        };
        if (kt0 < kt1) {
            load_kx(kt0, 0);
            it.barrier(sycl::access::fence_space::local_space);
            for (int kt = kt0; kt < kt1; ++kt) {
                const int cur = (kt - kt0) & 1;
                const int nxt = cur ^ 1;
                if (kt + 1 < kt1) {
                    load_kx(kt + 1, nxt);
                }
                fma_lane(lane, sh_pack + cur * pack_u32, sh_x + cur * ROWS * 16, 1.0f);
                if constexpr (WG == 16) {
                    fma_lane(lane + 16, sh_pack + cur * pack_u32, sh_x + cur * ROWS * 16, 1.0f);
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    const float kinv  = float(sycl::bit_cast<sycl::half>(uint16_t(0x1eee)));
    const float kbias = float(sycl::bit_cast<sycl::half>(uint16_t(0xc931)));
    const float aff   = 1024.0f * kinv + kbias;
    float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        sh_tile[lane] = acc0[r];
        sh_tile[32 + lane] = acc1[r];
        it.barrier(sycl::access::fence_space::local_space);
        if (lane < 8 && r < size_m) {
            const int g = lane * (WG == 32 ? 4 : 2);
            float s0 = 0.0f;
            float s1 = 0.0f;
            const int nsum = WG == 32 ? 4 : 2;
#pragma unroll
            for (int t = 0; t < nsum; ++t) {
                s0 += sh_tile[g + t];
                s1 += sh_tile[32 + g + t];
            }
            if constexpr (USE_I8) {
                s0 = kinv * s0 + aff * corr[r];
                s1 = kinv * s1 + aff * corr[r];
            }
            const int n0 = n_tile * 16;
            base[size_t(r) * size_t(size_n) + n0 + lane] = s0;
            base[size_t(r) * size_t(size_n) + n0 + 8 + lane] = s1;
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

// Hoisted-H path: pack from global, x ping-pong in SLM, NT N-tiles share x.
template <int bits, int ROWS, int NT>
void gemv_kernel_glb(const float * x, const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n, int n_kparts,
                     const sycl::half * lut,
                     sycl::nd_item<1> it, float * sh_x, float * sh_tile, uint32_t * sh_pack) {
    constexpr int WG = 32;
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int k_part = gid / n_groups_n;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + n_kparts - 1) / n_kparts;
    const int kt0 = k_part * chunk;
    const int kt1 = sycl::min(kslices, kt0 + chunk);

    float acc0[NT][ROWS];
    float acc1[NT][ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc0[t][r] = 0.0f;
            acc1[t][r] = 0.0f;
        }
    }

    auto load_x = [&](int kt, int slot) {
        float * xsl = sh_x + slot * ROWS * 16;
        if (lane < 16) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float v = 0.0f;
                if (r < size_m) {
                    v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
                }
                xsl[r * 16 + lane] = v;
            }
        }
    };

    auto fma_glb = [&](int t, const float * x16, const uint32_t * pack) {
        uint32_t w[8];
        extract8<bits>(pack, lane * 8, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
            const float val = lut ? float(lut[w[j]]) : float(decode_mul1(w[j]));
            float * dest = (j >= 4) ? acc1[t] : acc0[t];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                dest[r] += val * x16[r * 16 + row];
            }
        }
    };

    if (kt0 < kt1) {
        load_x(kt0, 0);
        it.barrier(sycl::access::fence_space::local_space);
        for (int kt = kt0; kt < kt1; ++kt) {
            const int cur = (kt - kt0) & 1;
            const int nxt = cur ^ 1;
            if (kt + 1 < kt1) {
                load_x(kt + 1, nxt);
            }
            const float * x16 = sh_x + cur * ROWS * 16;
            for (int t = 0; t < n_here; ++t) {
                const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
                fma_glb(t, x16, pack);
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sh_tile[t * 64 + lane] = acc0[t][r];
            sh_tile[t * 64 + 32 + lane] = acc1[t][r];
        }
        it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (lane < 8 && r < size_m && t < n_here) {
                const int g = t * 64 + lane * 4;
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    s0 += sh_tile[g + u];
                    s1 += sh_tile[g + 32 + u];
                }
                const int nbase = (n0 + t) * 16;
                base[size_t(r) * size_t(size_n) + nbase + lane] = s0;
                base[size_t(r) * size_t(size_n) + nbase + 8 + lane] = s1;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

// WG=16 matches Xe2 SIMD. Each lane does MMA lane and lane+16.
template <int bits, int ROWS, int NT>
void gemv_kernel_glb16(const float * x, const uint32_t * B, float * C,
                       int size_m, int size_k, int size_n, int packed_n, int n_kparts,
                       sycl::nd_item<1> it, float * sh_x, float * sh_tile) {
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int k_part = gid / n_groups_n;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + n_kparts - 1) / n_kparts;
    const int kt0 = k_part * chunk;
    const int kt1 = sycl::min(kslices, kt0 + chunk);

    float acc0[NT][ROWS];
    float acc1[NT][ROWS];
    float acc0h[NT][ROWS];
    float acc1h[NT][ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc0[t][r] = 0.0f;
            acc1[t][r] = 0.0f;
            acc0h[t][r] = 0.0f;
            acc1h[t][r] = 0.0f;
        }
    }

    auto load_x = [&](int kt) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float v = 0.0f;
            if (r < size_m) {
                v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
            }
            sh_x[r * 16 + lane] = v;
        }
    };

    if (kt0 < kt1) {
        load_x(kt0);
        it.barrier(sycl::access::fence_space::local_space);
        for (int kt = kt0; kt < kt1; ++kt) {
            const float * x16 = sh_x;
            for (int t = 0; t < n_here; ++t) {
                const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
                uint32_t w[8];
                uint32_t wh[8];
                extract8<bits>(pack, lane * 8, w);
                extract8<bits>(pack, (lane + 16) * 8, wh);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
                    const float val = float(decode_mul1(w[j]));
                    const float valh = float(decode_mul1(wh[j]));
                    float * dest = (j >= 4) ? acc1[t] : acc0[t];
                    float * desth = (j >= 4) ? acc1h[t] : acc0h[t];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        dest[r] += val * x16[r * 16 + row];
                        desth[r] += valh * x16[r * 16 + row];
                    }
                }
            }
            if (kt + 1 < kt1) {
                it.barrier(sycl::access::fence_space::local_space);
                load_x(kt + 1);
                it.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sh_tile[lane] = acc0[t][r];
            sh_tile[16 + lane] = acc0h[t][r];
            sh_tile[32 + lane] = acc1[t][r];
            sh_tile[48 + lane] = acc1h[t][r];
            it.barrier(sycl::access::fence_space::local_space);
            if (lane < 8 && r < size_m && t < n_here) {
                const int g = lane * 4;
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    s0 += sh_tile[g + u];
                    s1 += sh_tile[32 + g + u];
                }
                const int nbase = (n0 + t) * 16;
                base[size_t(r) * size_t(size_n) + nbase + lane] = s0;
                base[size_t(r) * size_t(size_n) + nbase + 8 + lane] = s1;
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
    }
}

// Barrier-free K loop: subgroup-16 shuffle for x, pack from global.
template <int bits, int ROWS, int NT>
void gemv_kernel_shuf(const float * x, const uint32_t * B, float * C,
                      int size_m, int size_k, int size_n, int packed_n, int n_kparts,
                      sycl::nd_item<1> it, float * sh_tile) {
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const auto sg = it.get_sub_group();
    const int sl = int(sg.get_local_linear_id());
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int k_part = gid / n_groups_n;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;
    const int chunk = (kslices + n_kparts - 1) / n_kparts;
    const int kt0 = k_part * chunk;
    const int kt1 = sycl::min(kslices, kt0 + chunk);

    float acc0[NT][ROWS];
    float acc1[NT][ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc0[t][r] = 0.0f;
            acc1[t][r] = 0.0f;
        }
    }

    for (int kt = kt0; kt < kt1; ++kt) {
        float xr0[ROWS], xr1[ROWS], xr8[ROWS], xr9[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            float my = 0.0f;
            if (r < size_m) {
                my = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + sl];
            }
            const int q = lane & 3;
            xr0[r] = sycl::select_from_group(sg, my, q * 2);
            xr1[r] = sycl::select_from_group(sg, my, q * 2 + 1);
            xr8[r] = sycl::select_from_group(sg, my, q * 2 + 8);
            xr9[r] = sycl::select_from_group(sg, my, q * 2 + 9);
        }
        for (int t = 0; t < n_here; ++t) {
            const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
            uint32_t w[8];
            extract8<bits>(pack, lane * 8, w);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int jm = j & 3;
                const float val = float(decode_mul1(w[j]));
                float * dest = (j >= 4) ? acc1[t] : acc0[t];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const float xr = (jm == 0) ? xr0[r] : (jm == 1) ? xr1[r] : (jm == 2) ? xr8[r] : xr9[r];
                    dest[r] += val * xr;
                }
            }
        }
    }

    float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sh_tile[t * 64 + lane] = acc0[t][r];
            sh_tile[t * 64 + 32 + lane] = acc1[t][r];
        }
        it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (lane < 8 && r < size_m && t < n_here) {
                const int g = t * 64 + lane * 4;
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    s0 += sh_tile[g + u];
                    s1 += sh_tile[g + 32 + u];
                }
                const int nbase = (n0 + t) * 16;
                base[size_t(r) * size_t(size_n) + nbase + lane] = s0;
                base[size_t(r) * size_t(size_n) + nbase + 8 + lane] = s1;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

template <int bits, int ROWS>
void gemv_kernel_slm(const float * x, const uint32_t * B, float * C,
                     int size_m, int size_k, int size_n, int packed_n,
                     const float * suh, int do_had, int n_kparts,
                     sycl::nd_item<1> it,
                     uint32_t * sh_pack, float * sh_tile, float * sh_x, float * sh_had) {
    constexpr int WG = 16;
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int gid = int(it.get_group(0));
    const int n_tile = gid % n_dest_tiles;
    const int k_part = gid / n_dest_tiles;
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0.0f;
    }
    auto dot_tile = [&](int kt, const float * x16) {
        for (int i = lane; i < pack_u32; i += WG) {
            sh_pack[i] = B[(size_t(kt) * size_t(ntiles) + size_t(n_tile)) * size_t(pack_u32) + i];
        }
        it.barrier(sycl::access::fence_space::local_space);
        decode_tile_wg<bits, WG>(sh_pack, lane, sh_tile);
        it.barrier(sycl::access::fence_space::local_space);
        if (lane < 16) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float d = 0.0f;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    d += sh_tile[k * 16 + lane] * x16[r * 16 + k];
                }
                acc[r] += d;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    };
    if (do_had) {
        const int nblk = size_k / 128;
        const int chunk = (nblk + n_kparts - 1) / n_kparts;
        const int b0 = k_part * chunk;
        const int b1 = sycl::min(nblk, b0 + chunk);
        for (int kb = b0; kb < b1; ++kb) {
            const int k0 = kb * 128;
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                had_in_row<WG>(x, suh, sh_had + r * 128, size_k, size_m, r, k0, lane, it);
            }
            for (int t = 0; t < 8; ++t) {
                if (lane < 16) {
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        sh_x[r * 16 + lane] = sh_had[r * 128 + t * 16 + lane];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                dot_tile(k0 / 16 + t, sh_x);
            }
        }
    } else {
        const int chunk = (kslices + n_kparts - 1) / n_kparts;
        const int kt0 = k_part * chunk;
        const int kt1 = sycl::min(kslices, kt0 + chunk);
        for (int kt = kt0; kt < kt1; ++kt) {
            if (lane < 16) {
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    float v = 0.0f;
                    if (r < size_m) {
                        v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
                    }
                    sh_x[r * 16 + lane] = v;
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
            dot_tile(kt, sh_x);
        }
    }
    if (lane < 16) {
        const int n = n_tile * 16 + lane;
        float * base = C + (n_kparts > 1 ? size_t(k_part) * size_t(size_m) * size_t(size_n) : 0);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r < size_m) {
                base[size_t(r) * size_t(size_n) + n] = acc[r];
            }
        }
    }
}

template <int bits>
void recon_nk_kernel(const uint32_t * B, sycl::half * W, int K, int dest_n, int packed_n,
                     sycl::nd_item<1> it, uint32_t * sh_pack, float * sh_tile) {
    constexpr int WG = 16;
    constexpr int pack_u32 = bits * 8;
    constexpr int KT = 8;
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int dest_ntiles = dest_n / 16;
    const int packed_ntiles = packed_n / 16;
    const int k_tiles = K / 16;
    const int n_tile = gid % dest_ntiles;
    const int k0 = (gid / dest_ntiles) * KT;
    for (int t = 0; t < KT; ++t) {
        const int k_tile = k0 + t;
        if (k_tile >= k_tiles) {
            break;
        }
        for (int i = lane; i < pack_u32; i += WG) {
            sh_pack[i] = B[(size_t(k_tile) * size_t(packed_ntiles) + size_t(n_tile)) * size_t(pack_u32) + i];
        }
        it.barrier(sycl::access::fence_space::local_space);
        decode_tile_wg<bits, WG>(sh_pack, lane, sh_tile);
        it.barrier(sycl::access::fence_space::local_space);
        const int n = n_tile * 16 + lane;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            W[size_t(n) * size_t(K) + size_t(k_tile) * 16 + k] = sycl::half(sh_tile[k * 16 + lane]);
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
}

// F32 fused decode, WG=32, NT N-tiles share one x slice. Pack from global (no pack SLM).
template <int bits, int ROWS, int NT>
void gemv_kernel_f32n(const float * x, const uint32_t * B, float * C,
                      int size_m, int size_k, int size_n, int packed_n,
                      sycl::nd_item<1> it, float * sh_x, float * sh_tile) {
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int n_dest_tiles = size_n / 16;
    const int n_groups_n = (n_dest_tiles + NT - 1) / NT;
    const int gid = int(it.get_group(0));
    const int n0 = (gid % n_groups_n) * NT;
    const int n_here = sycl::min(NT, n_dest_tiles - n0);
    const int ntiles = packed_n / 16;
    const int kslices = size_k / 16;

    float acc0[NT][ROWS];
    float acc1[NT][ROWS];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc0[t][r] = 0.0f;
            acc1[t][r] = 0.0f;
        }
    }

    for (int kt = 0; kt < kslices; ++kt) {
        if (lane < 16) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                float v = 0.0f;
                if (r < size_m) {
                    v = x[size_t(r) * size_t(size_k) + size_t(kt) * 16 + lane];
                }
                sh_x[r * 16 + lane] = v;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
        for (int t = 0; t < n_here; ++t) {
            const uint32_t * pack = B + (size_t(kt) * size_t(ntiles) + size_t(n0 + t)) * size_t(pack_u32);
            uint32_t w[8];
            extract8<bits>(pack, lane * 8, w);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int row = (lane % 4) * 2 + ((j % 2) ? 1 : 0) + ((j % 4) >= 2 ? 8 : 0);
                const float val = float(decode_mul1(w[j]));
                float * dest = (j >= 4) ? acc1[t] : acc0[t];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    dest[r] += val * sh_x[r * 16 + row];
                }
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            sh_tile[lane] = acc0[t][r];
            sh_tile[32 + lane] = acc1[t][r];
            it.barrier(sycl::access::fence_space::local_space);
            if (lane < 8 && r < size_m && t < n_here) {
                const int g = lane * 4;
                float s0 = 0.0f;
                float s1 = 0.0f;
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    s0 += sh_tile[g + u];
                    s1 += sh_tile[32 + g + u];
                }
                const int nbase = (n0 + t) * 16;
                float * rowp = C + size_t(r) * size_t(size_n);
                rowp[nbase + lane]     = s0;
                rowp[nbase + 8 + lane] = s1;
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
    }
}

template <int bits, int ROWS, int NT>
void launch_gemv_f32n(const float * x, const uint32_t * B, float * C,
                      int size_m, int size_k, int size_n, int packed_n, queue_ptr stream) {
    const int n_tiles = size_n / 16;
    const int n_groups_n = (n_tiles + NT - 1) / NT;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sh_x(sycl::range<1>(ROWS * 16), cgh);
        sycl::local_accessor<float, 1> sh_tile(sycl::range<1>(64), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(size_t(n_groups_n) * 32), sycl::range<1>(32)),
                [=](sycl::nd_item<1> it) {
                    gemv_kernel_f32n<bits, ROWS, NT>(x, B, C, size_m, size_k, size_n, packed_n, it,
                                                     get_pointer(sh_x), get_pointer(sh_tile));
                });
    });
}

const sycl::half * exl3_codebook_lut(queue_ptr stream) {
    static thread_local sycl::half * d = nullptr;
    static thread_local sycl::queue * last = nullptr;
    if (d && last == stream) {
        return d;
    }
    if (d && last) {
        sycl::free(d, *last);
        d = nullptr;
    }
    std::vector<sycl::half> h(65536);
    const sycl::half k_inv  = sycl::bit_cast<sycl::half>(uint16_t(0x1eee));
    const sycl::half k_bias = sycl::bit_cast<sycl::half>(uint16_t(0xc931));
    for (int i = 0; i < 65536; ++i) {
        uint32_t x = uint32_t(i) * 0x83DCD12Du;
        const uint32_t sum = 0x6400u
                + (x & 0xffu) + ((x >> 8) & 0xffu)
                + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu);
        h[i] = sycl::fma(sycl::bit_cast<sycl::half>(uint16_t(sum)), k_inv, k_bias);
    }
    d = sycl::malloc_device<sycl::half>(65536, *stream);
    stream->memcpy(d, h.data(), 65536 * sizeof(sycl::half)).wait();
    last = stream;
    return d;
}

template <int bits, int ROWS>
void launch_gemv(const float * x, const uint32_t * B, float * C,
                 int size_m, int size_k, int size_n, int packed_n,
                 const float * suh, int do_had, queue_ptr stream) {
    constexpr int pack_u32 = bits * 8;
    const int n_tiles = size_n / 16;
    int n_kparts = ggml_sycl_get_env("GGML_SYCL_EXL3_KSPLIT", 8);
    if (n_kparts < 1) {
        n_kparts = 1;
    }
    if (n_kparts > 32) {
        n_kparts = 32;
    }
    if (do_had) {
        const int nblk = size_k / 128;
        if (n_kparts > nblk) {
            n_kparts = nblk > 0 ? nblk : 1;
        }
    } else {
        const int ks = size_k / 16;
        if (n_kparts > ks) {
            n_kparts = ks > 0 ? ks : 1;
        }
    }
    float * out = C;
    if (n_kparts > 1) {
        static thread_local float * parts = nullptr;
        static thread_local size_t cap = 0;
        static thread_local sycl::queue * last = nullptr;
        const size_t need = size_t(n_kparts) * size_t(size_m) * size_t(size_n);
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
    const int groups = n_tiles * n_kparts;
    const int wg16 = ggml_sycl_get_env("GGML_SYCL_EXL3_WG16", 0);
    const int use_i8 = ggml_sycl_get_env("GGML_SYCL_EXL3_SIMT_I8", 0);
    auto submit_gemv = [&](auto wg_tag, auto i8_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        constexpr bool I8 = decltype(i8_tag)::value;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(size_t(pack_u32) * 2), cgh);
            sycl::local_accessor<float, 1>    sh_tile(sycl::range<1>(64), cgh);
            sycl::local_accessor<float, 1>    sh_x(sycl::range<1>(ROWS * 32), cgh);
            sycl::local_accessor<float, 1>    sh_had(sycl::range<1>(do_had ? ROWS * 128 : 1), cgh);
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(size_t(groups) * WG), sycl::range<1>(WG)),
                    [=](sycl::nd_item<1> it) {
                        gemv_kernel<bits, ROWS, WG, I8>(x, B, out, size_m, size_k, size_n, packed_n, suh, do_had,
                                                        n_kparts, it,
                                                        get_pointer(sh_pack), get_pointer(sh_tile),
                                                        get_pointer(sh_x), get_pointer(sh_had));
                    });
        });
    };
    if (wg16) {
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(pack_u32), cgh);
            sycl::local_accessor<float, 1>    sh_tile(sycl::range<1>(256), cgh);
            sycl::local_accessor<float, 1>    sh_x(sycl::range<1>(ROWS * 16), cgh);
            sycl::local_accessor<float, 1>    sh_had(sycl::range<1>(ROWS * 128), cgh);
            cgh.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(size_t(groups) * 16), sycl::range<1>(16)),
                    [=](sycl::nd_item<1> it) {
                        gemv_kernel_slm<bits, ROWS>(x, B, out, size_m, size_k, size_n, packed_n, suh, do_had,
                                                    n_kparts, it,
                                                    get_pointer(sh_pack), get_pointer(sh_tile),
                                                    get_pointer(sh_x), get_pointer(sh_had));
                    });
        });
    } else if (use_i8) {
        submit_gemv(std::integral_constant<int, 32>{}, std::integral_constant<bool, true>{});
    } else if (do_had == 0 && ggml_sycl_get_env("GGML_SYCL_EXL3_F32N", 0) != 0) {
        const int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_NT", 4);
        if (nt >= 8) {
            launch_gemv_f32n<bits, ROWS, 8>(x, B, out, size_m, size_k, size_n, packed_n, stream);
        } else if (nt <= 1) {
            launch_gemv_f32n<bits, ROWS, 1>(x, B, out, size_m, size_k, size_n, packed_n, stream);
        } else {
            launch_gemv_f32n<bits, ROWS, 4>(x, B, out, size_m, size_k, size_n, packed_n, stream);
        }
    } else if (do_had == 0 && ggml_sycl_get_env("GGML_SYCL_EXL3_GLB", 1)) {
        const int nt = ggml_sycl_get_env("GGML_SYCL_EXL3_GLB_NT", 4); // 8 spills on Xe2
        const int nt_use = nt < 1 ? 1 : nt;
        const int n_groups_n = (n_tiles + nt_use - 1) / nt_use;
        const int g_glb = n_groups_n * n_kparts;
        const int use16 = ggml_sycl_get_env("GGML_SYCL_EXL3_GLB16", 0);
        const sycl::half * lut = ggml_sycl_get_env("GGML_SYCL_EXL3_LUT", 0) ? exl3_codebook_lut(stream) : nullptr;
        auto submit_glb16 = [&](auto nt_tag) {
            constexpr int NT = decltype(nt_tag)::value;
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> sh_x(sycl::range<1>(ROWS * 16), cgh);
                sycl::local_accessor<float, 1> sh_tile(sycl::range<1>(64), cgh);
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(size_t(g_glb) * 16), sycl::range<1>(16)),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                            gemv_kernel_glb16<bits, ROWS, NT>(x, B, out, size_m, size_k, size_n, packed_n,
                                                              n_kparts, it, get_pointer(sh_x), get_pointer(sh_tile));
                        });
            });
        };
        auto submit_glb = [&](auto nt_tag) {
            constexpr int NT = decltype(nt_tag)::value;
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> sh_x(sycl::range<1>(ROWS * 32), cgh);
                sycl::local_accessor<float, 1> sh_tile(sycl::range<1>(size_t(NT) * 64), cgh);
                sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(size_t(2 * NT * pack_u32)), cgh);
                cgh.parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(size_t(g_glb) * 32), sycl::range<1>(32)),
                        [=](sycl::nd_item<1> it) {
                            gemv_kernel_glb<bits, ROWS, NT>(x, B, out, size_m, size_k, size_n, packed_n,
                                                            n_kparts, lut, it, get_pointer(sh_x), get_pointer(sh_tile),
                                                            get_pointer(sh_pack));
                        });
            });
        };
        auto pick_nt = [&](auto submit) {
            if (nt >= 8) {
                submit(std::integral_constant<int, 8>{});
            } else if (nt >= 4) {
                submit(std::integral_constant<int, 4>{});
            } else if (nt <= 1) {
                submit(std::integral_constant<int, 1>{});
            } else {
                submit(std::integral_constant<int, 2>{});
            }
        };
        if (use16) {
            pick_nt(submit_glb16);
        } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_SHUF", 0)) {
            auto submit_shuf = [&](auto nt_tag) {
                constexpr int NT = decltype(nt_tag)::value;
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> sh_tile(sycl::range<1>(size_t(NT) * 64), cgh);
                    cgh.parallel_for(
                            sycl::nd_range<1>(sycl::range<1>(size_t(g_glb) * 32), sycl::range<1>(32)),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                                gemv_kernel_shuf<bits, ROWS, NT>(x, B, out, size_m, size_k, size_n, packed_n,
                                                                 n_kparts, it, get_pointer(sh_tile));
                            });
                });
            };
            pick_nt(submit_shuf);
        } else {
            pick_nt(submit_glb);
        }
    } else {
        submit_gemv(std::integral_constant<int, 32>{}, std::integral_constant<bool, false>{});
    }
    if (n_kparts > 1) {
        const int np = n_kparts;
        const float * parts = out;
        const size_t mn = size_t(size_m) * size_t(size_n);
        stream->parallel_for(sycl::range<1>(mn), [=](sycl::id<1> i) {
            float s = 0.f;
            for (int p = 0; p < np; ++p) {
                s += parts[size_t(p) * mn + i];
            }
            C[i] = s;
        });
    }
}

template <int bits>
void launch_recon(const uint32_t * B, sycl::half * W, int K, int N, int packed_n, queue_ptr stream) {
    constexpr int pack_u32 = bits * 8;
    constexpr int KT = 8;
    const int n_tiles = N / 16;
    const int k_tiles = K / 16;
    const int k_groups = (k_tiles + KT - 1) / KT;
    const size_t groups = size_t(n_tiles) * size_t(k_groups);
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(pack_u32), cgh);
        sycl::local_accessor<float, 1>    sh_tile(sycl::range<1>(256), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(groups * 16), sycl::range<1>(16)),
                [=](sycl::nd_item<1> it) {
                    recon_nk_kernel<bits>(B, W, K, N, packed_n, it,
                                          get_pointer(sh_pack), get_pointer(sh_tile));
                });
    });
}

template <int bits>
void recon_had_tile_kernel(const uint32_t * B, sycl::half * W,
                           const float * suh, const float * svh,
                           int K, int dest_n, int packed_n, int xe2,
                           sycl::nd_item<1> it,
                           uint32_t * sh_pack, float * sh_16, float * tile, float * tmp) {
    constexpr int pack_u32 = bits * 8;
    const int lane = int(it.get_local_id(0));
    const int gid = int(it.get_group(0));
    const int dest_nblk = dest_n / 128;
    const int packed_ntiles = packed_n / 16;
    const int nb = gid % dest_nblk;
    const int kb = gid / dest_nblk;

    for (int tk = 0; tk < 8; ++tk) {
        for (int tn = 0; tn < 8; ++tn) {
            const int k_tile = kb * 8 + tk;
            const int n_tile = nb * 8 + tn;
            for (int i = lane; i < pack_u32; i += 32) {
                sh_pack[i] = B[(size_t(k_tile) * size_t(packed_ntiles) + size_t(n_tile)) * size_t(pack_u32) + i];
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (xe2) {
                if (lane < 16) {
#pragma unroll
                    for (int k = 0; k < 16; ++k) {
                        sh_16[k * 16 + lane] = float(decode_mul1(window16<bits>(sh_pack, lane * 16 + k)));
                    }
                }
            } else {
                decode_tile<bits>(sh_pack, lane, sh_16);
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (lane < 16) {
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    tile[(tk * 16 + k) * 128 + tn * 16 + lane] = sh_16[k * 16 + lane];
                }
            }
            it.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int k = 0; k < 128; ++k) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tile[k * 128 + i * 32 + lane] *= EXL3_HAD_ISQ;
        }
        it.barrier(sycl::access::fence_space::local_space);
        fwht128_slm<32>(tile + k * 128, lane, it);
    }
    for (int n = 0; n < 128; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tmp[i * 32 + lane] = tile[(i * 32 + lane) * 128 + n] * EXL3_HAD_ISQ;
        }
        it.barrier(sycl::access::fence_space::local_space);
        fwht128_slm<32>(tmp, lane, it);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tile[(i * 32 + lane) * 128 + n] = tmp[i * 32 + lane];
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = i * 32 + lane;
        const float su = suh[kb * 128 + k];
#pragma unroll
        for (int n = 0; n < 128; ++n) {
            const float v = tile[k * 128 + n] * su * svh[nb * 128 + n];
            W[size_t(nb * 128 + n) * size_t(K) + size_t(kb * 128 + k)] = sycl::half(v);
        }
    }
}

template <int bits>
void launch_recon_had(const uint32_t * B, sycl::half * W,
                      const float * suh, const float * svh,
                      int K, int N, int packed_n, queue_ptr stream) {
    constexpr int pack_u32 = bits * 8;
    const size_t groups = size_t(N / 128) * size_t(K / 128);
    const int xe2 = exl3_pack_xe2() ? 1 : 0;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint32_t, 1> sh_pack(sycl::range<1>(pack_u32), cgh);
        sycl::local_accessor<float, 1>    sh_16(sycl::range<1>(256), cgh);
        sycl::local_accessor<float, 1>    tile(sycl::range<1>(128 * 128), cgh);
        sycl::local_accessor<float, 1>    tmp(sycl::range<1>(128), cgh);
        cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(groups * 32), sycl::range<1>(32)),
                [=](sycl::nd_item<1> it) {
                    recon_had_tile_kernel<bits>(B, W, suh, svh, K, N, packed_n, xe2, it,
                            get_pointer(sh_pack), get_pointer(sh_16),
                            get_pointer(tile), get_pointer(tmp));
                });
    });
}

void launch_recon_had_bits(int bits, const uint32_t * B, sycl::half * W,
                           const float * suh, const float * svh,
                           int K, int N, int packed_n, queue_ptr stream) {
#define EXL3_SYCL_RECON_HAD_CASE(b) \
    case b: launch_recon_had<b>(B, W, suh, svh, K, N, packed_n, stream); break;
    switch (bits) {
        EXL3_SYCL_RECON_HAD_CASE(1)
        EXL3_SYCL_RECON_HAD_CASE(2)
        EXL3_SYCL_RECON_HAD_CASE(3)
        EXL3_SYCL_RECON_HAD_CASE(4)
        EXL3_SYCL_RECON_HAD_CASE(5)
        EXL3_SYCL_RECON_HAD_CASE(6)
        EXL3_SYCL_RECON_HAD_CASE(7)
        EXL3_SYCL_RECON_HAD_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_RECON_HAD_CASE
}

bool exl3_recon_had_want(int64_t M) {
    if (M < EXL3_RECON_HAD_MIN_M) {
        return false;
    }
    return ggml_sycl_get_env("GGML_SYCL_EXL3_RECON_HAD", 0) != 0;
}

void launch_gemv_bits(int bits, const float * x, const uint32_t * B, float * C,
                      int size_m, int size_k, int size_n, int packed_n,
                      const float * suh, int do_had, queue_ptr stream) {
#define EXL3_SYCL_GEMV_CASE(b) \
    case b: \
        if (size_m <= 1) { \
            launch_gemv<b, 1>(x, B, C, size_m, size_k, size_n, packed_n, suh, do_had, stream); \
        } else if (size_m <= 2) { \
            launch_gemv<b, 2>(x, B, C, size_m, size_k, size_n, packed_n, suh, do_had, stream); \
        } else { \
            launch_gemv<b, EXL3_GEMV_MAX_M>(x, B, C, size_m, size_k, size_n, packed_n, suh, do_had, stream); \
        } \
        break;
    switch (bits) {
        EXL3_SYCL_GEMV_CASE(1)
        EXL3_SYCL_GEMV_CASE(2)
        EXL3_SYCL_GEMV_CASE(3)
        EXL3_SYCL_GEMV_CASE(4)
        EXL3_SYCL_GEMV_CASE(5)
        EXL3_SYCL_GEMV_CASE(6)
        EXL3_SYCL_GEMV_CASE(7)
        EXL3_SYCL_GEMV_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_GEMV_CASE
}

void launch_recon_bits(int bits, const uint32_t * B, sycl::half * W,
                       int K, int N, int packed_n, queue_ptr stream) {
#define EXL3_SYCL_RECON_CASE(b) \
    case b: launch_recon<b>(B, W, K, N, packed_n, stream); break;
    switch (bits) {
        EXL3_SYCL_RECON_CASE(1)
        EXL3_SYCL_RECON_CASE(2)
        EXL3_SYCL_RECON_CASE(3)
        EXL3_SYCL_RECON_CASE(4)
        EXL3_SYCL_RECON_CASE(5)
        EXL3_SYCL_RECON_CASE(6)
        EXL3_SYCL_RECON_CASE(7)
        EXL3_SYCL_RECON_CASE(8)
        default: break;
    }
#undef EXL3_SYCL_RECON_CASE
}

} // namespace

bool ggml_sycl_exl3_available(int device) {
    static const int env = ggml_sycl_get_env("GGML_SYCL_EXL3", -1);
    if (env == 0) {
        return false;
    }
    if (device < 0 || device >= ggml_sycl_info().device_count) {
        return false;
    }
    if (env > 0) {
        return true;
    }
    return ggml_sycl_is_xe2(ggml_sycl_info().devices[device].hw_info.arch);
}

bool ggml_sycl_exl3_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!src0 || !src1 || !dst) {
        return false;
    }
    if (!ggml_is_exl3(src0->type)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t packed_n = src0->ne[1];
    const int64_t N = dst->ne[0];
    if (src1->ne[0] != K) {
        return false;
    }
    if (K <= 0 || packed_n <= 0 || N <= 0 || N > packed_n) {
        return false;
    }
    if ((K % 128) || (packed_n % 128) || (N % 128)) {
        return false;
    }
    if (ggml_nrows(dst) != ggml_nrows(src1)) {
        return false;
    }
    return true;
}

namespace {

const ggml_tensor * g_exl3_lookahead = nullptr;
const ggml_tensor * g_exl3_lookahead2 = nullptr;
const float * g_exl3_add_src = nullptr;
float * g_exl3_add_dst = nullptr;

struct exl3_pf {
    const uint32_t * B = nullptr;
    sycl::half * W = nullptr;
    int K = 0;
    int N = 0;
    int bits = 0;
    int packed_n = 0;
    sycl::event ev;
    bool pending = false;
};

exl3_pf g_exl3_pf;
exl3_pf g_exl3_pf2;
sycl::half * g_exl3_wbuf[3] = {};
size_t g_exl3_wcap[3] = {};
sycl::queue * g_exl3_wowner = nullptr;
sycl::queue * g_exl3_side = nullptr;
sycl::queue * g_exl3_side_seen = nullptr;

int exl3_slot_of(const sycl::half * W) {
    for (int i = 0; i < 3; ++i) {
        if (g_exl3_wbuf[i] == W) {
            return i;
        }
    }
    return 0;
}

int exl3_free_slot(int a, int b) {
    for (int i = 0; i < 3; ++i) {
        if (i != a && i != b) {
            return i;
        }
    }
    return 0;
}

sycl::half * exl3_wslot(queue_ptr stream, int slot, size_t need) {
    if (g_exl3_wowner != stream) {
        if (g_exl3_wowner) {
            for (int i = 0; i < 3; ++i) {
                if (g_exl3_wbuf[i]) {
                    sycl::free(g_exl3_wbuf[i], *g_exl3_wowner);
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            g_exl3_wbuf[i] = nullptr;
            g_exl3_wcap[i] = 0;
        }
        g_exl3_wowner = stream;
        g_exl3_pf.pending = false;
        g_exl3_pf2.pending = false;
    }
    if (!g_exl3_wbuf[slot] || g_exl3_wcap[slot] < need) {
        if (g_exl3_wbuf[slot]) {
            sycl::free(g_exl3_wbuf[slot], *stream);
        }
        g_exl3_wbuf[slot] = sycl::malloc_device<sycl::half>(need, *stream);
        g_exl3_wcap[slot] = need;
    }
    return g_exl3_wbuf[slot];
}

void exl3_launch_recon_q(int bits, const uint32_t * B, sycl::half * W,
                         int K, int N, int packed_n, queue_ptr q) {
    if (exl3_pack_xe2()) {
        launch_recon_xe2_bits(bits, B, W, K, N, packed_n, q);
        return;
    }
#if defined(__INTEL_LLVM_COMPILER)
    if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_RECON_ESIMD", 1)) {
        if (ggml_sycl_esimd::launch_exl3_recon_esimd_bits(bits, B, W, K, N, packed_n, q)) {
            return;
        }
    }
#endif
    launch_recon_bits(bits, B, W, K, N, packed_n, q);
}

void exl3_fwht128_view(ggml_tensor * t, void * data, int64_t rows) {
    memset(t, 0, sizeof(*t));
    t->type = GGML_TYPE_F32;
    t->ne[0] = 128;
    t->ne[1] = rows;
    t->ne[2] = 1;
    t->ne[3] = 1;
    t->nb[0] = sizeof(float);
    t->nb[1] = sizeof(float) * 128;
    t->nb[2] = t->nb[1] * (size_t) rows;
    t->nb[3] = t->nb[2];
    t->data = data;
}

struct exl3_pp_buckets {
    double pre = 0;
    double cvt = 0;
    double recon = 0;
    double gemm = 0;
    double post = 0;
    double mid = 0;
    double pre_b = 0;
    double cvt_b = 0;
    double post_b = 0;
    double mid_b = 0;
    int n = 0;
    int n_mid = 0;
    int armed = 0;
};

struct exl3_shape_b {
    int64_t K = 0;
    int64_t N = 0;
    double pre = 0;
    double post = 0;
    double pre_b = 0;
    double post_b = 0;
    int n = 0;
};

exl3_pp_buckets g_exl3_ppb;
exl3_shape_b g_exl3_sh[16];
int g_exl3_nsh = 0;

void exl3_shape_add(int64_t K, int64_t N, double pre, double post, double pre_b, double post_b) {
    for (int i = 0; i < g_exl3_nsh; ++i) {
        if (g_exl3_sh[i].K == K && g_exl3_sh[i].N == N) {
            g_exl3_sh[i].pre += pre;
            g_exl3_sh[i].post += post;
            g_exl3_sh[i].pre_b += pre_b;
            g_exl3_sh[i].post_b += post_b;
            g_exl3_sh[i].n += 1;
            return;
        }
    }
    if (g_exl3_nsh >= 16) {
        return;
    }
    exl3_shape_b & s = g_exl3_sh[g_exl3_nsh++];
    s.K = K;
    s.N = N;
    s.pre = pre;
    s.post = post;
    s.pre_b = pre_b;
    s.post_b = post_b;
    s.n = 1;
}

void exl3_ppb_dump() {
    if (g_exl3_ppb.n <= 0) {
        return;
    }
    const double d = (double) g_exl3_ppb.n;
    const int n_ub = (g_exl3_ppb.n % 400 == 0 && g_exl3_ppb.n >= 400) ? g_exl3_ppb.n / 400 : 2;
    const double ub = (double) n_ub;
    const auto gbs = [](double bytes, double ms) {
        return (ms > 0.0) ? (bytes / 1e9) / (ms / 1e3) : 0.0;
    };
    fprintf(stderr,
            "EXL3_SYCL buckets n=%d n_ub=%d"
            " pre=%.2f cvt=%.2f recon=%.2f gemm=%.2f post=%.2f mid=%.2f ms/ubatch"
            " pre_gbs=%.1f cvt_gbs=%.1f post_gbs=%.1f mid_gbs=%.1f"
            " sum_pre=%.2f sum_cvt=%.2f sum_recon=%.2f sum_gemm=%.2f sum_post=%.2f sum_mid=%.2f\n",
            g_exl3_ppb.n, n_ub,
            g_exl3_ppb.pre / ub, g_exl3_ppb.cvt / ub, g_exl3_ppb.recon / ub,
            g_exl3_ppb.gemm / ub, g_exl3_ppb.post / ub, g_exl3_ppb.mid / ub,
            gbs(g_exl3_ppb.pre_b, g_exl3_ppb.pre),
            gbs(g_exl3_ppb.cvt_b, g_exl3_ppb.cvt),
            gbs(g_exl3_ppb.post_b, g_exl3_ppb.post),
            gbs(g_exl3_ppb.mid_b, g_exl3_ppb.mid),
            g_exl3_ppb.pre, g_exl3_ppb.cvt, g_exl3_ppb.recon, g_exl3_ppb.gemm, g_exl3_ppb.post, g_exl3_ppb.mid);
    fprintf(stderr,
            "EXL3_SYCL buckets ms/call pre=%.3f cvt=%.3f recon=%.3f gemm=%.3f post=%.3f mid=%.3f n_mid=%d\n",
            g_exl3_ppb.pre / d, g_exl3_ppb.cvt / d, g_exl3_ppb.recon / d,
            g_exl3_ppb.gemm / d, g_exl3_ppb.post / d,
            g_exl3_ppb.n_mid > 0 ? g_exl3_ppb.mid / (double) g_exl3_ppb.n_mid : 0.0,
            g_exl3_ppb.n_mid);
    for (int i = 0; i < g_exl3_nsh; ++i) {
        const exl3_shape_b & s = g_exl3_sh[i];
        fprintf(stderr,
                "EXL3_SYCL shape K=%ld N=%ld n=%d pre=%.3fms/%.1fGB/s post=%.3fms/%.1fGB/s\n",
                (long) s.K, (long) s.N, s.n,
                s.pre / (double) s.n, gbs(s.pre_b, s.pre),
                s.post / (double) s.n, gbs(s.post_b, s.post));
    }
    fflush(stderr);
}

} // namespace

void ggml_sycl_exl3_set_lookahead(const ggml_tensor * next_mm) {
    g_exl3_lookahead = next_mm;
    g_exl3_lookahead2 = nullptr;
}

void ggml_sycl_mul_mat_exl3(ggml_backend_sycl_context & ctx,
                            const ggml_tensor * src0,
                            const ggml_tensor * src1,
                            ggml_tensor * dst,
                            const ggml_tensor * fuse_x,
                            const ggml_tensor * fuse_suh,
                            const ggml_tensor * fuse_svh,
                            const sycl::half * x_pre,
                            float * c_raw) {
    const ggml_tensor * x_src = fuse_x ? fuse_x : src1;
    GGML_ASSERT(x_src);
    GGML_ASSERT(ggml_sycl_exl3_shape_ok(src0, x_src, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(x_src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t K = src0->ne[0];
    const int64_t packed_n = src0->ne[1];
    const int64_t N = dst->ne[0];
    const int64_t M = ggml_nrows(x_src);
    const int bits = ggml_exl3_n_bits(src0->type);
    GGML_ASSERT(bits >= 1 && bits <= 8);
    const int do_had = fuse_x ? 1 : 0;
    if (do_had) {
        GGML_ASSERT(fuse_suh && ggml_is_contiguous(fuse_suh) && fuse_suh->type == GGML_TYPE_F32);
        GGML_ASSERT(fuse_suh->ne[0] == K);
    }

    queue_ptr stream = ctx.stream();
    const uint32_t * B = (const uint32_t *) src0->data;
    float * C = c_raw ? c_raw : (float *) dst->data;
    const float * suh = do_had ? (const float *) fuse_suh->data : nullptr;
    const float * svh = nullptr;
    if (fuse_svh) {
        GGML_ASSERT(ggml_is_contiguous(fuse_svh) && fuse_svh->type == GGML_TYPE_F32);
        GGML_ASSERT(fuse_svh->ne[0] == N);
        svh = (const float *) fuse_svh->data;
    }

    ggml_sycl_pool_alloc<float> x_f32(ctx.pool());
    const float * x = nullptr;
    if (!x_pre) {
        if (x_src->type == GGML_TYPE_F32) {
            x = (const float *) x_src->data;
        } else {
            const to_fp32_sycl_t to_fp32 = ggml_get_to_fp32_sycl(x_src->type, dst);
            GGML_ASSERT(to_fp32);
            x_f32.alloc((size_t) ggml_nelements(x_src));
            to_fp32(x_src->data, x_f32.get(), ggml_nelements(x_src), stream);
            x = x_f32.get();
        }
    }

    const bool want_xmx = M <= EXL3_GEMV_MAX_M && exl3_xmx_want();
    const bool want_i8 = M <= EXL3_GEMV_MAX_M && exl3_i8_want();
    const bool want_esimd = M <= EXL3_GEMV_MAX_M && exl3_esimd_want();
    const bool want_had = M > EXL3_GEMV_MAX_M && do_had && svh && exl3_recon_had_want(M);
    static thread_local int nlog = 0;
    static thread_local int nbig = 0;
    if (ggml_sycl_get_env("GGML_SYCL_EXL3_LOG", 0) &&
        (nlog < 8 || (M > EXL3_GEMV_MAX_M && nbig < 8))) {
        if (M > EXL3_GEMV_MAX_M) {
            nbig++;
        }
        nlog++;
        const char * path = "recon";
        if (M <= EXL3_GEMV_MAX_M) {
            if (exl3_pack_xe2()) {
                path = g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_ESIMD", 1)
                        ? "gemv-xe2-esimd" : "gemv-xe2-16";
            } else if (want_xmx) {
                path = "gemv-xmx";
            } else if (want_i8) {
                path = "gemv-i8";
            } else if (want_esimd) {
                path = "gemv-esimd";
            } else if (exl3_vec_want()) {
                path = "gemv-vec";
            } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_WG16", 0)) {
                path = "gemv-wg16";
            } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_SIMT_I8", 0)) {
                path = "gemv-simt-i8";
            } else if (exl3_i8n_want()) {
                path = "gemv-i8n";
            } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_F32N", 0)) {
                path = "gemv-f32n";
            } else if (!ggml_sycl_get_env("GGML_SYCL_EXL3_GLB", 1)) {
                path = "gemv-wg32";
            } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_GLB16", 0)) {
                path = "gemv-glb16";
            } else if (ggml_sycl_get_env("GGML_SYCL_EXL3_SHUF", 0)) {
                path = "gemv-shuf";
            } else {
                path = "gemv-glb";
            }
        } else if (want_had) {
            path = "recon_had";
        } else if (exl3_pack_xe2()) {
            if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_PACKED_XMX", 0)) {
                path = "gemm-xe2-xmx";
            } else if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_RECON_VN", 0)) {
                path = "recon-xe2-vn";
            } else if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_RECON_ESIMD", 0)) {
                path = "recon-xe2-esimd";
            } else {
                path = "recon-xe2-16";
            }
        } else if ((M % 8) == 0 && M <= 512 && g_ggml_sycl_enable_esimd
                && ggml_sycl_get_env("GGML_SYCL_EXL3_PACKED_XMX", 0)) {
            path = "packed-xmx";
        } else if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_RECON_ESIMD", 1)) {
            path = "recon-esimd";
        }
        fprintf(stderr, "EXL3_SYCL mul_mat bits=%d M=%ld K=%ld N=%ld packed_n=%ld path=%s%s\n",
                bits, (long) M, (long) K, (long) N, (long) packed_n, path,
                do_had ? (svh ? "+full+prehad" : "+prehad") : "");
        fflush(stderr);
    }

    if (M <= EXL3_GEMV_MAX_M) {
        const float * xg = x;
        ggml_sycl_pool_alloc<float> x_had(ctx.pool());
        int had_done = do_had;
        const int skip_had = ggml_sycl_get_env("GGML_SYCL_EXL3_SKIP_HAD", 0);
        if (do_had && !skip_had) {
            x_had.alloc((size_t) M * (size_t) K);
            launch_had_in_x(x, suh, x_had.get(), (int) K, (int) M, stream);
            xg = x_had.get();
            had_done = 0;
        } else if (skip_had) {
            had_done = 0;
        }
#if defined(__INTEL_LLVM_COMPILER)
        if (!exl3_pack_xe2() && (want_xmx || want_i8 || want_esimd || exl3_vec_want())) {
            bool ok = false;
            if (want_xmx) {
                ok = ggml_sycl_esimd::launch_exl3_gemv_xmx_bits(bits, xg, B, C,
                        (int) M, (int) K, (int) N, (int) packed_n, stream);
            } else if (want_i8) {
                ok = ggml_sycl_esimd::launch_exl3_gemv_i8_bits(bits, xg, B, C,
                        (int) M, (int) K, (int) N, (int) packed_n, stream);
            } else if (want_esimd) {
                ok = ggml_sycl_esimd::launch_exl3_gemv_esimd_bits(bits, xg, B, C,
                        (int) M, (int) K, (int) N, (int) packed_n, stream);
            } else {
                ok = ggml_sycl_esimd::launch_exl3_gemv_vec_bits(bits, xg, B, C,
                        (int) M, (int) K, (int) N, (int) packed_n, stream);
            }
            if (ok) {
                if (svh) {
                    launch_post_had_svh(C, svh, (int) N, (int) M, stream);
                }
                return;
            }
        }
#endif
        if (exl3_pack_xe2()) {
#if defined(__INTEL_LLVM_COMPILER)
            bool xe2_ok = false;
            if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_ESIMD", 1)) {
                xe2_ok = ggml_sycl_esimd::launch_exl3_gemv_xe2_esimd_bits(
                        bits, xg, B, C, (int) M, (int) K, (int) N, (int) packed_n, stream);
            }
            if (!xe2_ok)
#endif
            {
                launch_gemv_xe2_bits(bits, xg, B, C, (int) M, (int) K, (int) N, (int) packed_n, stream);
            }
        } else if (exl3_i8n_want() && !want_xmx && !want_i8 && !want_esimd &&
            (K % 16) == 0 && (N % 16) == 0) {
            const size_t nkt = size_t(K) / 16;
            ggml_sycl_pool_alloc<int8_t> x_i8(ctx.pool(), (size_t) M * (size_t) K);
            ggml_sycl_pool_alloc<float>  x_qs(ctx.pool(), (size_t) M * nkt);
            ggml_sycl_pool_alloc<int>    x_sum(ctx.pool(), (size_t) M * nkt);
            launch_quant_x_i8(xg, x_i8.get(), x_qs.get(), x_sum.get(), (int) K, (int) M, stream);
            launch_gemv_i8n_bits(bits, x_i8.get(), x_qs.get(), x_sum.get(), B, C,
                                 (int) M, (int) K, (int) N, (int) packed_n, stream);
        } else {
            launch_gemv_bits(bits, xg, B, C, (int) M, (int) K, (int) N, (int) packed_n,
                             had_done ? suh : nullptr, had_done, stream);
        }
        if (svh && !ggml_sycl_get_env("GGML_SYCL_EXL3_SKIP_HAD", 0)) {
            launch_post_had_svh(C, svh, (int) N, (int) M, stream);
        }
        if (ggml_sycl_get_env("GGML_SYCL_EXL3_TIME", 0)) {
            stream->wait();
            static thread_local auto tprev = std::chrono::steady_clock::now();
            static thread_local int ntm = 0;
            const auto now = std::chrono::steady_clock::now();
            if (ntm < 16) {
                const double ms = std::chrono::duration<double, std::milli>(now - tprev).count();
                fprintf(stderr, "EXL3_SYCL gemv time bits=%d M=%ld K=%ld N=%ld dt=%.2fms\n",
                        bits, (long) M, (long) K, (long) N, ntm == 0 ? 0.0 : ms);
                fflush(stderr);
                ntm++;
            }
            tprev = now;
        }
        return;
    }

#if defined(__INTEL_LLVM_COMPILER)
    if (!do_had && !want_had && !exl3_pack_xe2() && g_ggml_sycl_enable_esimd &&
            (M % 8) == 0 && M <= 512 && (N % 16) == 0 && (K % 16) == 0 &&
            ggml_sycl_get_env("GGML_SYCL_EXL3_PACKED_XMX", 0)) {
        ggml_sycl_pool_alloc<sycl::half> x_f16(ctx.pool());
        const sycl::half * xh = nullptr;
        if (x_src->type == GGML_TYPE_F16) {
            xh = (const sycl::half *) x_src->data;
        } else {
            const to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(GGML_TYPE_F32, dst);
            GGML_ASSERT(to_fp16);
            x_f16.alloc((size_t) M * (size_t) K);
            to_fp16(x, x_f16.get(), M * K, stream);
            xh = x_f16.get();
        }
        if (ggml_sycl_esimd::launch_exl3_gemm_packed_xmx_bits(
                    bits, xh, B, C, (int) M, (int) K, (int) N, (int) packed_n, stream)) {
            return;
        }
    }
#endif

#if GGML_SYCL_DNNL
    const int time_pp = ggml_sycl_get_env("GGML_SYCL_EXL3_TIME", 0);
    if (time_pp && !g_exl3_ppb.armed) {
        g_exl3_ppb.armed = 1;
        atexit(exl3_ppb_dump);
    }
    double ms_pre = 0;
    double ms_cvt = 0;
    double ms_post = 0;
    ggml_sycl_pool_alloc<sycl::half> x_f16(ctx.pool());
    const sycl::half * xh = x_pre;
    if (g_ggml_sycl_enable_dnn) {
        if (!xh) {
            if (do_had && !want_had) {
                if (time_pp) {
                    stream->wait();
                }
                const auto t_pre0 = std::chrono::steady_clock::now();
                x_f16.alloc((size_t) M * (size_t) K);
                launch_exl3_had_to_f16(x, x_f16.get(), suh, (M * K) / 128, K, stream);
                xh = x_f16.get();
                if (time_pp) {
                    stream->wait();
                    ms_pre = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_pre0).count();
                    g_exl3_ppb.pre_b += double(M) * double(K) * 6.0;
                }
            } else if (x_src->type == GGML_TYPE_F16) {
                xh = (const sycl::half *) x_src->data;
            } else {
                const to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(GGML_TYPE_F32, dst);
                GGML_ASSERT(to_fp16);
                x_f16.alloc((size_t) M * (size_t) K);
                to_fp16(x, x_f16.get(), M * K, stream);
                xh = x_f16.get();
            }
        }
    }
#if defined(__INTEL_LLVM_COMPILER)
    if (!do_had && !want_had && exl3_pack_xe2() && g_ggml_sycl_enable_esimd && xh
            && (N % 16) == 0 && (K % 16) == 0
            && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_PACKED_XMX", 0)) {
        if (ggml_sycl_esimd::launch_exl3_gemm_xe2_xmx_bits(
                    bits, xh, B, C, (int) M, (int) K, (int) N, (int) packed_n, stream)) {
            return;
        }
    }
    int slab = ggml_sycl_get_env("GGML_SYCL_EXL3_SLAB", 0);
    if (slab > 0 && slab < 128) {
        slab = 128;
    }
    slab -= slab % 128;
    const bool use_slab = g_ggml_sycl_enable_dnn && xh && !want_had
            && slab > 0 && K > slab && (K % 128) == 0 && (N % 16) == 0
            && (exl3_pack_xe2()
                || (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_RECON_ESIMD", 1)));
    if (use_slab) {
        const int nslabs = int((K + slab - 1) / slab);
        ggml_sycl_pool_alloc<sycl::half> w_slab(ctx.pool(), (size_t) 2 * (size_t) N * (size_t) slab);
        static thread_local sycl::queue * side = nullptr;
        static thread_local sycl::queue * side_seen = nullptr;
        if (!side || side_seen != stream) {
            delete side;
            side = new sycl::queue(stream->get_context(), stream->get_device(),
                                   sycl::property_list{sycl::property::queue::in_order()});
            side_seen = stream;
        }
        auto recon_slab = [&](int s, queue_ptr q) -> bool {
            const int k0 = s * slab;
            const int klen = int(std::min<int64_t>(slab, K - k0));
            sycl::half * W = w_slab.get() + size_t(s & 1) * size_t(N) * size_t(slab);
            if (exl3_pack_xe2()) {
                launch_recon_xe2_bits(bits, B, W, (int) K, (int) N, (int) packed_n,
                                      q, k0, klen, klen, 0);
                return true;
            }
            return ggml_sycl_esimd::launch_exl3_recon_esimd_bits(
                    bits, B, W, (int) K, (int) N, (int) packed_n, q, k0, klen, klen);
        };
        if (recon_slab(0, side)) {
            sycl::event ev_r[2];
            ev_r[0] = side->ext_oneapi_submit_barrier();
            sycl::event ev_g_prev;
            for (int s = 0; s < nslabs; ++s) {
                const int k0 = s * slab;
                const int klen = int(std::min<int64_t>(slab, K - k0));
                sycl::half * W = w_slab.get() + size_t(s & 1) * size_t(N) * size_t(slab);
                stream->ext_oneapi_submit_barrier({ev_r[s & 1]});
                DnnlGemmWrapper::gemm(ctx, (int) N, (int) M, klen,
                                      W, DnnlGemmWrapper::to_dt<sycl::half>(), 1, klen, (dnnl_dim_t) klen * N,
                                      xh + k0, DnnlGemmWrapper::to_dt<sycl::half>(), 1, (int) K, (dnnl_dim_t) M * K,
                                      C, DnnlGemmWrapper::to_dt<float>(), stream, 1, 1);
                sycl::event ev_g = stream->ext_oneapi_submit_barrier();
                if (s + 1 < nslabs) {
                    if (s >= 1) {
                        side->ext_oneapi_submit_barrier({ev_g_prev});
                    }
                    if (!recon_slab(s + 1, side)) {
                        break;
                    }
                    ev_r[(s + 1) & 1] = side->ext_oneapi_submit_barrier();
                }
                ev_g_prev = ev_g;
            }
            if (do_had && !want_had && svh) {
                launch_exl3_had_svh(C, svh, (M * N) / 128, N, stream, nullptr, nullptr);
            }
            return;
        }
    }
#endif
#endif

    const int do_pf = !want_had && ggml_sycl_get_env("GGML_SYCL_EXL3_PREFETCH", 1);
    const ggml_tensor * next_mm = g_exl3_lookahead;
    const ggml_tensor * next_mm2 = g_exl3_lookahead2;
    g_exl3_lookahead = nullptr;
    g_exl3_lookahead2 = nullptr;
    const float * add_src = g_exl3_add_src;
    float * add_dst = g_exl3_add_dst;
    g_exl3_add_src = nullptr;
    g_exl3_add_dst = nullptr;
    ggml_sycl_pool_alloc<sycl::half> w_pool(ctx.pool());
    sycl::half * W = nullptr;
    int hold_slot = -1;
    bool have_w = false;
    auto pf_match = [&](exl3_pf & p) {
        return p.pending && p.B == B && p.K == (int) K && p.N == (int) N &&
               p.bits == bits && p.packed_n == (int) packed_n;
    };
    if (do_pf && pf_match(g_exl3_pf)) {
        stream->ext_oneapi_submit_barrier({g_exl3_pf.ev});
        W = g_exl3_pf.W;
        hold_slot = exl3_slot_of(W);
        g_exl3_pf.pending = false;
        have_w = true;
    } else if (do_pf && pf_match(g_exl3_pf2)) {
        stream->ext_oneapi_submit_barrier({g_exl3_pf2.ev});
        W = g_exl3_pf2.W;
        hold_slot = exl3_slot_of(W);
        g_exl3_pf2.pending = false;
        have_w = true;
    } else {
        if (do_pf && g_exl3_pf.pending) {
            stream->ext_oneapi_submit_barrier({g_exl3_pf.ev});
            g_exl3_pf.pending = false;
        }
        if (do_pf && g_exl3_pf2.pending) {
            stream->ext_oneapi_submit_barrier({g_exl3_pf2.ev});
            g_exl3_pf2.pending = false;
        }
    }
    if (!have_w) {
        if (do_pf) {
            W = exl3_wslot(stream, 0, size_t(N) * size_t(K));
            hold_slot = 0;
        } else {
            w_pool.alloc((size_t) N * (size_t) K);
            W = w_pool.get();
        }
    }
    {
        static thread_local int npf = 0;
        if (do_pf && npf < 12 && ggml_sycl_get_env("GGML_SYCL_EXL3_LOG", 0)) {
            fprintf(stderr, "EXL3_SYCL pf bits=%d K=%ld N=%ld hit=%d next=%d next2=%d add=%d\n",
                    bits, (long) K, (long) N, (int) have_w, next_mm ? 1 : 0, next_mm2 ? 1 : 0,
                    add_src ? 1 : 0);
            fflush(stderr);
            npf++;
        }
    }
    std::chrono::steady_clock::time_point t0, t1, t2;
    if (time_pp) {
        stream->wait();
        t0 = std::chrono::steady_clock::now();
    }
    bool recon_ok = have_w;
    if (!have_w && want_had) {
        launch_recon_had_bits(bits, B, W, suh, svh, (int) K, (int) N, (int) packed_n, stream);
        recon_ok = true;
    } else if (!have_w && exl3_pack_xe2()) {
#if defined(__INTEL_LLVM_COMPILER)
        if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_RECON_VN", 0)) {
            recon_ok = ggml_sycl_esimd::launch_exl3_recon_xe2_vn_bits(
                    bits, B, W, (int) K, (int) N, (int) packed_n, stream);
        }
        if (!recon_ok && g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_XE2_RECON_ESIMD", 0)) {
            recon_ok = ggml_sycl_esimd::launch_exl3_recon_xe2_esimd_bits(
                    bits, B, W, (int) K, (int) N, (int) packed_n, stream);
        }
#endif
        if (!recon_ok) {
            launch_recon_xe2_bits(bits, B, W, (int) K, (int) N, (int) packed_n, stream);
            recon_ok = true;
        }
    } else if (!have_w) {
#if defined(__INTEL_LLVM_COMPILER)
        if (g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_EXL3_RECON_ESIMD", 1)) {
            recon_ok = ggml_sycl_esimd::launch_exl3_recon_esimd_bits(
                    bits, B, W, (int) K, (int) N, (int) packed_n, stream);
        }
#endif
        if (!recon_ok) {
            launch_recon_bits(bits, B, W, (int) K, (int) N, (int) packed_n, stream);
        }
    }
    const int wkn = recon_ok && !want_had && exl3_wkn();
    if (time_pp) {
        stream->wait();
        t1 = std::chrono::steady_clock::now();
    }

#if GGML_SYCL_DNNL
    if (g_ggml_sycl_enable_dnn && xh) {
        if (wkn) {
            DnnlGemmWrapper::gemm(ctx, (int) N, (int) M, (int) K,
                                  W, DnnlGemmWrapper::to_dt<sycl::half>(),
                                  (dnnl_dim_t) N, 1, (dnnl_dim_t) N * (dnnl_dim_t) K,
                                  xh, DnnlGemmWrapper::to_dt<sycl::half>(),
                                  1, (int) K, (dnnl_dim_t) M * (dnnl_dim_t) K,
                                  C, DnnlGemmWrapper::to_dt<float>(), stream, 1, 1);
        } else {
            DnnlGemmWrapper::row_gemm(ctx, (int) N, (int) M, (int) K,
                                      W, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      xh, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      C, DnnlGemmWrapper::to_dt<float>(), stream);
        }
        auto start_pf = [&](const ggml_tensor * mm, exl3_pf & slot, int wslot) {
            if (!mm || !mm->src[0] || !ggml_is_exl3(mm->src[0]->type)) {
                return;
            }
            const ggml_tensor * ns0 = mm->src[0];
            const int nK = (int) ns0->ne[0];
            const int nPacked = (int) ns0->ne[1];
            const int nN = (int) mm->ne[0];
            const int nbits = ggml_exl3_n_bits(ns0->type);
            if (nK <= 0 || nN <= 0 || nPacked < nN) {
                return;
            }
            if (!g_exl3_side || g_exl3_side_seen != stream) {
                delete g_exl3_side;
                g_exl3_side = new sycl::queue(stream->get_context(), stream->get_device(),
                                              sycl::property_list{sycl::property::queue::in_order()});
                g_exl3_side_seen = stream;
            }
            sycl::half * Wn = exl3_wslot(stream, wslot, size_t(nN) * size_t(nK));
            exl3_launch_recon_q(nbits, (const uint32_t *) ns0->data, Wn, nK, nN, nPacked, g_exl3_side);
            slot.B = (const uint32_t *) ns0->data;
            slot.W = Wn;
            slot.K = nK;
            slot.N = nN;
            slot.bits = nbits;
            slot.packed_n = nPacked;
            slot.ev = g_exl3_side->ext_oneapi_submit_barrier();
            slot.pending = true;
        };
        if (do_pf && next_mm) {
            const int keep2 = g_exl3_pf2.pending ? exl3_slot_of(g_exl3_pf2.W) : -1;
            const int s1 = exl3_free_slot(hold_slot, keep2);
            start_pf(next_mm, g_exl3_pf, s1);
            if (next_mm2 && !g_exl3_pf2.pending) {
                start_pf(next_mm2, g_exl3_pf2, exl3_free_slot(hold_slot, s1));
            }
        }
        if (time_pp) {
            stream->wait();
            t2 = std::chrono::steady_clock::now();
        }
        if (do_had && !want_had && svh) {
            const auto t_post0 = std::chrono::steady_clock::now();
            launch_exl3_had_svh(C, svh, (M * N) / 128, N, stream, add_src, add_dst);
            if (time_pp) {
                stream->wait();
                ms_post = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_post0).count();
                g_exl3_ppb.post_b += double(M) * double(N) * (add_src ? 12.0 : 8.0);
            }
        }
        if (time_pp) {
            const double recon_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double gemm_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
            g_exl3_ppb.pre += ms_pre;
            g_exl3_ppb.cvt += ms_cvt;
            g_exl3_ppb.recon += recon_ms;
            g_exl3_ppb.gemm += gemm_ms;
            g_exl3_ppb.post += ms_post;
            g_exl3_ppb.n += 1;
            exl3_shape_add(K, N, ms_pre, ms_post,
                           double(M) * double(K) * 6.0,
                           double(M) * double(N) * (add_src ? 12.0 : 8.0));
            static thread_local int ntm = 0;
            if (ntm < 16) {
                const double pre_bytes = double(M) * double(K) * 6.0;
                const double post_bytes = double(M) * double(N) * (add_src ? 12.0 : 8.0);
                const double pre_gbs = (ms_pre > 0) ? (pre_bytes / 1e9) / (ms_pre / 1e3) : 0.0;
                const double cvt_gbs = (ms_cvt > 0) ? (double(M) * double(K) * 6.0 / 1e9) / (ms_cvt / 1e3) : 0.0;
                const double post_gbs = (ms_post > 0) ? (post_bytes / 1e9) / (ms_post / 1e3) : 0.0;
                fprintf(stderr,
                        "EXL3_SYCL time bits=%d M=%ld K=%ld N=%ld"
                        " pre=%.2fms cvt=%.2fms recon=%.2fms gemm=%.2fms post=%.2fms"
                        " pre_gbs=%.1f cvt_gbs=%.1f post_gbs=%.1f\n",
                        bits, (long) M, (long) K, (long) N,
                        ms_pre, ms_cvt, recon_ms, gemm_ms, ms_post,
                        pre_gbs, cvt_gbs, post_gbs);
                fflush(stderr);
                ntm++;
            }
        }
        return;
    }
#endif

    const float * xg = x;
    ggml_sycl_pool_alloc<float> x_had_fb(ctx.pool());
    if (do_had && !want_had) {
        x_had_fb.alloc((size_t) M * (size_t) K);
        launch_had_in_x(x, suh, x_had_fb.get(), (int) K, (int) M, stream);
        xg = x_had_fb.get();
    }
    const int64_t n_chunk = EXL3_GEMV_MAX_M;
    for (int64_t m0 = 0; m0 < M; m0 += n_chunk) {
        const int64_t m_use = std::min(n_chunk, M - m0);
        launch_gemv_bits(bits, xg + m0 * K, B, C + m0 * N,
                         (int) m_use, (int) K, (int) N, (int) packed_n, nullptr, 0, stream);
    }
    if (do_had && !want_had && svh) {
        launch_post_had_svh(C, svh, (int) N, (int) M, stream);
    }
}

namespace {

bool exl3_is_cast(const ggml_tensor * t) {
    return t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_CONT);
}

bool exl3_is_had(const ggml_tensor * mm) {
    return mm && mm->op == GGML_OP_MUL_MAT &&
           ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           mm->type == GGML_TYPE_F32 &&
           mm->src[1] && mm->src[1]->ne[0] == 128;
}

bool exl3_from_node(const ggml_tensor * t, const ggml_tensor * root) {
    for (int hops = 0; t && hops < 6; ++hops) {
        if (t == root) {
            return true;
        }
        if (!exl3_is_cast(t)) {
            return false;
        }
        t = t->src[0];
    }
    return false;
}

bool exl3_row_vec(const ggml_tensor * v, int64_t ne0) {
    return v && v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) &&
           v->ne[0] == ne0 && v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1;
}

bool exl3_pick_scale(const ggml_tensor * mul, const ggml_tensor ** act, const ggml_tensor ** scale) {
    if (!mul || mul->op != GGML_OP_MUL) {
        return false;
    }
    if (mul->type != GGML_TYPE_F32 && mul->type != GGML_TYPE_F16) {
        return false;
    }
    const ggml_tensor * a = mul->src[0];
    const ggml_tensor * b = mul->src[1];
    if (!a || !b) {
        return false;
    }
    if (ggml_are_same_shape(a, mul) && exl3_row_vec(b, mul->ne[0]) && ggml_is_contiguous(a)) {
        *act = a;
        *scale = b;
        return true;
    }
    if (ggml_are_same_shape(b, mul) && exl3_row_vec(a, mul->ne[0]) && ggml_is_contiguous(b)) {
        *act = b;
        *scale = a;
        return true;
    }
    return false;
}

bool exl3_uses_ok(const ggml_cgraph * g, int begin, int end) {
    for (int k = begin; k < end; ++k) {
        if ((g->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(g, k) != 1) {
            return false;
        }
    }
    return true;
}

int exl3_follow_cast(const ggml_cgraph * g, int i, const ggml_tensor * cur, const ggml_tensor ** out) {
    *out = cur;
    while (i < g->n_nodes && exl3_is_cast(g->nodes[i]) && g->nodes[i]->src[0] == *out) {
        if ((g->nodes[i]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(g, i) != 1) {
            break;
        }
        *out = g->nodes[i];
        i++;
    }
    return i;
}

struct exl3_chain {
    const ggml_tensor * act = nullptr;
    const ggml_tensor * suh = nullptr;
    ggml_tensor * mm = nullptr;
    const ggml_tensor * svh = nullptr;
    ggml_tensor * dst = nullptr;
    int first = 0;
    int last = 0;
};

bool exl3_match_chain(const ggml_cgraph * cgraph, int node_idx, exl3_chain & c) {
    if (node_idx + 2 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * n0 = cgraph->nodes[node_idx];
    const ggml_tensor * act = nullptr;
    const ggml_tensor * suh = nullptr;
    if (!exl3_pick_scale(n0, &act, &suh)) {
        return false;
    }
    if (!ggml_is_contiguous_1(act) || act->ne[0] % 128 != 0) {
        return false;
    }
    if (act->nb[0] != ggml_element_size(act) || act->nb[1] != act->ne[0] * act->nb[0]) {
        return false;
    }
    if ((n0->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, node_idx) != 1) {
        return false;
    }

    const ggml_tensor * viewed = n0;
    const int after_mul = exl3_follow_cast(cgraph, node_idx + 1, n0, &viewed);
    if (after_mul >= cgraph->n_nodes || !exl3_uses_ok(cgraph, node_idx + 1, after_mul)) {
        return false;
    }
    ggml_tensor * had = cgraph->nodes[after_mul];
    if (!exl3_is_had(had)) {
        return false;
    }
    if (had->src[1] != viewed && !(exl3_is_cast(had->src[1]) && had->src[1]->src[0] == n0)) {
        return false;
    }
    if ((had->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, after_mul) != 1) {
        return false;
    }

    const ggml_tensor * viewed2 = had;
    int after_had = exl3_follow_cast(cgraph, after_mul + 1, had, &viewed2);
    if (after_had >= cgraph->n_nodes || !exl3_uses_ok(cgraph, after_mul + 1, after_had)) {
        return false;
    }
    ggml_tensor * mm = cgraph->nodes[after_had];
    if (mm->op != GGML_OP_MUL_MAT || !mm->src[0] || !ggml_is_exl3(mm->src[0]->type)) {
        mm = nullptr;
        const int scan_end = after_had + 8 < cgraph->n_nodes ? after_had + 8 : cgraph->n_nodes;
        for (int j = after_had; j < scan_end; ++j) {
            ggml_tensor * cand = cgraph->nodes[j];
            if (cand->op != GGML_OP_MUL_MAT || !cand->src[0] || !ggml_is_exl3(cand->src[0]->type)) {
                continue;
            }
            if (!exl3_from_node(cand->src[1], had)) {
                continue;
            }
            mm = cand;
            after_had = j;
            break;
        }
        if (!mm) {
            return false;
        }
    } else if (mm->src[1] != viewed2 && !exl3_from_node(mm->src[1], had)) {
        return false;
    }
    if (mm->type != GGML_TYPE_F32 || mm->src[0]->ne[0] != act->ne[0]) {
        return false;
    }
    if (!ggml_sycl_exl3_shape_ok(mm->src[0], act, mm)) {
        return false;
    }

    const ggml_tensor * svh = nullptr;
    ggml_tensor * dst = mm;
    int last = after_had;

    const int scan_hi = after_had + 16 < cgraph->n_nodes ? after_had + 16 : cgraph->n_nodes;
    for (int j = after_had + 1; j < scan_hi && !svh; ++j) {
        ggml_tensor * had2 = cgraph->nodes[j];
        if (!exl3_is_had(had2) || !(exl3_from_node(had2->src[1], mm))) {
            continue;
        }
        if (had2->flags & GGML_TENSOR_FLAG_OUTPUT) {
            continue;
        }
        const ggml_tensor * viewed4 = had2;
        const int after_had2 = exl3_follow_cast(cgraph, j + 1, had2, &viewed4);
        if (after_had2 >= cgraph->n_nodes) {
            continue;
        }
        ggml_tensor * mul_svh = cgraph->nodes[after_had2];
        const ggml_tensor * act2 = nullptr;
        const ggml_tensor * svh_t = nullptr;
        if (!exl3_pick_scale(mul_svh, &act2, &svh_t)) {
            continue;
        }
        if (!(act2 == viewed4 || exl3_from_node(act2, had2))) {
            continue;
        }
        if (svh_t->ne[0] != mm->ne[0] || ggml_nelements(mul_svh) != ggml_nelements(mm)) {
            continue;
        }
        svh = svh_t;
        dst = mul_svh;
        last = after_had2;
    }

    c.act = act;
    c.suh = suh;
    c.mm = mm;
    c.svh = svh;
    c.dst = dst;
    c.first = node_idx;
    c.last = last;
    return true;
}

void exl3_dump_nodes(const ggml_cgraph * g, int begin, int end, const char * tag) {
    fprintf(stderr, "EXL3_FFN %s nodes [%d,%d)\n", tag, begin, end);
    const int hi = end < g->n_nodes ? end : g->n_nodes;
    for (int i = begin; i < hi; ++i) {
        const ggml_tensor * t = g->nodes[i];
        fprintf(stderr, "  [%d] name=%s op=%s type=%s ne=[%ld,%ld,%ld,%ld]\n",
                i, t->name, ggml_op_name(t->op), ggml_type_name(t->type),
                (long) t->ne[0], (long) t->ne[1], (long) t->ne[2], (long) t->ne[3]);
        if (t->op == GGML_OP_GLU) {
            fprintf(stderr, "    glu_op=%d src0=%s src1=%s\n",
                    (int) ggml_get_glu_op(t),
                    t->src[0] ? t->src[0]->name : "",
                    t->src[1] ? t->src[1]->name : "");
        }
    }
    fflush(stderr);
}

const ggml_tensor * exl3_next_mm(const ggml_cgraph * g, int after) {
    for (int j = after + 1; j < g->n_nodes; ++j) {
        const ggml_tensor * n = g->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && n->src[0] && ggml_is_exl3(n->src[0]->type) &&
                n->src[1] && ggml_nrows(n->src[1]) > EXL3_GEMV_MAX_M) {
            return n;
        }
    }
    return nullptr;
}

bool exl3_match_resid_add(const ggml_cgraph * g, int after, const ggml_tensor * dst,
                          ggml_tensor ** add, const ggml_tensor ** resid, int * add_idx) {
    if (after + 1 >= g->n_nodes || !dst) {
        return false;
    }
    const int hi = after + 4 < g->n_nodes ? after + 4 : g->n_nodes;
    for (int j = after + 1; j < hi; ++j) {
        ggml_tensor * n = g->nodes[j];
        if (n->op != GGML_OP_ADD || n->type != GGML_TYPE_F32) {
            continue;
        }
        if (!ggml_is_contiguous(n) || !ggml_are_same_shape(n, dst)) {
            continue;
        }
        const bool s0 = n->src[0] == dst || exl3_from_node(n->src[0], dst);
        const bool s1 = n->src[1] == dst || exl3_from_node(n->src[1], dst);
        if (s0 == s1) {
            continue;
        }
        const ggml_tensor * r = s0 ? n->src[1] : n->src[0];
        if (!r || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || !ggml_are_same_shape(r, n)) {
            continue;
        }
        *add = n;
        *resid = r;
        *add_idx = j;
        return true;
    }
    return false;
}

int exl3_try_ffn_mid(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, const exl3_chain & up) {
    static const int fuse_pp = ggml_sycl_get_env("GGML_SYCL_EXL3_FUSE_PREFILL", 1);
    static int nfail = 0;
    const int dump = ggml_sycl_get_env("GGML_SYCL_EXL3_DUMP_FFN", 0);
    auto fail = [&](const char * why, int a, int b) {
        if (dump && nfail < 4 && up.mm && up.mm->ne[0] == 17408) {
            fprintf(stderr, "EXL3_FFN mid_fail %s up=[%d,%d] N=%ld\n",
                    why, up.first, up.last, (long) up.mm->ne[0]);
            exl3_dump_nodes(cgraph, a, b, why);
            nfail++;
        }
        return 0;
    };
    if (!fuse_pp || ggml_nrows(up.act) <= EXL3_GEMV_MAX_M || !up.svh) {
        return 0;
    }
    exl3_chain gate;
    if (!exl3_match_chain(cgraph, up.last + 1, gate)) {
        return fail("no_gate", up.last, up.last + 24);
    }
    if (gate.act != up.act || !gate.svh) {
        return fail("gate_act", up.first, gate.last + 8);
    }
    if (gate.mm->ne[0] != up.mm->ne[0] || ggml_nrows(gate.act) != ggml_nrows(up.act)) {
        return fail("gate_shape", up.first, gate.last + 8);
    }
    const ggml_tensor * gview = gate.dst;
    int gi = exl3_follow_cast(cgraph, gate.last + 1, gate.dst, &gview);
    if (gi >= cgraph->n_nodes) {
        return fail("no_glu_idx", gate.last, gate.last + 16);
    }
    ggml_tensor * glu = cgraph->nodes[gi];
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return fail("no_glu", gate.last, gate.last + 16);
    }
    if ((glu->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, gi) != 1) {
        return fail("glu_uses", gate.last, gi + 8);
    }
    const bool a_is_gate = exl3_from_node(glu->src[0], up.dst) && exl3_from_node(glu->src[1], gate.dst);
    const bool b_is_gate = exl3_from_node(glu->src[0], gate.dst) && exl3_from_node(glu->src[1], up.dst);
    if (!a_is_gate && !b_is_gate) {
        return fail("glu_src", gate.last, gi + 8);
    }
    const exl3_chain & ch_gate = a_is_gate ? up : gate;
    const exl3_chain & ch_up = a_is_gate ? gate : up;
    const ggml_tensor * dview = glu;
    int di = exl3_follow_cast(cgraph, gi + 1, glu, &dview);
    exl3_chain down;
    if (di >= cgraph->n_nodes || !exl3_match_chain(cgraph, di, down)) {
        return fail("no_down", gi, gi + 24);
    }
    if (!down.svh || down.mm->src[0]->ne[0] != up.mm->ne[0]) {
        return fail("down_shape", gi, down.last + 4);
    }
    if (!(down.act == glu || down.act == dview || exl3_from_node(down.act, glu))) {
        return fail("down_act", gi, down.last + 4);
    }
    if (!exl3_uses_ok(cgraph, gate.last + 1, down.last)) {
        return fail("uses", gate.last, down.last + 1);
    }

    static int ndump = 0;
    if (ndump < 3 && ggml_sycl_get_env("GGML_SYCL_EXL3_DUMP_FFN", 0)) {
        const int hi = down.last + 6 < cgraph->n_nodes ? down.last + 6 : cgraph->n_nodes;
        exl3_dump_nodes(cgraph, up.first, hi, "mid");
        ndump++;
    }

    const int64_t M = ggml_nrows(up.act);
    const int64_t N = up.mm->ne[0];
    const int64_t Kd = down.mm->src[0]->ne[0];
    ggml_sycl_pool_alloc<float> c_up(ctx.pool());
    ggml_sycl_pool_alloc<float> c_gate(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> x_down(ctx.pool());
    c_up.alloc((size_t) M * (size_t) N);
    c_gate.alloc((size_t) M * (size_t) N);
    x_down.alloc((size_t) M * (size_t) Kd);
    float * raw_first = a_is_gate ? c_gate.get() : c_up.get();
    float * raw_second = a_is_gate ? c_up.get() : c_gate.get();

    const int pf_early = ggml_sycl_get_env("GGML_SYCL_EXL3_PF_EARLY", 0);
    if (pf_early) {
        g_exl3_lookahead = gate.mm;
        g_exl3_lookahead2 = down.mm;
        ggml_sycl_mul_mat_exl3(ctx, up.mm->src[0], nullptr, up.dst, up.act, up.suh, nullptr, nullptr, raw_first);
        ggml_sycl_mul_mat_exl3(ctx, gate.mm->src[0], nullptr, gate.dst, gate.act, gate.suh, nullptr, nullptr, raw_second);
    } else {
        ggml_sycl_exl3_set_lookahead(gate.mm);
        ggml_sycl_mul_mat_exl3(ctx, up.mm->src[0], nullptr, up.dst, up.act, up.suh, nullptr, nullptr, raw_first);
        ggml_sycl_exl3_set_lookahead(down.mm);
        ggml_sycl_mul_mat_exl3(ctx, gate.mm->src[0], nullptr, gate.dst, gate.act, gate.suh, nullptr, nullptr, raw_second);
    }

    queue_ptr stream = ctx.stream();
    const int time_pp = ggml_sycl_get_env("GGML_SYCL_EXL3_TIME", 0);
    if (time_pp && !g_exl3_ppb.armed) {
        g_exl3_ppb.armed = 1;
        atexit(exl3_ppb_dump);
    }
    if (time_pp) {
        stream->wait();
    }
    const auto t_mid0 = std::chrono::steady_clock::now();
    launch_exl3_ffn_mid(c_gate.get(), c_up.get(),
                        (const float *) ch_gate.svh->data, (const float *) ch_up.svh->data,
                        (const float *) down.suh->data, x_down.get(),
                        (M * N) / 128, N, stream);
    if (time_pp) {
        stream->wait();
        const double ms_mid = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_mid0).count();
        g_exl3_ppb.mid += ms_mid;
        g_exl3_ppb.mid_b += double(M) * double(N) * 10.0;
        g_exl3_ppb.n_mid += 1;
        static thread_local int ntm = 0;
        if (ntm < 8) {
            const double gbs = (ms_mid > 0.0) ? (double(M) * double(N) * 10.0 / 1e9) / (ms_mid / 1e3) : 0.0;
            fprintf(stderr, "EXL3_SYCL mid M=%ld N=%ld ms=%.2f gbs=%.1f\n",
                    (long) M, (long) N, ms_mid, gbs);
            fflush(stderr);
            ntm++;
        }
    }

    ggml_tensor * add_n = nullptr;
    const ggml_tensor * resid = nullptr;
    int add_idx = -1;
    int last = down.last;
    static const int post_add = ggml_sycl_get_env("GGML_SYCL_EXL3_POST_ADD", 1);
    if (post_add && exl3_match_resid_add(cgraph, down.last, down.dst, &add_n, &resid, &add_idx)) {
        g_exl3_add_src = (const float *) resid->data;
        g_exl3_add_dst = (float *) add_n->data;
        last = add_idx;
    }
    ggml_sycl_exl3_set_lookahead(exl3_next_mm(cgraph, last));
    ggml_sycl_mul_mat_exl3(ctx, down.mm->src[0], nullptr, down.dst, down.act, down.suh, down.svh,
                           x_down.get(), nullptr);
    return last - up.first;
}

} // namespace

int ggml_sycl_try_exl3_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    if (!g_ggml_sycl_enable_fusion) {
        return 0;
    }
    static const int env = ggml_sycl_get_env("GGML_SYCL_EXL3_FUSE", 1);
    if (env == 0) {
        return 0;
    }
    if (!ggml_sycl_exl3_available(ctx.device)) {
        return 0;
    }
    exl3_chain ch;
    if (!exl3_match_chain(cgraph, node_idx, ch)) {
        return 0;
    }

    const int mid_skip = exl3_try_ffn_mid(ctx, cgraph, ch);
    if (mid_skip > 0) {
        return mid_skip;
    }

    if (ggml_nrows(ch.act) > EXL3_GEMV_MAX_M) {
        static const int fuse_pp = ggml_sycl_get_env("GGML_SYCL_EXL3_FUSE_PREFILL", 1);
        if (!fuse_pp && !(ch.svh && exl3_recon_had_want(ggml_nrows(ch.act)))) {
            return 0;
        }
        ggml_tensor * add_n = nullptr;
        const ggml_tensor * resid = nullptr;
        int add_idx = -1;
        int last = ch.last;
        static const int post_add = ggml_sycl_get_env("GGML_SYCL_EXL3_POST_ADD", 1);
        if (post_add && ch.svh && exl3_match_resid_add(cgraph, ch.last, ch.dst, &add_n, &resid, &add_idx)) {
            g_exl3_add_src = (const float *) resid->data;
            g_exl3_add_dst = (float *) add_n->data;
            last = add_idx;
        }
        ggml_sycl_exl3_set_lookahead(exl3_next_mm(cgraph, last));
        ggml_sycl_mul_mat_exl3(ctx, ch.mm->src[0], nullptr, ch.dst, ch.act, ch.suh, ch.svh);
        return last - node_idx;
    }

    ggml_sycl_mul_mat_exl3(ctx, ch.mm->src[0], nullptr, ch.dst, ch.act, ch.suh, ch.svh);
    return ch.last - node_idx;
}
