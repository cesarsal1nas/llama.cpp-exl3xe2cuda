#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <optional>
#include <unordered_map>
#include <vector>

#include "fattn-onednn.hpp"
#include "fattn-tile.hpp"
#include "convert.hpp"
#include "dequantize.hpp"
#include "fa_pkv_inv.hpp"
#include "fa_pkv_gen.hpp"

// set minimum query length to treat as prefill (32)
#define GGML_SYCL_FA_ONEDNN_MIN_Q 32

bool ggml_sycl_fattn_onednn_binds_kv(const ggml_tensor * K, const ggml_tensor * V) {
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }
    auto bindable = [](const ggml_tensor * t) {
        return t->nb[0] == sizeof(sycl::half) && t->nb[1] % sizeof(sycl::half) == 0 &&
               t->nb[2] % sizeof(sycl::half) == 0 && t->nb[3] % sizeof(sycl::half) == 0;
    };
    return bindable(K) && bindable(V);
}

bool ggml_sycl_flash_attn_ext_onednn_supported(const ggml_tensor * dst, bool use_shape_limit) {
#if !GGML_SYCL_DNNL
    GGML_UNUSED(dst);
    GGML_UNUSED(use_shape_limit);
    return false;
#else
    if (!g_ggml_sycl_fa_onednn) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // F16 KV: native SDPA at any KV length.
    // Non-F16: dequant to F16 then SDPA at prefill lengths. Only the
    // standard quantized KV cache types (Q4_0-Q8_0) and F32 are accepted
    // because their to_fp16_sycl conversion is verified. BF16 and IQ*
    // are excluded: BF16 needs a strided conversion kernel that does not
    // exist yet; IQ types are model-weight-only quants with no dequant
    // registration and are never used as KV caches.
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        auto kt = K->type, vt = V->type;
        bool k_ok = kt == GGML_TYPE_F32 || kt == GGML_TYPE_Q4_0 || kt == GGML_TYPE_Q4_1 ||
                    kt == GGML_TYPE_Q5_0 || kt == GGML_TYPE_Q5_1 || kt == GGML_TYPE_Q8_0;
        bool v_ok = vt == GGML_TYPE_F32 || vt == GGML_TYPE_Q4_0 || vt == GGML_TYPE_Q4_1 ||
                    vt == GGML_TYPE_Q5_0 || vt == GGML_TYPE_Q5_1 || vt == GGML_TYPE_Q8_0;
        if (!k_ok || !v_ok) {
            return false;
        }
        if (use_shape_limit && (Q->ne[1] < 32 || K->ne[1] < 1024)) {
            return false;
        }
        for (const ggml_tensor * t : {K, V}) {
            if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                return false;
            }
        }
    }
    // This is the improved SPDA gate. Rather than gating Alchemist GPUs from all SPDA features, we instead target only the failing shapes.
    // If the GPU being assessed isn't in the grouping below, it has full access to all SPDA shapes. Otherwise, if it's an Alchemist GPU, we block only the shapes with head sizes that fail.
    // It is much easier to compare the device to a small list of failing cases than to define all the passing ones.
    const gpu_arch arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
    bool support_spda = !(arch == gpu_arch::intel_gpu_dg2_g10 ||
                         arch == gpu_arch::intel_gpu_dg2_g11 ||
                         arch == gpu_arch::intel_gpu_dg2_g12);

    if (!support_spda && K->ne[0] == 64) {
        return false;
    }
    // Optional KV-length ceiling (GGML_SYCL_FA_ONEDNN_MAX_KV, 0 = unlimited). Escape hatch:
    // very long sequences make the fused SDPA slow enough to risk the xe driver watchdog on
    // some stacks; past the cap we fall back to the native FA kernel instead.
    if (g_ggml_sycl_fa_onednn_max_kv > 0 && K->ne[1] > g_ggml_sycl_fa_onednn_max_kv) {
        return false;
    }
    // gate for the following cases
    // 1. if the oneDNN graph Add node has no input --> skip
    // 2. types other than f16 need different logical_tensor declaration
    // 3. the mask must be shape [1, 1, q, seq]
    // 4. sinks: excludes attention sink (Xiao et al., 2024) that can't be modeled by oneDNN graph
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || sinks) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // K and V must share head_dim: the SDPA graph uses a single `d` for both.
    const int64_t d = K->ne[0];
    if (V->ne[0] != d || Q->ne[3] != 1) {
        return false;
    }
    // GQA must divide evenly.
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    static const int min_q = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MIN_Q", 4);
    if (use_shape_limit && Q->ne[1] < min_q) {
        return false;
    }
    return true;
#endif
}

#if GGML_SYCL_DNNL

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace dnnl;
using namespace dnnl::graph;

// strided src (f16 or f32) -> contiguous f16 [ne0,ne1,ne2,ne3] (ne0 innermost). nb* are BYTE strides.
template <typename src_t>
static void cont_to_f16_sycl(const char * src, sycl::half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, dpct::queue_ptr stream, float qsc = 1.0f) {
    const int64_t n = ne0 * ne1 * ne2 * ne3;
    if ((ne0 % 8) == 0 && n > 0) {
        const int64_t n8 = n / 8;
        const int64_t ne0v = ne0 / 8;
        stream->parallel_for(sycl::range<1>((size_t) n8), [=](sycl::id<1> ix) {
            int64_t i = (int64_t) ix[0];
            const int64_t i0v = i % ne0v; i /= ne0v;
            const int64_t i1 = i % ne1; i /= ne1;
            const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
            const src_t * p = (const src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + (i0v * 8);
            sycl::half * o = dst + ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0v * 8;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float x = (float) p[k];
                o[k] = (sycl::half) (qsc == 1.0f ? x : x * qsc);
            }
        });
        return;
    }
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t i0 = i % ne0; i /= ne0;
        const int64_t i1 = i % ne1; i /= ne1;
        const int64_t i2 = i % ne2; const int64_t i3 = i / ne2;
        const src_t * p = (const src_t *) (src + i1 * nb1 + i2 * nb2 + i3 * nb3) + i0;
        dst[gid] = (sycl::half) (qsc == 1.0f ? (float) (*p) : (float) (*p) * qsc);
    });
}

static void permute_sdpa_out_sycl(const sycl::half * out, float * dst,
        int64_t mb, int64_t H, int64_t q, int64_t d, dpct::queue_ptr stream) {
    const int64_t n = mb * H * q * d;
    if ((d % 8) == 0 && n > 0) {
        const int64_t n8 = n / 8;
        const int64_t dv = d / 8;
        stream->parallel_for(sycl::range<1>((size_t) n8), [=](sycl::id<1> ix) {
            int64_t i = (int64_t) ix[0];
            const int64_t e8 = i % dv; i /= dv;
            const int64_t t = i % q; i /= q;
            const int64_t h = i % H; const int64_t b = i / H;
            const int64_t e = e8 * 8;
            const sycl::half * op = out + e + t * d + h * q * d + b * H * q * d;
            float * dp = dst + e + h * d + t * d * H + b * d * H * q;
            const sycl::vec<sycl::half, 8> hv = *reinterpret_cast<const sycl::vec<sycl::half, 8> *>(op);
            sycl::vec<float, 8> fv;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                fv[k] = (float) hv[k];
            }
            *reinterpret_cast<sycl::vec<float, 8> *>(dp) = fv;
        });
        return;
    }
    stream->parallel_for(sycl::range<1>(n), [=](sycl::id<1> ix) {
        const int64_t gid = ix[0];
        int64_t       i   = gid;
        const int64_t e = i % d; i /= d;
        const int64_t t = i % q; i /= q;
        const int64_t h = i % H; const int64_t b = i / H;
        dst[e + h * d + t * d * H + b * d * H * q] = (float) out[gid];
    });
}

static void fa_jm_sdpa_f16(const sycl::half * Q, const sycl::half * K, const sycl::half * V, sycl::half * O,
        int H, int Hkv, int q, int seq, int d, float scale, dpct::queue_ptr stream) {
    constexpr int BM = 8;
    constexpr int BN = 16;
    constexpr int BK = 16;
    constexpr int D  = 256;
    if (d != D || (q % BM) != 0 || (seq % BN) != 0 || Hkv <= 0 || (H % Hkv) != 0) {
        return;
    }
    const int nqt = q / BM;
    const int gqa = H / Hkv;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm_s(sycl::range<1>((size_t) BM * BN), cgh);
        sycl::local_accessor<float, 1> slm_o(sycl::range<1>((size_t) BM * D), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) H * nqt * 16), sycl::range<1>(16)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                using namespace sycl::ext::oneapi::experimental::matrix;
                auto sg = it.get_sub_group();
                const int lane = (int) it.get_local_id(0);
                const int g    = (int) it.get_group(0);
                const int h    = g / nqt;
                const int qt   = g % nqt;
                const int t0   = qt * BM;
                const int hk   = h / gqa;
                const sycl::half * Qh = Q + ((size_t) h * q) * D;
                const sycl::half * Kh = K + ((size_t) hk * seq) * D;
                const sycl::half * Vh = V + ((size_t) hk * seq) * D;
                sycl::half * Oh = O + ((size_t) h * q) * D;
                const int q_base = seq - q;
                float m[BM];
                float lsum[BM];
#pragma unroll
                for (int i = 0; i < BM; ++i) {
                    m[i]    = -1.0e30f;
                    lsum[i] = 0.0f;
                }
                for (int e = lane; e < BM * D; e += 16) {
                    slm_o[e] = 0.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);
                for (int j0 = 0; j0 < seq; j0 += BN) {
                    joint_matrix<sycl::sub_group, float, use::accumulator, BM, BN> S;
                    joint_matrix_fill(sg, S, 0.0f);
                    for (int e0 = 0; e0 < D; e0 += BK) {
                        joint_matrix<sycl::sub_group, sycl::half, use::a, BM, BK, layout::row_major> A;
                        joint_matrix<sycl::sub_group, sycl::half, use::b, BK, BN, layout::col_major> B;
                        sycl::half * qp = const_cast<sycl::half *>(Qh + (size_t) t0 * D + e0);
                        sycl::half * kp = const_cast<sycl::half *>(Kh + (size_t) j0 * D + e0);
                        auto pa = sycl::address_space_cast<sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(qp);
                        auto pb = sycl::address_space_cast<sycl::access::address_space::global_space,
                            sycl::access::decorated::no>(kp);
                        joint_matrix_load(sg, A, pa, D);
                        joint_matrix_load(sg, B, pb, D);
                        joint_matrix_mad(sg, S, A, B, S);
                    }
                    float * sp = slm_s.get_multi_ptr<sycl::access::decorated::no>().get();
                    auto pc = sycl::address_space_cast<sycl::access::address_space::local_space,
                        sycl::access::decorated::no>(sp);
                    joint_matrix_store(sg, S, pc, BN, layout::row_major);
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (int i = 0; i < BM; ++i) {
                        const int t = t0 + i;
                        const int j = j0 + lane;
                        float s = slm_s[i * BN + lane];
                        const int keep = (t < q && j < seq && j <= q_base + t) ? 1 : 0;
                        s = keep ? (s * scale) : -1.0e30f;
                        const float mx   = sycl::reduce_over_group(sg, s, sycl::maximum<float>());
                        const float mnew = m[i] > mx ? m[i] : mx;
                        const float alpha = sycl::exp(m[i] - mnew);
                        float p = sycl::exp(s - mnew);
                        if (keep == 0) {
                            p = 0.0f;
                        }
                        const float psum = sycl::reduce_over_group(sg, p, sycl::plus<float>());
                        lsum[i] = lsum[i] * alpha + psum;
                        m[i]    = mnew;
                        slm_s[i * BN + lane] = p;
                        for (int e = lane; e < D; e += 16) {
                            slm_o[i * D + e] *= alpha;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                    for (int i = 0; i < BM; ++i) {
                        float * o = slm_o.get_multi_ptr<sycl::access::decorated::no>().get() + i * D;
#pragma unroll
                        for (int n = 0; n < BN; ++n) {
                            const float p = slm_s[i * BN + n];
                            if (p == 0.0f) {
                                continue;
                            }
                            const sycl::half * v = Vh + (size_t) (j0 + n) * D;
                            for (int e = lane; e < D; e += 16) {
                                o[e] += p * (float) v[e];
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
#pragma unroll
                for (int i = 0; i < BM; ++i) {
                    const int t = t0 + i;
                    const float inv = lsum[i] > 0.0f ? (1.0f / lsum[i]) : 0.0f;
                    for (int e = lane; e < D; e += 16) {
                        Oh[(size_t) t * D + e] = (sycl::half) (slm_o[i * D + e] * inv);
                    }
                }
            });
    });
}

struct fa_pkv_slot {
    const void * kdata = nullptr;
    const void * vdata = nullptr;
    int64_t d = 0;
    int64_t Hkv = 0;
    int64_t seq_cap = 0;
    int64_t filled = 0;
    uint64_t gen = 0;
    sycl::half * Kf = nullptr;
    sycl::half * Vf = nullptr;
};

static std::vector<fa_pkv_slot> g_fa_pkv;

static const void * fa_pkv_base_data(const ggml_tensor * t) {
    if (!t) {
        return nullptr;
    }
    const ggml_tensor * b = t;
    while (b->view_src) {
        b = b->view_src;
    }
    return b->data ? b->data : t->data;
}

static void fa_pkv_free_slot_bufs(fa_pkv_slot * sl, dpct::queue_ptr stream) {
    if (!sl) {
        return;
    }
    if (sl->Kf) {
        sycl::free(sl->Kf, *stream);
        sl->Kf = nullptr;
    }
    if (sl->Vf) {
        sycl::free(sl->Vf, *stream);
        sl->Vf = nullptr;
    }
    sl->filled  = 0;
    sl->seq_cap = 0;
    sl->gen     = 0;
}

static bool fa_q4_dense_hntd(const char * src, sycl::half * dst, int64_t seq, int64_t d, int64_t Hkv,
                             dpct::queue_ptr stream) {
    if (!src || !dst || seq < 1 || Hkv < 1 || (d % 32) != 0 || d < 32 || d > 256) {
        return false;
    }
    const int nblk = (int) (d / 32);
    const int64_t s01 = Hkv * (int64_t) nblk;
    const int64_t s02 = (int64_t) nblk;
    const int wgs = (int) d;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) seq * (size_t) Hkv * (size_t) wgs),
                          sycl::range<1>((size_t) wgs)),
        [=](sycl::nd_item<1> it) {
            const int row = (int) it.get_group(0);
            const int t = row / (int) Hkv;
            const int h = row - t * (int) Hkv;
            const int j = (int) it.get_local_id(0);
            const int ib = j / 32;
            const int jl = j - ib * 32;
            const block_q4_0 * b = (const block_q4_0 *) src + (int64_t) t * s01 + (int64_t) h * s02 + ib;
            const float scale = (float) b->d;
            const int vui = b->qs[jl < 16 ? jl : jl - 16];
            const int nib = (jl < 16) ? (vui & 0xF) : (vui >> 4);
            dst[((int64_t) h * seq + (int64_t) t) * d + j] = (sycl::half) (((float) nib - 8.0f) * scale);
        });
    return true;
}

static bool fa_q4_dense_hdts(const char * src, sycl::half * dst, int64_t seq, int64_t d, int64_t Hkv,
                             dpct::queue_ptr stream) {
    if (!src || !dst || seq < 1 || Hkv < 1 || (d % 32) != 0 || d < 32 || d > 256) {
        return false;
    }
    const int nblk = (int) (d / 32);
    const int64_t s01 = Hkv * (int64_t) nblk;
    const int64_t s02 = (int64_t) nblk;
    const int wgs = (int) d;
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t) seq * (size_t) Hkv * (size_t) wgs),
                          sycl::range<1>((size_t) wgs)),
        [=](sycl::nd_item<1> it) {
            const int row = (int) it.get_group(0);
            const int t = row / (int) Hkv;
            const int h = row - t * (int) Hkv;
            const int j = (int) it.get_local_id(0);
            const int ib = j / 32;
            const int jl = j - ib * 32;
            const block_q4_0 * b = (const block_q4_0 *) src + (int64_t) t * s01 + (int64_t) h * s02 + ib;
            const float scale = (float) b->d;
            const int vui = b->qs[jl < 16 ? jl : jl - 16];
            const int nib = (jl < 16) ? (vui & 0xF) : (vui >> 4);
            dst[(((int64_t) h * d) + j) * seq + t] = (sycl::half) (((float) nib - 8.0f) * scale);
        });
    return true;
}

static void fa_q4_pack_u4_k(const char * src, uint8_t * dst, sycl::half * sc, int8_t * zp,
                            int64_t seq, int64_t d, int64_t Hdst, int gqa, size_t nb_seq, size_t nb_head,
                            dpct::queue_ptr stream) {
    const int nblk = (int) (d / 32);
    const int64_t seq2 = seq / 2;
    stream->parallel_for(sycl::range<1>((size_t) Hdst * (size_t) d * (size_t) seq2), [=](sycl::id<1> gid) {
        int64_t i = (int64_t) gid[0];
        const int64_t t2 = i % seq2; i /= seq2;
        const int64_t j = i % d; i /= d;
        const int64_t hd = i;
        const int64_t h = hd / (int64_t) gqa;
        const int64_t t0 = t2 * 2;
        const int64_t t1 = t0 + 1;
        const int ib = (int) (j / 32);
        const int jl = (int) (j - (int64_t) ib * 32);
        const block_q4_0 * b0 = (const block_q4_0 *) (src + t0 * (int64_t) nb_seq + h * (int64_t) nb_head) + ib;
        const block_q4_0 * b1 = (const block_q4_0 *) (src + t1 * (int64_t) nb_seq + h * (int64_t) nb_head) + ib;
        const int vui0 = b0->qs[jl < 16 ? jl : jl - 16];
        const int vui1 = b1->qs[jl < 16 ? jl : jl - 16];
        const int n0 = (jl < 16) ? (vui0 & 0xF) : (vui0 >> 4);
        const int n1 = (jl < 16) ? (vui1 & 0xF) : (vui1 >> 4);
        dst[((hd * d + j) * seq + t0) / 2] = (uint8_t) ((n0 & 15) | ((n1 & 15) << 4));
        if (jl == 0) {
            sc[(hd * (int64_t) nblk + ib) * seq + t0] = b0->d;
            sc[(hd * (int64_t) nblk + ib) * seq + t1] = b1->d;
            zp[(hd * (int64_t) nblk + ib) * seq + t0] = (int8_t) 8;
            zp[(hd * (int64_t) nblk + ib) * seq + t1] = (int8_t) 8;
        }
    });
}

static void fa_q4_pack_u4(const char * src, uint8_t * dst, sycl::half * sc, int8_t * zp,
                          int64_t seq, int64_t d, int64_t Hdst, int gqa, size_t nb_seq, size_t nb_head,
                          dpct::queue_ptr stream) {
    const int nblk = (int) (d / 32);
    const int64_t nrows = seq * Hdst;
    const int64_t dh = d / 2;
    stream->parallel_for(sycl::range<1>((size_t) nrows), [=](sycl::id<1> gid) {
        const int64_t row = (int64_t) gid[0];
        const int64_t t = row / Hdst;
        const int64_t hd = row % Hdst;
        const int64_t h = hd / (int64_t) gqa;
        const block_q4_0 * b = (const block_q4_0 *) (src + t * (int64_t) nb_seq + h * (int64_t) nb_head);
        uint8_t * out = dst + (hd * seq + t) * dh;
        sycl::half * so = sc + (hd * seq + t) * nblk;
        int8_t * zo = zp + (hd * seq + t) * nblk;
        for (int ib = 0; ib < nblk; ib++) {
            so[ib] = b[ib].d;
            zo[ib] = (int8_t) 8;
            for (int j = 0; j < 16; j++) {
                const int j0 = j * 2;
                const int j1 = j0 + 1;
                const int vui0 = b[ib].qs[j0 < 16 ? j0 : j0 - 16];
                const int vui1 = b[ib].qs[j1 < 16 ? j1 : j1 - 16];
                const int n0 = (j0 < 16) ? (vui0 & 0xF) : (vui0 >> 4);
                const int n1 = (j1 < 16) ? (vui1 & 0xF) : (vui1 >> 4);
                out[ib * 16 + j] = (uint8_t) ((n0 & 15) | ((n1 & 15) << 4));
            }
        }
    });
}

// Dequant Q4_0 rows into F16 shadow [Hkv, seq_cap, d]. Returns false if args
// invalid or range exceeds seq_cap (no partial write, no launch).
static bool fa_pkv_q4_rows(const char * src, sycl::half * dst, int64_t t0, int64_t n_new, int64_t d,
                           int64_t Hkv, int64_t seq_cap, size_t nb_seq, size_t nb_head, dpct::queue_ptr stream) {
    if (!src || !dst || t0 < 0 || n_new < 1 || Hkv < 1 || seq_cap < 1 || d < 1) {
        return false;
    }
    if (t0 >= seq_cap || t0 + n_new > seq_cap) {
        return false;
    }
    const int64_t nrows = n_new * Hkv;
    const int nblk = (int) (d / 32);
    if (nrows <= 0 || nblk <= 0) {
        return false;
    }
    stream->parallel_for(sycl::range<1>((size_t) nrows), [=](sycl::id<1> gid) {
        const int64_t row = (int64_t) gid[0];
        const int64_t t = t0 + row / Hkv;
        const int64_t h = row % Hkv;
        const block_q4_0 * b = (const block_q4_0 *) (src + t * (int64_t) nb_seq + h * (int64_t) nb_head);
        sycl::half * out = dst + (h * seq_cap + t) * d;
        for (int ib = 0; ib < nblk; ib++) {
            const float scale = (float) b[ib].d;
            for (int j = 0; j < 16; j++) {
                const int vui = b[ib].qs[j];
                out[ib * 32 + j] = (sycl::half) (((float) (vui & 0xF) - 8.0f) * scale);
                out[ib * 32 + 16 + j] = (sycl::half) (((float) (vui >> 4) - 8.0f) * scale);
            }
        }
    });
    return true;
}

// PKV F16 shadow (GGML_SYCL_FA_ONEDNN_PKV):
//   0 off; 1 grow round-2048; >=2 fixed cap (default 32768, env PKV_CAP).
//   seq > cap: free this layer slot if present, return false (dense stage fallback).
// Default full-refresh [0,seq) - padded n_kv can be stale. PKV_INC=1 allows
// gen-gated incremental (set_rows host gen); still full-refresh if gen missing/mismatch.
static bool fa_pkv_stage(const ggml_tensor * K, const ggml_tensor * V, int64_t d, int64_t Hkv, int64_t seq,
                         dpct::queue_ptr stream, sycl::half ** K_ptr, sycl::half ** V_ptr,
                         std::array<int64_t, 5> * k_str, std::array<int64_t, 5> * v_str) {
    const int pkv_mode = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_PKV", 0);
    if (pkv_mode == 0) {
        return false;
    }
    if (!K || !V || !K->data || !V->data || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if ((d % 32) != 0 || Hkv < 1 || seq < 1) {
        return false;
    }
    auto is_pad = [](const ggml_tensor * t) {
        const bool non_dense = ((int64_t) t->ne[1] * t->nb[1] != t->nb[2]) && t->ne[2] > 1;
        return non_dense && ((int64_t) t->nb[2] < (int64_t) t->ne[1] * (int64_t) t->nb[1]);
    };
    if (!is_pad(K) || !is_pad(V)) {
        return false;
    }
    if (K->nb[1] != V->nb[1] || K->nb[2] != V->nb[2]) {
        return false;
    }

    int64_t seq_cap;
    if (pkv_mode >= 2) {
        seq_cap = 32768;
        const int env_cap = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_PKV_CAP", 0);
        if (env_cap > 0) {
            seq_cap = (int64_t) env_cap;
        }
    } else {
        seq_cap = fa_pkv_round_cap(seq, 2048);
    }
    const ggml_tensor * back = K->view_src ? K->view_src : K;
    while (back->view_src) {
        back = back->view_src;
    }
    if (back->ne[1] >= seq && seq_cap > back->ne[1]) {
        seq_cap = back->ne[1];
    }

    const void * kbase = fa_pkv_base_data(K);
    const void * vbase = fa_pkv_base_data(V);
    const bool allow_inc = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_PKV_INC", 0) != 0;
    const uint64_t live_gen = ggml_sycl_fa_pkv_get_gen(kbase);
    const bool have_gen = live_gen != 0;

    fa_pkv_slot * sl = nullptr;
    size_t sl_idx = (size_t) -1;
    for (size_t i = 0; i < g_fa_pkv.size(); i++) {
        if (g_fa_pkv[i].kdata == kbase) {
            sl = &g_fa_pkv[i];
            sl_idx = i;
            break;
        }
    }

    // Over-cap: free this layer only (not global), then dense fallback.
    if (seq > seq_cap) {
        if (sl) {
            try {
                stream->wait();
            } catch (...) {
                return false;
            }
            fa_pkv_free_slot_bufs(sl, stream);
            g_fa_pkv.erase(g_fa_pkv.begin() + (std::ptrdiff_t) sl_idx);
        }
        static int nskip = 0;
        if (nskip < 4) {
            nskip++;
            GGML_SYCL_XE2_LOG( "xe2mx fa pkv bound-skip seq=%ld cap=%ld (dense fallback)\n",
                    (long) seq, (long) seq_cap);
            fflush(stderr);
        }
        return false;
    }

    if (!sl) {
        g_fa_pkv.push_back({});
        sl = &g_fa_pkv.back();
        sl->kdata = kbase;
    }

    const size_t n = (size_t) Hkv * (size_t) seq_cap * (size_t) d;
    if (!sl->Kf || sl->d != d || sl->Hkv != Hkv || sl->seq_cap != seq_cap || sl->vdata != vbase) {
        sycl::half * oldk = sl->Kf;
        sycl::half * oldv = sl->Vf;
        const int64_t old_filled = sl->filled;
        const int64_t old_cap = sl->seq_cap;
        const int keep = oldk && oldv && sl->vdata == vbase && sl->d == d && sl->Hkv == Hkv
            && old_filled > 0 && old_cap > 0 && seq_cap > old_cap;
        try {
            stream->wait();
        } catch (...) {
            return false;
        }
        sycl::half * nk = nullptr;
        sycl::half * nv = nullptr;
        try {
            nk = sycl::malloc_device<sycl::half>(n, *stream);
        } catch (...) {
            nk = nullptr;
        }
        try {
            nv = sycl::malloc_device<sycl::half>(n, *stream);
        } catch (...) {
            nv = nullptr;
        }
        if (!nk || !nv) {
            if (nk) {
                sycl::free(nk, *stream);
            }
            if (nv) {
                sycl::free(nv, *stream);
            }
            return false;
        }
        int64_t kept_filled = 0;
        if (keep) {
            const int64_t ncopy = old_filled > old_cap ? old_cap : old_filled;
            if (ncopy > 0) {
                for (int64_t h = 0; h < sl->Hkv; h++) {
                    const size_t bytes = (size_t) ncopy * (size_t) sl->d * sizeof(sycl::half);
                    stream->memcpy(nk + h * seq_cap * sl->d, oldk + h * old_cap * sl->d, bytes);
                    stream->memcpy(nv + h * seq_cap * sl->d, oldv + h * old_cap * sl->d, bytes);
                }
                try {
                    stream->wait();
                } catch (...) {
                    sycl::free(nk, *stream);
                    sycl::free(nv, *stream);
                    return false;
                }
                kept_filled = ncopy;
            }
        }
        if (oldk) {
            sycl::free(oldk, *stream);
        }
        if (oldv) {
            sycl::free(oldv, *stream);
        }
        sl->Kf = nk;
        sl->Vf = nv;
        sl->d = d;
        sl->Hkv = Hkv;
        sl->seq_cap = seq_cap;
        sl->filled = kept_filled;
        sl->vdata = vbase;
        // Capacity change: drop gen match so decide() full-refreshes unless pure keep prefix.
        if (!keep) {
            sl->gen = 0;
        }
    }
    if (!sl->Kf || !sl->Vf) {
        return false;
    }

    fa_pkv_inv_in inv_in;
    inv_in.seq               = seq;
    inv_in.seq_cap           = seq_cap;
    inv_in.filled            = sl->filled;
    inv_in.slot_gen          = sl->gen;
    inv_in.live_gen          = live_gen;
    inv_in.have_live_gen     = have_gen;
    inv_in.allow_incremental = allow_inc;
    const fa_pkv_inv_out inv = fa_pkv_decide(inv_in);
    if (inv.reject) {
        return false;
    }

    if (inv.need_refresh) {
        if (inv.n_new < 1 || inv.t0 < 0 || inv.t0 + inv.n_new > seq_cap) {
            return false;
        }
        const bool ok_k = fa_pkv_q4_rows((const char *) K->data, sl->Kf, inv.t0, inv.n_new, d, Hkv, seq_cap,
                                         K->nb[1], K->nb[2], stream);
        const bool ok_v = fa_pkv_q4_rows((const char *) V->data, sl->Vf, inv.t0, inv.n_new, d, Hkv, seq_cap,
                                         V->nb[1], V->nb[2], stream);
        if (!ok_k || !ok_v) {
            // Do not publish filled/gen on failed write.
            return false;
        }
        sl->filled = inv.filled_after;
        sl->gen    = inv.gen_after;
    }

    static int nfire = 0;
    if (nfire < 4) {
        nfire++;
        GGML_SYCL_XE2_LOG( "xe2mx fa pkv seq=%ld cap=%ld t0=%ld n=%ld inc=%d gen=%llu Hkv=%ld d=%ld\n",
                (long) seq, (long) seq_cap, (long) inv.t0, (long) inv.n_new, allow_inc ? 1 : 0,
                (unsigned long long) sl->gen, (long) Hkv, (long) d);
        fflush(stderr);
    }
    *K_ptr = sl->Kf;
    *V_ptr = sl->Vf;
    const int64_t hs = seq_cap * d;
    *k_str = { Hkv * hs, hs, hs, d, 1 };
    *v_str = *k_str;
    return true;
}

struct sdpa_partition {
    compiled_partition          cp;
    std::vector<logical_tensor> ins;
    logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    size_t id_seqk = 0, id_seqq = 0, id_ninf = 0;
    int32_t q_len = 0, kv_len = 0;
    float   ninf = 0.f;
    bool   causal = false;
    bool   u4 = false;
    bool   ok = false;
    size_t id_ksc = (size_t) -1, id_vsc = (size_t) -1, id_kzp = (size_t) -1, id_vzp = (size_t) -1;
    sycl::half scale_h = (sycl::half) 0;
    bool hsc = false;
};

// Build + compile the contiguous-input GQA SDPA graph (MatMul->Divide->Add->SoftMax->MatMul), f32 out.
// Mirrors the hardware-verified scratch/onednn_sdpa_probe.cpp build_gqa (partitions=1, sdp_primitive_kernel_t).
static sdpa_partition build_sdpa(const engine & eng, int H, int Hkv, int q, int seq, int d,
                                 const std::array<int64_t, 5> & k_str, const std::array<int64_t, 5> & v_str) try {
    using ltype = logical_tensor::layout_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d}, s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, msk = {1, 1, 1, q, seq}, o_sz = {1, Hkv, rep, q, d};
    const ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t,  q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t,  kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);          // key is [.., seq, d]
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");     // score / (1/kq_scale) == score * kq_scale
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto mask   = logical_tensor(id++, t,  msk,  ltype::strided);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto madd   = op(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask}); madd.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, v_st);
    // f16 output is REQUIRED to hit sdp_primitive_kernel_t (the systolic micro-kernel); an f32 output
    // falls to larger_partition_kernel_t which materializes N^2 (confirmed: scratch/onednn_sdpa_kernel_probe.cpp).
    // converted to the f32 ggml dst in the permute below.
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);   // f16 contiguous [mb,Hkv,rep,q,d]
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    const int fpm = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_FPM", 0);
    if (fpm == 1) {
        g.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    } else if (fpm == 2) {
        g.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    } else if (fpm == 3) {
        g.set_fpmath_mode(dnnl::fpmath_mode::any, true);
    } else if (fpm == 4) {
        g.set_fpmath_mode(dnnl::fpmath_mode::tf32, true);
    }
    g.add_op(bmm1); g.add_op(sdiv); g.add_op(madd); g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        GGML_LOG_WARN("%s: oneDNN did not fuse the SDPA graph; falling back to TILE kernel\n", __func__);
        return E;   // ok stays false -> caller falls back to TILE
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = scale.get_id(); E.id_mask = mask.get_id();
    E.ok       = true;
    return E;
}
catch (const std::exception & e) {
    // compile() can reject a stride set the partitioner never inspects; memoise the failure so the
    // fallback costs one build rather than one per call.
    GGML_LOG_WARN("%s: oneDNN SDPA partition build failed (%s); falling back to TILE kernel\n", __func__, e.what());
    return {};
}

static sdpa_partition build_sdpa_causal(const engine & eng, int H, int Hkv, int q, int seq, int d,
                                          const std::array<int64_t, 5> & k_str, const std::array<int64_t, 5> & v_str,
                                          int kds) try {
    using ltype = logical_tensor::layout_type;
    using ptype = logical_tensor::property_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    t = dt::f16;
    const dt    fi = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_S16", 0) ? t : dt::f32;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, kv_sz = {1, Hkv, 1, seq, d},
                kds_sz = {1, Hkv, 1, d, seq},
                s_sz = {1, Hkv, rep, q, seq},
                sc = {1, 1, 1, 1, 1}, o_sz = {1, Hkv, rep, q, d};
    const ldims k_st(k_str.begin(), k_str.end()), v_st(v_str.begin(), v_str.end());
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t, q_sz, ltype::strided);
    auto key    = logical_tensor(id++, t, kds ? kds_sz : kv_sz, k_st);
    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    if (!kds) {
        bmm1.set_attr<bool>(op::attr::transpose_b, true);
    }
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    const int nq = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_NQ", 0);
    const int smul = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_SMUL", 0);
    const int hsc = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_HSC", 0);
    logical_tensor scale = hsc
        ? logical_tensor(id++, t, 0, ltype::strided, ptype::host_scalar)
        : logical_tensor(id++, t, sc, ltype::strided);
    logical_tensor scaled(id++, fi, s_sz, ltype::strided);
    op sdiv(id++, smul ? op::kind::Multiply : op::kind::Divide, "scale_div");
    logical_tensor attn = score;
    if (!nq) {
        sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});
        attn = scaled;
    }

    auto index_row = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gen_index_row = op(id++, op::kind::GenIndex, "gen_index_row");
    gen_index_row.set_attr<int64_t>(op::attr::axis, -2);
    gen_index_row.add_inputs({attn});
    gen_index_row.add_outputs({index_row});

    auto seq_len_kv = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto mask_add_out = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({index_row, seq_len_kv});
    mask_add.add_outputs({mask_add_out});

    auto seq_len_q = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto mask_sub_out = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto mask_sub = op(id++, op::kind::Subtract, "mask_sub");
    mask_sub.add_inputs({mask_add_out, seq_len_q});
    mask_sub.add_outputs({mask_sub_out});

    auto index_col = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gen_index_col = op(id++, op::kind::GenIndex, "gen_index_col");
    gen_index_col.set_attr<int64_t>(op::attr::axis, -1);
    gen_index_col.add_inputs({attn});
    gen_index_col.add_outputs({index_col});

    auto mask_ge_out = logical_tensor(id++, dt::boolean, s_sz, ltype::strided);
    auto mask_ge = op(id++, op::kind::GreaterEqual, "mask_ge");
    mask_ge.add_inputs({mask_sub_out, index_col});
    mask_ge.add_outputs({mask_ge_out});

    auto neg_inf = logical_tensor(id++, fi, 0, ltype::strided, ptype::host_scalar);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto mask_select = op(id++, op::kind::Select, "mask_select");
    mask_select.add_inputs({mask_ge_out, attn, neg_inf});
    mask_select.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto value  = logical_tensor(id++, t,  kv_sz, v_st);
    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    const int fpm = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_FPM", 0);
    if (fpm == 1) {
        g.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    } else if (fpm == 2) {
        g.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    } else if (fpm == 3) {
        g.set_fpmath_mode(dnnl::fpmath_mode::any, true);
    } else if (fpm == 4) {
        g.set_fpmath_mode(dnnl::fpmath_mode::tf32, true);
    }
    g.add_op(bmm1);
    if (!nq) {
        g.add_op(sdiv);
    }
    g.add_op(gen_index_row); g.add_op(mask_add); g.add_op(mask_sub);
    g.add_op(gen_index_col); g.add_op(mask_ge); g.add_op(mask_select);
    g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        return E;
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id(); E.id_k = key.get_id(); E.id_v = value.get_id();
    E.id_scale = nq ? (size_t) -1 : scale.get_id();
    E.id_seqk  = seq_len_kv.get_id(); E.id_seqq = seq_len_q.get_id(); E.id_ninf = neg_inf.get_id();
    E.q_len    = q;
    E.kv_len   = seq;
    E.ninf     = -std::numeric_limits<float>::infinity();
    E.causal   = true;
    E.hsc      = hsc && !nq;
    E.ok       = true;
    return E;
}
catch (const std::exception &) {
    return {};
}

static sdpa_partition build_sdpa_causal_u4(const engine & eng, int H, int Hkv, int q, int seq, int d) try {
    using ltype = logical_tensor::layout_type;
    using ptype = logical_tensor::property_type;
    using dt    = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const dt    fi = dt::f32, t = dt::f16;
    const int   ng = d / 32;
    const int   rep = H / Hkv;
    const ldims q_sz = {1, Hkv, rep, q, d}, k_sz = {1, Hkv, 1, d, seq}, v_sz = {1, Hkv, 1, seq, d},
                s_sz = {1, Hkv, rep, q, seq}, sc = {1, 1, 1, 1, 1}, o_sz = {1, Hkv, rep, q, d},
                ksc_sz = {1, Hkv, 1, ng, seq}, vsc_sz = {1, Hkv, 1, seq, ng};
    const std::vector<int64_t> gsh_k = {1, 1, 1, 32, 1};
    const std::vector<int64_t> gsh_v = {1, 1, 1, 1, 32};
    int64_t        id = 0;
    sdpa_partition E;

    auto query  = logical_tensor(id++, t, q_sz, ltype::strided);
    auto k_u4   = logical_tensor(id++, dt::u4, k_sz, ltype::strided);
    auto k_sc   = logical_tensor(id++, t, ksc_sz, ltype::strided);
    auto k_zp   = logical_tensor(id++, dt::s8, ksc_sz, ltype::strided);
    auto key    = logical_tensor(id++, t, k_sz, ltype::strided);
    auto deq_k  = op(id++, op::kind::DynamicDequantize, "deq_k");
    deq_k.set_attr<std::string>(op::attr::qtype, "per_group");
    deq_k.set_attr<std::vector<int64_t>>(op::attr::group_shape, gsh_k);
    deq_k.add_inputs({k_u4, k_sc, k_zp});
    deq_k.add_outputs({key});

    auto v_u4   = logical_tensor(id++, dt::u4, v_sz, ltype::strided);
    auto v_sc   = logical_tensor(id++, t, vsc_sz, ltype::strided);
    auto v_zp   = logical_tensor(id++, dt::s8, vsc_sz, ltype::strided);
    auto value  = logical_tensor(id++, t, v_sz, ltype::strided);
    auto deq_v  = op(id++, op::kind::DynamicDequantize, "deq_v");
    deq_v.set_attr<std::string>(op::attr::qtype, "per_group");
    deq_v.set_attr<std::vector<int64_t>>(op::attr::group_shape, gsh_v);
    deq_v.add_inputs({v_u4, v_sc, v_zp});
    deq_v.add_outputs({value});

    auto score  = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto bmm1   = op(id++, op::kind::MatMul, "bmm1");
    bmm1.add_inputs({query, key}); bmm1.add_outputs({score});

    auto scale  = logical_tensor(id++, t,  sc,   ltype::strided);
    auto scaled = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto sdiv   = op(id++, op::kind::Divide, "scale_div");
    sdiv.add_inputs({score, scale}); sdiv.add_outputs({scaled});

    auto index_row = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gen_index_row = op(id++, op::kind::GenIndex, "gen_index_row");
    gen_index_row.set_attr<int64_t>(op::attr::axis, -2);
    gen_index_row.add_inputs({scaled});
    gen_index_row.add_outputs({index_row});

    auto seq_len_kv = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto mask_add_out = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto mask_add = op(id++, op::kind::Add, "mask_add");
    mask_add.add_inputs({index_row, seq_len_kv});
    mask_add.add_outputs({mask_add_out});

    auto seq_len_q = logical_tensor(id++, dt::s32, 0, ltype::strided, ptype::host_scalar);
    auto mask_sub_out = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto mask_sub = op(id++, op::kind::Subtract, "mask_sub");
    mask_sub.add_inputs({mask_add_out, seq_len_q});
    mask_sub.add_outputs({mask_sub_out});

    auto index_col = logical_tensor(id++, dt::s32, s_sz, ltype::strided);
    auto gen_index_col = op(id++, op::kind::GenIndex, "gen_index_col");
    gen_index_col.set_attr<int64_t>(op::attr::axis, -1);
    gen_index_col.add_inputs({scaled});
    gen_index_col.add_outputs({index_col});

    auto mask_ge_out = logical_tensor(id++, dt::boolean, s_sz, ltype::strided);
    auto mask_ge = op(id++, op::kind::GreaterEqual, "mask_ge");
    mask_ge.add_inputs({mask_sub_out, index_col});
    mask_ge.add_outputs({mask_ge_out});

    auto neg_inf = logical_tensor(id++, fi, 0, ltype::strided, ptype::host_scalar);
    auto masked = logical_tensor(id++, fi, s_sz, ltype::strided);
    auto mask_select = op(id++, op::kind::Select, "mask_select");
    mask_select.add_inputs({mask_ge_out, scaled, neg_inf});
    mask_select.add_outputs({masked});

    auto probs  = logical_tensor(id++, t,  s_sz, ltype::strided);
    auto smax   = op(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked}); smax.add_outputs({probs});

    auto output = logical_tensor(id++, t,  o_sz, ltype::strided);
    auto bmm2   = op(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    g.add_op(deq_k); g.add_op(deq_v);
    g.add_op(bmm1); g.add_op(sdiv);
    g.add_op(gen_index_row); g.add_op(mask_add); g.add_op(mask_sub);
    g.add_op(gen_index_col); g.add_op(mask_ge); g.add_op(mask_select);
    g.add_op(smax); g.add_op(bmm2);
    g.finalize();

    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) {
        return E;
    }
    E.ins      = parts[0].get_input_ports();
    E.out      = parts[0].get_output_ports()[0];
    E.cp       = parts[0].compile(E.ins, {E.out}, eng);
    E.out      = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q     = query.get_id();
    E.id_k     = k_u4.get_id();
    E.id_v     = v_u4.get_id();
    E.id_ksc   = k_sc.get_id();
    E.id_vsc   = v_sc.get_id();
    E.id_kzp   = k_zp.get_id();
    E.id_vzp   = v_zp.get_id();
    E.id_scale = scale.get_id();
    E.id_seqk  = seq_len_kv.get_id();
    E.id_seqq  = seq_len_q.get_id();
    E.id_ninf  = neg_inf.get_id();
    E.q_len    = q;
    E.kv_len   = seq;
    E.ninf     = -std::numeric_limits<float>::infinity();
    E.causal   = true;
    E.u4       = true;
    E.ok       = true;
    return E;
}
catch (const std::exception &) {
    return {};
}

static void fa_mpool_kv(const sycl::half * K, const sycl::half * V, sycl::half * Kd, sycl::half * Vd,
                        int Hkv, int sink, int win, int seq, int group, int64_t src_hs, int dst_hs,
                        dpct::queue_ptr stream) {
    constexpr int D = 256;
    const int mid = seq - sink - win;
    const int n_pool = (mid + group - 1) / group;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) D), cgh);
        cgh.parallel_for(sycl::nd_range<1>(
                             sycl::range<1>((size_t) Hkv * (size_t) n_pool * (size_t) D),
                             sycl::range<1>((size_t) D)),
                         [=](sycl::nd_item<1> it) {
            const int j = (int) it.get_local_id(0);
            const int wg = (int) it.get_group(0);
            const int h = wg / n_pool;
            const int g = wg - h * n_pool;
            const int base = sink + g * group;
            int n = group;
            if (base + n > sink + mid) {
                n = sink + mid - base;
            }
            const sycl::half * ks = K + (int64_t) h * src_hs + (int64_t) base * D + j;
            const sycl::half * vs = V + (int64_t) h * src_hs + (int64_t) base * D + j;
            float sk = 0.f, sv = 0.f, ek = 0.f, ev = 0.f;
            for (int i = 0; i < n; ++i) {
                const float x = (float) ks[(int64_t) i * D];
                const float y = (float) vs[(int64_t) i * D];
                sk += x;
                sv += y;
                ek += x * x;
                ev += y * y;
            }
            const float inv = 1.0f / (float) n;
            const float mk = sk * inv;
            const float mv = sv * inv;
            auto sum_red = [&]() {
                it.barrier(sycl::access::fence_space::local_space);
                for (int stride = D >> 1; stride > 0; stride >>= 1) {
                    if (j < stride) {
                        red[j] += red[j + stride];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }
            };
            red[j] = mk * mk;
            sum_red();
            const float rms_mk = sycl::sqrt(red[0] / (float) D + 1.0e-12f);
            red[j] = mv * mv;
            sum_red();
            const float rms_mv = sycl::sqrt(red[0] / (float) D + 1.0e-12f);
            red[j] = ek;
            sum_red();
            const float rms_ek = sycl::sqrt(red[0] / ((float) n * (float) D) + 1.0e-12f);
            red[j] = ev;
            sum_red();
            const float rms_ev = sycl::sqrt(red[0] / ((float) n * (float) D) + 1.0e-12f);
            Kd[((int64_t) h * dst_hs + (int64_t) (sink + g) * D) + j] = (sycl::half) (mk * (rms_ek / rms_mk));
            Vd[((int64_t) h * dst_hs + (int64_t) (sink + g) * D) + j] = (sycl::half) (mv * (rms_ev / rms_mv));
        });
    });
}

static void fa_div_cos(const sycl::half * K, float * cos, int Hkv, int n_prefix, int64_t src_hs, dpct::queue_ptr stream) {
    constexpr int D = 256;
    const int ntok = n_prefix - 1;
    if (ntok <= 0) {
        return;
    }
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(3 * (size_t) D), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ntok * (size_t) D), sycl::range<1>((size_t) D)),
                         [=](sycl::nd_item<1> it) {
            const int j = (int) it.get_local_id(0);
            const int t = (int) it.get_group(0) + 1;
            float dot = 0.f, na = 0.f, nb = 0.f;
            for (int h = 0; h < Hkv; ++h) {
                const float x = (float) K[(int64_t) h * src_hs + (int64_t) (t - 1) * D + j];
                const float y = (float) K[(int64_t) h * src_hs + (int64_t) t * D + j];
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            red[j] = dot;
            red[D + j] = na;
            red[2 * D + j] = nb;
            it.barrier(sycl::access::fence_space::local_space);
            for (int stride = D >> 1; stride > 0; stride >>= 1) {
                if (j < stride) {
                    red[j] += red[j + stride];
                    red[D + j] += red[D + j + stride];
                    red[2 * D + j] += red[2 * D + j + stride];
                }
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (j == 0) {
                cos[t] = red[0] / sycl::sqrt(red[D] * red[2 * D] + 1.0e-12f);
            }
        });
    });
}

static void fa_div_gather(const sycl::half * K, const sycl::half * V, sycl::half * Kd, sycl::half * Vd,
                          const int * off, const uint8_t * merge, int Hkv, int n_blocks, int64_t src_hs, int64_t dst_hs,
                          dpct::queue_ptr stream) {
    constexpr int D = 256;
    constexpr int BLK = 16;
    if (n_blocks <= 0 || Hkv <= 0) {
        return;
    }
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) D), cgh);
        cgh.parallel_for(sycl::nd_range<1>(
                             sycl::range<1>((size_t) n_blocks * (size_t) Hkv * (size_t) D),
                             sycl::range<1>((size_t) D)),
                         [=](sycl::nd_item<1> it) {
            const int j = (int) it.get_local_id(0);
            const int grp = (int) it.get_group(0);
            const int b = grp / Hkv;
            const int h = grp - b * Hkv;
            const int base = b * BLK;
            const int dst_t = off[b];
            const sycl::half * ks = K + (int64_t) h * src_hs + (int64_t) base * D + j;
            const sycl::half * vs = V + (int64_t) h * src_hs + (int64_t) base * D + j;
            sycl::half * kd = Kd + (int64_t) h * dst_hs + (int64_t) dst_t * D + j;
            sycl::half * vd = Vd + (int64_t) h * dst_hs + (int64_t) dst_t * D + j;
            if (merge[b]) {
                float sk = 0.f, sv = 0.f, ek = 0.f, ev = 0.f;
                for (int i = 0; i < BLK; ++i) {
                    const float x = (float) ks[(int64_t) i * D];
                    const float y = (float) vs[(int64_t) i * D];
                    sk += x;
                    sv += y;
                    ek += x * x;
                    ev += y * y;
                }
                const float mk = sk / (float) BLK;
                const float mv = sv / (float) BLK;
                auto sum_red = [&](float v) {
                    red[j] = v;
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int stride = D >> 1; stride > 0; stride >>= 1) {
                        if (j < stride) {
                            red[j] += red[j + stride];
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    const float s = red[0];
                    it.barrier(sycl::access::fence_space::local_space);
                    return s;
                };
                const float rms_mk = sycl::sqrt(sum_red(mk * mk) / (float) D + 1.0e-12f);
                const float rms_mv = sycl::sqrt(sum_red(mv * mv) / (float) D + 1.0e-12f);
                const float rms_ek = sycl::sqrt(sum_red(ek) / ((float) BLK * (float) D) + 1.0e-12f);
                const float rms_ev = sycl::sqrt(sum_red(ev) / ((float) BLK * (float) D) + 1.0e-12f);
                kd[0] = (sycl::half) (mk * (rms_ek / rms_mk));
                vd[0] = (sycl::half) (mv * (rms_ev / rms_mv));
            } else {
                for (int i = 0; i < BLK; ++i) {
                    kd[(int64_t) i * D] = ks[(int64_t) i * D];
                    vd[(int64_t) i * D] = vs[(int64_t) i * D];
                }
            }
        });
    });
}

void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) try {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t d   = K->ne[0];   // head_dim
    const int64_t seq = K->ne[1];   // n_kv
    const int64_t Hkv = K->ne[2];   // n_head_kv
    const int64_t H   = Q->ne[2];   // n_head
    const int64_t q   = Q->ne[1];   // n_tok
    const int64_t mb  = Q->ne[3];   // batch (== 1, gated)

    float kq_scale = 1.0f;
    memcpy(&kq_scale, (const float *) dst->op_params + 0, sizeof(float));

    dpct::queue_ptr stream = ctx.stream();
    dnnl::engine    eng    = ctx.engine_dnnl(stream);
    dnnl::stream    strm   = ctx.stream_dnnl(stream);

    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);

    sycl::half * K_ptr = nullptr;
    sycl::half * V_ptr = nullptr;
    std::array<int64_t, 5> k_str{ Hkv * seq * d, seq * d, seq * d, d, 1 };
    std::array<int64_t, 5> v_str = k_str;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Kf_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vf_pool;
    // Live KV element count only (not full n_ctx). Pool OOM returns nullptr -- never deref.
    const size_t kv_n = (size_t) Hkv * (size_t) seq * (size_t) d;
    auto fa_oom_tile = [&](const char * why) {
        GGML_LOG_WARN("%s: %s (q=%lld seq=%lld Hkv=%lld d=%lld); falling back to TILE kernel\n",
                      __func__, why, (long long) q, (long long) seq, (long long) Hkv, (long long) d);
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
    };
    // Helper: hand out reserved space, or fall back to the pool (nullptr on OOM).
    auto stage_k = [&](size_t n) -> sycl::half * {
        if (extra.K_buffer_ptr) { return (sycl::half *) extra.K_buffer_ptr; }
        if (n == 0) { return nullptr; }
        Kf_pool.emplace(ctx.pool(), n);
        return Kf_pool->get();
    };
    auto stage_v = [&](size_t n) -> sycl::half * {
        if (extra.V_buffer_ptr) { return (sycl::half *) extra.V_buffer_ptr; }
        if (n == 0) { return nullptr; }
        Vf_pool.emplace(ctx.pool(), n);
        return Vf_pool->get();
    };

    auto elem_strides = [](const ggml_tensor * t) {
        const int64_t s1 = (int64_t) (t->nb[1] / t->nb[0]);
        const int64_t s2 = (int64_t) (t->nb[2] / t->nb[0]);
        const int64_t s3 = (int64_t) (t->nb[3] / t->nb[0]);
        return std::array<int64_t, 5>{ s3, s2, s2, s1, 1 };
    };

    const int want_u4 = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_U4", 0)
        && K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0
        && (d % 32) == 0 && (seq % 2) == 0 && q >= 32 && seq >= 1
        && Hkv > 0 && (H % Hkv) == 0;
    std::optional<ggml_sycl_pool_alloc<uint8_t>> Ku_pool;
    std::optional<ggml_sycl_pool_alloc<uint8_t>> Vu_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Ksc_pool;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> Vsc_pool;
    std::optional<ggml_sycl_pool_alloc<int8_t>> Kzp_pool;
    std::optional<ggml_sycl_pool_alloc<int8_t>> Vzp_pool;
    uint8_t * Ku_ptr = nullptr;
    uint8_t * Vu_ptr = nullptr;
    sycl::half * Ksc_ptr = nullptr;
    sycl::half * Vsc_ptr = nullptr;
    int8_t * Kzp_ptr = nullptr;
    int8_t * Vzp_ptr = nullptr;
    bool u4_run = false;
    bool kds_run = false;
    if (want_u4) {
        static std::unordered_map<std::string, sdpa_partition> u4cache;
        char ukey[128];
        snprintf(ukey, sizeof(ukey), "u4:%d:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
                 (long long) H, (long long) Hkv, (long long) q, (long long) seq, (long long) d);
        auto uit = u4cache.find(ukey);
        if (uit == u4cache.end()) {
            sdpa_partition P = build_sdpa_causal_u4(eng, (int) H, (int) Hkv, (int) q, (int) seq, (int) d);
            static int nrep = 0;
            if (nrep < 4) {
                nrep++;
                GGML_SYCL_XE2_LOG( "xe2mx fa u4 gqa build ok=%d q=%ld seq=%ld H=%ld Hkv=%ld d=%ld gqa=%ld\n",
                        P.ok ? 1 : 0, (long) q, (long) seq, (long) H, (long) Hkv, (long) d, (long) (H / Hkv));
                fflush(stderr);
            }
            uit = u4cache.emplace(ukey, std::move(P)).first;
        }
        if (uit->second.ok) {
            const size_t n4 = (size_t) Hkv * (size_t) seq * (size_t) (d / 2);
            const size_t nsc = (size_t) Hkv * (size_t) seq * (size_t) (d / 32);
            Ku_pool.emplace(ctx.pool(), n4);
            Vu_pool.emplace(ctx.pool(), n4);
            Ksc_pool.emplace(ctx.pool(), nsc);
            Vsc_pool.emplace(ctx.pool(), nsc);
            Kzp_pool.emplace(ctx.pool(), nsc);
            Vzp_pool.emplace(ctx.pool(), nsc);
            Ku_ptr = Ku_pool->get();
            Vu_ptr = Vu_pool->get();
            Ksc_ptr = Ksc_pool->get();
            Vsc_ptr = Vsc_pool->get();
            Kzp_ptr = Kzp_pool->get();
            Vzp_ptr = Vzp_pool->get();
            if (!Ku_ptr || !Vu_ptr || !Ksc_ptr || !Vsc_ptr || !Kzp_ptr || !Vzp_ptr) {
                Ku_pool.reset(); Vu_pool.reset();
                Ksc_pool.reset(); Vsc_pool.reset();
                Kzp_pool.reset(); Vzp_pool.reset();
                Ku_ptr = Vu_ptr = nullptr;
                Ksc_ptr = Vsc_ptr = nullptr;
                Kzp_ptr = Vzp_ptr = nullptr;
            } else {
                fa_q4_pack_u4_k((const char *) K->data, Ku_ptr, Ksc_ptr, Kzp_ptr, seq, d, Hkv, 1, K->nb[1], K->nb[2], stream);
                fa_q4_pack_u4((const char *) V->data, Vu_ptr, Vsc_ptr, Vzp_ptr, seq, d, Hkv, 1, V->nb[1], V->nb[2], stream);
                u4_run = true;
            }
        }
    }

    const int fa_time = ggml_sycl_get_env("GGML_SYCL_FA_TIME", 0);
    if (fa_time) {
        try { stream->wait(); } catch (...) {}
    }
    const auto fa_t0 = std::chrono::steady_clock::now();
    const bool pkv = !u4_run && fa_pkv_stage(K, V, d, Hkv, seq, stream, &K_ptr, &V_ptr, &k_str, &v_str);
    if (fa_time) {
        try { stream->wait(); } catch (...) {}
    }
    const double fa_ms_stage = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - fa_t0).count();
    if (u4_run) {
    } else if (pkv) {
    } else if (ggml_sycl_fattn_onednn_binds_kv(K, V)) {
        K_ptr = (sycl::half *) K->data;
        V_ptr = (sycl::half *) V->data;
        k_str = elem_strides(K);
        v_str = elem_strides(V);
    } else if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        K_ptr = stage_k(kv_n);
        V_ptr = stage_v(kv_n);
        if (!K_ptr || !V_ptr) {
            fa_oom_tile("F16 K/V stage pool OOM");
            return;
        }
        cont_to_f16_sycl<sycl::half>((const char *) K->data, K_ptr, d, seq, Hkv, mb, K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<sycl::half>((const char *) V->data, V_ptr, d, seq, Hkv, mb, V->nb[1], V->nb[2], V->nb[3], stream);
    } else if (ggml_is_quantized(K->type)) {
        // Quantized K/V: dequant to dense F16 using pool, same lifetime as F16 path.
        // Size by live seq (kv_n), not full n_ctx capacity.
        K_ptr = stage_k(kv_n);
        if (!K_ptr) {
            fa_oom_tile("quant K stage pool OOM");
            return;
        }
        {
            const char * K_data = (const char *)K->data;
            const bool k_non_dense = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
            const bool k_gemma = k_non_dense &&
                ((int64_t)K->nb[2] < (int64_t)K->ne[1] * (int64_t)K->nb[1]);
            const int want_kds = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_KDS", 0);
            if (want_kds && k_gemma && K->type == GGML_TYPE_Q4_0
                && fa_q4_dense_hdts(K_data, K_ptr, seq, d, Hkv, stream)) {
                k_str = { Hkv * d * seq, d * seq, d * seq, seq, 1 };
                kds_run = true;
            } else if (ggml_is_contiguously_allocated(K) && !k_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(K->type, dst);
                to_fp16(K_data, K_ptr, (int64_t) kv_n, stream);
            } else if (k_gemma && K->type == GGML_TYPE_Q4_0
                       && fa_q4_dense_hntd(K_data, K_ptr, seq, d, Hkv, stream)) {
            } else {
                const size_t bs = ggml_blck_size(K->type);
                const size_t ts = ggml_type_size(K->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(K->type);
                int64_t s01, s02, s03;
                if (k_gemma) {
                    const int64_t blk_per_row = (int64_t)K->ne[0] / bs;
                    s01 = (int64_t)Hkv * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)K->ne[1] * s01;
                } else {
                    s01 = (int64_t)K->nb[1] / ts;
                    s02 = (int64_t)K->nb[2] / ts;
                    s03 = (int64_t)K->nb[3] / ts;
                }
                to_fp16(K_data, K_ptr,
                        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                        s01, s02, s03, stream);
            }
        }
        // Quantized V: always dequant separately. Even when K and V share
        // the same underlying allocation (V is a view of K with the same
        // data pointer), their logical values differ because the quantized
        // elements at different positions/offsets represent different K/V
        // data. Master's F16 path also never aliases K and V.
        V_ptr = stage_v(kv_n);
        if (!V_ptr) {
            fa_oom_tile("quant V stage pool OOM");
            return;
        }
        {
            const char * V_data = (const char *)V->data;
            const bool v_non_dense = ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;
            const bool v_gemma = v_non_dense &&
                ((int64_t)V->nb[2] < (int64_t)V->ne[1] * (int64_t)V->nb[1]);
            if (ggml_is_contiguously_allocated(V) && !v_non_dense) {
                to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(V->type, dst);
                to_fp16(V_data, V_ptr, (int64_t) kv_n, stream);
            } else if (v_gemma && V->type == GGML_TYPE_Q4_0
                       && fa_q4_dense_hntd(V_data, V_ptr, seq, d, Hkv, stream)) {
            } else {
                const size_t bs = ggml_blck_size(V->type);
                const size_t ts = ggml_type_size(V->type);
                to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(V->type);
                int64_t s01, s02, s03;
                if (v_gemma) {
                    const int64_t blk_per_row = (int64_t)V->ne[0] / bs;
                    s01 = (int64_t)V->ne[2] * blk_per_row;
                    s02 = blk_per_row;
                    s03 = (int64_t)V->ne[1] * s01;
                } else {
                    s01 = (int64_t)V->nb[1] / ts;
                    s02 = (int64_t)V->nb[2] / ts;
                    s03 = (int64_t)V->nb[3] / ts;
                }
                to_fp16(V_data, V_ptr,
                        V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                        s01, s02, s03, stream);
            }
        }
    } else {
        // F32: strided copy to dense F16 via cont_to_f16_sycl<float>.
        K_ptr = stage_k(kv_n);
        V_ptr = stage_v(kv_n);
        if (!K_ptr || !V_ptr) {
            fa_oom_tile("F32 K/V stage pool OOM");
            return;
        }
        cont_to_f16_sycl<float>((const char *) K->data, K_ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3],
                                K->nb[1], K->nb[2], K->nb[3], stream);
        cont_to_f16_sycl<float>((const char *) V->data, V_ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3],
                                V->nb[1], V->nb[2], V->nb[3], stream);
    }

    int64_t fa_seq = seq;
    bool ht_on = false;
    // Approximation paths (HT/MPOOL/SPAN/STRIDE/DIV) need dense F16 HNTD layout with
    // token stride == d. That is true for PKV cache hits AND for reserved/pool dense
    // staging after pkv bound-skip - gate on layout, not the pkv bool (lying pkv=true
    // would also suppress FA_SPLIT/FA_JM which correctly key off !pkv).
    const bool kv_hntd = K_ptr && V_ptr && !u4_run && !kds_run && mb == 1
        && k_str[3] == d && v_str[3] == d
        && k_str[1] >= seq * d && v_str[1] >= seq * d
        && (k_str[1] % d) == 0 && (v_str[1] % d) == 0;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> HtK;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> HtV;
    if (kv_hntd) {
        int ht_sink = 0;
        int ht_win = 0;
        const char * hte = getenv("GGML_SYCL_FA_HT");
        if (hte && std::sscanf(hte, "%d,%d", &ht_sink, &ht_win) == 2
            && ht_sink > 0 && ht_win > 0 && q <= ht_win
            && seq > (int64_t) ht_sink + ht_win) {
            const int64_t nkv = (int64_t) ht_sink + ht_win;
            const int64_t src_hs = k_str[1];
            HtK.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
            HtV.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
            sycl::half * Kc = HtK->get();
            sycl::half * Vc = HtV->get();
            if (!Kc || !Vc) {
                HtK.reset();
                HtV.reset();
            } else {
            const size_t sink_b = (size_t) ht_sink * (size_t) d * sizeof(sycl::half);
            const size_t win_b = (size_t) ht_win * (size_t) d * sizeof(sycl::half);
            for (int64_t h = 0; h < Hkv; ++h) {
                sycl::half * ks = K_ptr + h * src_hs;
                sycl::half * vs = V_ptr + h * src_hs;
                sycl::half * kd = Kc + h * nkv * d;
                sycl::half * vd = Vc + h * nkv * d;
                stream->memcpy(kd, ks, sink_b);
                stream->memcpy(vd, vs, sink_b);
                stream->memcpy(kd + (int64_t) ht_sink * d, ks + (seq - ht_win) * d, win_b);
                stream->memcpy(vd + (int64_t) ht_sink * d, vs + (seq - ht_win) * d, win_b);
            }
            K_ptr = Kc;
            V_ptr = Vc;
            k_str = { Hkv * nkv * d, nkv * d, nkv * d, d, 1 };
            v_str = k_str;
            fa_seq = nkv;
            ht_on = true;
            }
        }
    }
    std::optional<ggml_sycl_pool_alloc<sycl::half>> MpK;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> MpV;
    // head_dim 128 (Qwen3.8 27B) and 256 both OK for mpool reshape
    if (!ht_on && kv_hntd && (d == 128 || d == 256)) {
        int mp_sink = 0;
        int mp_win = 0;
        int mp_group = 0;
        const char * mpe = getenv("GGML_SYCL_FA_MPOOL");
        if (mpe && std::sscanf(mpe, "%d,%d,%d", &mp_sink, &mp_win, &mp_group) == 3
            && mp_sink > 0 && mp_win > 0 && mp_group > 1 && q <= mp_win
            && seq > (int64_t) mp_sink + mp_win) {
            const int mid = (int) seq - mp_sink - mp_win;
            const int n_pool = (mid + mp_group - 1) / mp_group;
            const int nkv = mp_sink + n_pool + mp_win;
            const int64_t src_hs = k_str[1];
            const int64_t dst_hs = (int64_t) nkv * d;
            MpK.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
            MpV.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
            sycl::half * Kc = MpK->get();
            sycl::half * Vc = MpV->get();
            if (!Kc || !Vc) {
                MpK.reset();
                MpV.reset();
            } else {
            const size_t sink_b = (size_t) mp_sink * (size_t) d * sizeof(sycl::half);
            const size_t win_b = (size_t) mp_win * (size_t) d * sizeof(sycl::half);
            for (int64_t h = 0; h < Hkv; ++h) {
                sycl::half * ks = K_ptr + h * src_hs;
                sycl::half * vs = V_ptr + h * src_hs;
                sycl::half * kd = Kc + h * dst_hs;
                sycl::half * vd = Vc + h * dst_hs;
                stream->memcpy(kd, ks, sink_b);
                stream->memcpy(vd, vs, sink_b);
                stream->memcpy(kd + (int64_t) (mp_sink + n_pool) * d, ks + (seq - mp_win) * d, win_b);
                stream->memcpy(vd + (int64_t) (mp_sink + n_pool) * d, vs + (seq - mp_win) * d, win_b);
            }
            fa_mpool_kv(K_ptr, V_ptr, Kc, Vc, (int) Hkv, mp_sink, mp_win, (int) seq, mp_group, src_hs, dst_hs, stream);
            K_ptr = Kc;
            V_ptr = Vc;
            k_str = { Hkv * dst_hs, dst_hs, dst_hs, d, 1 };
            v_str = k_str;
            fa_seq = nkv;
            static int nmp = 0;
            if (nmp < 2) {
                nmp++;
                GGML_SYCL_XE2_LOG( "xe2mx fa mpool seq=%d nkv=%d group=%d\n", (int) seq, nkv, mp_group);
                fflush(stderr);
            }
            }
        }
    }
    std::optional<ggml_sycl_pool_alloc<sycl::half>> SpK;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> SpV;
    if (!ht_on && fa_seq == seq && kv_hntd && (d == 128 || d == 256)) {
        // FA_SPAN: a0,b0,a1,b1,win OR a0,b0,a1,b1,a2,b2,a3,b3,win (up to 4 prefix segs + window)
                int spv[10];
                int spn = 0;
                const char * spe = getenv("GGML_SYCL_FA_SPAN");
                if (spe) {
                    const char * p = spe;
                    while (spn < 10 && *p) {
                        int v = 0;
                        if (sscanf(p, "%d", &v) != 1) {
                            break;
                        }
                        spv[spn++] = v;
                        while (*p && *p != ',') {
                            ++p;
                        }
                        if (*p == ',') {
                            ++p;
                        }
                    }
                }
                const int sp_win = (spn == 5 || spn == 9) ? spv[spn - 1] : 0;
                const int n_pref = (spn == 5) ? 2 : (spn == 9) ? 4 : 0;
                if (sp_win > 0 && n_pref > 0 && q <= (int64_t) sp_win && seq > (int64_t) sp_win) {
                    const int64_t w0 = seq - sp_win;
                    int64_t seg_a[5];
                    int64_t seg_b[5];
                    int ns = 0;
                    auto push = [&](int64_t a, int64_t b) {
                        if (a < 0) {
                            a = 0;
                        }
                        if (b > w0) {
                            b = w0;
                        }
                        if (b > a && ns < n_pref) {
                            seg_a[ns] = a;
                            seg_b[ns] = b;
                            ns++;
                        }
                    };
                    for (int i = 0; i < n_pref; ++i) {
                        push(spv[2 * i], spv[2 * i + 1]);
                    }
            seg_a[ns] = w0;
            seg_b[ns] = seq;
            ns++;
            int64_t nkv = 0;
            for (int i = 0; i < ns; ++i) {
                nkv += seg_b[i] - seg_a[i];
            }
            if (nkv > 0 && nkv < seq) {
                const int64_t src_hs = k_str[1];
                const int64_t dst_hs = nkv * d;
                SpK.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                SpV.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                sycl::half * Kc = SpK->get();
                sycl::half * Vc = SpV->get();
                if (!Kc || !Vc) {
                    SpK.reset();
                    SpV.reset();
                } else {
                for (int64_t h = 0; h < Hkv; ++h) {
                    sycl::half * ks = K_ptr + h * src_hs;
                    sycl::half * vs = V_ptr + h * src_hs;
                    sycl::half * kd = Kc + h * dst_hs;
                    sycl::half * vd = Vc + h * dst_hs;
                    int64_t off = 0;
                    for (int i = 0; i < ns; ++i) {
                        const int64_t ntok = seg_b[i] - seg_a[i];
                        const size_t bytes = (size_t) ntok * (size_t) d * sizeof(sycl::half);
                        stream->memcpy(kd + off * d, ks + seg_a[i] * d, bytes);
                        stream->memcpy(vd + off * d, vs + seg_a[i] * d, bytes);
                        off += ntok;
                    }
                }
                K_ptr = Kc;
                V_ptr = Vc;
                k_str = { Hkv * dst_hs, dst_hs, dst_hs, d, 1 };
                v_str = k_str;
                fa_seq = nkv;
                static int nsp = 0;
                if (nsp < 2) {
                    nsp++;
                    GGML_SYCL_XE2_LOG( "xe2mx fa span seq=%d nkv=%d\n", (int) seq, (int) nkv);
                    fflush(stderr);
                }
                }
            }
        }
    }
    // FA_STRIDE=win,stride - keep last win + every stride-th prefix token (d128/256)
    // Uniform subsample often better quality than hard SPAN holes at similar nkv.
    if (!ht_on && fa_seq == seq && kv_hntd && (d == 128 || d == 256)) {
        int st_win = 0, st_stride = 0;
        const char * ste = getenv("GGML_SYCL_FA_STRIDE");
        if (ste && std::sscanf(ste, "%d,%d", &st_win, &st_stride) == 2
            && st_win > 0 && st_stride >= 2 && q <= st_win && seq > (int64_t) st_win) {
            const int64_t w0 = seq - st_win;
            const int64_t n_pref = (w0 + st_stride - 1) / st_stride; // ceil
            const int64_t nkv = n_pref + st_win;
            if (nkv > 0 && nkv < seq) {
                const int64_t src_hs = k_str[1];
                const int64_t dst_hs = nkv * d;
                SpK.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                SpV.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                sycl::half * Kc = SpK->get();
                sycl::half * Vc = SpV->get();
                if (!Kc || !Vc) {
                    SpK.reset();
                    SpV.reset();
                } else {
                for (int64_t h = 0; h < Hkv; ++h) {
                    sycl::half * ks = K_ptr + h * src_hs;
                    sycl::half * vs = V_ptr + h * src_hs;
                    sycl::half * kd = Kc + h * dst_hs;
                    sycl::half * vd = Vc + h * dst_hs;
                    int64_t off = 0;
                    for (int64_t t = 0; t < w0; t += st_stride) {
                        stream->memcpy(kd + off * d, ks + t * d, (size_t) d * sizeof(sycl::half));
                        stream->memcpy(vd + off * d, vs + t * d, (size_t) d * sizeof(sycl::half));
                        off++;
                    }
                    stream->memcpy(kd + off * d, ks + w0 * d, (size_t) st_win * (size_t) d * sizeof(sycl::half));
                    stream->memcpy(vd + off * d, vs + w0 * d, (size_t) st_win * (size_t) d * sizeof(sycl::half));
                }
                K_ptr = Kc;
                V_ptr = Vc;
                k_str = { Hkv * dst_hs, dst_hs, dst_hs, d, 1 };
                v_str = k_str;
                fa_seq = nkv;
                static int nst = 0;
                if (nst < 2) {
                    nst++;
                    GGML_SYCL_XE2_LOG( "xe2mx fa stride seq=%d nkv=%d win=%d stride=%d\n",
                            (int) seq, (int) nkv, st_win, st_stride);
                    fflush(stderr);
                }
                }
            }
        }
    }
    std::optional<ggml_sycl_pool_alloc<float>> DvCos;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> DvK;
    std::optional<ggml_sycl_pool_alloc<sycl::half>> DvV;
    std::optional<ggml_sycl_pool_alloc<int>> DvOff;
    std::optional<ggml_sycl_pool_alloc<uint8_t>> DvMerge;
    if (!ht_on && fa_seq == seq && kv_hntd && (d == 128 || d == 256)) {
        int dv_win = 0;
        int dv_budget = 0;
        const char * dve = getenv("GGML_SYCL_FA_DIV");
        if (dve && std::sscanf(dve, "%d,%d", &dv_win, &dv_budget) == 2
            && dv_win > 0 && dv_budget > 256 && q <= dv_win && seq > (int64_t) dv_win + dv_budget) {
            constexpr int BLK = 16;
            const int n_prefix = (int) seq - dv_win;
            const int n_blocks = n_prefix / BLK;
            const int rem = n_prefix - n_blocks * BLK;
            int M = (n_prefix - dv_budget + (BLK - 2)) / (BLK - 1);
            if (M > n_blocks) {
                M = n_blocks;
            }
            if (n_blocks > 1 && M > 0) {
                DvCos.emplace(ctx.pool(), (size_t) n_prefix);
                float * cos = DvCos->get();
                if (!cos) {
                    DvCos.reset();
                } else {
                fa_div_cos(K_ptr, cos, (int) Hkv, n_prefix, k_str[1], stream);
                std::vector<float> hcos((size_t) n_prefix);
                stream->memcpy(hcos.data(), cos, (size_t) n_prefix * sizeof(float)).wait();
                std::vector<float> score((size_t) n_blocks);
                for (int b = 0; b < n_blocks; ++b) {
                    float s = 0.f;
                    for (int i = 1; i < BLK; ++i) {
                        s += hcos[(size_t) b * BLK + i];
                    }
                    score[(size_t) b] = s;
                }
                std::vector<int> ord((size_t) n_blocks);
                std::iota(ord.begin(), ord.end(), 0);
                std::nth_element(ord.begin(), ord.begin() + M, ord.end(), [&](int a, int b) {
                    return score[(size_t) a] > score[(size_t) b];
                });
                std::vector<uint8_t> hmerge((size_t) n_blocks, 0);
                for (int i = 0; i < M; ++i) {
                    hmerge[(size_t) ord[(size_t) i]] = 1;
                }
                std::vector<int> hoff((size_t) n_blocks);
                int cursor = 0;
                for (int b = 0; b < n_blocks; ++b) {
                    hoff[(size_t) b] = cursor;
                    cursor += hmerge[(size_t) b] ? 1 : BLK;
                }
                const int block_out = cursor;
                const int prefix_out = cursor + rem;
                const int nkv = prefix_out + dv_win;
                if (nkv > 0 && nkv < (int) seq) {
                    const int64_t src_hs = k_str[1];
                    const int64_t dst_hs = (int64_t) nkv * d;
                    DvK.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                    DvV.emplace(ctx.pool(), (size_t) Hkv * (size_t) nkv * (size_t) d);
                    DvOff.emplace(ctx.pool(), (size_t) n_blocks);
                    DvMerge.emplace(ctx.pool(), (size_t) n_blocks);
                    sycl::half * Kc = DvK->get();
                    sycl::half * Vc = DvV->get();
                    int * offp = DvOff->get();
                    uint8_t * mergep = DvMerge->get();
                    if (!Kc || !Vc || !offp || !mergep) {
                        DvK.reset(); DvV.reset(); DvOff.reset(); DvMerge.reset();
                    } else {
                    stream->memcpy(offp, hoff.data(), (size_t) n_blocks * sizeof(int));
                    stream->memcpy(mergep, hmerge.data(), (size_t) n_blocks);
                    fa_div_gather(K_ptr, V_ptr, Kc, Vc, offp, mergep, (int) Hkv, n_blocks,
                                  src_hs, dst_hs, stream);
                    const size_t rem_b = (size_t) rem * (size_t) d * sizeof(sycl::half);
                    const size_t win_b = (size_t) dv_win * (size_t) d * sizeof(sycl::half);
                    for (int64_t h = 0; h < Hkv; ++h) {
                        sycl::half * ks = K_ptr + h * src_hs;
                        sycl::half * vs = V_ptr + h * src_hs;
                        sycl::half * kd = Kc + h * dst_hs;
                        sycl::half * vd = Vc + h * dst_hs;
                        if (rem > 0) {
                            stream->memcpy(kd + (int64_t) block_out * d, ks + (int64_t) (n_blocks * BLK) * d, rem_b);
                            stream->memcpy(vd + (int64_t) block_out * d, vs + (int64_t) (n_blocks * BLK) * d, rem_b);
                        }
                        stream->memcpy(kd + (int64_t) prefix_out * d, ks + (int64_t) n_prefix * d, win_b);
                        stream->memcpy(vd + (int64_t) prefix_out * d, vs + (int64_t) n_prefix * d, win_b);
                    }
                    K_ptr = Kc;
                    V_ptr = Vc;
                    k_str = { Hkv * dst_hs, dst_hs, dst_hs, d, 1 };
                    v_str = k_str;
                    fa_seq = nkv;
                    static int ndv = 0;
                    if (ndv < 2) {
                        ndv++;
                        GGML_SYCL_XE2_LOG( "xe2mx fa div seq=%d nkv=%d merge=%d\n", (int) seq, nkv, M);
                        fflush(stderr);
                    }
                    }
                }
                }
            }
        }
    }

    // divide-by-(1/scale) reproduces ggml's score *= kq_scale on the proven probe graph.
    //
    // The scale must not be uploaded with an async memcpy from a stack local: on the in-order
    // queue that copy waits behind the K/V staging kernels, and once those take long enough
    // (n_kv >= ~26k on B70) the host frame is recycled before the copy runs, feeding the SDPA a
    // garbage scale (output collapses to a repeated token). Write the scalar from a kernel
    // instead -- the value is captured into the command, so no host memory has to outlive the
    // call, and the enqueue stays async.
    const int nq = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_NQ", 0);
    const int smul = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_SMUL", 0);
    const int hsc = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_HSC", 0);
    const sycl::half scale_h = (sycl::half) (smul ? kq_scale : (1.0f / kq_scale));
    std::optional<ggml_sycl_pool_alloc<sycl::half>> scbuf;
    sycl::half * scale_dev = nullptr;
    if (!nq && !hsc) {
        scale_dev = (sycl::half *) extra.scale_buffer_ptr;
        if (!scale_dev) {
            scbuf.emplace(ctx.pool(), 1);
            scale_dev = scbuf->get();
        }
        if (!scale_dev) {
            fa_oom_tile("scale pool OOM");
            return;
        }
        stream->single_task([=]() { *scale_dev = scale_h; });
    }

    // f16 contiguous SDPA out [mb,H,q,d]
    std::optional<ggml_sycl_pool_alloc<sycl::half>> outf_pool;
    sycl::half * outf_ptr = (sycl::half *) extra.out_buffer_ptr;
    if (!outf_ptr) {
        outf_pool.emplace(ctx.pool(), (size_t) H * (size_t) q * (size_t) d);
        outf_ptr = outf_pool->get();
    }
    if (!outf_ptr) {
        fa_oom_tile("SDPA out pool OOM");
        return;
    }
    // Required K/V must be live before oneDNN execute (pool OOM above soft-fails compress paths only).
    if (u4_run) {
        if (!Ku_ptr || !Vu_ptr || !Ksc_ptr || !Vsc_ptr || !Kzp_ptr || !Vzp_ptr) {
            fa_oom_tile("u4 buffers missing");
            return;
        }
    } else if (!K_ptr || !V_ptr) {
        fa_oom_tile("K/V buffers missing");
        return;
    }

    // compile once per (device, shape, KV strides), reuse across layers/calls. Stride 2 always
    // repeats stride 1 and stride 4 is always 1, so the key covers every entry that can differ.
    static std::unordered_map<std::string, sdpa_partition> cache;
    int causal = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_CAUSAL", 1);
    // The implicit causal graph aligns queries to the end of the KV window: col <= row + fa_seq - q.
    // That is only the real mask when fa_seq is the true valid length (llama passes it in op_params[5]).
    // Padded n_kv would let each query see future tokens of this ubatch and stale padding cells.
    {
        const int32_t Lv = ggml_get_op_params_i32(dst, 5);
        if (causal && !u4_run) {
            if (Lv >= q && Lv <= seq && fa_seq == seq) {
                fa_seq = Lv;
            } else if (fa_seq == seq) {
                causal = 0;
            }
        }
        static int nlv = 0;
        if (nlv < 6) {
            nlv++;
            GGML_SYCL_XE2_LOG("xe2mx fa onednn valid_kv=%d seq=%ld q=%ld causal=%d fa_seq=%ld\n", (int) Lv, (long) seq, (long) q, causal, (long) fa_seq);
        }
    }
    char keyb[256];
    snprintf(keyb, sizeof(keyb), "%d:%d:%d:%d:%d:%d:%d:%d:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld:%lld", ggml_sycl_get_device(),
             ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_FPM", 0), causal, u4_run ? 1 : 0,
             ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_S16", 0), nq, smul, hsc,
             (long long) H, (long long) Hkv, (long long) q, (long long) fa_seq, (long long) d,
             (long long) k_str[0], (long long) k_str[1], (long long) k_str[3],
             (long long) v_str[0], (long long) v_str[1], (long long) v_str[3]);
    auto it = cache.find(keyb);
    if (it == cache.end()) {
        sdpa_partition P;
        if (u4_run) {
            P = build_sdpa_causal_u4(eng, (int) H, (int) Hkv, (int) q, (int) seq, (int) d);
        } else if (causal) {
            P = build_sdpa_causal(eng, (int) H, (int) Hkv, (int) q, (int) fa_seq, (int) d, k_str, v_str, kds_run ? 1 : 0);
            if (!P.ok) {
                GGML_SYCL_XE2_LOG( "xe2mx fa causal miss q=%d seq=%d kds=%d\n", (int) q, (int) fa_seq, kds_run ? 1 : 0);
                fflush(stderr);
            }
        }
        if (!P.ok && !u4_run && !kds_run
            && ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_S16", 0) == 0 && nq == 0 && smul == 0 && hsc == 0) {
            P = build_sdpa(eng, (int) H, (int) Hkv, (int) q, (int) fa_seq, (int) d, k_str, v_str);
        }
        it = cache.emplace(keyb, std::move(P)).first;
    }
    sdpa_partition & E = it->second;
    if (!E.ok) {
        ggml_sycl_flash_attn_ext_tile(ctx, dst);
        return;
    }
    if (E.hsc) {
        E.scale_h = scale_h;
    }

    std::optional<ggml_sycl_pool_alloc<sycl::half>> Qf_pool;
    sycl::half * Qf_ptr = (sycl::half *) extra.Q_buffer_ptr;
    if (!Qf_ptr) {
        Qf_pool.emplace(ctx.pool(), (size_t) H * (size_t) q * (size_t) d);
        Qf_ptr = Qf_pool->get();
    }
    if (!Qf_ptr) {
        fa_oom_tile("Q stage pool OOM");
        return;
    }
    cont_to_f16_sycl<float>((const char *) Q->data, Qf_ptr, d, q, H, mb, Q->nb[1], Q->nb[2], Q->nb[3], stream,
                            nq ? kq_scale : 1.0f);

    if (ggml_sycl_get_env("GGML_SYCL_FA_SPLIT", 0) == 2 && !u4_run && !kds_run && !pkv && causal && mb == 1
        && Hkv > 0 && (H % 2) == 0 && (Hkv % 2) == 0 && K_ptr && V_ptr && Qf_ptr && outf_ptr
        && k_str[4] == 1 && k_str[3] == d && v_str[4] == 1 && v_str[3] == d) {
        const int Hs = (int) (H / 2);
        const int Hkvs = (int) (Hkv / 2);
        const std::array<int64_t, 5> k_half{ (int64_t) Hkvs * seq * d, seq * d, seq * d, d, 1 };
        const std::array<int64_t, 5> v_half = k_half;
        static std::unordered_map<std::string, sdpa_partition> split_cache;
        char skey[256];
        snprintf(skey, sizeof(skey), "sp2:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d", ggml_sycl_get_device(),
                 ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_FPM", 0),
                 ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_S16", 0), nq, smul, hsc,
                 Hs, Hkvs, (int) q, (int) seq, (int) d);
        auto sit = split_cache.find(skey);
        if (sit == split_cache.end()) {
            sdpa_partition P = build_sdpa_causal(eng, Hs, Hkvs, (int) q, (int) seq, (int) d, k_half, v_half, 0);
            sit = split_cache.emplace(skey, std::move(P)).first;
        }
        sdpa_partition & Sp = sit->second;
        if (Sp.ok) {
            if (Sp.hsc) {
                Sp.scale_h = scale_h;
            }
            static sycl::queue * side = nullptr;
            static queue_ptr side_seen = nullptr;
            if (!side || side_seen != stream) {
                delete side;
                side = new sycl::queue(stream->get_context(), stream->get_device(),
                                       sycl::property_list{ sycl::property::queue::in_order() });
                side_seen = stream;
            }
            dnnl::engine eng1 = ctx.engine_dnnl(side);
            dnnl::stream strm1 = ctx.stream_dnnl(side);
            {
                dnnl::memory::desc sp({ 64 * 1024 * 1024 }, dnnl::memory::data_type::u8, { 1 });
                ctx.get_scratchpad_mem(sp, eng, stream);
                ctx.get_scratchpad_mem(sp, eng1, side);
            }
            {
                sycl::event evq = stream->ext_oneapi_submit_barrier();
                side->ext_oneapi_submit_barrier({ evq });
            }
            auto run_half = [&](int half, dnnl::engine & e, dnnl::stream & st) {
                sycl::half * Qh = Qf_ptr + (size_t) half * (size_t) Hs * (size_t) q * (size_t) d;
                sycl::half * Kh = K_ptr + (size_t) half * (size_t) Hkvs * (size_t) seq * (size_t) d;
                sycl::half * Vh = V_ptr + (size_t) half * (size_t) Hkvs * (size_t) seq * (size_t) d;
                sycl::half * Oh = outf_ptr + (size_t) half * (size_t) Hs * (size_t) q * (size_t) d;
                auto idp = [&](size_t r) -> void * {
                    if (r == Sp.id_q) return Qh;
                    if (r == Sp.id_k) return Kh;
                    if (r == Sp.id_v) return Vh;
                    if (Sp.id_scale != (size_t) -1 && r == Sp.id_scale) {
                        return Sp.hsc ? (void *) &Sp.scale_h : (void *) scale_dev;
                    }
                    if (r == Sp.id_mask) return (void *) mask->data;
                    if (r == Sp.id_seqk) return &Sp.kv_len;
                    if (r == Sp.id_seqq) return &Sp.q_len;
                    if (r == Sp.id_ninf) return &Sp.ninf;
                    return nullptr;
                };
                std::vector<tensor> ti;
                ti.reserve(Sp.ins.size());
                for (auto & lt : Sp.ins) {
                    const size_t rid = lt.get_id();
                    if (rid == Sp.id_seqk || rid == Sp.id_seqq || rid == Sp.id_ninf
                        || (Sp.hsc && rid == Sp.id_scale)) {
                        ti.push_back(tensor::make_scalar_tensor(lt, idp(rid)));
                    } else {
                        ti.emplace_back(lt, e, idp(rid));
                    }
                }
                tensor to(Sp.out, e, Oh);
                Sp.cp.execute(st, ti, {to});
            };
            run_half(0, eng, strm);
            run_half(1, eng1, strm1);
            side->wait();
            stream->wait();
            permute_sdpa_out_sycl(outf_ptr, (float *) dst->data, mb, H, q, d, stream);
            if (ggml_sycl_info().device_count > 1) {
                stream->wait_and_throw();
            }
            return;
        }
    }

    if (ggml_sycl_get_env("GGML_SYCL_FA_JM", 0) && !u4_run && !kds_run && !pkv
        && K_ptr && V_ptr && Qf_ptr && outf_ptr && mb == 1 && d == 256
        && (q % 8) == 0 && (seq % 16) == 0 && Hkv > 0 && (H % Hkv) == 0
        && k_str[4] == 1 && k_str[3] == d) {
        fa_jm_sdpa_f16(Qf_ptr, K_ptr, V_ptr, outf_ptr, (int) H, (int) Hkv, (int) q, (int) seq, (int) d, kq_scale, stream);
        permute_sdpa_out_sycl(outf_ptr, (float *) dst->data, mb, H, q, d, stream);
        if (ggml_sycl_info().device_count > 1) {
            stream->wait_and_throw();
        }
        return;
    }

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q)     return Qf_ptr;
        if (r == E.id_k)     return u4_run ? (void *) Ku_ptr : (void *) K_ptr;
        if (r == E.id_v)     return u4_run ? (void *) Vu_ptr : (void *) V_ptr;
        if (u4_run && r == E.id_ksc)   return Ksc_ptr;
        if (u4_run && r == E.id_vsc)   return Vsc_ptr;
        if (u4_run && r == E.id_kzp)   return Kzp_ptr;
        if (u4_run && r == E.id_vzp)   return Vzp_ptr;
        if (E.id_scale != (size_t) -1 && r == E.id_scale) return E.hsc ? (void *) &E.scale_h : (void *) scale_dev;
        if (r == E.id_mask)  return (void *) mask->data;
        if (r == E.id_seqk)  return &E.kv_len;
        if (r == E.id_seqq)  return &E.q_len;
        if (r == E.id_ninf)  return &E.ninf;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    for (auto & lt : E.ins) {
        const size_t rid = lt.get_id();
        if (E.causal && (rid == E.id_seqk || rid == E.id_seqq || rid == E.id_ninf
                         || (E.hsc && rid == E.id_scale))) {
            ti.push_back(tensor::make_scalar_tensor(lt, id2ptr(rid)));
        } else {
            ti.emplace_back(lt, eng, id2ptr(rid));
        }
    }
    tensor to(E.out, eng, outf_ptr);
    if (fa_time) {
        try { stream->wait(); } catch (...) {}
    }
    const auto fa_t1 = std::chrono::steady_clock::now();
    E.cp.execute(strm, ti, {to});

    permute_sdpa_out_sycl(outf_ptr, (float *) dst->data, mb, H, q, d, stream);
    // Single device needs no sync: the dnnl stream wraps this same in-order queue, so the SDPA
    // serializes with the staging kernels before it and the permute/pool reuse after it. The
    // garbage output formerly blamed on the missing sync here was the scale use-after-return
    // fixed above. Keep the conservative wait for multi-GPU, where other devices' streams can
    // race the pool:
    if (ggml_sycl_info().device_count > 1 || fa_time) {
        stream->wait_and_throw();
    }
    if (fa_time && q >= 512) {
        const double fa_ms_sdpa = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fa_t1).count();
        const double fa_ms_tot = fa_ms_stage + fa_ms_sdpa;
        static double sum_stage = 0, sum_sdpa = 0, sum_tot = 0;
        static int n_fa = 0;
        sum_stage += fa_ms_stage;
        sum_sdpa += fa_ms_sdpa;
        sum_tot += fa_ms_tot;
        n_fa++;
        if (n_fa <= 24 || (n_fa % 64) == 0) {
            GGML_SYCL_XE2_LOG(
                    "xe2mx fa_time q=%ld seq=%ld H=%ld Hkv=%ld pkv=%d stage=%.2f sdpa=%.2f tot=%.2f "
                    "sum_stage=%.0f sum_sdpa=%.0f sum_tot=%.0f n=%d\n",
                    (long) q, (long) seq, (long) H, (long) Hkv, pkv ? 1 : 0, fa_ms_stage, fa_ms_sdpa,
                    fa_ms_tot, sum_stage, sum_sdpa, sum_tot, n_fa);
            fflush(stderr);
        }
    }
}
catch (const std::exception & e) {
    // any oneDNN/SYCL failure is non-fatal: fall back to the existing kernel (strictly additive).
    GGML_LOG_WARN("%s: oneDNN SDPA failed (%s); falling back to TILE kernel\n", __func__, e.what());
    ggml_sycl_flash_attn_ext_tile(ctx, dst);
}

#endif // GGML_SYCL_DNNL
