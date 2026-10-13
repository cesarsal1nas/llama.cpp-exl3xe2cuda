#include <cstring>
#include <cstdio>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include "gdn-chunk.hpp"
#include "gdn-fla.hpp"

static thread_local ggml_backend_sycl_context * g_gdn_ctx = nullptr;
#include <cmath>
#if defined(__INTEL_LLVM_COMPILER)
    #include <sycl/ext/intel/esimd.hpp>
    #include <sycl/ext/intel/experimental/grf_size_properties.hpp>
    #define GGML_SYCL_GDN_HAS_ESIMD
#endif

// GGML_SYCL_FUSE_GDN_PRO: fold sigmoid(β) and/or E7 gate into this kernel.
// Formulas copied from element_wise so greedy identity holds.
struct gdn_pro_args {
    int           apply_sigmoid;
    int           apply_gate;
    const float * alpha;
    const float * dt;
    const float * a;
    int64_t       sa1, sa2, sa3;
};

static thread_local int   g_gdn_apply_l2 = 0;
static thread_local float g_gdn_l2eps    = 1.0e-6f;
static inline float gdn_sigmoid_f32(float x) {
    return 1.0f / (1.0f + sycl::exp(-x));
}

static inline float gdn_softplus_f32(float x) {
    const float ax = sycl::fabs(x);
    const float m  = sycl::fmax(x, 0.0f);
    return m + sycl::log1p(sycl::exp(-ax));
}

#ifdef GGML_SYCL_GDN_HAS_ESIMD
// Separate ESIMD COL2 instantiation. KEEP SYCL COL2 is untouched (not a runtime branch).
// One SIMD16 thread owns two columns of S_v=128 (full-column simd FMA, no warp reduce).
template <bool keep_rs_t>
ESIMD_INLINE void gated_delta_net_col2_esimd(const float * q, const float * k, const float * v, const float * gate,
                                             const float * beta, const float * curr_state, float * dst, float * state,
                                             int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
                                             int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                                             float scale, int64_t state_slot_stride, int Kst, const int32_t * state_rows,
                                             int64_t state_row_stride, gdn_pro_args pro, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int SV  = 128;
    constexpr int WGS = 16;
    const int gid = (int) it.get_group(0);
    const int lid = (int) it.get_local_id(0);
    const int wgs_h     = SV / (2 * WGS);  // 4
    const int h_idx     = (gid / wgs_h) % (int) H;
    const int sequence  = gid / (wgs_h * (int) H);
    const int col       = ((gid % wgs_h) * WGS + lid) * 2;
    if (col + 1 >= SV) {
        return;
    }

    const int64_t packed_stride    = H * SV * SV;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + (int64_t) h_idx * SV * SV;
    const int64_t state_out_offset = (sequence * H + h_idx) * SV * SV;
    state += state_out_offset;
    const float * state_in_col = curr_state + state_in_offset;
    float *       attn_data    = dst + (sequence * n_tokens * H + h_idx) * SV;

    auto load_col = [&](const float * p) {
        simd<float, SV> x;
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            x.template select<16, 1>(16 * i) = block_load<float, 16>(p + 16 * i);
        }
        return x;
    };
    auto store_col = [&](float * p, simd<float, SV> x) {
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            block_store<float, 16>(p + 16 * i, x.template select<16, 1>(16 * i));
        }
    };

    simd<float, SV> s0 = load_col(state_in_col + col * SV);
    simd<float, SV> s1 = load_col(state_in_col + (col + 1) * SV);

    for (int t = 0; t < (int) n_tokens; ++t) {
        const float * q_t = q + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
        const float * k_t = k + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;
        const int64_t gb  = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        auto e1 = [](float x) {
            simd<float, 1> s;
            s[0] = x;
            return exp(s)[0];
        };
        auto l1 = [](float x) {
            simd<float, 1> s;
            s[0] = x;
            return log(s)[0];
        };
        float beta_val = beta[gb];
        if (pro.apply_sigmoid) {
            beta_val = 1.0f / (1.0f + e1(-beta_val));
        }
        float g_in = gate[gb];
        if (pro.apply_gate) {
            const float al = pro.alpha[(int64_t) sequence * pro.sa3 + (int64_t) t * pro.sa2 + (int64_t) h_idx * pro.sa1];
            const float x  = al + pro.dt[h_idx];
            const float ax = x < 0.0f ? -x : x;
            const float m  = x > 0.0f ? x : 0.0f;
            g_in = (m + l1(1.0f + e1(-ax))) * pro.a[h_idx];
        }
        const float g_val = e1(g_in);
        simd<float, SV> kk = load_col(k_t);
        simd<float, SV> qq = load_col(q_t);
        const float kv0 = reduce<float>(s0 * kk, std::plus<>());
        const float kv1 = reduce<float>(s1 * kk, std::plus<>());
        const float d0  = (v_t[col] - g_val * kv0) * beta_val;
        const float d1  = (v_t[col + 1] - g_val * kv1) * beta_val;
        s0 = g_val * s0 + kk * d0;
        s1 = g_val * s1 + kk * d1;
        const float a0 = reduce<float>(s0 * qq, std::plus<>());
        const float a1 = reduce<float>(s1 * qq, std::plus<>());
        attn_data[col]     = a0 * scale;
        attn_data[col + 1] = a1 * scale;
        attn_data += SV * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < Kst) {
                float * st = state + (int64_t) target_slot * state_slot_stride;
                store_col(st + col * SV, s0);
                store_col(st + (col + 1) * SV, s1);
            }
        }
    }
}

template <bool keep_rs_t, int NH>
ESIMD_INLINE void gated_delta_net_col2_esimd_ilp(const float * q, const float * k, const float * v, const float * gate,
                                                   const float * beta, const float * curr_state, float * dst, float * state,
                                                   int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
                                                   int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                                                   float scale, int64_t state_slot_stride, int Kst, const int32_t * state_rows,
                                                   int64_t state_row_stride, gdn_pro_args pro, const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int SV  = 128;
    constexpr int WGS = 16;
    const int gid = (int) it.get_group(0);
    const int lid = (int) it.get_local_id(0);
    const int wgs_h    = SV / (2 * WGS);
    const int h_groups = (int) H / NH;
    const int h_base    = ((gid / wgs_h) % h_groups) * NH;
    const int sequence  = gid / (wgs_h * h_groups);
    const int col       = ((gid % wgs_h) * WGS + lid) * 2;
    if (col + 1 >= SV) {
        return;
    }

    const int64_t packed_stride = H * SV * SV;
    const int64_t seq_row       = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride    = state_rows ? state_row_stride : packed_stride;

    auto load_col = [&](const float * p) {
        simd<float, SV> x;
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            x.template select<16, 1>(16 * i) = block_load<float, 16>(p + 16 * i);
        }
        return x;
    };
    auto store_col = [&](float * p, simd<float, SV> x) {
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            block_store<float, 16>(p + 16 * i, x.template select<16, 1>(16 * i));
        }
    };
    auto e1 = [](float x) {
        simd<float, 1> s;
        s[0] = x;
        return exp(s)[0];
    };
    auto l1 = [](float x) {
        simd<float, 1> s;
        s[0] = x;
        return log(s)[0];
    };

    simd<float, SV> s0[NH];
    simd<float, SV> s1[NH];
    const float * state_in_col[NH];
    float * state_h[NH];
    float * attn_h[NH];
#pragma unroll
    for (int nh = 0; nh < NH; ++nh) {
        const int h_idx = h_base + nh;
        const int64_t state_in_offset  = seq_row * seq_stride + (int64_t) h_idx * SV * SV;
        const int64_t state_out_offset = (sequence * H + h_idx) * SV * SV;
        state_in_col[nh] = curr_state + state_in_offset;
        state_h[nh]      = state + state_out_offset;
        attn_h[nh]       = dst + (sequence * n_tokens * H + h_idx) * SV;
        s0[nh] = load_col(state_in_col[nh] + col * SV);
        s1[nh] = load_col(state_in_col[nh] + (col + 1) * SV);
    }

    for (int t = 0; t < (int) n_tokens; ++t) {
#pragma unroll
        for (int nh = 0; nh < NH; ++nh) {
            const int h_idx = h_base + nh;
            const float * q_t = q + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
            const float * k_t = k + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
            const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;
            const int64_t gb  = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
            float beta_val = beta[gb];
            if (pro.apply_sigmoid) {
                beta_val = 1.0f / (1.0f + e1(-beta_val));
            }
            float g_in = gate[gb];
            if (pro.apply_gate) {
                const float al = pro.alpha[(int64_t) sequence * pro.sa3 + (int64_t) t * pro.sa2 + (int64_t) h_idx * pro.sa1];
                const float x  = al + pro.dt[h_idx];
                const float ax = x < 0.0f ? -x : x;
                const float m  = x > 0.0f ? x : 0.0f;
                g_in = (m + l1(1.0f + e1(-ax))) * pro.a[h_idx];
            }
            const float g_val = e1(g_in);
            simd<float, SV> kk = load_col(k_t);
            simd<float, SV> qq = load_col(q_t);
            const float kv0 = reduce<float>(s0[nh] * kk, std::plus<>());
            const float kv1 = reduce<float>(s1[nh] * kk, std::plus<>());
            const float d0  = (v_t[col] - g_val * kv0) * beta_val;
            const float d1  = (v_t[col + 1] - g_val * kv1) * beta_val;
            s0[nh] = g_val * s0[nh] + kk * d0;
            s1[nh] = g_val * s1[nh] + kk * d1;
            const float a0 = reduce<float>(s0[nh] * qq, std::plus<>());
            const float a1 = reduce<float>(s1[nh] * qq, std::plus<>());
            attn_h[nh][col]     = a0 * scale;
            attn_h[nh][col + 1] = a1 * scale;
            attn_h[nh] += SV * H;
            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < Kst) {
                    float * st = state_h[nh] + (int64_t) target_slot * state_slot_stride;
                    store_col(st + col * SV, s0[nh]);
                    store_col(st + (col + 1) * SV, s1[nh]);
                }
            }
        }
    }
}

// KEEP-grid COL2: each ESIMD thread is one KEEP warp *lane* (simd<float,8> rows + SLM warp reduce).
// Not gated_delta_net_col2_esimd (full-column 2-col, KERNEL_CUTS gdn-esimd).
template <bool keep_rs_t>
ESIMD_INLINE void gated_delta_net_col2_esimd_warp(const float * q, const float * k, const float * v, const float * gate,
                                                  const float * beta, const float * curr_state, float * dst, float * state,
                                                  int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t sq1, int64_t sq2,
                                                  int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1,
                                                  int64_t sb2, int64_t sb3, float scale, int64_t state_slot_stride, int Kst,
                                                  const int32_t * state_rows, int64_t state_row_stride, gdn_pro_args pro,
                                                  const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int SV   = 128;
    constexpr int WARP = 16;
    constexpr int NW   = 4;
    constexpr int RPL  = SV / WARP;  // 8
    constexpr int NCG  = SV / (NW * 2);  // 16 col-groups, KEEP COL2
    slm_init<NW * WARP * 4>();

    const int gid  = (int) it.get_group(0);
    const int lid  = (int) it.get_local_id(0);
    const int lane = lid % WARP;
    const int warp = lid / WARP;
    const int col_group = gid % NCG;
    const int sequence  = (gid / NCG) % (int) n_seqs;
    const int h_idx     = gid / (NCG * (int) n_seqs);
    const int col       = (col_group * NW + warp) * 2;
    if (col + 1 >= SV || warp >= NW) {
        return;
    }

    const int64_t packed_stride    = H * SV * SV;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + (int64_t) h_idx * SV * SV;
    const int64_t state_out_offset = (sequence * H + h_idx) * SV * SV;
    state += state_out_offset;
    const float * state_in_col = curr_state + state_in_offset;
    float *       attn_data    = dst + (sequence * n_tokens * H + h_idx) * SV;

    simd<uint32_t, RPL> roff;
#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        roff[r] = (uint32_t) ((r * WARP + lane) * (int) sizeof(float));
    }
    simd<float, RPL> s0 = gather<float, RPL>(state_in_col + col * SV, roff);
    simd<float, RPL> s1 = gather<float, RPL>(state_in_col + (col + 1) * SV, roff);

    auto e1 = [](float x) {
        simd<float, 1> s;
        s[0] = x;
        return exp(s)[0];
    };
    auto l1 = [](float x) {
        simd<float, 1> s;
        s[0] = x;
        return log(s)[0];
    };
    auto warp_sum = [&](float x) {
        const uint32_t base = (uint32_t) (warp * WARP * 4);
        simd<float, 1> xv;
        xv[0] = x;
        slm_scatter<float, 1>(simd<uint32_t, 1>(base + (uint32_t) lane * 4), xv);
        barrier();
        simd<float, WARP> v = slm_block_load<float, WARP>(base);
        const float s = reduce<float>(v, std::plus<>());
        barrier();
        return s;
    };

    for (int t = 0; t < (int) n_tokens; ++t) {
        const float * q_t = q + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
        const float * k_t = k + (int64_t) sequence * sq3 + (int64_t) t * sq2 + (int64_t) h_idx * sq1;
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;
        const int64_t gb  = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        float beta_val = beta[gb];
        if (pro.apply_sigmoid) {
            beta_val = 1.0f / (1.0f + e1(-beta_val));
        }
        float g_in = gate[gb];
        if (pro.apply_gate) {
            const float al = pro.alpha[(int64_t) sequence * pro.sa3 + (int64_t) t * pro.sa2 + (int64_t) h_idx * pro.sa1];
            const float x  = al + pro.dt[h_idx];
            const float ax = x < 0.0f ? -x : x;
            const float m  = x > 0.0f ? x : 0.0f;
            g_in = (m + l1(1.0f + e1(-ax))) * pro.a[h_idx];
        }
        const float g_val = e1(g_in);
        simd<float, RPL> kk = gather<float, RPL>(k_t, roff);
        simd<float, RPL> qq = gather<float, RPL>(q_t, roff);
        float kv0 = warp_sum(reduce<float>(s0 * kk, std::plus<>()));
        float kv1 = warp_sum(reduce<float>(s1 * kk, std::plus<>()));
        const float d0 = (v_t[col] - g_val * kv0) * beta_val;
        const float d1 = (v_t[col + 1] - g_val * kv1) * beta_val;
        s0 = g_val * s0 + kk * d0;
        s1 = g_val * s1 + kk * d1;
        const float a0 = warp_sum(reduce<float>(s0 * qq, std::plus<>()));
        const float a1 = warp_sum(reduce<float>(s1 * qq, std::plus<>()));
        if (lane == 0) {
            attn_data[col]     = a0 * scale;
            attn_data[col + 1] = a1 * scale;
        }
        attn_data += SV * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < Kst) {
                float * st = state + (int64_t) target_slot * state_slot_stride;
                scatter<float, RPL>(st + col * SV, roff, s0);
                scatter<float, RPL>(st + (col + 1) * SV, roff, s1);
            }
        }
    }
}

template <bool keep_rs_t>
ESIMD_INLINE void gated_delta_net_tcol_esimd(const float * q, const float * k, const float * v, const float * gate,
                                              const float * beta, const float * curr_state, float * dst, float * state,
                                              int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
                                              int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                                              int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int Kst,
                                              const int32_t * state_rows, int64_t state_row_stride, gdn_pro_args pro,
                                              const sycl::nd_item<1> & it) {
    using namespace sycl::ext::intel::esimd;
    constexpr int SV = 128;
    constexpr uint32_t SLM_Q = 0;
    constexpr uint32_t SLM_K = SV * 4;
    slm_init<SV * 8>();

    const int lid = (int) it.get_local_id(0);
    const int gid = (int) it.get_group(0);
    const int h_idx = gid % (int) H;
    const int sequence = gid / (int) H;
    const int col = lid * 2;
    const int iq1 = neqk1 > 0 ? h_idx % (int) neqk1 : h_idx;
    const int iq3 = rq3 > 0 ? sequence / (int) rq3 : sequence;

    const int64_t packed_stride = H * SV * SV;
    const int64_t seq_row = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset = seq_row * seq_stride + (int64_t) h_idx * SV * SV;
    const int64_t state_out_offset = (sequence * H + h_idx) * SV * SV;
    state += state_out_offset;
    const float * state_in_col = curr_state + state_in_offset;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * SV;

    simd<float, SV> s0;
    simd<float, SV> s1;
#pragma unroll
    for (int i = 0; i < SV / 16; ++i) {
        s0.template select<16, 1>(16 * i) = block_load<float, 16>(state_in_col + col * SV + 16 * i);
        s1.template select<16, 1>(16 * i) = block_load<float, 16>(state_in_col + (col + 1) * SV + 16 * i);
    }

    for (int t = 0; t < (int) n_tokens; ++t) {
        if (lid < SV / 16) {
            const int i = lid * 16;
            const float * q_t = q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1;
            const float * k_t = k + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1;
            slm_block_store<float, 16>(SLM_Q + (uint32_t) i * 4u, block_load<float, 16>(q_t + i));
            slm_block_store<float, 16>(SLM_K + (uint32_t) i * 4u, block_load<float, 16>(k_t + i));
        }
        barrier();
        simd<float, SV> kk;
        simd<float, SV> qq;
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            kk.template select<16, 1>(16 * i) = slm_block_load<float, 16>(SLM_K + (uint32_t) (16 * i) * 4u);
            qq.template select<16, 1>(16 * i) = slm_block_load<float, 16>(SLM_Q + (uint32_t) (16 * i) * 4u);
        }
        const float * v_t = v + (int64_t) sequence * sv3 + (int64_t) t * sv2 + (int64_t) h_idx * sv1;
        const int64_t gb = (int64_t) sequence * sb3 + (int64_t) t * sb2 + (int64_t) h_idx * sb1;
        simd<float, 1> es;
        simd<float, 1> ls;
        float beta_val = beta[gb];
        if (pro.apply_sigmoid) {
            es[0] = -beta_val;
            beta_val = 1.0f / (1.0f + exp(es)[0]);
        }
        float g_in = gate[gb];
        if (pro.apply_gate) {
            const float al = pro.alpha[(int64_t) sequence * pro.sa3 + (int64_t) t * pro.sa2 + (int64_t) h_idx * pro.sa1];
            const float x = al + pro.dt[h_idx];
            const float ax = x < 0.0f ? -x : x;
            const float m = x > 0.0f ? x : 0.0f;
            es[0] = -ax;
            ls[0] = 1.0f + exp(es)[0];
            g_in = (m + log(ls)[0]) * pro.a[h_idx];
        }
        es[0] = g_in;
        const float g_val = exp(es)[0];
        const float kv0 = reduce<float>(s0 * kk, std::plus<>());
        const float kv1 = reduce<float>(s1 * kk, std::plus<>());
        const float d0 = (v_t[col] - g_val * kv0) * beta_val;
        const float d1 = (v_t[col + 1] - g_val * kv1) * beta_val;
        s0 = g_val * s0 + kk * d0;
        s1 = g_val * s1 + kk * d1;
        const float a0 = reduce<float>(s0 * qq, std::plus<>());
        const float a1 = reduce<float>(s1 * qq, std::plus<>());
        attn_data[col] = a0 * scale;
        attn_data[col + 1] = a1 * scale;
        attn_data += SV * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < Kst) {
                float * st = state + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                for (int i = 0; i < SV / 16; ++i) {
                    block_store<float, 16>(st + col * SV + 16 * i, s0.template select<16, 1>(16 * i));
                    block_store<float, 16>(st + (col + 1) * SV + 16 * i, s1.template select<16, 1>(16 * i));
                }
            }
        }
        barrier();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < SV / 16; ++i) {
            block_store<float, 16>(state + col * SV + 16 * i, s0.template select<16, 1>(16 * i));
            block_store<float, 16>(state + (col + 1) * SV + 16 * i, s1.template select<16, 1>(16 * i));
        }
    }
}
#endif

// Prefill COL2: 128-d S·k / S·q via joint_matrix 8×16×16 f16. Sequential t, f32 saxpy.
// Not FLA/WY, not ESIMD COL2, not kqreg, not f16 scalar scan. Default 0.
template <bool keep_rs_t>
static void gated_delta_net_col2_xmx(const float * q, const float * k, const float * v, const float * g,
                                     const float * beta, const float * curr_state, float * dst, float * state,
                                     int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1,
                                     int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, float scale,
                                     int64_t state_slot_stride, int Kst, const int32_t * state_rows,
                                     int64_t state_row_stride, gdn_pro_args pro, const sycl::uint3 neqk1_magic,
                                     const sycl::uint3 rq3_magic, sycl::half * slm_a, sycl::half * slm_b,
                                     float * slm_c) {
    using namespace sycl::ext::oneapi::experimental::matrix;
    constexpr int SV = 128, TM = 8, TN = 16, TK = 16, WARP = 16;
    auto           item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto           sg   = item.get_sub_group();
    const uint32_t h_idx    = item.get_group(2);
    const uint32_t sequence = item.get_group(1);
    const int      lane     = (int) item.get_local_id(2);
    const int      warp     = (int) item.get_local_id(1);
    const int      col0     = (int) item.get_group(0) * (int) item.get_local_range(1) + warp;
    const int      col      = col0 * 2;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    const int64_t packed_stride = H * SV * SV;
    const int64_t seq_row       = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride    = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_off  = seq_row * seq_stride + (int64_t) h_idx * SV * SV;
    state += (sequence * H + h_idx) * SV * SV;
    const float * state_in_col = curr_state + state_in_off;
    float *       attn_data    = dst + (sequence * n_tokens * H + h_idx) * SV;

    constexpr int RPL = SV / WARP;
    float         s0[RPL], s1[RPL];
#pragma unroll
    for (int r = 0; r < RPL; r++) {
        const int i = r * WARP + lane;
        s0[r]       = state_in_col[col * SV + i];
        s1[r]       = state_in_col[(col + 1) * SV + i];
    }

    auto pack_ab = [&](const float * vec, const float s[RPL], sycl::half * a, sycl::half * b) {
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            const int i = r * WARP + lane;
            a[r * TK + lane] = sycl::half(s[r]);
            b[r * TK + lane] = sycl::half(vec[i]);
        }
#pragma unroll
        for (int n = 8; n < TN; n++) {
            b[n * TK + lane] = sycl::half(0.f);
        }
    };
    auto xmx_dot = [&](sycl::half * a, sycl::half * b, float * cbuf) -> float {
        auto pa = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(a);
        auto pb = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(b);
        auto pc = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(cbuf);
        joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> acc;
        joint_matrix_fill(sg, acc, 0.f);
        joint_matrix<sycl::sub_group, sycl::half, use::a, TM, TK, layout::row_major> afrag;
        joint_matrix<sycl::sub_group, sycl::half, use::b, TK, TN, layout::col_major> bfrag;
        joint_matrix_load(sg, afrag, pa, TK);
        joint_matrix_load(sg, bfrag, pb, TK);
        joint_matrix_mad(sg, acc, afrag, bfrag, acc);
        joint_matrix_store(sg, acc, pc, TN, layout::row_major);
        float sum = 0.f;
#pragma unroll
        for (int r = 0; r < TM; r++) {
            sum += cbuf[r * TN + r];
        }
        return sum;
    };

    sycl::half * a0 = slm_a + (size_t) warp * (TM * TK);
    sycl::half * a1 = slm_a + (size_t) (4 + warp) * (TM * TK);
    sycl::half * bk = slm_b;
    sycl::half * bq = slm_b + TK * TN;
    float *      c0 = slm_c + (size_t) warp * (TM * TN);
    float *      c1 = slm_c + (size_t) (4 + warp) * (TM * TN);

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb  = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        float g_in = g[gb];
        if (pro.apply_gate) {
            const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
            g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
        }
        const float g_val = sycl::native::exp(g_in);

        pack_ab(k_t, s0, a0, bk);
        pack_ab(k_t, s1, a1, bk);
        item.barrier(sycl::access::fence_space::local_space);
        const float kv0 = xmx_dot(a0, bk, c0);
        const float kv1 = xmx_dot(a1, bk, c1);
        const float d0  = (v_t[col] - g_val * kv0) * beta_val;
        const float d1  = (v_t[col + 1] - g_val * kv1) * beta_val;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            const int i = r * WARP + lane;
            s0[r]       = g_val * s0[r] + k_t[i] * d0;
            s1[r]       = g_val * s1[r] + k_t[i] * d1;
        }
        pack_ab(q_t, s0, a0, bq);
        pack_ab(q_t, s1, a1, bq);
        item.barrier(sycl::access::fence_space::local_space);
        const float a0d = xmx_dot(a0, bq, c0);
        const float a1d = xmx_dot(a1, bq, c1);
        if (lane == 0) {
            attn_data[col]     = a0d * scale;
            attn_data[col + 1] = a1d * scale;
        }
        attn_data += SV * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < Kst) {
                float * st = state + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < RPL; r++) {
                    const int i = r * WARP + lane;
                    st[col * SV + i]       = s0[r];
                    st[(col + 1) * SV + i] = s1[r];
                }
            }
        }
    }
}

// Prefill COL2: 128-d S·k / S·q via SIMT dp4a on s8. Sequential t, f32 saxpy.
// Not joint_matrix/XMX, not ESIMD COL2, not f16 scalar, not FLA. Default 0.
template <bool keep_rs_t>
static void gated_delta_net_col2_dp4a(const float * q, const float * k, const float * v, const float * g,
                                      const float * beta, const float * curr_state, float * dst, float * state,
                                      int64_t H, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1,
                                      int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, float scale,
                                      int64_t state_slot_stride, int Kst, const int32_t * state_rows,
                                      int64_t state_row_stride, gdn_pro_args pro, const sycl::uint3 neqk1_magic,
                                      const sycl::uint3 rq3_magic) {
    constexpr int SV = 128, WARP = 16, RPL = 8;
    auto           item     = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item.get_group(2);
    const uint32_t sequence = item.get_group(1);
    const int      lane     = (int) item.get_local_id(2);
    const int      warp     = (int) item.get_local_id(1);
    const int      col      = ((int) item.get_group(0) * (int) item.get_local_range(1) + warp) * 2;
    const uint32_t iq1      = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3      = fastdiv(sequence, rq3_magic);
    const int64_t  packed_stride = H * SV * SV;
    const int64_t  seq_row       = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t  seq_stride    = state_rows ? state_row_stride : packed_stride;
    state += (sequence * H + h_idx) * SV * SV;
    const float * state_in_col = curr_state + seq_row * seq_stride + (int64_t) h_idx * SV * SV;
    float *       attn_data    = dst + (sequence * n_tokens * H + h_idx) * SV;

    float s0[RPL], s1[RPL];
#pragma unroll
    for (int r = 0; r < RPL; r++) {
        const int i = r * WARP + lane;
        s0[r]       = state_in_col[col * SV + i];
        s1[r]       = state_in_col[(col + 1) * SV + i];
    }

    auto pack8 = [&](const float x[RPL], float iscale, int & lo, int & hi) {
        auto qn = [&](int r) {
            int v = (int) sycl_rint_i(x[r] * iscale);
            v     = sycl::clamp(v, -127, 127);
            return (uint32_t) (uint8_t) (int8_t) v;
        };
        lo = (int) (qn(0) | (qn(1) << 8) | (qn(2) << 16) | (qn(3) << 24));
        hi = (int) (qn(4) | (qn(5) << 8) | (qn(6) << 16) | (qn(7) << 24));
    };
    auto amax8 = [&](const float x[RPL]) {
        float am = 0.f;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            am = sycl::fmax(am, sycl::fabs(x[r]));
        }
        return warp_reduce_max<WARP>(am);
    };
    auto dot_s8 = [&](int slo, int shi, int vlo, int vhi, float ss, float vs) {
        int acc = ggml_sycl_dp4a(slo, vlo, 0);
        acc     = ggml_sycl_dp4a(shi, vhi, acc);
        return warp_reduce_sum<WARP>((float) acc * ss * vs);
    };

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb  = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        float g_in = g[gb];
        if (pro.apply_gate) {
            const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
            g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
        }
        const float g_val = sycl::native::exp(g_in);

        float kreg[RPL], qreg[RPL];
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            const int i = r * WARP + lane;
            kreg[r]     = k_t[i];
            qreg[r]     = q_t[i];
        }
        const float amk = amax8(kreg);
        const float amq = amax8(qreg);
        const float am0 = amax8(s0);
        const float am1 = amax8(s1);
        const float isk = (amk > 1e-8f) ? 127.f / amk : 0.f;
        const float isq = (amq > 1e-8f) ? 127.f / amq : 0.f;
        const float is0 = (am0 > 1e-8f) ? 127.f / am0 : 0.f;
        const float is1 = (am1 > 1e-8f) ? 127.f / am1 : 0.f;
        const float sk  = (isk > 0.f) ? amk / 127.f : 0.f;
        const float sq  = (isq > 0.f) ? amq / 127.f : 0.f;
        const float ss0 = (is0 > 0.f) ? am0 / 127.f : 0.f;
        const float ss1 = (is1 > 0.f) ? am1 / 127.f : 0.f;
        int klo, khi, qlo, qhi, s0lo, s0hi, s1lo, s1hi;
        pack8(kreg, isk, klo, khi);
        pack8(qreg, isq, qlo, qhi);
        pack8(s0, is0, s0lo, s0hi);
        pack8(s1, is1, s1lo, s1hi);
        const float kv0 = dot_s8(s0lo, s0hi, klo, khi, ss0, sk);
        const float kv1 = dot_s8(s1lo, s1hi, klo, khi, ss1, sk);
        const float d0  = (v_t[col] - g_val * kv0) * beta_val;
        const float d1  = (v_t[col + 1] - g_val * kv1) * beta_val;
#pragma unroll
        for (int r = 0; r < RPL; r++) {
            s0[r] = g_val * s0[r] + kreg[r] * d0;
            s1[r] = g_val * s1[r] + kreg[r] * d1;
        }
        const float am0b = amax8(s0);
        const float am1b = amax8(s1);
        const float is0b = (am0b > 1e-8f) ? 127.f / am0b : 0.f;
        const float is1b = (am1b > 1e-8f) ? 127.f / am1b : 0.f;
        const float ss0b = (is0b > 0.f) ? am0b / 127.f : 0.f;
        const float ss1b = (is1b > 0.f) ? am1b / 127.f : 0.f;
        pack8(s0, is0b, s0lo, s0hi);
        pack8(s1, is1b, s1lo, s1hi);
        const float a0d = dot_s8(s0lo, s0hi, qlo, qhi, ss0b, sq);
        const float a1d = dot_s8(s1lo, s1hi, qlo, qhi, ss1b, sq);
        if (lane == 0) {
            attn_data[col]     = a0d * scale;
            attn_data[col + 1] = a1d * scale;
        }
        attn_data += SV * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < Kst) {
                float * st = state + (int64_t) target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < RPL; r++) {
                    const int i = r * WARP + lane;
                    st[col * SV + i]       = s0[r];
                    st[(col + 1) * SV + i] = s1[r];
                }
            }
        }
    }
}

template <int S_v, bool KDA, bool keep_rs_t, bool KQREG = false>
void gated_delta_net_sycl(const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K,
                          const int32_t *   state_rows,
                          int64_t           state_row_stride,
                          gdn_pro_args      pro,
                          int               pipe,
                          int               col2,
                          int               col4) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns one column (or two if col2), using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col0     = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int      col      = col4 ? col0 * 4 : (col2 ? col0 * 2 : col0);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // When state_rows is set, each sequence gathers a cache row (skip GET_ROWS).
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t packed_stride        = H * S_v * S_v;
    const int64_t seq_row              = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride           = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset      = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    const float * state_in_col = curr_state + state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    if (col >= S_v || (col4 && col + 3 >= S_v) || (col2 && col + 1 >= S_v)) {
        return;
    }
    float s_shard[rows_per_lane];
    float s_shard1[rows_per_lane];
    float s_shard2[rows_per_lane];
    float s_shard3[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = state_in_col[col * S_v + i];
        if (col4) {
            s_shard1[r] = state_in_col[(col + 1) * S_v + i];
            s_shard2[r] = state_in_col[(col + 2) * S_v + i];
            s_shard3[r] = state_in_col[(col + 3) * S_v + i];
        } else if (col2) {
            s_shard1[r] = state_in_col[(col + 1) * S_v + i];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    auto gdn_load_kqvb = [&](int t, float * kr, float * qr, float & vv, float & vv1, float & gv, float & bv) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            kr[r] = k_t[i];
            qr[r] = q_t[i];
        }
        vv = v_t[col];
        vv1 = col2 ? v_t[col + 1] : 0.f;
        float g_in = g[gb_offset];
        if (pro.apply_gate) {
            const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
            g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
        }
        gv = sycl::native::exp(g_in);
        bv = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb_offset]) : beta[gb_offset];
    };

    if constexpr (!KDA) {
    if (pipe && !col4) {
        float k_cur[rows_per_lane], q_cur[rows_per_lane];
        float k_nxt[rows_per_lane], q_nxt[rows_per_lane];
        float v_cur = 0.f, v_cur1 = 0.f, g_cur = 0.f, b_cur = 0.f;
        float v_nxt = 0.f, v_nxt1 = 0.f, g_nxt = 0.f, b_nxt = 0.f;
        if (n_tokens > 0) {
            gdn_load_kqvb(0, k_cur, q_cur, v_cur, v_cur1, g_cur, b_cur);
        }
        for (int t = 0; t < n_tokens; t++) {
            if (t + 1 < n_tokens) {
                gdn_load_kqvb(t + 1, k_nxt, q_nxt, v_nxt, v_nxt1, g_nxt, b_nxt);
            }
            if (col2) {
                float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv0 += s_shard[r] * k_cur[r];
                    kv1 += s_shard1[r] * k_cur[r];
                }
                kv0 = warp_reduce_sum<warp_size>(kv0);
                kv1 = warp_reduce_sum<warp_size>(kv1);
                const float d0 = (v_cur - g_cur * kv0) * b_cur;
                const float d1 = (v_cur1 - g_cur * kv1) * b_cur;
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r]  = g_cur * s_shard[r] + k_cur[r] * d0;
                    s_shard1[r] = g_cur * s_shard1[r] + k_cur[r] * d1;
                    a0 += s_shard[r] * q_cur[r];
                    a1 += s_shard1[r] * q_cur[r];
                }
                a0 = warp_reduce_sum<warp_size>(a0);
                a1 = warp_reduce_sum<warp_size>(a1);
                if (lane == 0) {
                    attn_data[col]     = a0 * scale;
                    attn_data[col + 1] = a1 * scale;
                }
            } else {
                float kv_shard = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_cur[r];
                }
                float kv_col = warp_reduce_sum<warp_size>(kv_shard);
                float delta_col = (v_cur - g_cur * kv_col) * b_cur;
                float attn_partial = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_cur * s_shard[r] + k_cur[r] * delta_col;
                    attn_partial += s_shard[r] * q_cur[r];
                }
                float attn_col = warp_reduce_sum<warp_size>(attn_partial);
                if (lane == 0) {
                    attn_data[col] = attn_col * scale;
                }
            }
            attn_data += S_v * H;
            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[col * S_v + i] = s_shard[r];
                        if (col2) {
                            curr_state[(col + 1) * S_v + i] = s_shard1[r];
                        }
                    }
                }
            }
            if (t + 1 < n_tokens) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    k_cur[r] = k_nxt[r];
                    q_cur[r] = q_nxt[r];
                }
                v_cur = v_nxt;
                v_cur1 = v_nxt1;
                g_cur = g_nxt;
                b_cur = b_nxt;
            }
        }
    } else if (col4) {
        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb_offset]) : beta[gb_offset];
            float g_in = g[gb_offset];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            const float g_val = sycl::native::exp(g_in);
            float kk[rows_per_lane];
            float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kk[r] = k_t[r * warp_size + lane];
                kv0 += s_shard[r] * kk[r];
                kv1 += s_shard1[r] * kk[r];
                kv2 += s_shard2[r] * kk[r];
                kv3 += s_shard3[r] * kk[r];
            }
            sycl::float2 kv01 = warp_reduce_sum<warp_size>(sycl::float2(kv0, kv1));
            sycl::float2 kv23 = warp_reduce_sum<warp_size>(sycl::float2(kv2, kv3));
            const float d0 = (v_t[col] - g_val * kv01.x()) * beta_val;
            const float d1 = (v_t[col + 1] - g_val * kv01.y()) * beta_val;
            const float d2 = (v_t[col + 2] - g_val * kv23.x()) * beta_val;
            const float d3 = (v_t[col + 3] - g_val * kv23.y()) * beta_val;
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const float qq = q_t[r * warp_size + lane];
                s_shard[r]  = g_val * s_shard[r] + kk[r] * d0;
                s_shard1[r] = g_val * s_shard1[r] + kk[r] * d1;
                s_shard2[r] = g_val * s_shard2[r] + kk[r] * d2;
                s_shard3[r] = g_val * s_shard3[r] + kk[r] * d3;
                a0 += s_shard[r] * qq;
                a1 += s_shard1[r] * qq;
                a2 += s_shard2[r] * qq;
                a3 += s_shard3[r] * qq;
            }
            sycl::float2 a01 = warp_reduce_sum<warp_size>(sycl::float2(a0, a1));
            sycl::float2 a23 = warp_reduce_sum<warp_size>(sycl::float2(a2, a3));
            if (lane == 0) {
                attn_data[col]     = a01.x() * scale;
                attn_data[col + 1] = a01.y() * scale;
                attn_data[col + 2] = a23.x() * scale;
                attn_data[col + 3] = a23.y() * scale;
            }
            attn_data += S_v * H;
            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * st = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        st[col * S_v + i]       = s_shard[r];
                        st[(col + 1) * S_v + i] = s_shard1[r];
                        st[(col + 2) * S_v + i] = s_shard2[r];
                        st[(col + 3) * S_v + i] = s_shard3[r];
                    }
                }
            }
        }
    } else if (col2) {
        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb_offset]) : beta[gb_offset];
            float g_in = g[gb_offset];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            const float g_val = sycl::native::exp(g_in);
            float kv0 = 0.f, kv1 = 0.f;
            if constexpr (KQREG) {
                float kk_r[rows_per_lane], qq_r[rows_per_lane];
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kk_r[r] = k_t[i];
                    qq_r[r] = q_t[i];
                }
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv0 += s_shard[r] * kk_r[r];
                    kv1 += s_shard1[r] * kk_r[r];
                }
                kv0 = warp_reduce_sum<warp_size>(kv0);
                kv1 = warp_reduce_sum<warp_size>(kv1);
                const float d0 = (v_t[col] - g_val * kv0) * beta_val;
                const float d1 = (v_t[col + 1] - g_val * kv1) * beta_val;
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r]  = g_val * s_shard[r] + kk_r[r] * d0;
                    s_shard1[r] = g_val * s_shard1[r] + kk_r[r] * d1;
                    a0 += s_shard[r] * qq_r[r];
                    a1 += s_shard1[r] * qq_r[r];
                }
                a0 = warp_reduce_sum<warp_size>(a0);
                a1 = warp_reduce_sum<warp_size>(a1);
                if (lane == 0) {
                    attn_data[col]     = a0 * scale;
                    attn_data[col + 1] = a1 * scale;
                }
            } else {
                float kk[rows_per_lane];
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kk[r] = k_t[r * warp_size + lane];
                    kv0 += s_shard[r] * kk[r];
                    kv1 += s_shard1[r] * kk[r];
                }
                sycl::float2 kv = warp_reduce_sum<warp_size>(sycl::float2(kv0, kv1));
                const float d0 = (v_t[col] - g_val * kv.x()) * beta_val;
                const float d1 = (v_t[col + 1] - g_val * kv.y()) * beta_val;
                float a0 = 0.f, a1 = 0.f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const float qq = q_t[r * warp_size + lane];
                    s_shard[r]  = g_val * s_shard[r] + kk[r] * d0;
                    s_shard1[r] = g_val * s_shard1[r] + kk[r] * d1;
                    a0 += s_shard[r] * qq;
                    a1 += s_shard1[r] * qq;
                }
                sycl::float2 a = warp_reduce_sum<warp_size>(sycl::float2(a0, a1));
                if (lane == 0) {
                    attn_data[col]     = a.x() * scale;
                    attn_data[col + 1] = a.y() * scale;
                }
            }
            attn_data += S_v * H;
            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * st = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        st[col * S_v + i]       = s_shard[r];
                        st[(col + 1) * S_v + i] = s_shard1[r];
                    }
                }
            }
        }
    } else {
        for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset;

        const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(*beta_t) : *beta_t;

            float g_in = *g_t;
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            const float g_val = sycl::native::exp(g_in);

            float kv_shard = 0.0f;
            float kk[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kk[r] = k_t[i];
                kv_shard += s_shard[r] * kk[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + kk[r] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }

        attn_data += S_v * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
        }
    }
    } else {
    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * S_v;

        const float beta_val = pro.apply_sigmoid ? gdn_sigmoid_f32(*beta_t) : *beta_t;

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += sycl::native::exp(g_t[i]) * s_shard[r] * k_t[i];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            float delta_col = (v_t[col] - kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = sycl::native::exp(g_t[i]) * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }

        attn_data += S_v * H;
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
            if (col4) {
                state[(col + 1) * S_v + i] = s_shard1[r];
                state[(col + 2) * S_v + i] = s_shard2[r];
                state[(col + 3) * S_v + i] = s_shard3[r];
            } else if (col2) {
                state[(col + 1) * S_v + i] = s_shard1[r];
            }
        }
    }
}

template <bool keep_rs_t, int TILE>
static void gated_delta_net_tcol128(const float *     q,
                                     const float *     k,
                                     const float *     v,
                                     const float *     g,
                                     const float *     beta,
                                     const float *     curr_state,
                                     float *           dst,
                                     float *           state,
                                     int64_t           H,
                                     int64_t           n_tokens,
                                     int64_t           sq1,
                                     int64_t           sq2,
                                     int64_t           sq3,
                                     int64_t           sv1,
                                     int64_t           sv2,
                                     int64_t           sv3,
                                     int64_t           sb1,
                                     int64_t           sb2,
                                     int64_t           sb3,
                                     const sycl::uint3 neqk1_magic,
                                     const sycl::uint3 rq3_magic,
                                     float             scale,
                                     int64_t           state_slot_stride,
                                     int               K,
                                     const int32_t *   state_rows,
                                     int64_t           state_row_stride,
                                     gdn_pro_args      pro,
                                     float *           qslm,
                                     float *           kslm,
                                     float *           gslm,
                                     float *           bslm) {
    constexpr int S_v      = 128;
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      tid      = (int) item_ct1.get_local_id(2);
    const int      nthr     = (int) item_ct1.get_local_range(2);
    const int      col      = (int) item_ct1.get_group(0) * nthr + tid;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t packed_stride    = H * S_v * S_v;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i];
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += TILE) {
        const int ntile = (int) sycl::min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);
        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            *reinterpret_cast<sycl::float4 *>(qslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(q_t + i);
            *reinterpret_cast<sycl::float4 *>(kslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(k_t + i);
        }
        if (tid < ntile) {
            const int     t  = t0 + tid;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            float          g_in = g[gb];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in           = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            gslm[tid] = g_in;
            bslm[tid] = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t      = qslm + tt * S_v;
            const float * k_t      = kslm + tt * S_v;
            const int     t        = t0 + tt;
            const float * v_t      = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const float   g_val    = sycl::native::exp(gslm[tt]);
            const float   beta_val = bslm[tt];

            float kv = 0.0f;
            #pragma unroll  // IGC 2.41.5 does not unroll this; s[128] spills
            for (int i = 0; i < S_v; i += 4) {
                const sycl::float4 kk = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                kv += s[i] * kk.x() + s[i + 1] * kk.y() + s[i + 2] * kk.z() + s[i + 3] * kk.w();
            }
            const float delta = (v_t[col] - g_val * kv) * beta_val;

            float attn = 0.0f;
            #pragma unroll  // IGC 2.41.5 does not unroll this; s[128] spills
            for (int i = 0; i < S_v; i += 4) {
                const sycl::float4 kk = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                const sycl::float4 qq = *reinterpret_cast<const sycl::float4 *>(q_t + i);
                s[i]     = g_val * s[i]     + kk.x() * delta;
                s[i + 1] = g_val * s[i + 1] + kk.y() * delta;
                s[i + 2] = g_val * s[i + 2] + kk.z() * delta;
                s[i + 3] = g_val * s[i + 3] + kk.w() * delta;
                attn += s[i] * qq.x() + s[i + 1] * qq.y() + s[i + 2] * qq.z() + s[i + 3] * qq.w();
            }
            attn_data[col] = attn * scale;
            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride + col * S_v;
#pragma unroll
                    for (int i = 0; i < S_v; i++) {
                        slot[i] = s[i];
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[col * S_v + i] = s[i];
        }
    }
}

template <bool keep_rs_t, int TILE>
static void gated_delta_net_tcol128_acc2(const float *     q,
                                         const float *     k,
                                         const float *     v,
                                         const float *     g,
                                         const float *     beta,
                                         const float *     curr_state,
                                         float *           dst,
                                         float *           state,
                                         int64_t           H,
                                         int64_t           n_tokens,
                                         int64_t           sq1,
                                         int64_t           sq2,
                                         int64_t           sq3,
                                         int64_t           sv1,
                                         int64_t           sv2,
                                         int64_t           sv3,
                                         int64_t           sb1,
                                         int64_t           sb2,
                                         int64_t           sb3,
                                         const sycl::uint3 neqk1_magic,
                                         const sycl::uint3 rq3_magic,
                                         float             scale,
                                         int64_t           state_slot_stride,
                                         int               K,
                                         const int32_t *   state_rows,
                                         int64_t           state_row_stride,
                                         gdn_pro_args      pro,
                                         float *           qslm,
                                         float *           kslm,
                                         float *           gslm,
                                         float *           bslm) {
    constexpr int S_v      = 128;
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      tid      = (int) item_ct1.get_local_id(2);
    const int      nthr     = (int) item_ct1.get_local_range(2);
    const int      col      = (int) item_ct1.get_group(0) * nthr + tid;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t packed_stride    = H * S_v * S_v;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i];
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += TILE) {
        const int ntile = (int) sycl::min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);
        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            *reinterpret_cast<sycl::float4 *>(qslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(q_t + i);
            *reinterpret_cast<sycl::float4 *>(kslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(k_t + i);
        }
        if (tid < ntile) {
            const int     t  = t0 + tid;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            float          g_in = g[gb];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in           = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            gslm[tid] = g_in;
            bslm[tid] = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t      = qslm + tt * S_v;
            const float * k_t      = kslm + tt * S_v;
            const int     t        = t0 + tt;
            const float * v_t      = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const float   g_val    = sycl::native::exp(gslm[tt]);
            const float   beta_val = bslm[tt];

            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 8) {
                const sycl::float4 k0 = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                const sycl::float4 k1 = *reinterpret_cast<const sycl::float4 *>(k_t + i + 4);
                kv += s[i] * k0.x() + s[i + 1] * k0.y() + s[i + 2] * k0.z() + s[i + 3] * k0.w();
                kv += s[i + 4] * k1.x() + s[i + 5] * k1.y() + s[i + 6] * k1.z() + s[i + 7] * k1.w();
            }
            const float delta = (v_t[col] - g_val * kv) * beta_val;

            float attn = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 8) {
                const sycl::float4 k0 = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                const sycl::float4 k1 = *reinterpret_cast<const sycl::float4 *>(k_t + i + 4);
                const sycl::float4 q0 = *reinterpret_cast<const sycl::float4 *>(q_t + i);
                const sycl::float4 q1 = *reinterpret_cast<const sycl::float4 *>(q_t + i + 4);
                s[i]     = g_val * s[i]     + k0.x() * delta;
                s[i + 1] = g_val * s[i + 1] + k0.y() * delta;
                s[i + 2] = g_val * s[i + 2] + k0.z() * delta;
                s[i + 3] = g_val * s[i + 3] + k0.w() * delta;
                attn += s[i] * q0.x() + s[i + 1] * q0.y() + s[i + 2] * q0.z() + s[i + 3] * q0.w();
                s[i + 4] = g_val * s[i + 4] + k1.x() * delta;
                s[i + 5] = g_val * s[i + 5] + k1.y() * delta;
                s[i + 6] = g_val * s[i + 6] + k1.z() * delta;
                s[i + 7] = g_val * s[i + 7] + k1.w() * delta;
                attn += s[i + 4] * q1.x() + s[i + 5] * q1.y() + s[i + 6] * q1.z() + s[i + 7] * q1.w();
            }
            attn_data[col] = attn * scale;
            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride + col * S_v;
#pragma unroll
                    for (int i = 0; i < S_v; i++) {
                        slot[i] = s[i];
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[col * S_v + i] = s[i];
        }
    }
}

template <bool keep_rs_t, int TILE>
static void gated_delta_net_tcol128_l2(const float *     q,
                                       const float *     k,
                                       const float *     v,
                                       const float *     g,
                                       const float *     beta,
                                       const float *     curr_state,
                                       float *           dst,
                                       float *           state,
                                       int64_t           H,
                                       int64_t           n_tokens,
                                       int64_t           sq1,
                                       int64_t           sq2,
                                       int64_t           sq3,
                                       int64_t           sv1,
                                       int64_t           sv2,
                                       int64_t           sv3,
                                       int64_t           sb1,
                                       int64_t           sb2,
                                       int64_t           sb3,
                                       const sycl::uint3 neqk1_magic,
                                       const sycl::uint3 rq3_magic,
                                       float             scale,
                                       int64_t           state_slot_stride,
                                       int               K,
                                       const int32_t *   state_rows,
                                       int64_t           state_row_stride,
                                       gdn_pro_args      pro,
                                       float             l2eps,
                                       float *           qslm,
                                       float *           kslm,
                                       float *           gslm,
                                       float *           bslm) {
    constexpr int S_v      = 128;
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      tid      = (int) item_ct1.get_local_id(2);
    const int      nthr     = (int) item_ct1.get_local_range(2);
    const int      col      = (int) item_ct1.get_group(0) * nthr + tid;
    const int      warp_id  = tid / WARP_SIZE;
    const int      lane     = tid % WARP_SIZE;
    const int      nwarps   = nthr / WARP_SIZE;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t packed_stride    = H * S_v * S_v;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i];
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += TILE) {
        const int ntile = (int) sycl::min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);
        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            *reinterpret_cast<sycl::float4 *>(qslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(q_t + i);
            *reinterpret_cast<sycl::float4 *>(kslm + tt * S_v + i) =
                *reinterpret_cast<const sycl::float4 *>(k_t + i);
        }
        if (tid < ntile) {
            const int     t  = t0 + tid;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            float          g_in = g[gb];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in           = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            gslm[tid] = g_in;
            bslm[tid] = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int round = 0; round < (ntile + nwarps - 1) / nwarps; round++) {
            const int tt = round * nwarps + warp_id;
            if (tt < ntile) {
                float * qv = qslm + tt * S_v;
                float * kv = kslm + tt * S_v;
                float q2 = 0.0f;
                float k2 = 0.0f;
                for (int i = lane; i < S_v; i += WARP_SIZE) {
                    const float qe = qv[i];
                    const float ke = kv[i];
                    q2 += qe * qe;
                    k2 += ke * ke;
                }
                q2 = warp_reduce_sum<WARP_SIZE>(q2);
                k2 = warp_reduce_sum<WARP_SIZE>(k2);
                const float qs = sycl::rsqrt(sycl::fmax(q2, l2eps * l2eps));
                const float ks = sycl::rsqrt(sycl::fmax(k2, l2eps * l2eps));
                for (int i = lane; i < S_v; i += WARP_SIZE) {
                    qv[i] *= qs;
                    kv[i] *= ks;
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t      = qslm + tt * S_v;
            const float * k_t      = kslm + tt * S_v;
            const int     t        = t0 + tt;
            const float * v_t      = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const float   g_val    = sycl::native::exp(gslm[tt]);
            const float   beta_val = bslm[tt];

            float kv = 0.0f;
            #pragma unroll  // IGC 2.41.5 does not unroll this; s[128] spills
            for (int i = 0; i < S_v; i += 4) {
                const sycl::float4 kk = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                kv += s[i] * kk.x() + s[i + 1] * kk.y() + s[i + 2] * kk.z() + s[i + 3] * kk.w();
            }
            const float delta = (v_t[col] - g_val * kv) * beta_val;

            float attn = 0.0f;
            #pragma unroll  // IGC 2.41.5 does not unroll this; s[128] spills
            for (int i = 0; i < S_v; i += 4) {
                const sycl::float4 kk = *reinterpret_cast<const sycl::float4 *>(k_t + i);
                const sycl::float4 qq = *reinterpret_cast<const sycl::float4 *>(q_t + i);
                s[i]     = g_val * s[i]     + kk.x() * delta;
                s[i + 1] = g_val * s[i + 1] + kk.y() * delta;
                s[i + 2] = g_val * s[i + 2] + kk.z() * delta;
                s[i + 3] = g_val * s[i + 3] + kk.w() * delta;
                attn += s[i] * qq.x() + s[i + 1] * qq.y() + s[i + 2] * qq.z() + s[i + 3] * qq.w();
            }
            attn_data[col] = attn * scale;
            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride + col * S_v;
#pragma unroll
                    for (int i = 0; i < S_v; i++) {
                        slot[i] = s[i];
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[col * S_v + i] = s[i];
        }
    }
}

template <bool keep_rs_t, int TILE>
static void gated_delta_net_tcol128_hf(const float *     q,
                                       const float *     k,
                                       const float *     v,
                                       const float *     g,
                                       const float *     beta,
                                       const float *     curr_state,
                                       float *           dst,
                                       float *           state,
                                       int64_t           H,
                                       int64_t           n_tokens,
                                       int64_t           sq1,
                                       int64_t           sq2,
                                       int64_t           sq3,
                                       int64_t           sv1,
                                       int64_t           sv2,
                                       int64_t           sv3,
                                       int64_t           sb1,
                                       int64_t           sb2,
                                       int64_t           sb3,
                                       const sycl::uint3 neqk1_magic,
                                       const sycl::uint3 rq3_magic,
                                       float             scale,
                                       int64_t           state_slot_stride,
                                       int               K,
                                       const int32_t *   state_rows,
                                       int64_t           state_row_stride,
                                       gdn_pro_args      pro,
                                       sycl::half *      qslm,
                                       sycl::half *      kslm,
                                       float *           gslm,
                                       float *           bslm) {
    constexpr int S_v = 128;
    using h4          = sycl::vec<sycl::half, 4>;
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      tid      = (int) item_ct1.get_local_id(2);
    const int      nthr     = (int) item_ct1.get_local_range(2);
    const int      col      = (int) item_ct1.get_group(0) * nthr + tid;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t packed_stride    = H * S_v * S_v;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    sycl::half s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = sycl::half(curr_state[i]);
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += TILE) {
        const int ntile = (int) sycl::min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);
        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const sycl::float4 qf = *reinterpret_cast<const sycl::float4 *>(q_t + i);
            const sycl::float4 kf = *reinterpret_cast<const sycl::float4 *>(k_t + i);
            *reinterpret_cast<h4 *>(qslm + tt * S_v + i) =
                h4(sycl::half(qf.x()), sycl::half(qf.y()), sycl::half(qf.z()), sycl::half(qf.w()));
            *reinterpret_cast<h4 *>(kslm + tt * S_v + i) =
                h4(sycl::half(kf.x()), sycl::half(kf.y()), sycl::half(kf.z()), sycl::half(kf.w()));
        }
        if (tid < ntile) {
            const int     t  = t0 + tid;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            float          g_in = g[gb];
            if (pro.apply_gate) {
                const float al = pro.alpha[sequence * pro.sa3 + t * pro.sa2 + h_idx * pro.sa1];
                g_in           = gdn_softplus_f32(al + pro.dt[h_idx]) * pro.a[h_idx];
            }
            gslm[tid] = g_in;
            bslm[tid] = pro.apply_sigmoid ? gdn_sigmoid_f32(beta[gb]) : beta[gb];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int tt = 0; tt < ntile; tt++) {
            const sycl::half * q_t      = qslm + tt * S_v;
            const sycl::half * k_t      = kslm + tt * S_v;
            const int          t        = t0 + tt;
            const float *      v_t      = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const float        g_val    = sycl::native::exp(gslm[tt]);
            const float        beta_val = bslm[tt];
            const sycl::half   gh       = sycl::half(g_val);

            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 4) {
                const h4 kk = *reinterpret_cast<const h4 *>(k_t + i);
                kv += float(s[i] * kk.x()) + float(s[i + 1] * kk.y()) + float(s[i + 2] * kk.z()) +
                      float(s[i + 3] * kk.w());
            }
            const sycl::half dh = sycl::half((v_t[col] - g_val * kv) * beta_val);

            float attn = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 4) {
                const h4 kk = *reinterpret_cast<const h4 *>(k_t + i);
                const h4 qq = *reinterpret_cast<const h4 *>(q_t + i);
                s[i]     = gh * s[i]     + kk.x() * dh;
                s[i + 1] = gh * s[i + 1] + kk.y() * dh;
                s[i + 2] = gh * s[i + 2] + kk.z() * dh;
                s[i + 3] = gh * s[i + 3] + kk.w() * dh;
                attn += float(s[i] * qq.x()) + float(s[i + 1] * qq.y()) + float(s[i + 2] * qq.z()) +
                        float(s[i + 3] * qq.w());
            }
            attn_data[col] = attn * scale;
            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride + col * S_v;
#pragma unroll
                    for (int i = 0; i < S_v; i++) {
                        slot[i] = float(s[i]);
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[col * S_v + i] = float(s[i]);
        }
    }
}

// Prefill: cooperative SLM tile of q/k/v/g/beta, then the same serial recurrence.
// Decode n=1 SLM k/q is KERNEL_CUTS. This path is T>=tile only. GGML_SYCL_GDN_PREFILL_TILE=16|32.
template <bool keep_rs_t>
static void gated_delta_net_sycl_tile128(const float *     q,
                                         const float *     k,
                                         const float *     v,
                                         const float *     g,
                                         const float *     beta,
                                         const float *     curr_state,
                                         float *           dst,
                                         float *           state,
                                         int64_t           H,
                                         int64_t           n_tokens,
                                         int64_t           sq1,
                                         int64_t           sq2,
                                         int64_t           sq3,
                                         int64_t           sv1,
                                         int64_t           sv2,
                                         int64_t           sv3,
                                         int64_t           sb1,
                                         int64_t           sb2,
                                         int64_t           sb3,
                                         const sycl::uint3 neqk1_magic,
                                         const sycl::uint3 rq3_magic,
                                         float             scale,
                                         int64_t           state_slot_stride,
                                         int               K,
                                         const int32_t *   state_rows,
                                         int64_t           state_row_stride,
                                         int               tile,
                                         float *           qslm,
                                         float *           kslm,
                                         float *           vslm,
                                         float *           gslm,
                                         float *           bslm) {
    constexpr int S_v = 128;
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    const int      lane     = item_ct1.get_local_id(2);
    const int      col      = item_ct1.get_group(0) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    const int      tid      = (int) item_ct1.get_local_id(1) * (int) item_ct1.get_local_range(2) + lane;
    const int      nthr     = (int) item_ct1.get_local_range(1) * (int) item_ct1.get_local_range(2);

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;
    const int64_t packed_stride    = H * S_v * S_v;
    const int64_t seq_row          = state_rows ? (int64_t) state_rows[sequence] : sequence;
    const int64_t seq_stride       = state_rows ? state_row_stride : packed_stride;
    const int64_t state_in_offset  = seq_row * seq_stride + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size     = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t0 = 0; t0 < n_tokens; t0 += tile) {
        const int ntile = (int) sycl::min((int64_t) tile, n_tokens - t0);
        for (int e = tid; e < ntile * S_v; e += nthr) {
            const int tt = e / S_v;
            const int i  = e % S_v;
            const int t  = t0 + tt;
            qslm[e] = (q + iq3 * sq3 + t * sq2 + iq1 * sq1)[i];
            kslm[e] = (k + iq3 * sq3 + t * sq2 + iq1 * sq1)[i];
            vslm[e] = (v + sequence * sv3 + t * sv2 + h_idx * sv1)[i];
        }
        for (int tt = tid; tt < ntile; tt += nthr) {
            const int t = t0 + tt;
            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            gslm[tt] = g[gb_offset];
            bslm[tt] = beta[gb_offset];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t      = qslm + tt * S_v;
            const float * k_t      = kslm + tt * S_v;
            const float   g_val    = sycl::native::exp(gslm[tt]);
            const float   beta_val = bslm[tt];
            const float   v_col    = vslm[tt * S_v + col];

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += s_shard[r] * k_t[i];
            }
            const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
            const float delta_col = (v_col - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = g_val * s_shard[r] + k_t[i] * delta_col;
                attn_partial += s_shard[r] * q_t[i];
            }
            const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int t           = t0 + tt;
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        slot[col * S_v + i] = s_shard[r];
                    }
                }
            }
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool keep_rs_t>
static void launch_gated_delta_net_tile128(const float *   q_d,
                                           const float *   k_d,
                                           const float *   v_d,
                                           const float *   g_d,
                                           const float *   b_d,
                                           const float *   s_d,
                                           float *         dst_d,
                                           float *         state_d,
                                           int64_t         H,
                                           int64_t         n_tokens,
                                           int64_t         n_seqs,
                                           int64_t         sq1,
                                           int64_t         sq2,
                                           int64_t         sq3,
                                           int64_t         sv1,
                                           int64_t         sv2,
                                           int64_t         sv3,
                                           int64_t         sb1,
                                           int64_t         sb2,
                                           int64_t         sb3,
                                           int64_t         neqk1,
                                           int64_t         rq3,
                                           float           scale,
                                           int64_t         state_slot_stride,
                                           int             K,
                                           const int32_t * state_rows,
                                           int64_t         state_row_stride,
                                           int             tile,
                                           dpct::queue_ptr stream) {
    constexpr int S_v       = 128;
    const int     warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
    const int     num_warps = 4;
    dpct::dim3    grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dpct::dim3    block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);
    const int         slm_vec     = tile * S_v;
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> qslm(sycl::range<1>(slm_vec), cgh);
        sycl::local_accessor<float, 1> kslm(sycl::range<1>(slm_vec), cgh);
        sycl::local_accessor<float, 1> vslm(sycl::range<1>(slm_vec), cgh);
        sycl::local_accessor<float, 1> gslm(sycl::range<1>(tile), cgh);
        sycl::local_accessor<float, 1> bslm(sycl::range<1>(tile), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gated_delta_net_sycl_tile128<keep_rs_t>(
                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2,
                                 sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_rows,
                                 state_row_stride, tile, qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 vslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   const int32_t * state_rows,
                                   int64_t         state_row_stride,
                                   gdn_pro_args    pro,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    static const int fla = ggml_sycl_get_env("GGML_SYCL_GDN_FLA", 0);
    if (fla && !KDA && S_v == 128 && n_tokens >= 64 && !pro.apply_sigmoid && !pro.apply_gate) {
        if (g_gdn_ctx && gdn_fla_try_launch<keep_rs_t>(*g_gdn_ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                                                       n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3,
                                                       neqk1, rq3, scale, state_slot_stride, K, state_rows,
                                                       state_row_stride, stream)) {
            return;
        }
    }
    static const int chunk = ggml_sycl_get_env("GGML_SYCL_GDN_CHUNK", 0);
    static const int chunk_par = ggml_sycl_get_env("GGML_SYCL_GDN_CHUNK_PAR", 0);
    const bool chunk32 = (!KDA && S_v == 128 && n_tokens >= 32 && !pro.apply_sigmoid && !pro.apply_gate) &&
                         ((chunk == 32 && !keep_rs_t) ||
                          (chunk_par == 32 && (!keep_rs_t || n_tokens >= 1024)));
    if (chunk32) {
        const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
        const int num_warps = 4;
        constexpr int C = 32;
        constexpr int D = 128;
        const int nch = (int) ((n_tokens + C - 1) / C);
        const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
        const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);
        const size_t need = (size_t) nch * (size_t) H * (size_t) n_seqs * (size_t) (C * C);
        static float * Tbuf = nullptr;
        static float * Kqbuf = nullptr;
        static size_t  cap  = 0;
        if (need > cap) {
            if (Tbuf) {
                stream->wait();
                sycl::free(Tbuf, *stream);
                sycl::free(Kqbuf, *stream);
            }
            Tbuf  = sycl::malloc_device<float>(need, *stream);
            Kqbuf = sycl::malloc_device<float>(need, *stream);
            cap   = need;
        }
        float * Tptr  = Tbuf;
        float * Kqptr = Kqbuf;
        {
            dpct::dim3 grid_t(H, n_seqs, nch);
            dpct::dim3 block_t(128, 1, 1);
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> kslm(sycl::range<1>(C * D), cgh);
                sycl::local_accessor<float, 1> qslm(sycl::range<1>(C * D), cgh);
                sycl::local_accessor<float, 1> gslm(sycl::range<1>(C), cgh);
                sycl::local_accessor<float, 1> bslm(sycl::range<1>(C), cgh);
                sycl::local_accessor<float, 1> aslm(sycl::range<1>(C * C), cgh);
                cgh.parallel_for(sycl::nd_range<3>(grid_t * block_t, block_t),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     gdn_chunk32_build_T(k_d, q_d, g_d, b_d, Tptr, Kqptr, H, n_tokens, n_seqs, nch, sq1,
                                                         sq2, sq3, sb1, sb2, sb3, neqk1_magic, rq3_magic,
                                                         kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         bslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                                         aslm.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        {
            dpct::dim3 grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
            dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> kslm(sycl::range<1>(C * D), cgh);
                sycl::local_accessor<float, 1> qslm(sycl::range<1>(C * D), cgh);
                sycl::local_accessor<float, 1> vslm(sycl::range<1>(C * D), cgh);
                sycl::local_accessor<float, 1> gslm(sycl::range<1>(C), cgh);
                sycl::local_accessor<float, 1> bslm(sycl::range<1>(C), cgh);
                sycl::local_accessor<float, 1> Tslm(sycl::range<1>(C * C), cgh);
                sycl::local_accessor<float, 1> Kqslm(sycl::range<1>(C * C), cgh);
                cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                 [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                     gated_delta_net_chunk32_cols(
                                         q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, Tptr, Kqptr, H, n_tokens, nch,
                                         sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale,
                                         kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         vslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         bslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         Tslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                         Kqslm.get_multi_ptr<sycl::access::decorated::no>().get());
                                 });
            });
        }
        return;
    }
    static const int tile = ggml_sycl_get_env("GGML_SYCL_GDN_PREFILL_TILE", 0);
    if (!KDA && S_v == 128 && tile >= 16 && tile <= 32 && n_tokens >= tile && !pro.apply_sigmoid && !pro.apply_gate) {
        launch_gated_delta_net_tile128<keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs,
                                                  sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale,
                                                  state_slot_stride, K, state_rows, state_row_stride, tile, stream);
        return;
    }
    const int gdn_tcol = ggml_sycl_get_env("GGML_SYCL_GDN_TCOL", 1);
    const int gdn_tcole = ggml_sycl_get_env("GGML_SYCL_GDN_TCOLE", 0);
    if constexpr (!KDA) {
#ifdef GGML_SYCL_GDN_HAS_ESIMD
        if (gdn_tcole && S_v == 128 && n_tokens >= 16) {
            constexpr int WG = 64;
            const size_t nwg = (size_t) H * (size_t) n_seqs;
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nwg * WG), sycl::range<1>(WG)),
                sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                    gated_delta_net_tcol_esimd<keep_rs_t>(
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3, sb1,
                        sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, pro, it);
                });
            return;
        }
#endif
        if (gdn_tcol && S_v == 128 && n_tokens >= 16 && g_gdn_apply_l2) {
            constexpr int TILE = 32;
            constexpr int Svt  = 128;
            constexpr int NTHR = 128;
            dpct::dim3    grid_dims(H, n_seqs, (Svt + NTHR - 1) / NTHR);
            dpct::dim3    block_dims(NTHR, 1, 1);
            const sycl::uint3 neqk1_magic_t = init_fastdiv_values(neqk1);
            const sycl::uint3 rq3_magic_t   = init_fastdiv_values(rq3);
            const float l2eps = g_gdn_l2eps;
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                    [=](sycl::nd_item<3> /*item_ct1*/) {
                        gated_delta_net_tcol128_l2<keep_rs_t, TILE>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                            state_row_stride, pro, l2eps,
                            qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
            return;
        }
        if (gdn_tcol && S_v == 128 && n_tokens >= 16 &&
            ggml_sycl_get_env("GGML_SYCL_GDN_ACC2", 0)) {
            constexpr int TILE = 32;
            constexpr int Svt  = 128;
            constexpr int NTHR = 128;
            dpct::dim3    grid_dims(H, n_seqs, (Svt + NTHR - 1) / NTHR);
            dpct::dim3    block_dims(NTHR, 1, 1);
            const sycl::uint3 neqk1_magic_t = init_fastdiv_values(neqk1);
            const sycl::uint3 rq3_magic_t   = init_fastdiv_values(rq3);
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                    [=](sycl::nd_item<3> /*item_ct1*/) {
                        gated_delta_net_tcol128_acc2<keep_rs_t, TILE>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                            state_row_stride, pro,
                            qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
            return;
        }
        if (gdn_tcol && S_v == 128 && n_tokens >= 16 &&
            ggml_sycl_get_env("GGML_SYCL_GDN_HF", 0)) {
            constexpr int TILE = 32;
            constexpr int Svt  = 128;
            constexpr int NTHR = 128;
            dpct::dim3    grid_dims(H, n_seqs, (Svt + NTHR - 1) / NTHR);
            dpct::dim3    block_dims(NTHR, 1, 1);
            const sycl::uint3 neqk1_magic_t = init_fastdiv_values(neqk1);
            const sycl::uint3 rq3_magic_t   = init_fastdiv_values(rq3);
            stream->submit([&](sycl::handler & cgh) {
                sycl::local_accessor<sycl::half, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<sycl::half, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                sycl::local_accessor<float, 1>      gslm(sycl::range<1>(TILE), cgh);
                sycl::local_accessor<float, 1>      bslm(sycl::range<1>(TILE), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<128> },
                    [=](sycl::nd_item<3> /*item_ct1*/) {
                        gated_delta_net_tcol128_hf<keep_rs_t, TILE>(
                            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                            sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                            state_row_stride, pro,
                            qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                            bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            });
            return;
        }
        if (gdn_tcol && S_v == 128 && n_tokens >= 16) {
            constexpr int TILE = 32;
            constexpr int Svt  = 128;
            const int nthr_t = ggml_sycl_get_env("GGML_SYCL_GDN_TCOLN", 128);
            const int NTHR = (nthr_t == 64) ? 64 : 128;
            const int tcolgrf = ggml_sycl_get_env("GGML_SYCL_GDN_TCOLGRF", 256);
            const int tcolsg = ggml_sycl_get_env("GGML_SYCL_GDN_TCOLSG", 0);
            dpct::dim3    grid_dims(H, n_seqs, (Svt + NTHR - 1) / NTHR);
            dpct::dim3    block_dims(NTHR, 1, 1);
            const sycl::uint3 neqk1_magic_t = init_fastdiv_values(neqk1);
            const sycl::uint3 rq3_magic_t   = init_fastdiv_values(rq3);
            if (tcolgrf == 256 && tcolsg == 16) {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                    sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(16)]] {
                            gated_delta_net_tcol128<keep_rs_t, TILE>(
                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                                state_row_stride, pro, qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            } else if (tcolgrf == 256 && tcolsg == 32) {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                    sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(32)]] {
                            gated_delta_net_tcol128<keep_rs_t, TILE>(
                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                                state_row_stride, pro, qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            } else if (tcolgrf == 256) {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                    sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                        [=](sycl::nd_item<3> /*item_ct1*/) {
                            gated_delta_net_tcol128<keep_rs_t, TILE>(
                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                                state_row_stride, pro, qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            } else {
                stream->submit([&](sycl::handler & cgh) {
                    sycl::local_accessor<float, 1> qslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> kslm(sycl::range<1>(TILE * Svt), cgh);
                    sycl::local_accessor<float, 1> gslm(sycl::range<1>(TILE), cgh);
                    sycl::local_accessor<float, 1> bslm(sycl::range<1>(TILE), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        [=](sycl::nd_item<3> /*item_ct1*/) {
                            gated_delta_net_tcol128<keep_rs_t, TILE>(
                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                sb1, sb2, sb3, neqk1_magic_t, rq3_magic_t, scale, state_slot_stride, K, state_rows,
                                state_row_stride, pro, qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                bslm.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
            }
            return;
        }
    }
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
    const int gdn_pipe = ggml_sycl_get_env("GGML_SYCL_GDN_PIPE", 0);
    int gdn_col2 = 0;
    int gdn_col4 = 0;
    if constexpr (!KDA) {
        if (S_v == 128 && n_tokens >= 16) {
            const auto arch = ggml_sycl_info().devices[ggml_sycl_get_device()].hw_info.arch;
            const int col2_def = ggml_sycl_is_xe2_dgpu(arch) ? 1 : 0;
            gdn_col2 = ggml_sycl_get_env("GGML_SYCL_GDN_COL2", col2_def);
            gdn_col4 = ggml_sycl_get_env("GGML_SYCL_GDN_COL4", 0);
            if (gdn_col4) {
                gdn_col2 = 0;
            }
        }
    }
    const int gdn_kqreg = (gdn_col2 && ggml_sycl_get_env("GGML_SYCL_GDN_KQREG", 0)) ? 1 : 0;
    const int gdn_esimd = (gdn_col2 && ggml_sycl_get_env("GGML_SYCL_GDN_ESIMD", 0)) ? 1 : 0;
    const int gdn_esimd_warp = (gdn_col2 && ggml_sycl_get_env("GGML_SYCL_GDN_ESIMD_WARP", 0)) ? 1 : 0;
    const int gdn_xmx = (gdn_col2 && ggml_sycl_get_env("GGML_SYCL_GDN_XMX", 0)) ? 1 : 0;
    const int gdn_dp4a = (gdn_col2 && ggml_sycl_get_env("GGML_SYCL_GDN_DP4A", 0)) ? 1 : 0;
    int gdn_ilp = 0;
    if (gdn_col2 && S_v == 128 && (H % 2) == 0) {
        gdn_ilp = ggml_sycl_get_env("GGML_SYCL_GDN_ILP", 0);
    }
    if (gdn_dp4a && S_v == 128) {
        constexpr int WARP = 16;
        constexpr int NW   = 4;
        const int cols_per_wg = NW * 2;
        dpct::dim3 grid_dims(H, n_seqs, (S_v + cols_per_wg - 1) / cols_per_wg);
        dpct::dim3 block_dims(WARP, NW, 1);
        const sycl::uint3 neqk1_magic_d = init_fastdiv_values(neqk1);
        const sycl::uint3 rq3_magic_d   = init_fastdiv_values(rq3);
        stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 gated_delta_net_col2_dp4a<keep_rs_t>(
                                     q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1,
                                     sv2, sv3, sb1, sb2, sb3, scale, state_slot_stride, K, state_rows, state_row_stride,
                                     pro, neqk1_magic_d, rq3_magic_d);
                             });
        return;
    }
    if (gdn_xmx && S_v == 128) {
        constexpr int WARP = 16;
        constexpr int NW   = 4;
        const int cols_per_wg = NW * 2;
        dpct::dim3 grid_dims(H, n_seqs, (S_v + cols_per_wg - 1) / cols_per_wg);
        dpct::dim3 block_dims(WARP, NW, 1);
        const sycl::uint3 neqk1_magic_x = init_fastdiv_values(neqk1);
        const sycl::uint3 rq3_magic_x   = init_fastdiv_values(rq3);
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<sycl::half, 1> aslm(sycl::range<1>(8 * 8 * 16), cgh);
            sycl::local_accessor<sycl::half, 1> bslm(sycl::range<1>(2 * 16 * 16), cgh);
            sycl::local_accessor<float, 1>      cslm(sycl::range<1>(8 * 8 * 16), cgh);
            cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                             [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 gated_delta_net_col2_xmx<keep_rs_t>(
                                     q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1,
                                     sv2, sv3, sb1, sb2, sb3, scale, state_slot_stride, K, state_rows, state_row_stride,
                                     pro, neqk1_magic_x, rq3_magic_x,
                                     aslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     bslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                     cslm.get_multi_ptr<sycl::access::decorated::no>().get());
                             });
        });
        return;
    }
#ifdef GGML_SYCL_GDN_HAS_ESIMD
    if (gdn_esimd_warp && S_v == 128) {
        constexpr int WARP = 16;
        constexpr int NW   = 4;
        constexpr int NCG  = 128 / (NW * 2);
        const size_t nwg   = (size_t) NCG * (size_t) n_seqs * (size_t) H;
        const size_t wg    = (size_t) WARP * NW;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nwg * wg), sycl::range<1>(wg)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                gated_delta_net_col2_esimd_warp<keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                           n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale,
                                                           state_slot_stride, K, state_rows, state_row_stride, pro, it);
            });
        return;
    }
    if (gdn_esimd && S_v == 128) {
        constexpr int WGS = 16;
        const int wgs_h   = 128 / (2 * WGS);
        const size_t nwg  = (size_t) wgs_h * (size_t) H * (size_t) n_seqs;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nwg * WGS), sycl::range<1>(WGS)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                gated_delta_net_col2_esimd<keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1,
                                                      sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale, state_slot_stride,
                                                      K, state_rows, state_row_stride, pro, it);
            });
        return;
    }
    if (gdn_ilp && S_v == 128) {
        constexpr int WGS = 16;
        constexpr int NH  = 2;
        const int wgs_h  = 128 / (2 * WGS);
        const size_t nwg = (size_t) wgs_h * (size_t) (H / NH) * (size_t) n_seqs;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nwg * WGS), sycl::range<1>(WGS)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                gated_delta_net_col2_esimd_ilp<keep_rs_t, NH>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                               sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale,
                                                               state_slot_stride, K, state_rows, state_row_stride, pro, it);
            });
        return;
    }
#endif

    int num_warps = 4;
    if (n_tokens >= 256) {
        const int pw = ggml_sycl_get_env("GGML_SYCL_GDN_PREFILL_WARPS", 0);
        if (pw == 8 || pw == 16) {
            num_warps = pw;
        }
    }
    const int cols_per_wg = num_warps * (gdn_col4 ? 4 : (gdn_col2 ? 2 : 1));
    dpct::dim3 grid_dims(H, n_seqs, (S_v + cols_per_wg - 1) / cols_per_wg);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);
    const int gdn_pipe_use = gdn_col4 ? 0 : gdn_pipe;

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                                                       state_rows, state_row_stride, pro, gdn_pipe_use, gdn_col2, gdn_col4);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, KDA, keep_rs_t>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens,
                                                                       sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2,
                                                                       sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                                                       state_rows, state_row_stride, pro, gdn_pipe_use, gdn_col2, gdn_col4);
                                     });
            }
            break;
        case 64: {
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                                state_rows, state_row_stride, pro, gdn_pipe_use, gdn_col2, gdn_col4);
                                        });
            }
            break;
        }
        case 128: {
            {
                constexpr int sv = 128;
                const int grf = ggml_sycl_get_env("GGML_SYCL_GDN_GRF", 0);
                if (gdn_kqreg) {
                    stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                            [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                                gated_delta_net_sycl<sv, KDA, keep_rs_t, true>(
                                                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                    sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                                    state_rows, state_row_stride, pro, 0, gdn_col2, gdn_col4);
                                            });
                } else if (grf == 256) {
                    stream->parallel_for(
                        sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                        sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
                        [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                            gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                state_rows, state_row_stride, pro, gdn_pipe_use, gdn_col2, gdn_col4);
                        });
                } else {
                    stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                            [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                                gated_delta_net_sycl<sv, KDA, keep_rs_t>(
                                                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2,
                                                    sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                                    state_rows, state_row_stride, pro, gdn_pipe_use, gdn_col2, gdn_col4);
                                            });
                }
            }
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    int   apply_l2 = 0;
    float l2eps    = 1.0e-6f;
    const int want_l2 = ggml_sycl_get_env("GGML_SYCL_GDN_L2", 0);
    if (want_l2 &&
        ggml_sycl_get_env("GGML_SYCL_GDN_TCOL", 1) &&
        src_v && src_v->ne[0] == 128 && src_v->ne[2] >= 16) {
        auto peel = [](ggml_tensor * t) -> ggml_tensor * {
            while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
                         t->op == GGML_OP_TRANSPOSE)) {
                t = t->view_src ? t->view_src : t->src[0];
            }
            return t;
        };
        ggml_tensor * uq = peel(src_q);
        ggml_tensor * uk = peel(src_k);
        if (uq && uq->op == GGML_OP_L2_NORM && uq->src[0] && uq->src[0]->type == GGML_TYPE_F32 &&
            uk && uk->op == GGML_OP_L2_NORM && uk->src[0] && uk->src[0]->type == GGML_TYPE_F32 &&
            ggml_is_contiguous_rows(uq->src[0]) && ggml_is_contiguous_rows(uk->src[0])) {
            memcpy(&l2eps, uq->op_params, sizeof(float));
            src_q     = uq->src[0];
            src_k     = uk->src[0];
            apply_l2  = want_l2 ? 1 : 0;
        }
    }

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;
    const int32_t * state_rows      = nullptr;
    int64_t         state_row_stride = 0;

    // Read the recurrent cache directly when src_state is a GET_ROWS gather
    // of a packed F32 row. Skips the 3 MiB/layer copy on Qwen3.5 decode.
    {
        const ggml_tensor * st = src_state;
        while (st && (st->op == GGML_OP_RESHAPE || st->op == GGML_OP_VIEW ||
                      st->op == GGML_OP_PERMUTE || st->op == GGML_OP_TRANSPOSE)) {
            st = st->view_src ? st->view_src : st->src[0];
        }
        if (st && st->op == GGML_OP_GET_ROWS && st->src[0] && st->src[1] &&
            st->src[0]->type == GGML_TYPE_F32 && st->src[1]->type == GGML_TYPE_I32 &&
            st->src[0]->ne[0] == S_v * S_v * H && st->src[0]->nb[0] == sizeof(float) &&
            (st->src[0]->nb[1] % sizeof(float)) == 0) {
            s_d              = (const float *) st->src[0]->data;
            state_rows       = (const int32_t *) st->src[1]->data;
            state_row_stride = (int64_t) (st->src[0]->nb[1] / sizeof(float));
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    if (state_rows == nullptr) {
        GGML_ASSERT(ggml_is_contiguous(src_state));
    }

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    gdn_pro_args pro{};
    g_gdn_apply_l2 = apply_l2;
    g_gdn_l2eps    = l2eps;
    static const int gdn_pro_mask = ggml_sycl_get_env("GGML_SYCL_FUSE_GDN_PRO", 0);
    auto unwrap = [](const ggml_tensor * t) -> const ggml_tensor * {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE ||
                     t->op == GGML_OP_TRANSPOSE)) {
            t = t->view_src ? t->view_src : t->src[0];
        }
        return t;
    };
    if (!kda && gdn_pro_mask && (n_tokens <= 4 || n_tokens >= 16)) {
        const ggml_tensor * b = unwrap(src_beta);
        if ((gdn_pro_mask & 1) && b && b->op == GGML_OP_UNARY && ggml_get_unary_op(b) == GGML_UNARY_OP_SIGMOID &&
            b->src[0] && b->src[0]->type == GGML_TYPE_F32 && ggml_are_same_shape(b, b->src[0])) {
            b_d                = (const float *) b->src[0]->data;
            pro.apply_sigmoid  = 1;
        }
        const ggml_tensor * gnode = unwrap(src_g);
        if ((gdn_pro_mask & 2) && gnode && gnode->op == GGML_OP_MUL && gnode->src[0] && gnode->src[1]) {
            const ggml_tensor * sp = gnode->src[0];
            const ggml_tensor * w  = gnode->src[1];
            if (sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) {
                sp = gnode->src[1];
                w  = gnode->src[0];
            }
            const ggml_tensor * add = (sp && sp->op == GGML_OP_UNARY) ? unwrap(sp->src[0]) : nullptr;
            if (add && add->op == GGML_OP_ADD && add->src[0] && add->src[1] && w && w->type == GGML_TYPE_F32 &&
                ggml_nelements(w) == H && ggml_nelements(add->src[1]) == H && add->src[0]->type == GGML_TYPE_F32) {
                const ggml_tensor * al = add->src[0];
                pro.apply_gate = 1;
                pro.alpha      = (const float *) al->data;
                pro.dt         = (const float *) add->src[1]->data;
                pro.a          = (const float *) w->data;
                pro.sa1        = (int64_t) (al->nb[0] / sizeof(float));
                pro.sa2        = (int64_t) (al->nb[1] / sizeof(float));
                pro.sa3        = (int64_t) (al->nb[2] / sizeof(float));
            }
        }
    }

    dpct::queue_ptr stream = ctx.stream();
    g_gdn_ctx = &ctx;

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, pro, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, pro, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, pro, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_rows, state_row_stride, pro, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache);
}
