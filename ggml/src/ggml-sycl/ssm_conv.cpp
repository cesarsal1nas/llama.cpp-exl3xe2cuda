#include "ssm_conv.hpp"
#include "concat.hpp"
#include "fusion.hpp"
#include "common.hpp"
#include "norm.hpp"

#include <cstdio>
#include <cstring>

using namespace sycl;

static inline float ssm_silu(float x) {
    return x / (1.0f + sycl::native::exp(-x));
}

static inline float ssm_dot4(float a0, float b0, float a1, float b1, float a2, float b2, float a3, float b3) {
    float s = sycl::fma(a0, b0, 0.0f);
    s = sycl::fma(a1, b1, s);
    s = sycl::fma(a2, b2, s);
    s = sycl::fma(a3, b3, s);
    return s;
}

static inline float4 ssm_load_split(
    const float * state,
    const float * x,
    int logical,
    int c,
    int n_st,
    int C,
    int n_t,
    int s) {
    if (logical < n_st) {
        const float * base = state + (static_cast<size_t>(s) * static_cast<size_t>(n_st) * static_cast<size_t>(C));
        return float4(
            base[logical + (c + 0) * n_st],
            base[logical + (c + 1) * n_st],
            base[logical + (c + 2) * n_st],
            base[logical + (c + 3) * n_st]);
    }
    const float * xp = x + (static_cast<size_t>(s) * static_cast<size_t>(C) * static_cast<size_t>(n_t)) +
                       (static_cast<size_t>(logical - n_st) * static_cast<size_t>(C)) + static_cast<size_t>(c);
    return *reinterpret_cast<const float4 *>(xp);
}

template <bool apply_silu>
static void kernel_ssm_conv_split(
    queue & q,
    const float * state,
    const float * x,
    const float * weights,
    float * dst,
    int n_st,
    int C,
    int n_t,
    int n_s) {
    constexpr int VEC = 4;
    int tok = ggml_sycl_get_env("GGML_SYCL_SSM_TOK", 0);
    if (tok != 32 && tok != 128) {
        tok = 64;
    }
    const int TOK     = tok;
    const int n_c4    = C / VEC;
    const int n_tb    = (n_t + TOK - 1) / TOK;
    const size_t total_work = static_cast<size_t>(n_c4) * static_cast<size_t>(n_tb) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    q.submit([&](handler & h) {
        h.parallel_for(
            nd_range<1>(range<1>(num_work_groups * work_group_size), range<1>(work_group_size)),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int c4 = static_cast<int>(idx % static_cast<size_t>(n_c4));
                const int tb = static_cast<int>((idx / static_cast<size_t>(n_c4)) % static_cast<size_t>(n_tb));
                const int s  = static_cast<int>(idx / (static_cast<size_t>(n_c4) * static_cast<size_t>(n_tb)));
                const int c  = c4 * VEC;
                const int t0 = tb * TOK;
                const int t1 = t0 + TOK < n_t ? t0 + TOK : n_t;

                const float4 wa = *reinterpret_cast<const float4 *>(weights + (c + 0) * 4);
                const float4 wb = *reinterpret_cast<const float4 *>(weights + (c + 1) * 4);
                const float4 wc = *reinterpret_cast<const float4 *>(weights + (c + 2) * 4);
                const float4 wd = *reinterpret_cast<const float4 *>(weights + (c + 3) * 4);
                const float4 w0 = float4(wa.x(), wb.x(), wc.x(), wd.x());
                const float4 w1 = float4(wa.y(), wb.y(), wc.y(), wd.y());
                const float4 w2 = float4(wa.z(), wb.z(), wc.z(), wd.z());
                const float4 w3 = float4(wa.w(), wb.w(), wc.w(), wd.w());

                float4 x0 = ssm_load_split(state, x, t0 + 0, c, n_st, C, n_t, s);
                float4 x1 = ssm_load_split(state, x, t0 + 1, c, n_st, C, n_t, s);
                float4 x2 = ssm_load_split(state, x, t0 + 2, c, n_st, C, n_t, s);

                float * ybase = dst + (static_cast<size_t>(s) * static_cast<size_t>(C) * static_cast<size_t>(n_t));

                for (int t = t0; t < t1; ++t) {
                    const float4 x3 = ssm_load_split(state, x, t + 3, c, n_st, C, n_t, s);
                    const float s0 = ssm_dot4(x0.x(), w0.x(), x1.x(), w1.x(), x2.x(), w2.x(), x3.x(), w3.x());
                    const float s1 = ssm_dot4(x0.y(), w0.y(), x1.y(), w1.y(), x2.y(), w2.y(), x3.y(), w3.y());
                    const float s2 = ssm_dot4(x0.z(), w0.z(), x1.z(), w1.z(), x2.z(), w2.z(), x3.z(), w3.z());
                    const float s3 = ssm_dot4(x0.w(), w0.w(), x1.w(), w1.w(), x2.w(), w2.w(), x3.w(), w3.w());
                    float4 sumf = float4(
                        apply_silu ? ssm_silu(s0) : s0,
                        apply_silu ? ssm_silu(s1) : s1,
                        apply_silu ? ssm_silu(s2) : s2,
                        apply_silu ? ssm_silu(s3) : s3);
                    *reinterpret_cast<float4 *>(ybase + static_cast<size_t>(t) * static_cast<size_t>(C) + c) = sumf;
                    x0 = x1;
                    x1 = x2;
                    x2 = x3;
                }
            });
    });
}

template <bool apply_silu>
static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(global_range, local_range),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                const int token   = static_cast<int>(idx % n_t);
                const int channel = static_cast<int>((idx / n_t) % d_inner);
                const int seq     = static_cast<int>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

                const float *s = src_data
                    + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
                    + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
                    + static_cast<size_t>(token);

                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

                float sumf = 0.0f;
                for (int i0 = 0; i0 < d_conv; ++i0) {
                    sumf += s[i0] * c[i0];
                }

                const size_t dst_idx =
                    static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
                    static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
                    static_cast<size_t>(channel);

                dst_data[dst_idx] = apply_silu ? ssm_silu(sumf) : sumf;
            }
        );
    });
}

inline void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];
    ggml_tensor * out  = silu_dst ? silu_dst : dst;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type  == GGML_TYPE_F32);

    try {
        queue *q = ctx.stream();

        if (ggml_sycl_ssm_conv_can_split(dst)) {
            const ggml_tensor * concat = src0;
            const ggml_tensor * st     = concat->src[0];
            const ggml_tensor * x      = concat->src[1];
            const int n_st = static_cast<int>(st->ne[0]);
            const int C    = static_cast<int>(st->ne[1]);
            const int n_t  = static_cast<int>(x->ne[0]);
            const int n_s  = static_cast<int>(st->ne[2]);
            GGML_ASSERT(ggml_is_contiguous(out));
            GGML_ASSERT(out->ne[0] == C && out->ne[1] == n_t && out->ne[2] == n_s);
            const float * state   = static_cast<const float *>(st->data);
            const float * x_data  = static_cast<const float *>(x->data);
            const float * weights = static_cast<const float *>(src1->data);
            float * dst_data      = static_cast<float *>(out->data);
            GGML_ASSERT(state && x_data && weights && dst_data);
            if (silu_dst) {
                kernel_ssm_conv_split<true>(*q, state, x_data, weights, dst_data, n_st, C, n_t, n_s);
            } else {
                kernel_ssm_conv_split<false>(*q, state, x_data, weights, dst_data, n_st, C, n_t, n_s);
            }
            return;
        }

        const int d_conv   = src1->ne[0];
        const int ncs      = src0->ne[0];
        const int d_inner  = src0->ne[1];
        const int n_t      = out->ne[1];
        const int n_s      = out->ne[2];

        GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
        GGML_ASSERT(src0->ne[1] == d_inner);
        GGML_ASSERT(src1->ne[1] == d_inner);

        GGML_ASSERT(out->ne[0] == d_inner);
        GGML_ASSERT(out->ne[1] == n_t);
        GGML_ASSERT(out->ne[2] == n_s);

        GGML_ASSERT(src0->nb[0] == sizeof(float));
        GGML_ASSERT(src1->nb[0] == sizeof(float));

        GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

        const int src_stride_inner = ncs;
        const int src_stride_seq   = ncs * d_inner;
        const int dst_stride_token = d_inner;
        const int dst_stride_seq   = d_inner * n_t;

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        float *dst_data       = static_cast<float *>(out->data);

        GGML_ASSERT(src_data && weights && dst_data);

        if (silu_dst) {
            kernel_ssm_conv<true>(
                *q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, dst_stride_token, dst_stride_seq);
        } else {
            kernel_ssm_conv<false>(
                *q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, dst_stride_token, dst_stride_seq);
        }

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2, silu_dst ? " : fused silu" : "");
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst);
}

static ggml_tensor * g_conv_concat_done_ssm[96];
static ggml_tensor * g_conv_concat_done_silu[96];
static ggml_tensor * g_conv_concat_done_cpy[192];
static int g_conv_concat_done_n = 0;
static int g_conv_concat_done_cpy_n = 0;

static void kernel_conv_state_from_split(
        queue & q,
        const float * state,
        const float * x,
        float * dest,
        int n_st,
        int C,
        int n_t,
        int n_s,
        int col0,
        int dest_seq) {
    const size_t nout = static_cast<size_t>(n_s) * static_cast<size_t>(C) * static_cast<size_t>(n_st);
    q.parallel_for(range<1>(nout), [=](id<1> id) {
        size_t idx = static_cast<size_t>(id[0]);
        const int i = static_cast<int>(idx % static_cast<size_t>(n_st));
        idx /= static_cast<size_t>(n_st);
        const int c = static_cast<int>(idx % static_cast<size_t>(C));
        const int s = static_cast<int>(idx / static_cast<size_t>(C));
        const int logical = col0 + i;
        float v;
        if (logical < n_st) {
            v = state[static_cast<size_t>(s) * static_cast<size_t>(n_st) * static_cast<size_t>(C) +
                      static_cast<size_t>(logical) + static_cast<size_t>(c) * static_cast<size_t>(n_st)];
        } else {
            const int t = logical - n_st;
            v = x[static_cast<size_t>(s) * static_cast<size_t>(C) * static_cast<size_t>(n_t) +
                  static_cast<size_t>(t) * static_cast<size_t>(C) + static_cast<size_t>(c)];
        }
        dest[static_cast<size_t>(s) * static_cast<size_t>(dest_seq) + static_cast<size_t>(i) +
             static_cast<size_t>(c) * static_cast<size_t>(n_st)] = v;
    });
}

bool ggml_sycl_try_conv_concat_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * concat = cgraph->nodes[node_idx];
    if (!ggml_sycl_is_gdn_conv_concat(concat)) {
        return false;
    }
    ggml_tensor * ssm  = nullptr;
    ggml_tensor * silu = nullptr;
    int ssm_idx = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_SSM_CONV && n->src[0] == concat) {
            ssm     = n;
            ssm_idx = j;
            break;
        }
    }
    if (!ssm || !ggml_sycl_ssm_conv_can_split(ssm)) {
        return false;
    }
    if (ssm_idx + 1 < cgraph->n_nodes &&
        ggml_sycl_can_fuse(cgraph, ssm_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        silu = cgraph->nodes[ssm_idx + 1];
    }
    const int nocat = ggml_sycl_get_env("GGML_SYCL_XE2MX_CONV_NOCAT", 0);
    ggml_tensor * cpy_hit[8];
    int cpy_col[8];
    int n_cpy = 0;
    bool cpy_ok = true;
    if (nocat) {
        const ggml_tensor * st = concat->src[0];
        const ggml_tensor * x  = concat->src[1];
        const int n_st = static_cast<int>(st->ne[0]);
        const int C    = static_cast<int>(st->ne[1]);
        const int n_t  = static_cast<int>(x->ne[0]);
        const int n_s  = static_cast<int>(st->ne[2]);
        const int ncs  = n_st + n_t;
        for (int j = node_idx + 1; j < cgraph->n_nodes && cpy_ok; ++j) {
            ggml_tensor * n = cgraph->nodes[j];
            if (!n || n->op != GGML_OP_CPY || !n->src[0] || !n->src[1]) {
                continue;
            }
            if (n->src[0]->view_src != concat || n->src[0]->op != GGML_OP_VIEW) {
                continue;
            }
            if (n->src[0]->type != GGML_TYPE_F32 || n->src[1]->type != GGML_TYPE_F32) {
                cpy_ok = false;
                break;
            }
            if (n->src[0]->ne[0] != n_st || n->src[0]->ne[1] != C || n->src[0]->ne[2] != n_s) {
                cpy_ok = false;
                break;
            }
            if ((n->src[0]->view_offs % sizeof(float)) != 0) {
                cpy_ok = false;
                break;
            }
            const int col0 = static_cast<int>(n->src[0]->view_offs / sizeof(float));
            if (col0 < 0 || col0 + n_st > ncs) {
                cpy_ok = false;
                break;
            }
            if (!n->src[1]->data || n->src[1]->nb[0] != sizeof(float) ||
                ggml_nelements(n->src[1]) != (int64_t) n_st * C * n_s) {
                cpy_ok = false;
                break;
            }
            const int dest_seq = n_s > 1 ? static_cast<int>(n->src[1]->nb[1] / sizeof(float)) : n_st * C;
            if (dest_seq < n_st * C) {
                cpy_ok = false;
                break;
            }
            if (n_cpy >= 8) {
                cpy_ok = false;
                break;
            }
            cpy_hit[n_cpy] = n;
            cpy_col[n_cpy] = col0;
            n_cpy++;
            (void) dest_seq;
        }
        if (!cpy_ok) {
            n_cpy = 0;
        }
    }
    if (!(nocat && cpy_ok)) {
        ggml_sycl_concat_conv_tail(ctx, concat);
    } else if (n_cpy > 0) {
        const ggml_tensor * st = concat->src[0];
        const ggml_tensor * x  = concat->src[1];
        const int n_st = static_cast<int>(st->ne[0]);
        const int C    = static_cast<int>(st->ne[1]);
        const int n_t  = static_cast<int>(x->ne[0]);
        const int n_s  = static_cast<int>(st->ne[2]);
        queue_ptr stream = ctx.stream();
        for (int k = 0; k < n_cpy; ++k) {
            const int dest_seq = n_s > 1 ? static_cast<int>(cpy_hit[k]->src[1]->nb[1] / sizeof(float)) : n_st * C;
            kernel_conv_state_from_split(
                *stream, static_cast<const float *>(st->data), static_cast<const float *>(x->data),
                static_cast<float *>(cpy_hit[k]->src[1]->data), n_st, C, n_t, n_s, cpy_col[k], dest_seq);
            if (g_conv_concat_done_cpy_n < 192) {
                g_conv_concat_done_cpy[g_conv_concat_done_cpy_n++] = cpy_hit[k];
            }
        }
    }
    ggml_sycl_ssm_conv(ctx, ssm, silu);
    if (silu && ggml_sycl_get_env("GGML_SYCL_FUSE_CONV_L2", 0) && ggml_is_contiguous(silu) &&
        silu->type == GGML_TYPE_F32) {
        ggml_tensor * l2q = nullptr;
        ggml_tensor * l2k = nullptr;
        for (int j = ssm_idx + 1; j < cgraph->n_nodes && j < ssm_idx + 48; ++j) {
            ggml_tensor * n = cgraph->nodes[j];
            if (!n || n->op != GGML_OP_L2_NORM || !n->src[0] || n->src[0]->ne[0] != 128) {
                continue;
            }
            const ggml_tensor * v = n->src[0];
            while (v && (v->op == GGML_OP_VIEW || v->op == GGML_OP_RESHAPE || v->op == GGML_OP_PERMUTE ||
                         v->op == GGML_OP_TRANSPOSE)) {
                if (v->view_src == silu) {
                    break;
                }
                v = v->view_src ? v->view_src : v->src[0];
            }
            if (!v || v->view_src != silu) {
                continue;
            }
            if (!l2q) {
                l2q = n;
            } else if (!l2k) {
                l2k = n;
                break;
            }
        }
        if (l2q && l2k && l2q->src[0]->ne[1] == l2k->src[0]->ne[1] &&
            l2q->src[0]->ne[0] == 128 && ggml_is_contiguous_rows(l2q->src[0]) &&
            ggml_is_contiguous_rows(l2k->src[0]) && l2q->data && l2k->data) {
            ggml_sycl_op_l2_norm(ctx, l2q);
            ggml_sycl_op_l2_norm(ctx, l2k);
            static int ncl2 = 0;
            if (ncl2 < 4) {
                ncl2++;
                GGML_SYCL_XE2_LOG( "xe2mx conv-l2 hit C=%ld nt=%ld heads=%ld\n",
                        (long) silu->ne[0], (long) silu->ne[1],
                        (long) (l2q->src[0]->ne[1] + l2k->src[0]->ne[1]));
                fflush(stderr);
            }
        }
    }
    if (g_conv_concat_done_n < 96) {
        g_conv_concat_done_ssm[g_conv_concat_done_n]  = ssm;
        g_conv_concat_done_silu[g_conv_concat_done_n] = silu;
        g_conv_concat_done_n++;
    }
    return true;
}

void ggml_sycl_conv_concat_fuse_begin(void) {
    g_conv_concat_done_n = 0;
    g_conv_concat_done_cpy_n = 0;
}

int ggml_sycl_conv_cpy_skip_done(const ggml_tensor * node) {
    for (int k = 0; k < g_conv_concat_done_cpy_n; ++k) {
        if (node != g_conv_concat_done_cpy[k]) {
            continue;
        }
        g_conv_concat_done_cpy[k] = nullptr;
        return 1;
    }
    return 0;
}

int ggml_sycl_ssm_conv_skip_done(ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * node = cgraph->nodes[node_idx];
    for (int k = 0; k < g_conv_concat_done_n; ++k) {
        if (node != g_conv_concat_done_ssm[k]) {
            continue;
        }
        int skip = 1;
        if (g_conv_concat_done_silu[k] && node_idx + 1 < cgraph->n_nodes &&
            cgraph->nodes[node_idx + 1] == g_conv_concat_done_silu[k]) {
            skip = 2;
        }
        g_conv_concat_done_ssm[k]  = nullptr;
        g_conv_concat_done_silu[k] = nullptr;
        return skip;
    }
    return 0;
}
