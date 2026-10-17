#include "ssm_scan.hpp"
#include "common.hpp"
#include "gemm.hpp"
// Chunked SSD scan. Matches ssm_scan_f32_group layout. Strides are host ints; never touch a
// ggml_tensor inside a kernel. GGML_SYCL_SSM_SSD=1 (2 = probe), K==1, one sequence, d_state==128.

bool ggml_sycl_ssm_ssd(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const float * pre_dt, const float * pre_da) {
    const int mode = ggml_sycl_get_env("GGML_SYCL_SSM_SSD", 0);
    if (!pre_dt || !pre_da || (mode != 1 && mode != 2)) return false;

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];
    const ggml_tensor * src6 = dst->src[6];

    const int64_t d_state  = src0->ne[0];
    const int64_t head_dim = src0->ne[1];
    const int64_t n_head   = src1->ne[1];
    const int64_t n_group  = src4->ne[1];
    const int64_t n_tok    = src1->ne[2];
    const int64_t n_seq    = src1->ne[3];
    const int64_t K        = ggml_get_op_params_i32(dst, 0);

    constexpr int Q = 64;
    if (K < 1 || K > Q || n_seq != 1 || d_state != 128 || head_dim <= 0 || head_dim % 8 != 0) return false;
    if (n_group <= 0 || n_head % n_group != 0 || n_tok < Q) return false;
    if (mode == 2) return false;

#if GGML_SYCL_DNNL
    const int x_nb2  = (int) src1->nb[2];
    const int b_nb2  = (int) src4->nb[2];
    const int c_nb2  = (int) src5->nb[2];
    const int s0_nb2 = (int) src0->nb[2];
    const int s0_nb3 = (int) src0->nb[3];
    const int a_nb1  = (int) src3->nb[1];

    const int nh = (int) n_head;
    const int hd = (int) head_dim;
    const int ds = (int) d_state;
    const int ng = (int) n_group;
    const int nt = (int) n_tok;
    const int d_inner = nh * hd;
    const int heads_per_group = nh / ng;
    const int64_t s_off_bytes = ggml_nelements(src1) * (int64_t) sizeof(float);

    const char * x_base = (const char *) src1->data;
    const char * a_base = (const char *) src3->data;
    const char * b_base = (const char *) src4->data;
    const char * c_base = (const char *) src5->data;
    const char * s0_raw = (const char *) src0->data;
    const int32_t * slot_ptr = (const int32_t *) src6->data;
    float * y_base = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    const int n_chunks = (nt + Q - 1) / Q;
    constexpr int max_super = 32;
    const int sc_cap = std::min(n_chunks, max_super);
    const int k_slots = (int) K;
    const int c_snap = (k_slots > 1) ? ((nt - k_slots) / Q) : -1;

    ggml_sycl_pool_alloc<float> cs_b(ctx.pool());
    ggml_sycl_pool_alloc<float> exp_pos_b(ctx.pool());
    ggml_sycl_pool_alloc<float> exp_neg_b(ctx.pool());
    ggml_sycl_pool_alloc<float> hstate_b(ctx.pool());
    ggml_sycl_pool_alloc<float> hsnap_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> b_pack_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> c_pack_b(ctx.pool());
    ggml_sycl_pool_alloc<float> cb_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> m_mask_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> x_dt_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> b_w_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> c_sc_b(ctx.pool());
    ggml_sycl_pool_alloc<float> y_intra_b(ctx.pool());
    ggml_sycl_pool_alloc<float> y_prior_b(ctx.pool());
    ggml_sycl_pool_alloc<float> dstate_b(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> s_prev_b(ctx.pool());

    const size_t cs_elems = (size_t) n_chunks * (size_t) nh * Q;
    cs_b.alloc(cs_elems);
    exp_pos_b.alloc(cs_elems);
    exp_neg_b.alloc(cs_elems);
    hstate_b.alloc((size_t) nh * (size_t) hd * (size_t) ds);
    if (k_slots > 1) {
        hsnap_b.alloc((size_t) nh * (size_t) hd * (size_t) ds);
    }
    b_pack_b.alloc((size_t) sc_cap * (size_t) ng * Q * (size_t) ds);
    c_pack_b.alloc((size_t) sc_cap * (size_t) ng * Q * (size_t) ds);
    cb_b.alloc((size_t) sc_cap * (size_t) ng * Q * Q);
    m_mask_b.alloc((size_t) sc_cap * (size_t) nh * Q * Q);
    x_dt_b.alloc((size_t) sc_cap * (size_t) nh * Q * (size_t) hd);
    b_w_b.alloc((size_t) sc_cap * (size_t) nh * Q * (size_t) ds);
    c_sc_b.alloc((size_t) sc_cap * (size_t) nh * Q * (size_t) ds);
    y_intra_b.alloc((size_t) sc_cap * (size_t) nh * Q * (size_t) hd);
    y_prior_b.alloc((size_t) sc_cap * (size_t) nh * Q * (size_t) hd);
    dstate_b.alloc((size_t) sc_cap * (size_t) nh * (size_t) hd * (size_t) ds);
    s_prev_b.alloc((size_t) sc_cap * (size_t) nh * (size_t) hd * (size_t) ds);

    float * cs          = cs_b.get();
    float * exp_pos     = exp_pos_b.get();
    float * exp_neg     = exp_neg_b.get();
    float * hs          = hstate_b.get();
    float * hsnap       = (k_slots > 1) ? hsnap_b.get() : nullptr;
    sycl::half * b_pack = b_pack_b.get();
    sycl::half * c_pack = c_pack_b.get();
    float * cb          = cb_b.get();
    sycl::half * m_mask = m_mask_b.get();
    sycl::half * x_dt   = x_dt_b.get();
    sycl::half * b_w    = b_w_b.get();
    sycl::half * c_sc   = c_sc_b.get();
    float * y_intra     = y_intra_b.get();
    float * y_prior     = y_prior_b.get();
    float * dstate      = dstate_b.get();
    sycl::half * s_prev = s_prev_b.get();

    // 1. Chunk-local inclusive prefix sum of softplus(dt) + precomputed exponentials
    // Layout of cs, exp_pos, exp_neg: [n_chunks, nh, Q] so lane accesses are contiguous.
    stream->parallel_for(sycl::range<1>((size_t) n_chunks * (size_t) nh), [=](sycl::id<1> id) {
        const int idx = (int) id[0];
        const int h = idx % nh;
        const int c = idx / nh;
        const int t0 = c * Q;
        const int qlen = std::min(Q, nt - t0);
        const float a_val = *(const float *) (a_base + (size_t) h * (size_t) a_nb1);
        const size_t base_off = (size_t) idx * Q;
        float run = 0.0f;
        float rbuf[Q];
#pragma unroll
        for (int lane = 0; lane < Q; ++lane) {
            const int tok = t0 + lane;
            if (tok < nt) {
                run += pre_dt[(size_t) tok * (size_t) nh + (size_t) h];
            }
            rbuf[lane] = run;
            cs[base_off + (size_t) lane] = run;
            exp_pos[base_off + (size_t) lane] = sycl::exp(a_val * run);
        }
        const float cs_last = (qlen > 0) ? rbuf[qlen - 1] : 0.0f;
#pragma unroll
        for (int lane = 0; lane < Q; ++lane) {
            exp_neg[base_off + (size_t) lane] = sycl::exp(a_val * (cs_last - rbuf[lane]));
        }
    });

    // 2. Copy initial state S0 [nh, hd, ds] into contiguous hs (reading slot_ptr on device)
    stream->parallel_for(sycl::range<1>((size_t) nh * (size_t) hd * (size_t) ds), [=](sycl::id<1> id) {
        const int idx = (int) id[0];
        const int s = idx % ds;
        const int rem = idx / ds;
        const int d = rem % hd;
        const int h = rem / hd;
        const int32_t slot = slot_ptr[0];
        const char * s0_base = s0_raw + (size_t) slot * (size_t) s0_nb3;
        const float * src = (const float *) (s0_base + (size_t) h * (size_t) s0_nb2) + (size_t) d * (size_t) ds;
        hs[idx] = src[s];
    });

    for (int c0 = 0; c0 < n_chunks; c0 += sc_cap) {
        const int sc = std::min(sc_cap, n_chunks - c0);
        const int batch_g = sc * ng;
        const int batch_h = sc * nh;

        // Pack B and C for this super-batch of chunks [sc * ng, Q, ds]
        stream->parallel_for(sycl::range<1>((size_t) batch_g * Q * (size_t) ds), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int s = idx % ds;
            const int rem1 = idx / ds;
            const int lane = rem1 % Q;
            const int rem2 = rem1 / Q;
            const int g = rem2 % ng;
            const int cl = rem2 / ng;
            const int tok = (c0 + cl) * Q + lane;
            if (tok < nt) {
                const float * bsrc = (const float *) (b_base + (size_t) tok * (size_t) b_nb2) + (size_t) g * (size_t) ds;
                const float * csrc = (const float *) (c_base + (size_t) tok * (size_t) c_nb2) + (size_t) g * (size_t) ds;
                b_pack[idx] = (sycl::half) bsrc[s];
                c_pack[idx] = (sycl::half) csrc[s];
            } else {
                b_pack[idx] = (sycl::half) 0.0f;
                c_pack[idx] = (sycl::half) 0.0f;
            }
        });

        // Batched GEMM: cb[cl * ng + g, tout, tin] = sum_s B[tin, s] * C[tout, s]
        DnnlGemmWrapper::gemm(ctx, Q, Q, ds,
            b_pack, DnnlGemmWrapper::to_dt<sycl::half>(), 1, ds, Q * ds,
            c_pack, DnnlGemmWrapper::to_dt<sycl::half>(), 1, ds, Q * ds,
            cb, DnnlGemmWrapper::to_dt<float>(), stream, batch_g, batch_g);

        // Materialize causal mask M [batch_h, Q, Q]
        static const int mask_v = ggml_sycl_get_env("GGML_SYCL_SSM_SSD_PACK3D", 0);
        if (mask_v == 2 && (Q % 8) == 0) {
            // 8 consecutive tin per item, float cb loads, half8 store
            stream->parallel_for(sycl::range<1>((size_t) batch_h * Q * Q / 8), [=](sycl::id<1> id) {
                const int idx = (int) id[0] * 8;
                const int tin0 = idx % Q;
                const int rem1 = idx / Q;
                const int tout = rem1 % Q;
                const int bh = rem1 / Q;
                const int h = bh % nh;
                const int cl = bh / nh;
                const int chunk = c0 + cl;
                const int t0 = chunk * Q;
                const int qlen = sycl::min(Q, nt - t0);
                sycl::vec<sycl::half, 8> o(0.0f);
                if (tout < qlen && tin0 <= tout) {
                    const int g = h / heads_per_group;
                    const float a_val = *(const float *) (a_base + (size_t) h * (size_t) a_nb1);
                    const size_t cs_base = ((size_t) chunk * (size_t) nh + (size_t) h) * Q;
                    const float cs_out = cs[cs_base + (size_t) tout];
                    const size_t cbi = ((size_t) (cl * ng + g) * Q + (size_t) tout) * Q + (size_t) tin0;
                    for (int j = 0; j < 8; ++j) {
                        const int tin = tin0 + j;
                        if (tin <= tout) {
                            o[j] = (sycl::half) (sycl::exp(a_val * (cs_out - cs[cs_base + (size_t) tin])) * cb[cbi + j]);
                        }
                    }
                }
                *(sycl::vec<sycl::half, 8> *) (m_mask + idx) = o;
            });
        } else {
        stream->parallel_for(sycl::range<1>((size_t) batch_h * Q * Q), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int tin = idx % Q;
            const int rem1 = idx / Q;
            const int tout = rem1 % Q;
            const int bh = rem1 / Q;
            const int h = bh % nh;
            const int cl = bh / nh;
            const int chunk = c0 + cl;
            const int t0 = chunk * Q;
            const int qlen = std::min(Q, nt - t0);
            float val = 0.0f;
            if (tout < qlen && tin <= tout) {
                const int g = h / heads_per_group;
                const float a_val = *(const float *) (a_base + (size_t) h * (size_t) a_nb1);
                const size_t cs_base = ((size_t) chunk * (size_t) nh + (size_t) h) * Q;
                const float cs_out = cs[cs_base + (size_t) tout];
                const float cs_in  = cs[cs_base + (size_t) tin];
                const float cb_val = cb[((size_t) (cl * ng + g) * Q + (size_t) tout) * Q + (size_t) tin];
                val = sycl::exp(a_val * (cs_out - cs_in)) * cb_val;
            }
            m_mask[idx] = (sycl::half) val;
        });
        }

        // Fused Pack X_dt [batch_h, Q, hd] + B_weighted [batch_h, Q, ds] + C_scaled [batch_h, Q, ds]
        // Uses precomputed exp_neg and exp_pos (zero sycl::exp evaluations in this kernel!)
        static const int pack_v = ggml_sycl_get_env("GGML_SYCL_SSM_SSD_PACK3D", 0);
        if (pack_v == 2 && (ds % 8) == 0 && (hd % 8) == 0) {
            // 8 consecutive s per work-item, 16-byte half stores
            const int nv = ds / 8;
            stream->parallel_for(sycl::range<2>((size_t) batch_h * Q, (size_t) nv), [=](sycl::id<2> id) {
                const int rowi = (int) id[0];
                const int s0 = (int) id[1] * 8;
                const int bh = rowi / Q;
                const int lane = rowi % Q;
                const int h = bh % nh;
                const int cl = bh / nh;
                const int chunk = c0 + cl;
                const int t0 = chunk * Q;
                const int tok = t0 + lane;
                const int qlen = sycl::min(Q, nt - t0);
                const size_t row = (size_t) rowi;
                if (s0 < hd) {
                    sycl::vec<sycl::half, 8> xo(0.0f);
                    if (tok < nt) {
                        const float * xp = (const float *) (x_base + (size_t) tok * (size_t) x_nb2) + (size_t) h * (size_t) hd + (size_t) s0;
                        const float dts = pre_dt[(size_t) tok * (size_t) nh + (size_t) h];
                        sycl::vec<float, 4> a = *(const sycl::vec<float, 4> *) xp;
                        sycl::vec<float, 4> b = *(const sycl::vec<float, 4> *) (xp + 4);
                        for (int j = 0; j < 4; ++j) { xo[j] = (sycl::half) (a[j] * dts); xo[4 + j] = (sycl::half) (b[j] * dts); }
                    }
                    *(sycl::vec<sycl::half, 8> *) (x_dt + row * (size_t) hd + (size_t) s0) = xo;
                }
                sycl::vec<sycl::half, 8> bo(0.0f), co(0.0f);
                if (lane < qlen) {
                    const int g = h / heads_per_group;
                    const size_t cs_idx = ((size_t) chunk * (size_t) nh + (size_t) h) * Q + (size_t) lane;
                    const size_t bc_idx = ((size_t) (cl * ng + g) * Q + (size_t) lane) * (size_t) ds + (size_t) s0;
                    const float en = exp_neg[cs_idx];
                    const float ep = exp_pos[cs_idx];
                    const sycl::vec<sycl::half, 8> bi = *(const sycl::vec<sycl::half, 8> *) (b_pack + bc_idx);
                    const sycl::vec<sycl::half, 8> ci = *(const sycl::vec<sycl::half, 8> *) (c_pack + bc_idx);
                    for (int j = 0; j < 8; ++j) { bo[j] = (sycl::half) ((float) bi[j] * en); co[j] = (sycl::half) ((float) ci[j] * ep); }
                }
                *(sycl::vec<sycl::half, 8> *) (b_w + row * (size_t) ds + (size_t) s0) = bo;
                *(sycl::vec<sycl::half, 8> *) (c_sc + row * (size_t) ds + (size_t) s0) = co;
            });
        } else {
        stream->parallel_for(sycl::range<1>((size_t) batch_h * Q * (size_t) ds), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int s = idx % ds;
            const int rem1 = idx / ds;
            const int lane = rem1 % Q;
            const int bh = rem1 / Q;
            const int h = bh % nh;
            const int cl = bh / nh;
            const int chunk = c0 + cl;
            const int t0 = chunk * Q;
            const int tok = t0 + lane;
            const int qlen = std::min(Q, nt - t0);

            if (s < hd) {
                float xval = 0.0f;
                if (tok < nt) {
                    const float xv = *((const float *) (x_base + (size_t) tok * (size_t) x_nb2) + (size_t) h * (size_t) hd + (size_t) s);
                    const float dt_sp = pre_dt[(size_t) tok * (size_t) nh + (size_t) h];
                    xval = xv * dt_sp;
                }
                x_dt[((size_t) bh * Q + (size_t) lane) * (size_t) hd + (size_t) s] = (sycl::half) xval;
            }

            float bw_val = 0.0f;
            float cs_val = 0.0f;
            if (lane < qlen) {
                const int g = h / heads_per_group;
                const size_t cs_idx = ((size_t) chunk * (size_t) nh + (size_t) h) * Q + (size_t) lane;
                const size_t bc_idx = ((size_t) (cl * ng + g) * Q + (size_t) lane) * (size_t) ds + (size_t) s;
                bw_val = (float) b_pack[bc_idx] * exp_neg[cs_idx];
                cs_val = (float) c_pack[bc_idx] * exp_pos[cs_idx];
            }
            b_w[idx]  = (sycl::half) bw_val;
            c_sc[idx] = (sycl::half) cs_val;
        });
        }

        // Intra-chunk Y across all (sc * nh) in one GEMM: y_intra[bh, tout, d] = sum_tin x_dt[bh, tin, d] * m_mask[bh, tout, tin]
        DnnlGemmWrapper::gemm(ctx, hd, Q, Q,
            x_dt, DnnlGemmWrapper::to_dt<sycl::half>(), hd, 1, Q * hd,
            m_mask, DnnlGemmWrapper::to_dt<sycl::half>(), 1, Q, Q * Q,
            y_intra, DnnlGemmWrapper::to_dt<float>(), stream, batch_h, batch_h);

        // Chunk state deltas across all (sc * nh) in one GEMM: dstate[bh, d, s] = sum_lane b_w[bh, lane, s] * x_dt[bh, lane, d]
        DnnlGemmWrapper::gemm(ctx, ds, hd, Q,
            b_w, DnnlGemmWrapper::to_dt<sycl::half>(), ds, 1, Q * ds,
            x_dt, DnnlGemmWrapper::to_dt<sycl::half>(), hd, 1, Q * hd,
            dstate, DnnlGemmWrapper::to_dt<float>(), stream, batch_h, batch_h);

        // Scan state across the sc chunks in registers, recording pre-chunk state s_prev[cl, h, d, s]
        if (pack_v == 2 && (ds % 4) == 0) {
            // 4 consecutive s per item: float4 dstate loads, half4 s_prev stores
            stream->parallel_for(sycl::range<1>((size_t) nh * (size_t) hd * (size_t) ds / 4), [=](sycl::id<1> id) {
                const int idx = (int) id[0] * 4;
                const int h = idx / (hd * ds);
                sycl::vec<float, 4> cur = *(const sycl::vec<float, 4> *) (hs + idx);
                const size_t stride_hds = (size_t) nh * (size_t) hd * (size_t) ds;
                for (int cl = 0; cl < sc; ++cl) {
                    const int chunk = c0 + cl;
                    if (hsnap && chunk == c_snap) {
                        *(sycl::vec<float, 4> *) (hsnap + idx) = cur;
                    }
                    const int t0 = chunk * Q;
                    const int qlen = sycl::min(Q, nt - t0);
                    const float decay = exp_pos[((size_t) chunk * (size_t) nh + (size_t) h) * Q + (size_t) (qlen - 1)];
                    const size_t off = (size_t) cl * stride_hds + (size_t) idx;
                    *(sycl::vec<sycl::half, 4> *) (s_prev + off) = cur.convert<sycl::half>();
                    cur = cur * decay + *(const sycl::vec<float, 4> *) (dstate + off);
                }
                *(sycl::vec<float, 4> *) (hs + idx) = cur;
            });
        } else {
        stream->parallel_for(sycl::range<1>((size_t) nh * (size_t) hd * (size_t) ds), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int h = idx / (hd * ds);
            float cur = hs[idx];
            const size_t stride_hds = (size_t) nh * (size_t) hd * (size_t) ds;
            for (int cl = 0; cl < sc; ++cl) {
                const int chunk = c0 + cl;
                if (hsnap && chunk == c_snap) {
                    hsnap[idx] = cur;
                }
                const int t0 = chunk * Q;
                const int qlen = std::min(Q, nt - t0);
                const float decay = exp_pos[((size_t) chunk * (size_t) nh + (size_t) h) * Q + (size_t) (qlen - 1)];
                const size_t off = (size_t) cl * stride_hds + (size_t) idx;
                s_prev[off] = (sycl::half) cur;
                cur = cur * decay + dstate[off];
            }
            hs[idx] = cur;
        });
        }

        // Prior-state Y across all (sc * nh) in one GEMM: y_prior[bh, tout, d] = sum_s s_prev[bh, d, s] * c_sc[bh, tout, s]
        DnnlGemmWrapper::gemm(ctx, hd, Q, ds,
            s_prev, DnnlGemmWrapper::to_dt<sycl::half>(), 1, ds, hd * ds,
            c_sc, DnnlGemmWrapper::to_dt<sycl::half>(), 1, ds, Q * ds,
            y_prior, DnnlGemmWrapper::to_dt<float>(), stream, batch_h, batch_h);

        // Combine y_prior + y_intra into output y [nt, nh, hd]
        const int tok_start = c0 * Q;
        const int tok_count = std::min(sc * Q, nt - tok_start);
        if (pack_v == 2 && (hd % 4) == 0) {
            stream->parallel_for(sycl::range<1>((size_t) tok_count * (size_t) nh * (size_t) hd / 4), [=](sycl::id<1> id) {
                const int idx = (int) id[0] * 4;
                const int d = idx % hd;
                const int rem = idx / hd;
                const int h = rem % nh;
                const int rel_tok = rem / nh;
                const int cl = rel_tok / Q;
                const int lane = rel_tok % Q;
                const int tok = tok_start + rel_tok;
                const size_t bh = (size_t) cl * (size_t) nh + (size_t) h;
                const size_t src_idx = (bh * Q + (size_t) lane) * (size_t) hd + (size_t) d;
                const sycl::vec<float, 4> a = *(const sycl::vec<float, 4> *) (y_prior + src_idx);
                const sycl::vec<float, 4> b = *(const sycl::vec<float, 4> *) (y_intra + src_idx);
                *(sycl::vec<float, 4> *) (y_base + (size_t) tok * (size_t) d_inner + (size_t) h * (size_t) hd + (size_t) d) = a + b;
            });
        } else {
        stream->parallel_for(sycl::range<1>((size_t) tok_count * (size_t) nh * (size_t) hd), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int d = idx % hd;
            const int rem = idx / hd;
            const int h = rem % nh;
            const int rel_tok = rem / nh;
            const int cl = rel_tok / Q;
            const int lane = rel_tok % Q;
            const int tok = tok_start + rel_tok;
            const size_t bh = (size_t) cl * (size_t) nh + (size_t) h;
            const size_t src_idx = (bh * Q + (size_t) lane) * (size_t) hd + (size_t) d;
            const float yp = y_prior[src_idx];
            y_base[(size_t) tok * (size_t) d_inner + (size_t) h * (size_t) hd + (size_t) d] =
                yp + y_intra[src_idx];
        });
        }
    }

    // 4. Write final state hs (and K > 1 rollback snapshots if enabled) back to dst state buffer (no host wait!)
    if (k_slots == 1) {
        stream->parallel_for(sycl::range<1>((size_t) nh * (size_t) hd * (size_t) ds), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int s = idx % ds;
            const int rem = idx / ds;
            const int d = rem % hd;
            const int h = rem / hd;
            float * dsts = (float *) ((char *) y_base + (size_t) s_off_bytes + (size_t) h * (size_t) s0_nb2) + (size_t) d * (size_t) ds;
            dsts[s] = hs[idx];
        });
    } else {
        const int tok_snap = c_snap * Q;
        stream->parallel_for(sycl::range<1>((size_t) nh * (size_t) hd * (size_t) ds), [=](sycl::id<1> id) {
            const int idx = (int) id[0];
            const int s = idx % ds;
            const int rem = idx / ds;
            const int d = rem % hd;
            const int h = rem / hd;
            const int g = h / heads_per_group;
            float st = hsnap[idx];
            for (int tok = tok_snap; tok < nt; ++tok) {
                const int64_t at = (int64_t) tok * (int64_t) nh + (int64_t) h;
                const float dt_sp = pre_dt[at];
                const float dA = pre_da[at];
                const float xv = *((const float *) (x_base + (size_t) tok * (size_t) x_nb2) + (size_t) h * (size_t) hd + (size_t) d);
                const float bv = *((const float *) (b_base + (size_t) tok * (size_t) b_nb2) + (size_t) g * (size_t) ds + (size_t) s);
                st = st * dA + bv * (xv * dt_sp);
                const int slot = nt - 1 - tok;
                if (slot > 0 && slot < k_slots) {
                    float * snap_dst = (float *) ((char *) y_base + (size_t) s_off_bytes + (size_t) slot * (size_t) s0_nb3 + (size_t) h * (size_t) s0_nb2) + (size_t) d * (size_t) ds;
                    snap_dst[s] = st;
                }
            }
            float * dsts = (float *) ((char *) y_base + (size_t) s_off_bytes + (size_t) h * (size_t) s0_nb2) + (size_t) d * (size_t) ds;
            dsts[s] = hs[idx];
        });
    }

    return true;
#else
    return false;
#endif
}
