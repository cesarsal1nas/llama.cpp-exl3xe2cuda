#pragma once

// FLA/WY GDN prefill, BT=16. GGML_SYCL_GDN_FLA=1 (default 0).
// oneDNN batched GEMM for intra-chunk KK^T. New TU — not gdn_chunk32_* C=32 T-build.
// https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/wy_fast.py
// https://github.com/ggml-org/llama.cpp/pull/26001

#if GGML_SYCL_DNNL
#include "gemm.hpp"
#endif

static inline float gdn_fla_exp_clamp(float x) {
    return sycl::native::exp(sycl::fmin(sycl::fmax(x, -80.f), 80.f));
}

static void gdn_fla_pack_k(const float * k, float * kpack, int64_t H, int64_t n_tokens, int64_t n_seqs, int nch,
                           int64_t sq1, int64_t sq2, int64_t sq3, const sycl::uint3 neqk1_magic,
                           const sycl::uint3 rq3_magic) {
    constexpr int BT = 16;
    constexpr int D  = 128;
    auto          it  = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const size_t  gid = it.get_global_id(0);
    const size_t  n   = (size_t) n_seqs * (size_t) H * (size_t) nch * (size_t) BT * (size_t) D;
    if (gid >= n) {
        return;
    }
    const int d    = (int) (gid % (size_t) D);
    size_t    rem  = gid / (size_t) D;
    const int tloc = (int) (rem % (size_t) BT);
    rem /= (size_t) BT;
    const int ich = (int) (rem % (size_t) nch);
    rem /= (size_t) nch;
    const int h   = (int) (rem % (size_t) H);
    const int seq = (int) (rem / (size_t) H);
    const int t   = ich * BT + tloc;
    float     v   = 0.f;
    if (t < (int) n_tokens) {
        const uint32_t iq1 = fastmodulo((uint32_t) h, neqk1_magic);
        const uint32_t iq3 = fastdiv((uint32_t) seq, rq3_magic);
        v                  = (k + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1)[d];
    }
    kpack[gid] = v;
}

// One WG per (seq,head,chunk). KK^T from oneDNN, scale+tril invert, write T[BT,BT].
static void gdn_fla_solve_T(const float * akk, const float * g, const float * beta, float * Tglob, int64_t H,
                            int64_t n_tokens, int nch, int64_t sb1, int64_t sb2, int64_t sb3, float * aslm,
                            float * gslm, float * bslm, float * tslm) {
    constexpr int BT = 16;
    auto          it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const int     b  = (int) it.get_group(0);
    const int     tid = (int) it.get_local_id(0);
    const int     ich = b % nch;
    int           rem = b / nch;
    const int     h   = rem % (int) H;
    const int     seq = rem / (int) H;
    const int     t0  = ich * BT;
    const int     nC  = (int) sycl::min((int64_t) BT, n_tokens - (int64_t) t0);

    const float * Ak = akk + (size_t) b * (size_t) (BT * BT);
    for (int p = tid; p < BT * BT; p += 32) {
        aslm[p] = Ak[p];
    }
    if (tid < BT) {
        if (tid < nC) {
            const int64_t gb = (int64_t) seq * sb3 + (int64_t) (t0 + tid) * sb2 + (int64_t) h * sb1;
            gslm[tid]        = g[gb];
            bslm[tid]        = beta[gb];
        } else {
            gslm[tid] = 0.f;
            bslm[tid] = 0.f;
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < BT; ++i) {
            acc += gslm[i];
            gslm[i] = acc;
        }
        for (int i = 0; i < BT; ++i) {
            for (int j = 0; j < BT; ++j) {
                if (i <= j) {
                    aslm[i * BT + j] = 0.f;
                } else {
                    aslm[i * BT + j] *= bslm[j] * gdn_fla_exp_clamp(gslm[j] - gslm[i]);
                }
            }
        }
        for (int i = 0; i < BT; ++i) {
            for (int j = 0; j < BT; ++j) {
                tslm[i * BT + j] = (i == j) ? 1.f : 0.f;
            }
        }
        for (int i = 1; i < BT; ++i) {
            for (int j = 0; j < i; ++j) {
                float acc2 = aslm[i * BT + j];
                for (int kk = j + 1; kk < i; ++kk) {
                    acc2 += aslm[i * BT + kk] * tslm[kk * BT + j];
                }
                tslm[i * BT + j] = -acc2;
            }
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    float * T = Tglob + (size_t) b * (size_t) (BT * BT);
    for (int p = tid; p < BT * BT; p += 32) {
        T[p] = tslm[p];
    }
}

template <bool keep_rs_t>
static void gdn_fla_scan(const float * q, const float * k, const float * v, const float * g, const float * beta,
                         const float * curr_state, float * dst, float * state, const float * Tglob, int64_t H,
                         int64_t n_tokens, int nch, int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2,
                         int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, const sycl::uint3 neqk1_magic,
                         const sycl::uint3 rq3_magic, float scale, int64_t state_slot_stride, int Kst,
                         const int32_t * state_rows, int64_t state_row_stride, float * kslm, float * qslm, float * vslm,
                         float * gslm, float * bslm, float * Tslm) {
    constexpr int BT = 16;
    constexpr int D  = 128;
    auto          it = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h    = it.get_group(2);
    const uint32_t seq  = it.get_group(1);
    const int      lane = (int) it.get_local_id(2);
    const int      col  = (int) it.get_group(0) * (int) it.get_local_range(1) + (int) it.get_local_id(1);
    const int      tid  = (int) it.get_local_id(1) * (int) it.get_local_range(2) + lane;
    const int      nthr = (int) it.get_local_range(1) * (int) it.get_local_range(2);
    if (col >= D) {
        return;
    }
    constexpr int warp_size     = 16;
    constexpr int rows_per_lane = D / warp_size;
    const uint32_t iq1          = fastmodulo(h, neqk1_magic);
    const uint32_t iq3          = fastdiv(seq, rq3_magic);
    const int64_t packed        = H * D * D;
    const int64_t seq_row       = state_rows ? (int64_t) state_rows[seq] : (int64_t) seq;
    const int64_t seq_stride    = state_rows ? state_row_stride : packed;
    const float * state_in      = curr_state + seq_row * seq_stride + (int64_t) h * D * D + (int64_t) col * D;
    float *       attn_data     = dst + ((int64_t) seq * n_tokens * H + (int64_t) h) * D;
    float *       state_out     = state + ((int64_t) seq * H + (int64_t) h) * D * D;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in[r * warp_size + lane];
    }

    for (int ich = 0; ich < nch; ++ich) {
        const int t0 = ich * BT;
        const int nC = (int) sycl::min((int64_t) BT, n_tokens - (int64_t) t0);
        for (int e = tid; e < BT * D; e += nthr) {
            const int tt = e / D;
            const int i  = e % D;
            if (tt >= nC) {
                kslm[e] = 0.f;
                qslm[e] = 0.f;
                vslm[e] = 0.f;
                continue;
            }
            const int t = t0 + tt;
            qslm[e]     = (q + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1)[i];
            kslm[e]     = (k + (int64_t) iq3 * sq3 + (int64_t) t * sq2 + (int64_t) iq1 * sq1)[i];
            vslm[e]     = (v + (int64_t) seq * sv3 + (int64_t) t * sv2 + (int64_t) h * sv1)[i];
        }
        for (int tt = tid; tt < BT; tt += nthr) {
            if (tt >= nC) {
                gslm[tt] = 0.f;
                bslm[tt] = 0.f;
                continue;
            }
            const int64_t gb = (int64_t) seq * sb3 + (int64_t) (t0 + tt) * sb2 + (int64_t) h * sb1;
            gslm[tt]         = g[gb];
            bslm[tt]         = beta[gb];
        }
        const int     bidx = ((int) seq * (int) H + (int) h) * nch + ich;
        const float * Tg   = Tglob + (size_t) bidx * (size_t) (BT * BT);
        for (int e = tid; e < BT * BT; e += nthr) {
            Tslm[e] = Tg[e];
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (tid == 0) {
            float acc = 0.f;
            for (int i = 0; i < BT; ++i) {
                acc += gslm[i];
                gslm[i] = acc;
            }
        }
        it.barrier(sycl::access::fence_space::local_space);

        float kS[BT];
        for (int j = 0; j < BT; ++j) {
            float         sh = 0.f;
            const float * kj = kslm + j * D;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                sh += s_shard[r] * kj[r * warp_size + lane];
            }
            kS[j] = bslm[j] * gdn_fla_exp_clamp(gslm[j]) * warp_reduce_sum<warp_size>(sh);
        }

        float Utilde[BT];
        for (int i = 0; i < BT; ++i) {
            float U = 0.f;
            float P = 0.f;
            for (int j = 0; j < BT; ++j) {
                const float Tji = Tslm[j * BT + i];
                U += vslm[j * D + col] * bslm[j] * Tji;
                P += Tji * kS[j];
            }
            Utilde[i] = U - P;
        }

        for (int i = 0; i < nC; ++i) {
            float         inter = 0.f;
            const float * qi    = qslm + i * D;
            const float   gi    = gdn_fla_exp_clamp(gslm[i]);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                inter += s_shard[r] * qi[r * warp_size + lane];
            }
            inter = gi * warp_reduce_sum<warp_size>(inter);
            float intra = 0.f;
            for (int j = 0; j <= i; ++j) {
                float         dot = 0.f;
                const float * kj  = kslm + j * D;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    dot += kj[r * warp_size + lane] * qi[r * warp_size + lane];
                }
                dot = warp_reduce_sum<warp_size>(dot);
                intra += dot * gdn_fla_exp_clamp(gslm[j] - gslm[i]) * Utilde[j];
            }
            if (lane == 0) {
                attn_data[(t0 + i) * (int) H * D + col] = (inter + intra) * scale;
            }
            if constexpr (keep_rs_t) {
                // per-token state only on last chunk (Kst snapshots of the tail)
            }
        }

        const float Glast = gdn_fla_exp_clamp(gslm[nC - 1]);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= Glast;
            const int key = r * warp_size + lane;
            for (int i = 0; i < nC; ++i) {
                s_shard[r] += kslm[i * D + key] * Utilde[i] * gdn_fla_exp_clamp(gslm[nC - 1] - gslm[i]);
            }
        }
        if constexpr (keep_rs_t) {
            if (ich == nch - 1) {
                for (int s = 0; s < Kst; ++s) {
                    float * st = state_out + (int64_t) s * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        st[col * D + r * warp_size + lane] = s_shard[r];
                    }
                }
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out[col * D + r * warp_size + lane] = s_shard[r];
    }
}

template <bool keep_rs_t>
static bool gdn_fla_try_launch(ggml_backend_sycl_context & ctx, const float * q_d, const float * k_d, const float * v_d,
                               const float * g_d, const float * b_d, const float * s_d, float * dst_d, float * state_d,
                               int64_t H, int64_t n_tokens, int64_t n_seqs, int64_t sq1, int64_t sq2, int64_t sq3,
                               int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3,
                               int64_t neqk1, int64_t rq3, float scale, int64_t state_slot_stride, int K,
                               const int32_t * state_rows, int64_t state_row_stride, dpct::queue_ptr stream) {
#if !GGML_SYCL_DNNL
    (void) ctx;
    (void) q_d;
    (void) k_d;
    (void) v_d;
    (void) g_d;
    (void) b_d;
    (void) s_d;
    (void) dst_d;
    (void) state_d;
    (void) H;
    (void) n_tokens;
    (void) n_seqs;
    (void) sq1;
    (void) sq2;
    (void) sq3;
    (void) sv1;
    (void) sv2;
    (void) sv3;
    (void) sb1;
    (void) sb2;
    (void) sb3;
    (void) neqk1;
    (void) rq3;
    (void) scale;
    (void) state_slot_stride;
    (void) K;
    (void) state_rows;
    (void) state_row_stride;
    (void) stream;
    return false;
#else
    constexpr int BT = 16;
    constexpr int D  = 128;
    if (n_tokens < 64 || n_seqs < 1) {
        return false;
    }
    const int              nch         = (int) ((n_tokens + BT - 1) / BT);
    const int64_t          batch       = n_seqs * H * (int64_t) nch;
    const sycl::uint3      neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3      rq3_magic   = init_fastdiv_values(rq3);
    static float *         kpack       = nullptr;
    static float *         akk         = nullptr;
    static float *         Tbuf        = nullptr;
    static size_t          cap_k       = 0;
    static size_t          cap_a       = 0;
    const size_t           need_k      = (size_t) batch * (size_t) BT * (size_t) D;
    const size_t           need_a      = (size_t) batch * (size_t) (BT * BT);
    if (need_k > cap_k) {
        if (kpack) {
            stream->wait();
            sycl::free(kpack, *stream);
        }
        kpack = sycl::malloc_device<float>(need_k, *stream);
        cap_k = need_k;
    }
    if (need_a > cap_a) {
        if (akk) {
            stream->wait();
            sycl::free(akk, *stream);
            sycl::free(Tbuf, *stream);
        }
        akk   = sycl::malloc_device<float>(need_a, *stream);
        Tbuf  = sycl::malloc_device<float>(need_a, *stream);
        cap_a = need_a;
    }
    float * kpack_p = kpack;
    float * akk_p   = akk;
    float * Tptr    = Tbuf;
    const size_t pack_n = need_k;
    const size_t wg     = 256;
    const size_t ng     = ((pack_n + wg - 1) / wg) * wg;
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(ng), sycl::range<1>(wg)),
                         [=](sycl::nd_item<1> /*it*/) {
                             gdn_fla_pack_k(k_d, kpack_p, H, n_tokens, n_seqs, nch, sq1, sq2, sq3, neqk1_magic,
                                            rq3_magic);
                         });

    using dt = DnnlGemmWrapper::dt;
    // C[b,i,j] = K[b,i,:] · K[b,j,:]  => A[m=BT,k=D] @ B[k=D,n=BT] with B = K^T
    // B = K^T view: [batch, k=D, n=BT], B[d,i] = K[i,d] at i*D+d → strb0(k)=1, strb1(n)=D
    DnnlGemmWrapper::gemm(ctx, BT, BT, D, kpack_p, dt::f32, /*stra0*/ 1, /*stra1*/ D, /*stra2*/ BT * D, kpack_p, dt::f32,
                          /*strb0*/ 1, /*strb1*/ D, /*strb2*/ BT * D, akk_p, dt::f32, stream, (dnnl_dim_t) batch,
                          (dnnl_dim_t) batch);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> aslm(sycl::range<1>(BT * BT), cgh);
        sycl::local_accessor<float, 1> gslm(sycl::range<1>(BT), cgh);
        sycl::local_accessor<float, 1> bslm(sycl::range<1>(BT), cgh);
        sycl::local_accessor<float, 1> tslm(sycl::range<1>(BT * BT), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) batch * 32), sycl::range<1>(32)),
                         [=](sycl::nd_item<1> /*it*/) {
                             gdn_fla_solve_T(akk_p, g_d, b_d, Tptr, H, n_tokens, nch, sb1, sb2, sb3,
                                             aslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                             gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                             bslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                             tslm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });

    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;
    const int num_warps = 4;
    dpct::dim3 grid_dims((unsigned) H, (unsigned) n_seqs, (unsigned) ((D + num_warps - 1) / num_warps));
    dpct::dim3 block_dims((unsigned) (warp_size <= D ? warp_size : D), (unsigned) num_warps, 1);
    // nd_range<3> in this tree is (z,y,x) ~ (H, n_seqs, col_groups) matching COL2/chunk32
    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> kslm(sycl::range<1>(BT * D), cgh);
        sycl::local_accessor<float, 1> qslm(sycl::range<1>(BT * D), cgh);
        sycl::local_accessor<float, 1> vslm(sycl::range<1>(BT * D), cgh);
        sycl::local_accessor<float, 1> gslm(sycl::range<1>(BT), cgh);
        sycl::local_accessor<float, 1> bslm(sycl::range<1>(BT), cgh);
        sycl::local_accessor<float, 1> Tslm(sycl::range<1>(BT * BT), cgh);
        cgh.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                         [=](sycl::nd_item<3> /*it*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             gdn_fla_scan<keep_rs_t>(
                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, Tptr, H, n_tokens, nch, sq1, sq2, sq3,
                                 sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K,
                                 state_rows, state_row_stride,
                                 kslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 qslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 vslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 gslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 bslm.get_multi_ptr<sycl::access::decorated::no>().get(),
                                 Tslm.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
    });
    return true;
#endif
}
