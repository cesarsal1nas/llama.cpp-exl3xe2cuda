#pragma once

// Fused GDN prefill chunk !KDA D=128 C=32. GGML_SYCL_GDN_CHUNK=32.
// Pass 1: one T=(I+A)^{-1} per (head,seq,chunk). Pass 2: 4-col apply.
// o uses S_start. Decode stays serial (n_tokens < 32).
// exp(gcs[j]-gcs[i]) overflows on Qwopus gates (HTTP 500). Clamp.

static inline float gdn_exp_clamp(float x) {
    return sycl::native::exp(sycl::fmin(sycl::fmax(x, -80.f), 80.f));
}

static void gdn_chunk32_build_T(const float *     k,
                                const float *     q,
                                const float *     g,
                                const float *     beta,
                                float *           Tglob,
                                float *           Kqglob,
                                int64_t           H,
                                int64_t           n_tokens,
                                int64_t           n_seqs,
                                int64_t           nch,
                                int64_t           sq1,
                                int64_t           sq2,
                                int64_t           sq3,
                                int64_t           sb1,
                                int64_t           sb2,
                                int64_t           sb3,
                                const sycl::uint3 neqk1_magic,
                                const sycl::uint3 rq3_magic,
                                float *           kslm,
                                float *           qslm,
                                float *           gslm,
                                float *           bslm,
                                float *           aslm) {
    constexpr int D = 128;
    constexpr int C = 32;
    auto           item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h    = item.get_group(2);
    const uint32_t seq  = item.get_group(1);
    const int      ich  = (int) item.get_group(0);
    const int      tid  = (int) item.get_local_id(2);
    const int      nthr = (int) item.get_local_range(2);
    const uint32_t iq1  = fastmodulo(h, neqk1_magic);
    const uint32_t iq3  = fastdiv(seq, rq3_magic);
    const int      t0   = ich * C;
    const int      nC   = (int) sycl::min((int64_t) C, n_tokens - t0);

    for (int e = tid; e < C * D; e += nthr) {
        const int tt = e / D;
        const int i  = e % D;
        if (tt >= nC) {
            kslm[e] = 0.f;
            qslm[e] = 0.f;
            continue;
        }
        kslm[e] = (k + iq3 * sq3 + (t0 + tt) * sq2 + iq1 * sq1)[i];
        qslm[e] = (q + iq3 * sq3 + (t0 + tt) * sq2 + iq1 * sq1)[i];
    }
    for (int tt = tid; tt < C; tt += nthr) {
        if (tt >= nC) {
            gslm[tt] = 0.f;
            bslm[tt] = 0.f;
            continue;
        }
        const int64_t gb = seq * sb3 + (t0 + tt) * sb2 + h * sb1;
        gslm[tt]         = g[gb];
        bslm[tt]         = beta[gb];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        float acc = 0.f;
        for (int i = 0; i < C; ++i) {
            acc += gslm[i];
            gslm[i] = acc;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int p = tid; p < C * C; p += nthr) {
        const int i = p / C;
        const int j = p % C;
        if (i <= j) {
            aslm[p] = 0.f;
            continue;
        }
        float         dot = 0.f;
        const float * ki  = kslm + i * D;
        const float * kj  = kslm + j * D;
        for (int d = 0; d < D; ++d) {
            dot += ki[d] * kj[d];
        }
        aslm[p] = bslm[j] * dot * gdn_exp_clamp(gslm[j] - gslm[i]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    float * T = Tglob + (((int64_t) seq * H + h) * nch + ich) * (C * C);
    if (tid == 0) {
        for (int i = 0; i < C; ++i) {
            for (int j = 0; j < C; ++j) {
                T[i * C + j] = (i == j) ? 1.f : 0.f;
            }
        }
        for (int i = 1; i < C; ++i) {
            for (int j = 0; j < i; ++j) {
                float acc2 = aslm[i * C + j];
                for (int kk = j + 1; kk < i; ++kk) {
                    acc2 += aslm[i * C + kk] * T[kk * C + j];
                }
                T[i * C + j] = -acc2;
            }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    float * Kq = Kqglob + (((int64_t) seq * H + h) * nch + ich) * (C * C);
    for (int p = tid; p < C * C; p += nthr) {
        const int i = p / C;
        const int j = p % C;
        if (j > i) {
            Kq[j * C + i] = 0.f;
            continue;
        }
        float         dot = 0.f;
        const float * kj  = kslm + j * D;
        const float * qi  = qslm + i * D;
        for (int d = 0; d < D; ++d) {
            dot += kj[d] * qi[d];
        }
        Kq[j * C + i] = dot * gdn_exp_clamp(gslm[j] - gslm[i]);
    }
}

static void gated_delta_net_chunk32_cols(const float *     q,
                                         const float *     k,
                                         const float *     v,
                                         const float *     g,
                                         const float *     beta,
                                         const float *     curr_state,
                                         float *           dst,
                                         float *           state,
                                         const float *     Tglob,
                                         const float *     Kqglob,
                                         int64_t           H,
                                         int64_t           n_tokens,
                                         int64_t           nch,
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
                                         float *           kslm,
                                         float *           qslm,
                                         float *           vslm,
                                         float *           gslm,
                                         float *           bslm,
                                         float *           Tslm,
                                         float *           Kqslm) {
    constexpr int D = 128;
    constexpr int C = 32;
    auto           item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h    = item.get_group(2);
    const uint32_t seq  = item.get_group(1);
    const int      lane = item.get_local_id(2);
    const int      col  = item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int      tid  = (int) item.get_local_id(1) * (int) item.get_local_range(2) + lane;
    const int      nthr = (int) item.get_local_range(1) * (int) item.get_local_range(2);
    const uint32_t iq1  = fastmodulo(h, neqk1_magic);
    const uint32_t iq3  = fastdiv(seq, rq3_magic);

    constexpr int warp_size     = ggml_sycl_get_physical_warp_size() < D ? ggml_sycl_get_physical_warp_size() : D;
    constexpr int rows_per_lane = (D + warp_size - 1) / warp_size;
    const int64_t packed        = H * D * D;
    float *       attn_data     = dst + (seq * n_tokens * H + h) * D;
    float *       state_out     = state + (seq * H + h) * D * D;
    const float * state_in      = curr_state + seq * packed + h * D * D + col * D;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = state_in[r * warp_size + lane];
    }

    for (int t0 = 0; t0 < (int) n_tokens; t0 += C) {
        const int nC  = (int) sycl::min((int64_t) C, n_tokens - t0);
        const int ich = t0 / C;
        for (int e = tid; e < C * D; e += nthr) {
            const int tt = e / D;
            const int i  = e % D;
            if (tt >= nC) {
                kslm[e] = 0.f;
                qslm[e] = 0.f;
                vslm[e] = 0.f;
                continue;
            }
            const int t = t0 + tt;
            qslm[e]     = (q + iq3 * sq3 + t * sq2 + iq1 * sq1)[i];
            kslm[e]     = (k + iq3 * sq3 + t * sq2 + iq1 * sq1)[i];
            vslm[e]     = (v + seq * sv3 + t * sv2 + h * sv1)[i];
        }
        for (int tt = tid; tt < C; tt += nthr) {
            if (tt >= nC) {
                gslm[tt] = 0.f;
                bslm[tt] = 0.f;
                continue;
            }
            const int64_t gb = seq * sb3 + (t0 + tt) * sb2 + h * sb1;
            gslm[tt]         = g[gb];
            bslm[tt]         = beta[gb];
        }
        const float * Tg = Tglob + (((int64_t) seq * H + h) * nch + ich) * (C * C);
        const float * Kqg = Kqglob + (((int64_t) seq * H + h) * nch + ich) * (C * C);
        for (int e = tid; e < C * C; e += nthr) {
            Tslm[e]  = Tg[e];
            Kqslm[e] = Kqg[e];
        }
        item.barrier(sycl::access::fence_space::local_space);

        if (tid == 0) {
            float acc = 0.f;
            for (int i = 0; i < C; ++i) {
                acc += gslm[i];
                gslm[i] = acc;
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        float kS[C];
        for (int j = 0; j < C; ++j) {
            float         sh = 0.f;
            const float * kj = kslm + j * D;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                sh += s_shard[r] * kj[r * warp_size + lane];
            }
            kS[j] = bslm[j] * sycl::native::exp(gslm[j]) * warp_reduce_sum<warp_size>(sh);
        }

        float Utilde[C];
        for (int i = 0; i < C; ++i) {
            float U = 0.f;
            float P = 0.f;
            for (int j = 0; j < C; ++j) {
                const float Tji = Tslm[j * C + i];
                U += vslm[j * D + col] * bslm[j] * Tji;
                P += Tji * kS[j];
            }
            Utilde[i] = U - P;
        }

        for (int i = 0; i < nC; ++i) {
            float         inter = 0.f;
            const float * qi    = qslm + i * D;
            const float   gi    = sycl::native::exp(gslm[i]);
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                inter += s_shard[r] * qi[r * warp_size + lane];
            }
            inter = gi * warp_reduce_sum<warp_size>(inter);
            float intra = 0.f;
            for (int j = 0; j <= i; ++j) {
                intra += Kqslm[j * C + i] * Utilde[j];
            }
            if (lane == 0) {
                attn_data[(t0 + i) * H * D + col] = (inter + intra) * scale;
            }
        }

        const float Glast = sycl::native::exp(gslm[nC - 1]);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] *= Glast;
            const int key = r * warp_size + lane;
            for (int i = 0; i < nC; ++i) {
                s_shard[r] += kslm[i * D + key] * Utilde[i] * sycl::native::exp(gslm[nC - 1] - gslm[i]);
            }
        }
        item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        state_out[col * D + r * warp_size + lane] = s_shard[r];
    }
}
