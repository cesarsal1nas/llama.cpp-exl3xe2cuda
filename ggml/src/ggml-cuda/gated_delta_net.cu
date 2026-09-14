#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
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

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <int S_v, bool keep_rs_t, int TILE>
__global__ void gated_delta_net_cuda_col(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * curr_state, float * dst, float * state,
    int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    const uint3 neqk1_magic, const uint3 rq3_magic,
    float scale, int64_t state_slot_stride, int K);

template <int S_v, int NWARPS, int COLS, bool KDA, bool keep_rs_t, int TILE>
__global__ void gated_delta_net_cuda_prefill(
    const float * q, const float * k, const float * v, const float * g, const float * beta,
    const float * curr_state, float * dst, float * state,
    int64_t H, int64_t n_tokens, int64_t n_seqs,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    const uint3 neqk1_magic, const uint3 rq3_magic,
    float scale, int64_t state_slot_stride, int K);

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    // S_v=128 !KDA prefill: one thread per column. Warp-tile and WY are slower here.
    if constexpr (!KDA) {
        if (n_tokens >= 16 && S_v == 128) {
            dim3 grid_dims(H, n_seqs, 2);
            dim3 block_dims(64, 1, 1);
            const ggml_cuda_kernel_launch_params pp =
                ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
            ggml_cuda_kernel_launch(gated_delta_net_cuda_col<128, keep_rs_t, 32>, pp,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            return;
        }
    }

    // Other prefill: more columns per block + shared q/k tile.
    if (n_tokens >= 16 && S_v == 64 && warp_size <= S_v && (S_v % warp_size) == 0) {
        constexpr int NWARPS = 8;
        constexpr int COLS   = 2;
        dim3 grid_dims(H, n_seqs, S_v / (NWARPS * COLS));
        dim3 block_dims(warp_size, NWARPS, 1);
        const ggml_cuda_kernel_launch_params pp =
            ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
        ggml_cuda_kernel_launch(gated_delta_net_cuda_prefill<64, NWARPS, COLS, KDA, keep_rs_t, 32>, pp,
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
        return;
    }

    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// One thread owns one state column. Same recurrence, no warp reduce.
// Prefill S_v=128 !KDA: 128 regs of state, q/k in smem.
template <int S_v, bool keep_rs_t, int TILE>
__global__ void __launch_bounds__(128, 1)
gated_delta_net_cuda_col(const float * q,
                         const float * k,
                         const float * v,
                         const float * g,
                         const float * beta,
                         const float * curr_state,
                         float *       dst,
                         float *       state,
                         int64_t       H,
                         int64_t       n_tokens,
                         int64_t       n_seqs,
                         int64_t       sq1,
                         int64_t       sq2,
                         int64_t       sq3,
                         int64_t       sv1,
                         int64_t       sv2,
                         int64_t       sv3,
                         int64_t       sb1,
                         int64_t       sb2,
                         int64_t       sb3,
                         const uint3   neqk1_magic,
                         const uint3   rq3_magic,
                         float         scale,
                         int64_t       state_slot_stride,
                         int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      col      = (int) blockIdx.z * (int) blockDim.x + (int) threadIdx.x;
    const int      tid      = (int) threadIdx.x;
    const int      nthr     = (int) blockDim.x;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    float * attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;

    float s[S_v];
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i];
    }

    __shared__ float q_sm[TILE * S_v];
    __shared__ float k_sm[TILE * S_v];
    __shared__ float g_sm[TILE];
    __shared__ float b_sm[TILE];

    GGML_UNUSED(n_seqs);
    ggml_cuda_pdl_sync();

    for (int t0 = 0; t0 < n_tokens; t0 += TILE) {
        const int ntile = (int) min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);
        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            reinterpret_cast<float4 *>(q_sm + tt * S_v)[i / 4] =
                *reinterpret_cast<const float4 *>(q_t + i);
            reinterpret_cast<float4 *>(k_sm + tt * S_v)[i / 4] =
                *reinterpret_cast<const float4 *>(k_t + i);
        }
        if (tid < ntile) {
            const int t = t0 + tid;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            g_sm[tid] = g[gb];
            b_sm[tid] = beta[gb];
        }
        __syncthreads();

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t = q_sm + tt * S_v;
            const float * k_t = k_sm + tt * S_v;
            const int     t   = t0 + tt;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const float   g_val    = expf(g_sm[tt]);
            const float   beta_val = b_sm[tt];

            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 4) {
                const float4 kk = *reinterpret_cast<const float4 *>(k_t + i);
                kv += s[i] * kk.x + s[i + 1] * kk.y + s[i + 2] * kk.z + s[i + 3] * kk.w;
            }
            const float delta = (v_t[col] - g_val * kv) * beta_val;

            float attn = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i += 4) {
                const float4 kk = *reinterpret_cast<const float4 *>(k_t + i);
                const float4 qq = *reinterpret_cast<const float4 *>(q_t + i);
                s[i]     = g_val * s[i]     + kk.x * delta;
                s[i + 1] = g_val * s[i + 1] + kk.y * delta;
                s[i + 2] = g_val * s[i + 2] + kk.z * delta;
                s[i + 3] = g_val * s[i + 3] + kk.w * delta;
                attn += s[i] * qq.x + s[i + 1] * qq.y + s[i + 2] * qq.z + s[i + 3] * qq.w;
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
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int i = 0; i < S_v; i++) {
            state[col * S_v + i] = s[i];
        }
    }
}

// Prefill: same recurrence as gated_delta_net_cuda, but one smem tile of q/k/v/g/beta
// shared by every warp in the block (the decode kernel reloads q/k per warp).
// COLS=2: each warp owns two adjacent state columns (float2 reduce).
template <int S_v, int NWARPS, int COLS, bool KDA, bool keep_rs_t, int TILE>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * NWARPS, 2)
gated_delta_net_cuda_prefill(const float * q,
                             const float * k,
                             const float * v,
                             const float * g,
                             const float * beta,
                             const float * curr_state,
                             float *       dst,
                             float *       state,
                             int64_t       H,
                             int64_t       n_tokens,
                             int64_t       n_seqs,
                             int64_t       sq1,
                             int64_t       sq2,
                             int64_t       sq3,
                             int64_t       sv1,
                             int64_t       sv2,
                             int64_t       sv3,
                             int64_t       sb1,
                             int64_t       sb2,
                             int64_t       sb3,
                             const uint3   neqk1_magic,
                             const uint3   rq3_magic,
                             float         scale,
                             int64_t       state_slot_stride,
                             int           K) {
    constexpr int warp_size     = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(COLS == 1 || COLS == 2, "COLS must be 1 or 2");
    static_assert(S_v % (NWARPS * COLS) == 0, "block must cover S_v columns exactly");
    static_assert(S_v % 4 == 0, "float4 row loads");

    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col0     = (blockIdx.z * NWARPS + threadIdx.y) * COLS;
    const int      tid      = threadIdx.y * blockDim.x + threadIdx.x;
    const int      nthr     = NWARPS * warp_size;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float * attn_data = dst;

    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float s0[rows_per_lane];
    float s1[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s0[r] = curr_state[col0 * S_v + i];
        if constexpr (COLS == 2) {
            s1[r] = curr_state[(col0 + 1) * S_v + i];
        }
    }

    __shared__ float q_sm[TILE * S_v];
    __shared__ float k_sm[TILE * S_v];
    __shared__ float g_sm[TILE];
    __shared__ float b_sm[TILE];

    GGML_UNUSED(n_seqs);
    ggml_cuda_pdl_sync();

    for (int t0 = 0; t0 < n_tokens; t0 += TILE) {
        const int ntile = (int) min((int64_t) TILE, n_tokens - t0);
        const int n4    = ntile * (S_v / 4);

        for (int e = tid; e < n4; e += nthr) {
            const int tt = e / (S_v / 4);
            const int i  = (e % (S_v / 4)) * 4;
            const int t  = t0 + tt;
            const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
            reinterpret_cast<float4 *>(q_sm + tt * S_v)[i / 4] =
                *reinterpret_cast<const float4 *>(q_t + i);
            reinterpret_cast<float4 *>(k_sm + tt * S_v)[i / 4] =
                *reinterpret_cast<const float4 *>(k_t + i);
        }
        for (int tt = tid; tt < ntile; tt += nthr) {
            const int t = t0 + tt;
            const int64_t gb = sequence * sb3 + t * sb2 + h_idx * sb1;
            b_sm[tt] = beta[gb];
            if constexpr (!KDA) {
                g_sm[tt] = g[gb];
            }
        }
        __syncthreads();

        for (int tt = 0; tt < ntile; tt++) {
            const float * q_t = q_sm + tt * S_v;
            const float * k_t = k_sm + tt * S_v;
            const int     t   = t0 + tt;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;
            const int64_t gb  = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float   beta_val = b_sm[tt];
            const float   v0 = v_t[col0];
            const float   v1 = COLS == 2 ? v_t[col0 + 1] : 0.0f;

            if constexpr (!KDA) {
                const float g_val = expf(g_sm[tt]);

                float kv0 = 0.0f;
                float kv1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    kv0 += s0[r] * k_t[i];
                    if constexpr (COLS == 2) {
                        kv1 += s1[r] * k_t[i];
                    }
                }
                if constexpr (COLS == 2) {
                    float2 kv = warp_reduce_sum<warp_size>(make_float2(kv0, kv1));
                    kv0 = kv.x;
                    kv1 = kv.y;
                } else {
                    kv0 = warp_reduce_sum<warp_size>(kv0);
                }

                const float d0 = (v0 - g_val * kv0) * beta_val;
                const float d1 = (v1 - g_val * kv1) * beta_val;

                float attn0 = 0.0f;
                float attn1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    s0[r] = g_val * s0[r] + k_t[i] * d0;
                    attn0 += s0[r] * q_t[i];
                    if constexpr (COLS == 2) {
                        s1[r] = g_val * s1[r] + k_t[i] * d1;
                        attn1 += s1[r] * q_t[i];
                    }
                }
                if constexpr (COLS == 2) {
                    float2 at = warp_reduce_sum<warp_size>(make_float2(attn0, attn1));
                    attn0 = at.x;
                    attn1 = at.y;
                } else {
                    attn0 = warp_reduce_sum<warp_size>(attn0);
                }

                if (lane == 0) {
                    attn_data[col0] = attn0 * scale;
                    if constexpr (COLS == 2) {
                        attn_data[col0 + 1] = attn1 * scale;
                    }
                }
            } else {
                const float * g_t = g + gb * S_v;

                float kv0 = 0.0f;
                float kv1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int   i  = r * warp_size + lane;
                    const float gi = expf(g_t[i]);
                    kv0 += gi * s0[r] * k_t[i];
                    if constexpr (COLS == 2) {
                        kv1 += gi * s1[r] * k_t[i];
                    }
                }
                if constexpr (COLS == 2) {
                    float2 kv = warp_reduce_sum<warp_size>(make_float2(kv0, kv1));
                    kv0 = kv.x;
                    kv1 = kv.y;
                } else {
                    kv0 = warp_reduce_sum<warp_size>(kv0);
                }

                const float d0 = (v0 - kv0) * beta_val;
                const float d1 = (v1 - kv1) * beta_val;

                float attn0 = 0.0f;
                float attn1 = 0.0f;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int   i  = r * warp_size + lane;
                    const float gi = expf(g_t[i]);
                    s0[r] = gi * s0[r] + k_t[i] * d0;
                    attn0 += s0[r] * q_t[i];
                    if constexpr (COLS == 2) {
                        s1[r] = gi * s1[r] + k_t[i] * d1;
                        attn1 += s1[r] * q_t[i];
                    }
                }
                if constexpr (COLS == 2) {
                    float2 at = warp_reduce_sum<warp_size>(make_float2(attn0, attn1));
                    attn0 = at.x;
                    attn1 = at.y;
                } else {
                    attn0 = warp_reduce_sum<warp_size>(attn0);
                }

                if (lane == 0) {
                    attn_data[col0] = attn0 * scale;
                    if constexpr (COLS == 2) {
                        attn_data[col0 + 1] = attn1 * scale;
                    }
                }
            }

            attn_data += S_v * H;

            if constexpr (keep_rs_t) {
                const int target_slot = (int) n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * slot = state + target_slot * state_slot_stride;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        slot[col0 * S_v + i] = s0[r];
                        if constexpr (COLS == 2) {
                            slot[(col0 + 1) * S_v + i] = s1[r];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            state[col0 * S_v + i] = s0[r];
            if constexpr (COLS == 2) {
                state[(col0 + 1) * S_v + i] = s1[r];
            }
        }
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

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

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

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

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
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
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
