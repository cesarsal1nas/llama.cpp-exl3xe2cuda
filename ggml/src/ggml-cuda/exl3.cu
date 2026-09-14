#include "exl3.cuh"
#include "convert.cuh"
#include "mmid.cuh"
#include "exl3/exl3_gemv.cuh"
#include "exl3/exl3_gemv_int8.cuh"
#include "exl3/exl3_gemm.cuh"
#include "exl3/exl3_moe.cuh"
#include "exl3/exl3_recon.cuh"
#include "exl3/exl3_recon_had.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static __global__ void exl3_sum_planes_kernel(const float * src, float * dst, int n, int n_split) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float s = 0.0f;
    for (int z = 0; z < n_split; ++z) {
        s += src[(size_t) z * (size_t) n + i];
    }
    dst[i] = s;
}

static void exl3_sum_planes(const float * src, float * dst, int n, int n_split, cudaStream_t stream) {
    const int bs = 256;
    const int nb = (n + bs - 1) / bs;
    exl3_sum_planes_kernel<<<nb, bs, 0, stream>>>(src, dst, n, n_split);
}

static int exl3_recon_had_env() {
    static const int v = [] {
        const char * e = getenv("GGML_EXL3_RECON_HAD");
        if (!e) {
            return -1;
        }
        return atoi(e);
    }();
    return v;
}

static int exl3_gemv_want_fp16() {
    static const int on = [] {
        const char * e = getenv("GGML_EXL3_GEMV");
        return e && strcmp(e, "fp16") == 0 ? 1 : 0;
    }();
    return on;
}

static bool exl3_allow_recon(const ggml_tensor * src0) {
    if (ggml_exl3_get_codebook() != 0) {
        return false;
    }
    if (src0->ne[2] > 1 || src0->ne[3] > 1) {
        return false;
    }
    if (src0->view_src && (src0->view_src->ne[2] > 1 || src0->view_src->ne[3] > 1)) {
        return false;
    }
    return true;
}

static void exl3_gemv_cuda_rows(
        const half * x, const uint16_t * B, float * C,
        int M, int K, int N, int bits, cudaStream_t stream, int packed_n) {
    for (int m0 = 0; m0 < M; m0 += EXL3_GEMV_MAX_M) {
        const int m = M - m0 < EXL3_GEMV_MAX_M ? M - m0 : EXL3_GEMV_MAX_M;
        exl3_gemv_cuda(x + (size_t) m0 * K, B, C + (size_t) m0 * N,
            m, K, N, bits, stream, packed_n);
    }
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const ggml_tensor * fuse_x, const ggml_tensor * fuse_suh, const ggml_tensor * fuse_svh) {
    const int64_t K = src0->ne[0];
    const int64_t packed_n = src0->ne[1];
    const int64_t N = dst->ne[0];
    const ggml_tensor * x_src = fuse_x ? fuse_x : src1;
    GGML_ASSERT(x_src);
    const int64_t M = ggml_nrows(x_src);

    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(x_src->ne[0] == K);
    GGML_ASSERT(N > 0 && N <= packed_n && ggml_nrows(dst) == M);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(K % 128 == 0 && packed_n % 128 == 0 && N % 128 == 0);
    if (ggml_exl3_get_pack() != 0) {
        GGML_ABORT("EXL3 xe2-16 pack is SYCL-only; CUDA keeps v3 MMA files");
    }

    cudaStream_t stream = ctx.stream();
    const int bits = ggml_exl3_n_bits(src0->type);
    const char * recon_env = getenv("GGML_EXL3_RECONSTRUCT");
    const bool force_reconstruct = recon_env && atoi(recon_env) != 0;
    const char * gemm_env = getenv("GGML_EXL3_GEMM");
    const bool force_gemm = gemm_env && atoi(gemm_env) != 0;
    const int had_env = exl3_recon_had_env();
    const bool allow_recon = exl3_allow_recon(src0);
    const bool use_recon_had = allow_recon && !force_gemm && had_env != 0 && fuse_x && fuse_suh && fuse_svh &&
        M > EXL3_GEMV_MAX_M && (had_env > 0 || M >= EXL3_RECON_HAD_MIN_M);

    if (use_recon_had) {
        GGML_ASSERT(fuse_x->type == GGML_TYPE_F32 || fuse_x->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(fuse_x));
        GGML_ASSERT(fuse_suh->type == GGML_TYPE_F32);
        GGML_ASSERT(fuse_svh->type == GGML_TYPE_F32);
        GGML_ASSERT(fuse_svh->ne[0] >= N);
        ggml_cuda_mul_mat_exl3_recon_had(ctx, src0, src1, dst, fuse_x->data, fuse_x->type,
            (const float *) fuse_suh->data, (const float *) fuse_svh->data);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (M <= EXL3_GEMV_MAX_M && fuse_x) {
        GGML_ASSERT(fuse_x->type == GGML_TYPE_F32 || fuse_x->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(fuse_x));
        ggml_cuda_pool_alloc<float> x_f16to32(ctx.pool());
        const float * x_f32 = nullptr;
        if (fuse_x->type == GGML_TYPE_F16) {
            const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
            GGML_ASSERT(to_fp32);
            x_f16to32.alloc(ggml_nelements(fuse_x));
            to_fp32(fuse_x->data, x_f16to32.get(), ggml_nelements(fuse_x), stream);
            x_f32 = x_f16to32.get();
        } else {
            x_f32 = (const float *) fuse_x->data;
        }
        ggml_cuda_pool_alloc<float> xcopy(ctx.pool());
        if (fuse_x->data && dst->data) {
            const int64_t x0 = (int64_t) fuse_x->data;
            const int64_t x1 = x0 + (int64_t) ggml_nbytes(fuse_x);
            const int64_t d0 = (int64_t) dst->data;
            const int64_t d1 = d0 + (int64_t) ggml_nbytes(dst);
            if ((d0 <= x0 && x0 < d1) || (x0 <= d0 && d0 < x1)) {
                xcopy.alloc(ggml_nelements(fuse_x));
                CUDA_CHECK(cudaMemcpyAsync(xcopy.get(), fuse_x->data, ggml_nbytes(fuse_x),
                                           cudaMemcpyDeviceToDevice, stream));
                x_f32 = xcopy.get();
            }
        }
        const float * suh = nullptr;
        if (fuse_suh) {
            GGML_ASSERT(fuse_suh->type == GGML_TYPE_F32);
            suh = (const float *) fuse_suh->data;
        }
        const float * svh = nullptr;
        if (fuse_svh) {
            GGML_ASSERT(fuse_svh->type == GGML_TYPE_F32);
            GGML_ASSERT(fuse_svh->ne[0] >= N);
            svh = (const float *) fuse_svh->data;
        }
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        if (!exl3_gemv_want_fp16() && ggml_exl3_get_codebook() == 0 &&
                exl3_gemv_int8_shape_ok((int) K, (int) N, bits)) {
            if (exl3_gemv_int8_cuda(x_f32, (const uint16_t *) src0->data, (float *) dst->data,
                    (int) M, (int) K, (int) N, bits, suh, true, svh,
                    nullptr, nsm, stream, (int) packed_n)) {
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        exl3_gemv_cuda_fused(x_f32, suh, (const uint16_t *) src0->data, (float *) dst->data,
            (int) M, (int) K, (int) N, bits, stream, svh, (int) packed_n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool());
    const half * x = nullptr;
    if (src1->type == GGML_TYPE_F16) {
        x = (const half *) src1->data;
    } else {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16);
        src1_f16.alloc(ggml_nelements(src1));
        to_fp16(src1->data, src1_f16.get(), ggml_nelements(src1), stream);
        x = src1_f16.get();
    }

    if (M <= EXL3_GEMV_MAX_M) {
        exl3_gemv_cuda(x, (const uint16_t *) src0->data, (float *) dst->data,
            (int) M, (int) K, (int) N, bits, stream, (int) packed_n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // GEMV is M<=8. Packed GEMM is 8<M<192 (M-tiles looped in-kernel, grid.y=1).
    // Recon is default for M>=192 unless GGML_EXL3_GEMM=1.
    if (allow_recon && (N < packed_n || force_reconstruct ||
            (!force_gemm && M >= EXL3_RECON_HAD_MIN_M) ||
            !exl3_gemm_shape_ok((int) K, (int) N, bits))) {
        if (N < packed_n && fuse_x && fuse_suh && fuse_svh) {
            ggml_cuda_mul_mat_exl3_recon_had(ctx, src0, src1, dst, fuse_x->data, fuse_x->type,
                (const float *) fuse_suh->data, (const float *) fuse_svh->data);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        ggml_cuda_mul_mat_exl3_recon_full(ctx, src0, src1, dst, x);
        return;
    }

    if (!exl3_gemm_shape_ok((int) K, (int) N, bits)) {
        exl3_gemv_cuda_rows(x, (const uint16_t *) src0->data, (float *) dst->data,
            (int) M, (int) K, (int) N, bits, stream, (int) packed_n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int n_locks = exl3_gemm_lock_elems((int) M, (int) N);
    ggml_cuda_pool_alloc<int> locks(ctx.pool());
    locks.alloc((size_t) n_locks);
    CUDA_CHECK(cudaMemsetAsync(locks.get(), 0, (size_t) n_locks * sizeof(int), stream));
    const int n_sms = ggml_cuda_info().devices[ctx.device].nsm;
    exl3_gemm_cuda(x, (const uint16_t *) src0->data, (float *) dst->data,
        (int) M, (int) K, (int) N, bits, locks.get(), n_sms, stream);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" bool ggml_cuda_exl3_reconstruct_had_prefix_f16(
        const ggml_tensor * w,
        const ggml_tensor * suh,
        const ggml_tensor * svh,
        int64_t n_cols,
        void * dst_host_f16) {
    if (!w || !w->data || !suh || !suh->data || !svh || !svh->data || !dst_host_f16) {
        return false;
    }
    if (!ggml_is_exl3(w->type) || suh->type != GGML_TYPE_F32 || svh->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    if (K % 128 != 0 || N % 128 != 0 || n_cols <= 0 || n_cols > N || n_cols % 128 != 0) {
        return false;
    }

    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, w->data) != cudaSuccess || attr.devicePointer == nullptr) {
        cudaGetLastError();
        return false;
    }
    ggml_cuda_set_device(attr.device);

    const size_t nbytes = (size_t) K * (size_t) n_cols * sizeof(half);
    half * dst_dev = nullptr;
    if (cudaMalloc(&dst_dev, nbytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const int bits = ggml_exl3_n_bits(w->type);
    exl3_reconstruct_had_prefix(
            (const char *) w->data, dst_dev,
            (const float *) suh->data, (const float *) svh->data,
            (int) K, (int) N, (int) n_cols, bits, 0);
    const cudaError_t kerr = cudaGetLastError();
    const cudaError_t merr = (kerr == cudaSuccess)
            ? cudaMemcpy(dst_host_f16, dst_dev, nbytes, cudaMemcpyDeviceToHost)
            : kerr;
    cudaFree(dst_dev);
    return merr == cudaSuccess;
}

void ggml_cuda_mul_mat_exl3_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];
    const ggml_tensor * suh  = dst->src[3];
    const ggml_tensor * svh  = dst->src[4];

    GGML_ASSERT(ggml_is_exl3(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] > 1);
    GGML_ASSERT(suh && svh);
    GGML_ASSERT(suh->type == GGML_TYPE_F32 && svh->type == GGML_TYPE_F32);
    GGML_ASSERT(suh->ne[0] == src0->ne[0] && suh->ne[1] == src0->ne[2]);
    GGML_ASSERT(svh->ne[0] == dst->ne[0] && svh->ne[1] == src0->ne[2]);
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(suh));
    GGML_ASSERT(ggml_is_contiguous(svh));
    GGML_ASSERT(src1->ne[0] == src0->ne[0]);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->ne[1] == src1->ne[2]);
    GGML_ASSERT(ids->ne[0] == dst->ne[1]);
    GGML_ASSERT(dst->ne[0] == src0->ne[1]);
    GGML_ASSERT(dst->ne[2] == src1->ne[2]);
    GGML_ASSERT(ggml_exl3_get_pack() == 0);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = src1->ne[2];
    const int64_t n_inst = n_tokens * n_expert_used;
    if (n_inst <= 0) {
        return;
    }
    const int bits = ggml_exl3_n_bits(src0->type);
    const int K = (int) src0->ne[0];
    const int N = (int) dst->ne[0];
    const int packed_n = (int) src0->ne[1];
    cudaStream_t stream = ctx.stream();
    const int n_sms = ggml_cuda_info().devices[ctx.device].nsm;

    exl3_gemv_id id = {};
    id.ids = (const int32_t *) ids->data;
    id.n_expert_used = (int) n_expert_used;
    id.ids_nb0 = ids->nb[0];
    id.ids_nb1 = ids->nb[1];
    id.expert_stride = src0->nb[2];
    id.x_ne1 = (int) src1->ne[1];
    id.suh_2d = 1;

    id.n_inst = (int) n_inst;

    if (n_tokens <= 1) {
        exl3_gemv_cuda_fused_id(
            (const float *) src1->data, (const float *) suh->data, (const uint16_t *) src0->data,
            (float *) dst->data, 1, K, N, bits, stream, (const float *) svh->data, packed_n, id);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src1->nb[1] % src1->nb[0] == 0);
    GGML_ASSERT(src1->nb[2] % src1->nb[1] == 0);

    const int n_experts = (int) src0->ne[2];
    const int si1 = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_inst);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_inst);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_launch_mm_ids_helper(
        (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, (int) n_tokens, (int) n_expert_used, (int) src1->ne[1], si1, sis1, false, stream);
    CUDA_CHECK(cudaGetLastError());

    if (n_tokens > EXL3_GEMV_MAX_M && exl3_gemm_shape_ok(K, N, bits)) {
        ggml_cuda_pool_alloc<half> A_had(ctx.pool(), n_inst * (size_t) K);
        ggml_cuda_pool_alloc<float> C_cmp(ctx.pool(), n_inst * (size_t) N);
        exl3_moe_pre_had_cuda(
            (const float *) src1->data, (const float *) suh->data, A_had.get(),
            (const int32_t *) ids->data, ids_dst.get(),
            (int) n_inst, K, (int) n_expert_used, (int) src1->ne[1],
            ids->nb[0], ids->nb[1], stream);
        CUDA_CHECK(cudaGetLastError());
        const int n_m_max = ((int) n_tokens + 15) / 16;
        exl3_gemm_cuda_id(
            A_had.get(), (const uint16_t *) src0->data, C_cmp.get(),
            K, N, bits, n_m_max, n_experts,
            nullptr, n_sms, expert_bounds.get(), src0->nb[2],
            ids_dst.get(), (const float *) svh->data, (float *) dst->data, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    id.bounds = expert_bounds.get();
    id.ids_src = ids_src1.get();
    id.ids_dst = ids_dst.get();
    id.n_experts = n_experts;
    exl3_gemv_cuda_fused_id(
        (const float *) src1->data, (const float *) suh->data, (const uint16_t *) src0->data,
        (float *) dst->data, EXL3_GEMV_MAX_M, K, N, bits, stream, (const float *) svh->data, packed_n, id);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_exl3_id_glu(ggml_backend_cuda_context & ctx,
        ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu, ggml_tensor * down) {
    const ggml_tensor * Wg = gate->src[0];
    const ggml_tensor * Wu = up->src[0];
    const ggml_tensor * src1 = up->src[1];
    const ggml_tensor * ids  = up->src[2];
    const ggml_tensor * suh_g = gate->src[3];
    const ggml_tensor * svh_g = gate->src[4];
    const ggml_tensor * suh_u = up->src[3];
    const ggml_tensor * svh_u = up->src[4];

    GGML_ASSERT(gate->src[1] == src1 && gate->src[2] == ids);
    GGML_ASSERT(ggml_is_exl3(Wg->type) && ggml_is_exl3(Wu->type));
    GGML_ASSERT(suh_g && svh_g && suh_u && svh_u);
    GGML_ASSERT(ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, glu));

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = src1->ne[2];
    const int64_t n_inst = n_tokens * n_expert_used;
    if (n_inst <= 0) {
        return;
    }

    const int bits_g = ggml_exl3_n_bits(Wg->type);
    const int bits_u = ggml_exl3_n_bits(Wu->type);
    const int K = (int) Wg->ne[0];
    const int N = (int) gate->ne[0];
    const int packed_n = (int) Wg->ne[1];
    cudaStream_t stream = ctx.stream();
    const int n_sms = ggml_cuda_info().devices[ctx.device].nsm;

    exl3_gemv_id id = {};
    id.ids = (const int32_t *) ids->data;
    id.n_expert_used = (int) n_expert_used;
    id.ids_nb0 = ids->nb[0];
    id.ids_nb1 = ids->nb[1];
    id.x_ne1 = (int) src1->ne[1];
    id.suh_2d = 1;
    id.n_inst = (int) n_inst;

    auto run_down_gemv = [&]() {
        if (!down) {
            return;
        }
        const ggml_tensor * Wd = down->src[0];
        const ggml_tensor * suh_d = down->src[3];
        const ggml_tensor * svh_d = down->src[4];
        GGML_ASSERT(Wd && suh_d && svh_d && down->src[2] == ids);
        id.expert_stride = Wd->nb[2];
        id.x_ne1 = (int) glu->ne[1];
        exl3_gemv_cuda_fused_id(
            (const float *) glu->data, (const float *) suh_d->data, (const uint16_t *) Wd->data,
            (float *) down->data, id.n_experts ? EXL3_GEMV_MAX_M : 1,
            (int) Wd->ne[0], (int) down->ne[0], ggml_exl3_n_bits(Wd->type),
            stream, (const float *) svh_d->data, (int) Wd->ne[1], id);
    };

    if (n_tokens <= 1) {
        id.expert_stride = Wg->nb[2];
        exl3_gemv_cuda_fused_id(
            (const float *) src1->data, (const float *) suh_g->data, (const uint16_t *) Wg->data,
            (float *) gate->data, 1, K, N, bits_g, stream, (const float *) svh_g->data, packed_n, id);
        id.expert_stride = Wu->nb[2];
        exl3_gemv_cuda_fused_id(
            (const float *) src1->data, (const float *) suh_u->data, (const uint16_t *) Wu->data,
            (float *) up->data, 1, K, N, bits_u, stream, (const float *) svh_u->data, packed_n, id);
        exl3_swiglu_f32_cuda((const float *) gate->data, (const float *) up->data,
            (float *) glu->data, n_inst * (int64_t) N, stream);
        run_down_gemv();
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(ids->nb[0] == sizeof(int32_t));
    const int n_experts = (int) Wg->ne[2];
    const int si1 = (int) (ids->nb[1] / sizeof(int32_t));
    const int sis1 = (int) (src1->nb[2] / src1->nb[1]);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_inst);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_inst);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_launch_mm_ids_helper(
        (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, (int) n_tokens, (int) n_expert_used, (int) src1->ne[1], si1, sis1, false, stream);

    static const int moe_ffn_off = [] {
        const char * e = getenv("GGML_EXL3_MOE_FFN");
        return (e && atoi(e) == 1) ? 0 : 1;
    }();
    if (!moe_ffn_off && n_tokens > EXL3_GEMV_MAX_M && down && ggml_is_exl3(down->src[0]->type) &&
            down->src[3] && down->src[4] && (int) down->src[0]->ne[0] == N &&
            exl3_moe_ffn_ok(K, N, bits_g, bits_u, ggml_exl3_n_bits(down->src[0]->type))) {
        const ggml_tensor * Wd = down->src[0];
        const int Nd = (int) down->ne[0];
        ggml_cuda_pool_alloc<half> A_g(ctx.pool(), n_inst * (size_t) K);
        ggml_cuda_pool_alloc<half> A_u(ctx.pool(), n_inst * (size_t) K);
        ggml_cuda_pool_alloc<half> A_d(ctx.pool(), n_inst * (size_t) N);
        ggml_cuda_pool_alloc<float> C_g(ctx.pool(), n_inst * (size_t) N);
        ggml_cuda_pool_alloc<float> C_u(ctx.pool(), n_inst * (size_t) N);
        ggml_cuda_pool_alloc<float> C_d(ctx.pool(), n_inst * (size_t) Nd);
        ggml_cuda_pool_alloc<int> locks(ctx.pool(), exl3_moe_lock_n(n_experts));
        CUDA_CHECK(cudaMemsetAsync(locks.get(), 0, sizeof(int) * (size_t) exl3_moe_lock_n(n_experts), stream));
        exl3_moe_ffn_cuda(
            (const float *) src1->data,
            (const float *) suh_g->data, (const float *) suh_u->data, (const float *) down->src[3]->data,
            (const float *) svh_g->data, (const float *) svh_u->data, (const float *) down->src[4]->data,
            (const uint16_t *) Wg->data, (const uint16_t *) Wu->data, (const uint16_t *) Wd->data,
            A_g.get(), A_u.get(), A_d.get(), C_g.get(), C_u.get(), C_d.get(), (float *) down->data,
            (const int32_t *) ids->data, ids_dst.get(), expert_bounds.get(),
            locks.get(), n_sms,
            K, N, bits_g, n_experts, (int) n_expert_used, (int) src1->ne[1],
            ids->nb[0], ids->nb[1], Wg->nb[2], Wu->nb[2], Wd->nb[2], stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (n_tokens > EXL3_GEMV_MAX_M && exl3_gemm_shape_ok(K, N, bits_g) && exl3_gemm_shape_ok(K, N, bits_u)) {
        ggml_cuda_pool_alloc<half> A_g(ctx.pool(), n_inst * (size_t) K);
        ggml_cuda_pool_alloc<half> A_u(ctx.pool(), n_inst * (size_t) K);
        const size_t n_c = n_inst * (size_t) N;
        const size_t c_pad = 4096;
        ggml_cuda_pool_alloc<float> C_g(ctx.pool(), n_c + c_pad);
        ggml_cuda_pool_alloc<float> C_u(ctx.pool(), n_c);
        exl3_moe_pre_had_cuda(
            (const float *) src1->data, (const float *) suh_g->data, A_g.get(),
            (const int32_t *) ids->data, ids_dst.get(),
            (int) n_inst, K, (int) n_expert_used, (int) src1->ne[1],
            ids->nb[0], ids->nb[1], stream);
        exl3_moe_pre_had_cuda(
            (const float *) src1->data, (const float *) suh_u->data, A_u.get(),
            (const int32_t *) ids->data, ids_dst.get(),
            (int) n_inst, K, (int) n_expert_used, (int) src1->ne[1],
            ids->nb[0], ids->nb[1], stream);
        static const int snap = [] {
            const char * e = getenv("GGML_EXL3_ID_SNAP");
            return e ? atoi(e) : 0;
        }();
        ggml_cuda_pool_alloc<half> A_u_save;
        if (snap) {
            A_u_save.alloc(ctx.pool(), n_inst * (size_t) K);
            CUDA_CHECK(cudaMemcpyAsync(A_u_save.get(), A_u.get(), n_inst * (size_t) K * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        }
        const int n_m_max = ((int) n_tokens + 15) / 16;
        const int id_nx = exl3_gemm_id_nx();
        const int tile_n_g = exl3_gemm_id_tile_n(K, N, bits_g);
        const int n_lock = (id_nx > 0) ? exl3_gemm_id_lock_n(n_experts, n_m_max, N, tile_n_g) : 0;
        ggml_cuda_pool_alloc<int> gemm_locks;
        int * gate_locks = nullptr;
        if (n_lock > 0) {
            gemm_locks.alloc(ctx.pool(), (size_t) n_lock);
            CUDA_CHECK(cudaMemsetAsync(gemm_locks.get(), 0, sizeof(int) * (size_t) n_lock, stream));
            CUDA_CHECK(cudaMemsetAsync(C_g.get(), 0, sizeof(float) * n_c, stream));
            gate_locks = gemm_locks.get();
        }
        static const int pad_chk = [] {
            const char * e = getenv("GGML_EXL3_ID_PAD");
            return e ? atoi(e) : 0;
        }();
        if (pad_chk && gate_locks) {
            std::vector<float> sent(c_pad, 1234.5f);
            CUDA_CHECK(cudaMemcpyAsync(C_g.get() + n_c, sent.data(), c_pad * sizeof(float), cudaMemcpyHostToDevice, stream));
        }
        exl3_gemm_cuda_id(
            A_g.get(), (const uint16_t *) Wg->data, C_g.get(),
            K, N, bits_g, n_m_max, n_experts,
            gate_locks, n_sms, expert_bounds.get(), Wg->nb[2],
            nullptr, nullptr, nullptr, stream);
        if (snap && gate_locks) {
            static int snap_left = snap;
            if (snap_left > 0) {
                const size_t n_a = n_inst * (size_t) K;
                std::vector<half> h0(n_a);
                std::vector<half> h1(n_a);
                CUDA_CHECK(cudaMemcpyAsync(h0.data(), A_u_save.get(), n_a * sizeof(half), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(h1.data(), A_u.get(), n_a * sizeof(half), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                int n_diff = 0;
                for (size_t i = 0; i < n_a; ++i) {
                    if (h0[i] != h1[i]) {
                        n_diff++;
                    }
                }
                fprintf(stderr, "exl3 id_snap A_u diffs=%d/%zu\n", n_diff, n_a);
                snap_left--;
            }
        }
        if (pad_chk && gate_locks) {
            static int pad_left = pad_chk;
            if (pad_left > 0) {
                std::vector<float> sent(c_pad);
                CUDA_CHECK(cudaMemcpyAsync(sent.data(), C_g.get() + n_c, c_pad * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                int n_hit = 0;
                float first = 1234.5f;
                for (size_t i = 0; i < c_pad; ++i) {
                    if (sent[i] != 1234.5f) {
                        if (n_hit == 0) {
                            first = sent[i];
                        }
                        n_hit++;
                    }
                }
                fprintf(stderr, "exl3 id_pad n_tokens=%d n_inst=%d n_m_max=%d hits=%d/%zu first=%g\n",
                    (int) n_tokens, (int) n_inst, n_m_max, n_hit, c_pad, first);
                pad_left--;
            }
        }
        static int cmp_mode = [] {
            const char * e = getenv("GGML_EXL3_ID_CMP");
            return e ? atoi(e) : 0;
        }();
        static int cmp_left = cmp_mode >= 0 ? cmp_mode : 1000000;
        if (cmp_left > 0 && gate_locks && cmp_mode != 0) {
            const size_t n_c = n_inst * (size_t) N;
            ggml_cuda_pool_alloc<float> C_split(ctx.pool(), n_c);
            CUDA_CHECK(cudaMemcpyAsync(C_split.get(), C_g.get(), n_c * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            exl3_gemm_id_nx_override() = 4;
            exl3_gemm_cuda_id(
                A_g.get(), (const uint16_t *) Wg->data, C_g.get(),
                K, N, bits_g, n_m_max, n_experts,
                gate_locks, n_sms, expert_bounds.get(), Wg->nb[2],
                nullptr, nullptr, nullptr, stream);
            exl3_gemm_id_nx_override() = -1;
            std::vector<float> h_split(n_c);
            std::vector<float> h_full(n_c);
            CUDA_CHECK(cudaMemcpyAsync(h_split.data(), C_split.get(), n_c * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_full.data(), C_g.get(), n_c * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            double max_abs = 0;
            double sum_abs = 0;
            double sum_full = 0;
            int n_bad = 0;
            int first_bad = -1;
            for (size_t i = 0; i < n_c; ++i) {
                const double d = fabs((double) h_split[i] - (double) h_full[i]);
                const double a = fabs((double) h_full[i]);
                sum_abs += d;
                sum_full += a;
                if (d > max_abs) {
                    max_abs = d;
                }
                if (d > 1e-3 * (1.0 + a)) {
                    n_bad++;
                    if (first_bad < 0) {
                        first_bad = (int) i;
                    }
                }
            }
            int n_nan = 0;
            int n_inf = 0;
            for (size_t i = 0; i < n_c; ++i) {
                if (isnan(h_split[i])) {
                    n_nan++;
                }
                if (isinf(h_split[i])) {
                    n_inf++;
                }
            }
            fprintf(stderr, "exl3 id_cmp n_tokens=%d n_inst=%d N=%d max_abs=%.6g mean_abs=%.6g mean_full=%.6g n_bad=%d/%zu n_nan=%d n_inf=%d first_bad=%d",
                (int) n_tokens, (int) n_inst, N, max_abs, sum_abs / (double) n_c, sum_full / (double) n_c,
                n_bad, n_c, n_nan, n_inf, first_bad);
            if (first_bad >= 0) {
                fprintf(stderr, " split=%g full=%g", h_split[first_bad], h_full[first_bad]);
            }
            fprintf(stderr, " c0 split=%g %g %g %g full=%g %g %g %g\n",
                h_split[0], h_split[1], h_split[2], h_split[3],
                h_full[0], h_full[1], h_full[2], h_full[3]);
            // cmp_mode < 0: keep chunked C. > 0: keep full C.
            if (cmp_mode < 0) {
                CUDA_CHECK(cudaMemcpyAsync(C_g.get(), C_split.get(), n_c * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            }
            cmp_left--;
        }
        static const int replay = [] {
            const char * e = getenv("GGML_EXL3_ID_REPLAY");
            return e ? atoi(e) : 0;
        }();
        static const int atcmp = [] {
            const char * e = getenv("GGML_EXL3_ID_ATCMP");
            return e ? atoi(e) : 0;
        }();
        if (atcmp && gate_locks) {
            static int at_left = atcmp;
            if (at_left > 0) {
                ggml_cuda_pool_alloc<float> C_at(ctx.pool(), n_c);
                CUDA_CHECK(cudaMemcpyAsync(C_at.get(), C_g.get(), n_c * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                exl3_gemm_id_nx_override() = 4;
                // write path: unset atomic via sadd=0 by clearing env-equivalent: launch with locks but we need sadd=0.
                // Temporarily zero C and run write by using a second ID without ATOMIC - getenv still set.
                // Use override 4 and a host write launch: memset then kernel with sadd 0 via nx override only.
                CUDA_CHECK(cudaMemsetAsync(C_g.get(), 0, n_c * sizeof(float), stream));
                unsetenv("GGML_EXL3_ID_ATOMIC");
                exl3_gemm_cuda_id(
                    A_g.get(), (const uint16_t *) Wg->data, C_g.get(),
                    K, N, bits_g, n_m_max, n_experts,
                    gate_locks, n_sms, expert_bounds.get(), Wg->nb[2],
                    nullptr, nullptr, nullptr, stream);
                setenv("GGML_EXL3_ID_ATOMIC", "1", 1);
                exl3_gemm_id_nx_override() = -1;
                std::vector<float> ha(n_c);
                std::vector<float> hw(n_c);
                CUDA_CHECK(cudaMemcpyAsync(ha.data(), C_at.get(), n_c * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(hw.data(), C_g.get(), n_c * sizeof(float), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                double max_abs = 0;
                int n_bad = 0;
                for (size_t i = 0; i < n_c; ++i) {
                    const double d = fabs((double) ha[i] - (double) hw[i]);
                    if (d > max_abs) {
                        max_abs = d;
                    }
                    if (d > 1e-3 * (1.0 + fabs((double) hw[i]))) {
                        n_bad++;
                    }
                }
                fprintf(stderr, "exl3 atcmp max_abs=%.6g n_bad=%d/%zu at0=%g %g wr0=%g %g\n",
                    max_abs, n_bad, n_c, ha[0], ha[1], hw[0], hw[1]);
                at_left--;
            }
        }
        if (replay && gate_locks) {
            const size_t n_c = n_inst * (size_t) N;
            ggml_cuda_pool_alloc<float> C_dummy(ctx.pool(), n_c);
            exl3_gemm_id_nx_override() = 4;
            exl3_gemm_cuda_id(
                A_g.get(), (const uint16_t *) Wg->data, C_dummy.get(),
                K, N, bits_g, n_m_max, n_experts,
                gate_locks, n_sms, expert_bounds.get(), Wg->nb[2],
                nullptr, nullptr, nullptr, stream);
            exl3_gemm_id_nx_override() = -1;
        }
        exl3_gemm_cuda_id(
            A_u.get(), (const uint16_t *) Wu->data, C_u.get(),
            K, N, bits_u, n_m_max, n_experts,
            nullptr, n_sms, expert_bounds.get(), Wu->nb[2],
            nullptr, nullptr, nullptr, stream);
        if (down && ggml_is_exl3(down->src[0]->type) && down->src[3] && down->src[4] &&
                (int) down->src[0]->ne[0] == N &&
                exl3_gemm_shape_ok((int) down->src[0]->ne[0], (int) down->ne[0], ggml_exl3_n_bits(down->src[0]->type))) {
            const ggml_tensor * Wd = down->src[0];
            const int Kd = N;
            const int Nd = (int) down->ne[0];
            ggml_cuda_pool_alloc<half> A_d(ctx.pool(), n_inst * (size_t) Kd);
            ggml_cuda_pool_alloc<float> C_d(ctx.pool(), n_inst * (size_t) Nd);
            exl3_moe_swiglu_pre_down_cuda(
                C_g.get(), C_u.get(),
                (const float *) svh_g->data, (const float *) svh_u->data,
                (const float *) down->src[3]->data, A_d.get(),
                (const int32_t *) ids->data, ids_dst.get(),
                (int) n_inst, N, (int) n_expert_used, ids->nb[0], ids->nb[1], stream);
            exl3_gemm_cuda_id(
                A_d.get(), (const uint16_t *) Wd->data, C_d.get(),
                Kd, Nd, ggml_exl3_n_bits(Wd->type), n_m_max, n_experts,
                nullptr, n_sms, expert_bounds.get(), Wd->nb[2],
                ids_dst.get(), (const float *) down->src[4]->data, (float *) down->data, stream);
        } else {
            exl3_moe_post_had_swiglu_cuda(
                C_g.get(), C_u.get(),
                (const float *) svh_g->data, (const float *) svh_u->data, (float *) glu->data,
                (const int32_t *) ids->data, ids_dst.get(),
                (int) n_inst, N, (int) n_expert_used, ids->nb[0], ids->nb[1], stream);
            if (down) {
                id.bounds = expert_bounds.get();
                id.ids_src = ids_src1.get();
                id.ids_dst = ids_dst.get();
                id.n_experts = n_experts;
                run_down_gemv();
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    id.bounds = expert_bounds.get();
    id.ids_src = ids_src1.get();
    id.ids_dst = ids_dst.get();
    id.n_experts = n_experts;
    id.expert_stride = Wg->nb[2];
    exl3_gemv_cuda_fused_id(
        (const float *) src1->data, (const float *) suh_g->data, (const uint16_t *) Wg->data,
        (float *) gate->data, EXL3_GEMV_MAX_M, K, N, bits_g, stream, (const float *) svh_g->data, packed_n, id);
    id.expert_stride = Wu->nb[2];
    exl3_gemv_cuda_fused_id(
        (const float *) src1->data, (const float *) suh_u->data, (const uint16_t *) Wu->data,
        (float *) up->data, EXL3_GEMV_MAX_M, K, N, bits_u, stream, (const float *) svh_u->data, packed_n, id);
    exl3_swiglu_f32_cuda((const float *) gate->data, (const float *) up->data,
        (float *) glu->data, n_inst * (int64_t) N, stream);
    run_down_gemv();
    CUDA_CHECK(cudaGetLastError());
}
