#include "common.cuh"
#include "fwht.cuh"

#include <type_traits>

template <int N, typename src_t>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const src_t * src, float * dst, const int64_t n_rows, const float scale,
                          const float * scale_in, const int64_t k_in, const int64_t s_in,
                          const float * scale_out, const int64_t k_out, const int64_t s_out) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int     col = i * warp_size + lane;
        const int64_t e   = r * (int64_t) N + col;
        float         v   = (float) src[col] * scale;
        if (scale_in) {
            v *= scale_in[(e % k_in) * s_in];
        }
        reg[i] = v;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int     col = i * warp_size + lane;
        float         v   = reg[i];
        if (scale_out) {
            const int64_t e = r * (int64_t) N + col;
            v *= scale_out[(e % k_out) * s_out];
        }
        dst[col] = v;
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                       const ggml_tensor * scale_in, const ggml_tensor * scale_out, const void * src_data) {
    if (ggml_nelements(src) != ggml_nelements(dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const void *  src_d = src_data ? src_data : src->data;
    float *       dst_d = (float *) dst->data;

    const float * scale_in_d  = nullptr;
    const float * scale_out_d = nullptr;
    int64_t       k_in        = 0;
    int64_t       s_in        = 0;
    int64_t       k_out       = 0;
    int64_t       s_out       = 0;

    if (scale_in) {
        GGML_ASSERT(scale_in->type == GGML_TYPE_F32);
        GGML_ASSERT(scale_in->ne[0] > 0);
        scale_in_d = (const float *) scale_in->data;
        k_in       = scale_in->ne[0];
        s_in       = scale_in->nb[0] / sizeof(float);
        GGML_ASSERT(s_in >= 1);
    }
    if (scale_out) {
        GGML_ASSERT(scale_out->type == GGML_TYPE_F32);
        GGML_ASSERT(scale_out->ne[0] > 0);
        scale_out_d = (const float *) scale_out->data;
        k_out       = scale_out->ne[0];
        s_out       = scale_out->nb[0] / sizeof(float);
        GGML_ASSERT(s_out >= 1);
    }

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    auto launch = [&](auto src_typed) {
        using src_t = std::remove_const_t<std::remove_pointer_t<decltype(src_typed)>>;
        switch (n) {
            case 64:
                ggml_cuda_kernel_launch(fwht_cuda<64, src_t>, launch_params, src_typed, dst_d, rows, scale,
                                        scale_in_d, k_in, s_in, scale_out_d, k_out, s_out);
                return true;
            case 128:
                ggml_cuda_kernel_launch(fwht_cuda<128, src_t>, launch_params, src_typed, dst_d, rows, scale,
                                        scale_in_d, k_in, s_in, scale_out_d, k_out, s_out);
                return true;
            case 256:
                ggml_cuda_kernel_launch(fwht_cuda<256, src_t>, launch_params, src_typed, dst_d, rows, scale,
                                        scale_in_d, k_in, s_in, scale_out_d, k_out, s_out);
                return true;
            case 512:
                ggml_cuda_kernel_launch(fwht_cuda<512, src_t>, launch_params, src_typed, dst_d, rows, scale,
                                        scale_in_d, k_in, s_in, scale_out_d, k_out, s_out);
                return true;
            default:
                return false;
        }
    };

    if (src->type == GGML_TYPE_F16) {
        return launch((const half *) src_d);
    }
    return launch((const float *) src_d);
}
