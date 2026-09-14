#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale * x[i] + bias;
    }
}

static __global__ void scale_f32_vec4(const float * x, float * dst, const float scale, const float bias, const int64_t nvec) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nvec; i += stride) {
        float4 v = reinterpret_cast<const float4 *>(x)[i];
        v.x = scale * v.x + bias;
        v.y = scale * v.y + bias;
        v.z = scale * v.z + bias;
        v.w = scale * v.w + bias;
        reinterpret_cast<float4 *>(dst)[i] = v;
    }
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    if (nelements >= 4 && (nelements & 3) == 0 &&
        ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
        const int64_t nvec = nelements / 4;
        const int64_t num_blocks = (nvec + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(scale_f32_vec4, launch_params, x, dst, scale, bias, nvec);
        return;
    }

    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
}
