#pragma once

#include "common.cuh"

#define EXL3_GEMV_MAX_M 8

// fuse_x / fuse_suh: F32 activations and suh for the fused input path (suh*x + H128 inside gemv).
// fuse_svh: output H128 * svh (INT8 epilogue, or a cheap in-place kernel after fp16 gemv).
void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const ggml_tensor * fuse_x = nullptr, const ggml_tensor * fuse_suh = nullptr,
                            const ggml_tensor * fuse_svh = nullptr);

void ggml_cuda_mul_mat_exl3_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_mul_mat_exl3_id_glu(ggml_backend_cuda_context & ctx,
        ggml_tensor * gate, ggml_tensor * up, ggml_tensor * glu, ggml_tensor * down = nullptr);
