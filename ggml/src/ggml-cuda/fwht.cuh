#include "common.cuh"

// src_data overrides src->data when a fused MUL wrote that buffer. scale_* may be null.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                       const ggml_tensor * scale_in = nullptr, const ggml_tensor * scale_out = nullptr,
                       const void * src_data = nullptr);
