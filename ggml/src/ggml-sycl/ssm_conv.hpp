#pragma once

#include "common.hpp"

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr);
void ggml_sycl_conv_concat_fuse_begin(void);
bool ggml_sycl_try_conv_concat_fuse(ggml_backend_sycl_context & ctx, struct ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_ssm_conv_skip_done(struct ggml_cgraph * cgraph, int node_idx);
