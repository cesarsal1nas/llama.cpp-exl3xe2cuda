#pragma once

#include "common.hpp"

void ggml_sycl_ssm_scan(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// Lab SSD. False means the sequential scan must run.
bool ggml_sycl_ssm_ssd(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
        const float * pre_dt, const float * pre_da);
