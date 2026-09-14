#ifndef GGML_SYCL_EXL3_HPP
#define GGML_SYCL_EXL3_HPP

#include "common.hpp"

#ifndef EXL3_GEMV_MAX_M
#define EXL3_GEMV_MAX_M 8
#endif

#ifndef EXL3_RECON_HAD_MIN_M
#define EXL3_RECON_HAD_MIN_M 192
#endif

bool ggml_sycl_exl3_available(int device);
bool ggml_sycl_exl3_shape_ok(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_sycl_mul_mat_exl3(ggml_backend_sycl_context & ctx,
                            const ggml_tensor * src0,
                            const ggml_tensor * src1,
                            ggml_tensor * dst,
                            const ggml_tensor * fuse_x = nullptr,
                            const ggml_tensor * fuse_suh = nullptr,
                            const ggml_tensor * fuse_svh = nullptr,
                            const sycl::half * x_pre = nullptr,
                            float * c_raw = nullptr);

// Returns nodes to skip (0 = no fuse). Match at GGML_OP_MUL.
int ggml_sycl_try_exl3_fuse(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

// Next PP EXL3 mul_mat, or null. Consumed by ggml_sycl_mul_mat_exl3.
void ggml_sycl_exl3_set_lookahead(const ggml_tensor * next_mm);

#endif
