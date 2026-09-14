#ifndef GGML_SYCL_FWHT_HPP
#define GGML_SYCL_FWHT_HPP

#include "common.hpp"

// Fast Walsh-Hadamard transform, the fast path for a MUL_MAT whose src0 ggml has
// tagged GGML_HINT_SRC0_IS_HADAMARD. src0 is not read at all. Returns false if the
// shape is not one this can serve, in which case the caller must fall through to the
// ordinary mat-mul dispatch.
bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                       const ggml_tensor * scale_in = nullptr, const ggml_tensor * scale_out = nullptr,
                       const void * src_data = nullptr);

// MUL(x, suh) + H128 or H128 + MUL(y, svh). Returns nodes consumed, or 0.
int ggml_sycl_try_fwht_scale_fuse(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx);

#endif  // GGML_SYCL_FWHT_HPP
