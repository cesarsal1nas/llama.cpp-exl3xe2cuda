#ifndef GGML_SYCL_FUSION_HPP
#define GGML_SYCL_FUSION_HPP

#include <initializer_list>

#include "common.hpp"

// Backend-side fusability test. `ops` names a candidate op sequence starting at cgraph node
// `node_idx`, and `unary_ops` the GGML_UNARY_OP each GGML_OP_UNARY in `ops` must carry, in
// order; the result is true only if ggml considers that subgraph fusable *and* the SYCL
// kernel which would service it accepts the tensors involved (types, shapes, contiguity).
//
// Lives in its own translation unit because it grows a branch per supported op sequence.
bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops);

bool ggml_sycl_is_gdn_conv_concat(const ggml_tensor * concat);
bool ggml_sycl_ssm_conv_can_split(const ggml_tensor * ssm);

#endif  // GGML_SYCL_FUSION_HPP
