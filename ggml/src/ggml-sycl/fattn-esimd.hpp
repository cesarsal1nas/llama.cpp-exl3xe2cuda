#ifndef GGML_SYCL_FATTN_ESIMD_HPP
#define GGML_SYCL_FATTN_ESIMD_HPP

#include "common.hpp"

// Decode flash attention (1..4 query tokens) for q4_0 K/V caches with grouped-query attention,
// written in ESIMD for Xe2. One work-group per (kv head, key split); the work-group holds all
// query heads that share the kv head (RQ = gqa ratio rows) so every K/V byte is read once per
// token, not once per query head. Partials (unnormalized O, running max, running sum) go to the
// same buffers/layout the generic flash_attn_combine_results kernel expects.
bool ggml_sycl_flash_attn_ext_esimd_supported(const ggml_tensor * dst);
void ggml_sycl_flash_attn_ext_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
bool ggml_sycl_flash_attn_ext_esimd_prefill_dpas_on();
void ggml_sycl_flash_attn_ext_esimd_prefill_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                                 const int8_t * q8, const float * qc, float * parts,
                                                 sycl::float2 * meta);
void ggml_sycl_flash_attn_ext_esimd_prefill_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_ESIMD_HPP
