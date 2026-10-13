#ifndef GGML_SYCL_XE2MX_HPP
#define GGML_SYCL_XE2MX_HPP

#include "common.hpp"

void ggml_sycl_mul_mat_xe2mx(ggml_backend_sycl_context & ctx,
                              const ggml_tensor * src0,
                              const ggml_tensor * src1,
                              ggml_tensor * dst);

// Activation scratch is one slot. A second SYCL device must not reuse it.
void xe2mx_scratch_bind(int device);

void ggml_sycl_xe2mx_graph_begin();
// Size MG defer slots and drop host LO packs before SYCL begin_recording.
// The waits in mg_run are alloc-only; they are illegal on a recording queue.
void ggml_sycl_xe2mx_graph_record_prepare(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);
void ggml_sycl_xe2mx_qbatch_prepare(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph);

bool ggml_sycl_xe2mx_mul_mat_glu(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_s4gmlp_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

bool ggml_sycl_xe2mx_mul_mat_add(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_mul_mat_silu_mul(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_ovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_f2_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_mg_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);
// GGML_SYCL_XE2MX_BATCH=1 (default off): batch small-M XE2MX4 MUL_MATs with different src1 into one launch.
int ggml_sycl_xe2mx_batch_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);
int ggml_sycl_xe2mx_fn_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

int ggml_sycl_xe2mx_s4ovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

// Dual-queue LO s4.s4 gems that share the same A (gate||up). Not p0_u4.
int ggml_sycl_xe2mx_loovl_run(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx);

bool ggml_sycl_q8_xmx(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                       ggml_tensor * dst);

bool ggml_sycl_xe2mx_s8dw_on(const ggml_tensor * t);
size_t ggml_sycl_xe2mx_s8dw_nbytes(const ggml_tensor * t);
bool ggml_sycl_xe2mx_s8dw_upload(ggml_tensor * t, const void * host, size_t size, sycl::queue & q);

bool ggml_sycl_xe2mx_lo_on(const ggml_tensor * t);
size_t ggml_sycl_xe2mx_lo_nbytes(const ggml_tensor * t);
bool ggml_sycl_xe2mx_lo_upload(ggml_tensor * t, const void * host, size_t size, sycl::queue & q, bool force = false);
void * ggml_sycl_xe2mx_lo_packed(const void * key);
// Drop the device LO pack once every host mirror exists. Prefill re-uploads it.
void ggml_sycl_xe2mx_lo_hostpack_release(sycl::queue & q);
void ggml_sycl_xe2mx_lo_rekey(const void * oldk, const void * newk);
// true when t->data holds original/SOA XE2MX4 bytes (dual-store default).
// false when LO packed s4 lives in t->data (LO_INLINE) - callers must not dequant as block_xe2mx4.
bool ggml_sycl_xe2mx_lo_bytes_are_soa(const ggml_tensor * t);
// true when this weight is LO and t->data is the packed s4+scale blob (no separate SOA copy).
bool ggml_sycl_xe2mx_lo_inline_on(const ggml_tensor * t);

#endif
