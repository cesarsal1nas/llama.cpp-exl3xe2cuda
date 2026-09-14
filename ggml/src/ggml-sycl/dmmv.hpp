//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"


// IQ4_XS x q8_1(SoA) matvec for ncols_dst == 1 on the ESIMD path. Returns false
// (nothing launched) when ESIMD is unavailable/disabled or the shape does not fit.
// true when the ESIMD IQ4_XS decode kernel is compiled in and enabled
// (GGML_SYCL_ENABLE_ESIMD and GGML_SYCL_IQ4_ESIMD, both default 1)
bool ggml_sycl_iq4_esimd_available();

// vy must be quantized with quantize_and_reorder_q8_1_soa_iq4pair (both).
// glu: dst = silu(gate . y) * (up . y), SWIGLU only (false = declined, nothing launched).
bool ggml_sycl_mul_mat_vec_iq4_xs_q8_1_glu_esimd(enum ggml_glu_op glu_op, const void * vup, const void * vgate,
                                                 const void * vy, float * dst, int ncols, int nrows,
                                                 dpct::queue_ptr stream);
// ncols_y = 1..4 y columns (MTP verification batches); column c of vy at
// (char *) vy + c * stride_y_bytes, dst column c at dst + c * stride_dst.
bool ggml_sycl_mul_mat_vec_iq4_xs_q8_1_esimd(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                             int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);
// One launch over nmat=2..3 IQ4_XS GEMVs that share y (sibling z/β/α). vx[i]/dst[i]/nrows[i]/stride_dst[i].
bool ggml_sycl_mul_mat_vec_iq4_xs_q8_1_esimd_sib(const void * const * vx, float * const * dst, const int * nrows,
                                                 int nmat, const void * vy, int ncols, int ncols_y, int stride_y_bytes,
                                                 const int * stride_dst, dpct::queue_ptr stream);
bool ggml_sycl_mul_mat_vec_iq4_xs_q8_1_glu_esimd(enum ggml_glu_op glu_op, const void * vup, const void * vgate,
                                                 const void * vy, float * dst, int ncols, int nrows,
                                                 int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);
// largest ncols_y the ESIMD IQ4_XS / Q4_K / Q5_K q8_1 kernels take
constexpr int GGML_SYCL_IQ4_ESIMD_MAX_NCOLS = 4;

// ESIMD IQ4_XS x q8_1 dpas (XMX) kernel; vy must be quantized with
// quantize_and_reorder_q8_1_soa_iq4nat. Taken for the shapes ggml_sycl_iq4_dpas_shape_ok
// admits (ncols_y == 4, long rows); false = declined, nothing launched.
bool ggml_sycl_iq4_dpas_available();
bool ggml_sycl_iq4_dpas_shape_ok(int ncols, int nrows, int ncols_y);
bool ggml_sycl_mul_mat_vec_iq4_xs_q8_1_dpas(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                            int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);

// Prefill f16 XMX dpas GEMM after IQ4 panel dequant. GGML_SYCL_IQ4_PANEL_XMX.
bool ggml_sycl_panel_f16_xmx_gemm(const sycl::half * W, const sycl::half * X, float * dst, int M, int N, int K, int ldc,
                                  dpct::queue_ptr stream);

// ESIMD Q4_K x q8_1 dp4a kernel, 1..4 y columns; vy must be quantized with
// quantize_and_reorder_q8_1_soa_q5k_slots. Lab opt-in (GGML_SYCL_Q4K_ESIMD).
bool ggml_sycl_q4k_esimd_available();
bool ggml_sycl_mul_mat_vec_q4_K_q8_1_esimd(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                           int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);
// One launch over nmat=2 Q4_K GEMVs that share y (GDN β‖α, nrows==48). Default off.
bool ggml_sycl_mul_mat_vec_q4_K_q8_1_esimd_sib(const void * const * vx, float * const * dst, const int * nrows,
                                               int nmat, const void * vy, int ncols, int ncols_y, int stride_y_bytes,
                                               const int * stride_dst, dpct::queue_ptr stream);
// Fused up+gate+SWIGLU. vy = quantize_and_reorder_q8_1_soa_q5k_slots. false = declined.
bool ggml_sycl_mul_mat_vec_q4_K_q8_1_glu_esimd(enum ggml_glu_op glu_op, const void * vup, const void * vgate,
                                               const void * vy, float * dst, int ncols, int nrows, int ncols_y,
                                               int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);

// ESIMD Q5_K x q8_1 dp4a kernel, 1..4 y columns; vy must be quantized with
// quantize_and_reorder_q8_1_soa_q5k_slots. false = declined, nothing launched.
bool ggml_sycl_q5k_esimd_available();
bool ggml_sycl_mul_mat_vec_q5_K_q8_1_esimd(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                           int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);

// ESIMD Q6_K x q8_1 dp4a kernel, 1..4 y columns; vy is quantize_and_reorder_q8_1_soa
// (natural sub-block order). false = declined, nothing launched.
bool ggml_sycl_q6k_esimd_available();
bool ggml_sycl_mul_mat_vec_q6_K_q8_1_esimd(const void * vx, const void * vy, float * dst, int ncols, int nrows,
                                           int ncols_y, int stride_y_bytes, int stride_dst, dpct::queue_ptr stream);

void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

#endif // GGML_SYCL_DMMV_HPP
