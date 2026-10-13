#ifndef GGML_SYCL_XE2MX_A8_HPP
#define GGML_SYCL_XE2MX_A8_HPP

#include "common.hpp"

bool ggml_sycl_xe2mx_prefill(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M);

bool ggml_sycl_xe2mx_prefill_n8(ggml_backend_sycl_context & ctx,
                                 const uint8_t * packed,
                                 const int8_t * A,
                                 const float * da,
                                 float * C,
                                 int64_t K,
                                 int64_t N,
                                 int64_t M);

bool ggml_sycl_xe2mx_ksplit(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M);

bool ggml_sycl_xe2mx_xmx(ggml_backend_sycl_context & ctx,
                          const uint8_t * packed,
                          const int8_t * A,
                          const float * da,
                          float * C,
                          int64_t K,
                          int64_t N,
                          int64_t M);

// Up to 4 XE2MX4 weights sharing one quantized A (same K), one launch. Rows match ggml_sycl_xe2mx_xmx per weight.
// N is the packed weight width (multiple of 16). Ndst is the stored width and the
// dst row stride. Pass Ndst null when every member stores all N columns.
bool ggml_sycl_xe2mx_xmx_mg(ggml_backend_sycl_context & ctx,
                            const uint8_t * const * vx,
                            float * const * C,
                            const int64_t * N,
                            const int64_t * Ndst,
                            int nw,
                            const int8_t * A,
                            const float * da,
                            int64_t K,
                            int64_t M);

// GGML_SYCL_XE2MX_BATCH: up to XE2MX_BATCH_MAX independent XE2MX4 matmuls in one launch. Unlike _mg the
// entries do not share A: each carries its own quantized A/da and K/N. Every entry runs the launch_soa body
// (same dpas tile, xorm, store mask, n_row_groups = N/16) on its own workgroup range. All entries must have
// the same M (2..31, picks the tile like ggml_sycl_xe2mx_xmx). N multiple of 16, Ndst == N.
constexpr int XE2MX_BATCH_MAX = 16;
struct xe2mx_batch_ent {
    const uint8_t * vx;
    const int8_t *  A;
    const float *   da;
    float *         C;
    int64_t         K;
    int64_t         N;
    int64_t         M;
};

bool ggml_sycl_xe2mx_xmx_batch(ggml_backend_sycl_context & ctx, const xe2mx_batch_ent * ents, int nw);

bool ggml_sycl_xe2mx_xmx_glu(ggml_backend_sycl_context & ctx,
                               const uint8_t * pg,
                               const uint8_t * pu,
                               const int8_t * A,
                               const float * da,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M);

bool ggml_sycl_xe2mx_xmx_pair(ggml_backend_sycl_context & ctx,
                                const uint8_t * p0,
                                const uint8_t * p1,
                                const int8_t * A,
                                const float * da,
                                float * C0,
                                float * C1,
                                int64_t K,
                                int64_t N,
                                int64_t M);

bool ggml_sycl_xe2mx_xmx_fn(ggml_backend_sycl_context & ctx,
                             const uint8_t * p0,
                             const uint8_t * p1,
                             const int8_t * A,
                             const float * da,
                             float * C0,
                             float * C1,
                             int64_t K,
                             int64_t N0,
                             int64_t N1,
                             int64_t M);

bool ggml_sycl_xe2mx_xmx2(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M);

bool ggml_sycl_xe2mx_xmx1(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M);

bool ggml_sycl_xe2mx_xmx8(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M);

bool ggml_sycl_xe2mx_xmx_nt(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M);

bool ggml_sycl_xe2mx_blk(ggml_backend_sycl_context & ctx,
                          const uint8_t * packed,
                          const int8_t * A,
                          const float * da,
                          float * C,
                          int64_t K,
                          int64_t N,
                          int64_t M);

bool ggml_sycl_xe2mx_wg(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M);

bool ggml_sycl_xe2mx_isa(ggml_backend_sycl_context & ctx,
                           const uint8_t * packed,
                           const int8_t * A,
                           const float * da,
                           float * C,
                           int64_t K,
                           int64_t N,
                           int64_t M);

bool ggml_sycl_xe2mx_isa_h(ggml_backend_sycl_context & ctx,
                              const uint8_t * packed,
                              const sycl::half * xh,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M);

bool ggml_sycl_xe2mx_isa_hf(ggml_backend_sycl_context & ctx,
                             const uint8_t * packed,
                             const sycl::half * xh,
                             float * C,
                             int64_t K,
                             int64_t N,
                             int64_t M);

bool ggml_sycl_xe2mx_xmx_f32(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * x,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M);

bool ggml_sycl_xe2mx_gemv_f32(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * x,
                               float * C,
                               int64_t K,
                               int64_t N);

bool ggml_sycl_xe2mx_gemm_s8(ggml_backend_sycl_context & ctx,
                              const int8_t * W8,
                              const uint16_t * d16,
                              const int8_t * A,
                              const float * da,
                              float * C,
                              int64_t K,
                              int64_t N,
                              int64_t M,
                              int64_t pitch);

bool ggml_sycl_xe2mx_gemm_s8_vnni(ggml_backend_sycl_context & ctx,
                                   const int8_t * vnni,
                                   const uint16_t * d16,
                                   const int8_t * A,
                                   const float * da,
                                   float * C,
                                   int64_t K,
                                   int64_t N,
                                   int64_t M);

bool ggml_sycl_xe2mx_gemm_s4t(ggml_backend_sycl_context & ctx,
                               const uint8_t * packed,
                               const float * scw,
                               const int8_t * A,
                               const float * sa,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M,
                               const float * glu_up);

bool ggml_sycl_xe2mx_gemm_s4t_glu(ggml_backend_sycl_context & ctx,
                                   const uint8_t * pg,
                                   const float * scg,
                                   const uint8_t * pu,
                                   const float * scu,
                                   const int8_t * A,
                                   const float * sa,
                                   float * C,
                                   int64_t K,
                                   int64_t N,
                                   int64_t M);

bool ggml_sycl_xe2mx_gemm_q8(ggml_backend_sycl_context & ctx,
                               const void * q8,
                               const int8_t * A,
                               const float * da,
                               float * C,
                               int64_t K,
                               int64_t N,
                               int64_t M);

bool xe2mx_quant_a_s4_nibble(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                             int64_t M, int64_t K, int64_t g, float * za = nullptr);

bool xe2mx_pack_hf_s4(queue_ptr stream, const sycl::half * xh, const float * sa, uint8_t * A4,
                      int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s4_nibble_had(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s4_nibble_had_off(queue_ptr stream, const sycl::half * xh, uint8_t * A4, float * sa,
                                     int64_t M, int64_t lda, int64_t koff, int64_t klen, int64_t g);

bool xe2mx_quant_a_s4_nibble_had_f32(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                     int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s4_nibble_had_f32_rte(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                         int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s4_nibble_had_f32_rte_off(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                             int64_t M, int64_t lda, int64_t koff, int64_t klen, int64_t g);

bool xe2mx_quant_a_s8_had(queue_ptr stream, const sycl::half * xh, uint8_t * A8, float * sa,
                          int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s8_had_f32_rte(queue_ptr stream, const float * x, uint8_t * A8, float * sa,
                                  int64_t M, int64_t K, int64_t g);

bool xe2mx_quant_a_s4_nibble_f32(queue_ptr stream, const float * x, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g, float * za = nullptr);

bool xe2mx_rms_mul_hilp_pack(queue_ptr stream, const float * x, const float * mul, float * y,
                             uint8_t * A4, float * sa, int64_t M, int64_t K, int64_t g, float eps);

bool xe2mx_add_rms_mul_hilp_pack(queue_ptr stream, const float * a, const float * b, float * add_dst,
                                 const float * mul, float * y, uint8_t * A4, float * sa,
                                 int64_t M, int64_t K, int64_t g, float eps, int drop_add);

bool xe2mx_silu_up_hilp_pack(queue_ptr stream, const float * gate, const float * up, float * y,
                             uint8_t * A4, float * sa, int64_t M, int64_t K, int64_t g);

void xe2mx_s44_quant_w_gptq(queue_ptr stream, float * src, int8_t * w, float * sc, float * bsum,
                            int64_t K, int64_t N, int64_t g);

bool ggml_sycl_xe2mx_lo_s44(ggml_backend_sycl_context & ctx,
                             const uint8_t * a4,
                             const float * sa,
                             const uint8_t * w4,
                             const float * scw,
                             float * C,
                             int64_t K,
                             int64_t N,
                             int64_t M,
                             int64_t g);

bool ggml_sycl_xe2mx_lo_hf_s44(ggml_backend_sycl_context & ctx,
                                const sycl::half * xh,
                                const float * sa,
                                const uint8_t * w4,
                                const float * scw,
                                float * C,
                                int64_t K,
                                int64_t N,
                                int64_t M,
                                int64_t g);

bool ggml_sycl_xe2mx_s44_pack_xf(queue_ptr stream, const void * asrc, const float * xf,
                                 int64_t M, int64_t K);

bool ggml_sycl_xe2mx_s44_rms_hilp(queue_ptr stream, const void * asrc, const float * x, const float * mul,
                                  float * y, int64_t M, int64_t K, float eps);

bool ggml_sycl_xe2mx_s44_add_rms_hilp(queue_ptr stream, const void * asrc, const float * a, const float * b,
                                      float * add_dst, const float * mul, float * y, int64_t M, int64_t K,
                                      float eps, int drop_add);

bool ggml_sycl_xe2mx_s44_add_rms_bufs(queue_ptr stream, const void * asrc, const float * y, int64_t M, int64_t K,
                                      uint8_t ** A4, float ** sa, int * gout = nullptr);

bool ggml_sycl_xe2mx_s44_silu_hilp(queue_ptr stream, const void * asrc, const float * gate, const float * up,
                                   float * y, int64_t M, int64_t K);

bool ggml_sycl_xe2mx_lo_s44_glu(ggml_backend_sycl_context & ctx,
                                 const uint8_t * a4,
                                 const float * sa,
                                 const uint8_t * wg,
                                 const float * scg,
                                 const uint8_t * wu,
                                 const float * scu,
                                 float * C,
                                 int64_t K,
                                 int64_t N,
                                 int64_t M,
                                 int64_t g);

#endif
