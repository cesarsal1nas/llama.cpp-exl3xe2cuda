#pragma once
// Prefill IQ4 panel GEMM on XMX fp16 dpas. GGML_SYCL_IQ4_PANEL_XMX=1.
// Probe-proven on B65: identity 8x16x16, VNNI B[(k/2)*32+n*2+(k&1)], copy_from, WG=16.
// No ESIMD control-flow: M%8==0 and N%16==0 required. Remainder N stays on oneDNN.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace ggml_sycl_esimd {

ESIMD_INLINE void panel_f16_xmx_tile(const sycl::half * W, const sycl::half * X, float * C, int M, int N, int K,
                                     int ldc, int gid) {
    using namespace sycl::ext::intel::esimd;
    using namespace sycl::ext::intel::esimd::xmx;
    constexpr int RM = 8;
    constexpr int RN = 16;
    constexpr int RK = 16;
    const int ntn = N / RN;
    const int tm  = gid / ntn;
    const int tn  = gid - tm * ntn;
    const int m0  = tm * RM;
    const int n0  = tn * RN;
    simd<float, RM * RN> acc(0.f);
    for (int k0 = 0; k0 < K; k0 += RK) {
        simd<sycl::half, RM * RK> A;
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            simd<sycl::half, RK> row;
            row.copy_from(W + (size_t) (m0 + r) * (size_t) K + k0);
            A.template select<RK, 1>(r * RK) = row;
        }
        simd<sycl::half, RK * RN> B;
#pragma unroll
        for (int n = 0; n < RN; ++n) {
            simd<sycl::half, RK> row;
            row.copy_from(X + (size_t) (n0 + n) * (size_t) K + k0);
#pragma unroll
            for (int k = 0; k < RK; ++k) {
                B[(k / 2) * (RN * 2) + n * 2 + (k & 1)] = row[k];
            }
        }
        acc = dpas<8, RM, float, float, sycl::half, sycl::half>(acc, B, A);
    }
#pragma unroll
    for (int r = 0; r < RM; ++r) {
#pragma unroll
        for (int n = 0; n < RN; ++n) {
            C[(size_t) (m0 + r) + (size_t) (n0 + n) * (size_t) ldc] = acc[r * RN + n];
        }
    }
    (void) M;
}

} // namespace ggml_sycl_esimd
