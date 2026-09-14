#include "dmmv.hpp"
#include "panel-xmx.hpp"

#include <cstdio>

#if defined(__INTEL_LLVM_COMPILER)
    #include <sycl/ext/intel/experimental/grf_size_properties.hpp>
    #define GGML_SYCL_PANEL_XMX_HAS_ESIMD
#endif

bool ggml_sycl_panel_f16_xmx_gemm(const sycl::half * W, const sycl::half * X, float * dst, int M, int N, int K, int ldc,
                                  dpct::queue_ptr stream) {
#ifdef GGML_SYCL_PANEL_XMX_HAS_ESIMD
    if (W == nullptr || X == nullptr || dst == nullptr || stream == nullptr) {
        return false;
    }
    if (M <= 0 || N <= 0 || K <= 0 || (K % 16) != 0 || (M % 8) != 0) {
        return false;
    }
    constexpr int RM = 8;
    constexpr int RN = 16;
    constexpr int WG = 16;
    const int N_aln = (N / RN) * RN;
    if (N_aln <= 0) {
        return false;
    }
    try {
        const int tiles = (M / RM) * (N_aln / RN);
        const int gsz   = ((tiles + WG - 1) / WG) * WG;
        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) gsz), sycl::range<1>(WG)),
            sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> },
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                int tile = (int) it.get_global_id(0);
                tile     = tile < tiles ? tile : (tiles - 1);
                ggml_sycl_esimd::panel_f16_xmx_tile(W, X, dst, M, N_aln, K, ldc, tile);
            });
        stream->wait();
    } catch (const sycl::exception & e) {
        fprintf(stderr, "PANEL_XMX sycl M=%d N=%d K=%d ldc=%d: %s\n", M, N, K, ldc, e.what());
        fflush(stderr);
        return false;
    }
    return true;
#else
    GGML_UNUSED(W); GGML_UNUSED(X); GGML_UNUSED(dst); GGML_UNUSED(M); GGML_UNUSED(N);
    GGML_UNUSED(K); GGML_UNUSED(ldc); GGML_UNUSED(stream);
    return false;
#endif
}
