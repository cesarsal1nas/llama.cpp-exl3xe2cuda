#include "fwht.hpp"

#include <cmath>
#if defined(__INTEL_LLVM_COMPILER)
#include <sycl/ext/intel/esimd.hpp>
#endif

extern int g_ggml_sycl_enable_esimd;
#define P 1.0f
#define N -1.0f

// constant Hadamard matrix via Paley I construction
static constexpr float H12[12][12] = {
    { P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, P, P, P, N, N, N, P, N },
    { P, N, N, P, N, P, P, P, N, N, N, P },
    { P, P, N, N, P, N, P, P, P, N, N, N },
    { P, N, P, N, N, P, N, P, P, P, N, N },
    { P, N, N, P, N, N, P, N, P, P, P, N },
    { P, N, N, N, P, N, N, P, N, P, P, P },
    { P, P, N, N, N, P, N, N, P, N, P, P },
    { P, P, P, N, N, N, P, N, N, P, N, P },
    { P, P, P, P, N, N, N, P, N, N, P, N },
    { P, N, P, P, P, N, N, N, P, N, N, P },
    { P, P, N, P, P, P, N, N, N, P, N, N }
};

static constexpr float H20[20][20] = {
    { P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P },
    { P, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N },
    { P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P },
    { P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P },
    { P, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N },
    { P, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N, N },
    { P, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N, N },
    { P, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P, N },
    { P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N, P },
    { P, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P, N },
    { P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N, P },
    { P, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P, N },
    { P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P, P },
    { P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P, P },
    { P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P, P },
    { P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N, P },
    { P, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N, N },
    { P, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P, N },
    { P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N, P },
    { P, P, N, N, P, P, P, P, N, P, N, P, N, N, N, N, P, P, N, N }
};

#undef P
#undef N

template <int N>
static void fwht_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                        const float scale, const sycl::nd_item<2> & item) {
    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int el_w = N / WARP_SIZE;
    static_assert(el_w >= 1 && N % WARP_SIZE == 0, "row must be a whole number of sub-group widths");

    float     reg[el_w];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * WARP_SIZE + lane] * scale;
    }

    // Butterflies inside the sub-group. The partner of a lane with bit h clear is the
    // lower index of the pair, so it takes the sum and the upper takes lower - upper.
#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // Butterflies across registers: h is a multiple of WARP_SIZE, so the partner of
    // element i*WARP_SIZE + lane lives in reg[i + h/WARP_SIZE] on the same lane.
#pragma unroll
    for (int h = WARP_SIZE; h < N; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * WARP_SIZE + lane] = reg[i];
    }
}

template <int N>
static void launch_fwht(const float * src, float * dst, const int64_t n_rows, const float scale,
                        dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             fwht_kernel<N>(src, dst, n_rows, scale, item);
                         });
}

// EXL3 H128: two rows/thread, 16 rows/WG, native xor permute. Same butterflies as fwht_kernel<128>.
#if WARP_SIZE == 16
static void fwht128_pair_kernel(const float * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
                                const float scale, const float * sin, const int64_t kin, const float * sout,
                                const int64_t kout, const sycl::nd_item<2> & item) {
    constexpr int SG = 16;
    constexpr int EL = 8;
    const sycl::sub_group sg = item.get_sub_group();
    const int lane = int(sg.get_local_linear_id());
    const int64_t r0 = int64_t(item.get_global_id(0)) * 2;
    const int64_t r1 = r0 + 1;

    float a[EL];
    float b[EL];
    if (r0 < n_rows) {
        const float * s = src + r0 * 128;
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            const int col = i * SG + lane;
            float v = s[col] * scale;
            if (sin) {
                v *= sin[(r0 * 128 + col) % kin];
            }
            a[i] = v;
        }
    } else {
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            a[i] = 0.f;
        }
    }
    if (r1 < n_rows) {
        const float * s = src + r1 * 128;
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            const int col = i * SG + lane;
            float v = s[col] * scale;
            if (sin) {
                v *= sin[(r1 * 128 + col) % kin];
            }
            b[i] = v;
        }
    } else {
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            b[i] = 0.f;
        }
    }

#pragma unroll
    for (int h = 1; h < SG; h *= 2) {
#pragma unroll
        for (int j = 0; j < EL; ++j) {
            const float a2 = dpct::permute_sub_group_by_xor(sg, a[j], h, SG);
            const float b2 = dpct::permute_sub_group_by_xor(sg, b[j], h, SG);
            a[j] = (lane & h) == 0 ? a[j] + a2 : a2 - a[j];
            b[j] = (lane & h) == 0 ? b[j] + b2 : b2 - b[j];
        }
    }
#pragma unroll
    for (int h = SG; h < 128; h *= 2) {
        const int step = h / SG;
#pragma unroll
        for (int j = 0; j < EL; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float ax = a[j + k];
                const float ay = a[j + k + step];
                const float bx = b[j + k];
                const float by = b[j + k + step];
                a[j + k] = ax + ay;
                a[j + k + step] = ax - ay;
                b[j + k] = bx + by;
                b[j + k + step] = bx - by;
            }
        }
    }

    if (r0 < n_rows) {
        float * d = dst + r0 * 128;
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            const int col = i * SG + lane;
            float v = a[i];
            if (sout) {
                v *= sout[(r0 * 128 + col) % kout];
            }
            d[col] = v;
        }
    }
    if (r1 < n_rows) {
        float * d = dst + r1 * 128;
#pragma unroll
        for (int i = 0; i < EL; ++i) {
            const int col = i * SG + lane;
            float v = b[i];
            if (sout) {
                v *= sout[(r1 * 128 + col) % kout];
            }
            d[col] = v;
        }
    }
}

#if defined(__INTEL_LLVM_COMPILER)
using namespace sycl::ext::intel::esimd;

ESIMD_INLINE void fwht16_reg(simd<float, 16> & x) {
    {
        const simd<float, 8> e = x.select<8, 2>(0);
        const simd<float, 8> o = x.select<8, 2>(1);
        x.select<8, 2>(0) = e + o;
        x.select<8, 2>(1) = e - o;
    }
    {
        simd<float, 8> lo, hi;
        lo.select<2, 1>(0) = x.select<2, 1>(0);
        lo.select<2, 1>(2) = x.select<2, 1>(4);
        lo.select<2, 1>(4) = x.select<2, 1>(8);
        lo.select<2, 1>(6) = x.select<2, 1>(12);
        hi.select<2, 1>(0) = x.select<2, 1>(2);
        hi.select<2, 1>(2) = x.select<2, 1>(6);
        hi.select<2, 1>(4) = x.select<2, 1>(10);
        hi.select<2, 1>(6) = x.select<2, 1>(14);
        simd<float, 8> s = lo + hi;
        simd<float, 8> d = lo - hi;
        x.select<2, 1>(0) = s.select<2, 1>(0);
        x.select<2, 1>(2) = d.select<2, 1>(0);
        x.select<2, 1>(4) = s.select<2, 1>(2);
        x.select<2, 1>(6) = d.select<2, 1>(2);
        x.select<2, 1>(8) = s.select<2, 1>(4);
        x.select<2, 1>(10) = d.select<2, 1>(4);
        x.select<2, 1>(12) = s.select<2, 1>(6);
        x.select<2, 1>(14) = d.select<2, 1>(6);
    }
    {
        simd<float, 8> lo, hi;
        lo.select<4, 1>(0) = x.select<4, 1>(0);
        lo.select<4, 1>(4) = x.select<4, 1>(8);
        hi.select<4, 1>(0) = x.select<4, 1>(4);
        hi.select<4, 1>(4) = x.select<4, 1>(12);
        simd<float, 8> s = lo + hi;
        simd<float, 8> d = lo - hi;
        x.select<4, 1>(0) = s.select<4, 1>(0);
        x.select<4, 1>(4) = d.select<4, 1>(0);
        x.select<4, 1>(8) = s.select<4, 1>(4);
        x.select<4, 1>(12) = d.select<4, 1>(4);
    }
    {
        const simd<float, 8> lo = x.select<8, 1>(0);
        const simd<float, 8> hi = x.select<8, 1>(8);
        x.select<8, 1>(0) = lo + hi;
        x.select<8, 1>(8) = lo - hi;
    }
}

ESIMD_INLINE void fwht128_reg(simd<float, 16> v[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        fwht16_reg(v[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        const simd<float, 16> x = v[i];
        const simd<float, 16> y = v[i + 1];
        v[i] = x + y;
        v[i + 1] = x - y;
    }
#pragma unroll
    for (int i = 0; i < 8; i += 4) {
        const simd<float, 16> x0 = v[i];
        const simd<float, 16> y0 = v[i + 2];
        const simd<float, 16> x1 = v[i + 1];
        const simd<float, 16> y1 = v[i + 3];
        v[i] = x0 + y0;
        v[i + 2] = x0 - y0;
        v[i + 1] = x1 + y1;
        v[i + 3] = x1 - y1;
    }
    {
        const simd<float, 16> x0 = v[0];
        const simd<float, 16> y0 = v[4];
        const simd<float, 16> x1 = v[1];
        const simd<float, 16> y1 = v[5];
        const simd<float, 16> x2 = v[2];
        const simd<float, 16> y2 = v[6];
        const simd<float, 16> x3 = v[3];
        const simd<float, 16> y3 = v[7];
        v[0] = x0 + y0;
        v[4] = x0 - y0;
        v[1] = x1 + y1;
        v[5] = x1 - y1;
        v[2] = x2 + y2;
        v[6] = x2 - y2;
        v[3] = x3 + y3;
        v[7] = x3 - y3;
    }
}

static void launch_fwht128_esimd(const float * src, float * dst, const int64_t n_rows, const float scale,
                                 dpct::queue_ptr stream) {
    constexpr int ROWS = 2;
    const int64_t groups = (n_rows + ROWS - 1) / ROWS;
    stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(groups)), sycl::range<1>(1)),
                         [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                             const int64_t r0 = int64_t(it.get_group(0)) * ROWS;
#pragma unroll
                             for (int t = 0; t < ROWS; ++t) {
                                 const int64_t r = r0 + t;
                                 if (r >= n_rows) {
                                     continue;
                                 }
                                 simd<float, 16> v[8];
#pragma unroll
                                 for (int i = 0; i < 8; ++i) {
                                     v[i].copy_from(src + r * 128 + i * 16);
                                     v[i] = v[i] * scale;
                                 }
                                 fwht128_reg(v);
#pragma unroll
                                 for (int i = 0; i < 8; ++i) {
                                     v[i].copy_to(dst + r * 128 + i * 16);
                                 }
                             }
                         });
}
#endif

static void launch_fwht128(const float * src, float * dst, const int64_t n_rows, const float scale,
                           dpct::queue_ptr stream, const float * sin = nullptr, int64_t kin = 0,
                           const float * sout = nullptr, int64_t kout = 0) {
#if defined(__INTEL_LLVM_COMPILER)
    if (!sin && !sout && g_ggml_sycl_enable_esimd && ggml_sycl_get_env("GGML_SYCL_FWHT_ESIMD", 1) > 0) {
        launch_fwht128_esimd(src, dst, n_rows, scale, stream);
        return;
    }
#endif
    constexpr int SG = 16;
    constexpr int RPB = 16;
    const int64_t pairs = (n_rows + 1) / 2;
    const int64_t num_blocks = (pairs + RPB - 1) / RPB;
    const sycl::range<2> global(num_blocks * RPB, SG);
    const sycl::range<2> local(RPB, SG);
    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SG)]] {
                             fwht128_pair_kernel(src, dst, n_rows, scale, sin, kin, sout, kout, item);
                         });
}
#endif

template <int N, int m>
static void kronecker_kernel(const float * __restrict__ src,
                             float * __restrict__ dst,
                             const int64_t            n_rows,
                             const float              scale,
                             const sycl::nd_item<2> & item) {
    static_assert(m == 12 || m == 20, "block size has to be 12 or 20.");

    const sycl::sub_group sg = item.get_sub_group();

    const int64_t r = item.get_global_id(0);
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    constexpr int blocks_per_group = N / m;
    constexpr int el_w             = blocks_per_group / WARP_SIZE;
    static_assert(el_w >= 1 && blocks_per_group % WARP_SIZE == 0, "blocks_per_group must be a multiple of WARP_SIZE");
    float     reg[el_w * m];
    const int lane = sg.get_local_linear_id();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;

#pragma unroll
        for (int j = 0; j < m; ++j) {
            reg[i * m + j] = src[b_idx * m + j] * scale;
        }
    }

#pragma unroll
    for (int b = 0; b < el_w; ++b) {
        float z[m] = { 0.0f };

#pragma unroll
        for (int i = 0; i < m; ++i) {
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const float h = (m == 12 ? H12[j][i] : H20[j][i]);
                z[i] += reg[b * m + j] * h;
            }
        }

#pragma unroll
        for (int i = 0; i < m; ++i) {
            reg[b * m + i] = z[i];
        }
    }

#pragma unroll
    for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
#pragma unroll
            for (int k = 0; k < m; ++k) {
                const float val  = reg[j * m + k];
                const float val2 = dpct::permute_sub_group_by_xor(sg, val, h, WARP_SIZE);

                reg[j * m + k] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

#pragma unroll
    for (int h = WARP_SIZE; h < blocks_per_group; h *= 2) {
        const int step = h / WARP_SIZE;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int s = 0; s < step; ++s) {
#pragma unroll
                for (int k = 0; k < m; ++k) {
                    const float x = reg[(j + s) * m + k];
                    const float y = reg[(j + s + step) * m + k];

                    reg[(j + s) * m + k]        = x + y;
                    reg[(j + s + step) * m + k] = x - y;
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int b_idx = i * WARP_SIZE + lane;
#pragma unroll
        for (int k = 0; k < m; ++k) {
            dst[b_idx * m + k] = reg[i * m + k];
        }
    }
}

template <int N, int m>
static void launch_kronecker(const float *   src,
                             float *         dst,
                             const int64_t   n_rows,
                             const float     scale,
                             dpct::queue_ptr stream) {
    constexpr int rows_per_block = 4;

    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // dim 1 is the fastest-varying, so a sub-group is exactly one row's WARP_SIZE lanes.
    const sycl::range<2> global(num_blocks * rows_per_block, WARP_SIZE);
    const sycl::range<2> local(rows_per_block, WARP_SIZE);

    stream->parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             kronecker_kernel<N, m>(src, dst, n_rows, scale, item);
                         });
}

bool ggml_sycl_op_fwht(ggml_backend_sycl_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                       const ggml_tensor * scale_in, const ggml_tensor * scale_out, const void * src_data) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_nelements(src) != ggml_nelements(dst)) {
        return false;
    }
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int     n    = (int) src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float *   src_d  = src_data ? (const float *) src_data : (const float *) src->data;
    float *         dst_d  = (float *) dst->data;
    dpct::queue_ptr stream = ctx.stream();

    const float * sin = nullptr;
    const float * sout = nullptr;
    int64_t kin = 0;
    int64_t kout = 0;
    if (scale_in) {
        if (scale_in->type != GGML_TYPE_F32 || scale_in->ne[0] <= 0) {
            return false;
        }
        sin = (const float *) scale_in->data;
        kin = scale_in->ne[0];
    }
    if (scale_out) {
        if (scale_out->type != GGML_TYPE_F32 || scale_out->ne[0] <= 0) {
            return false;
        }
        sout = (const float *) scale_out->data;
        kout = scale_out->ne[0];
    }

    const float scale = 1.0f / std::sqrt((float) n);

    switch (n) {
        case 64:
            if (sin || sout) {
                return false;
            }
            launch_fwht<64>(src_d, dst_d, rows, scale, stream);
            return true;
        case 128:
#if WARP_SIZE == 16
            launch_fwht128(src_d, dst_d, rows, scale, stream, sin, kin, sout, kout);
#else
            if (sin || sout) {
                return false;
            }
            launch_fwht<128>(src_d, dst_d, rows, scale, stream);
#endif
            return true;
        case 256:
            launch_fwht<256>(src_d, dst_d, rows, scale, stream);
            return true;
        case 512:
            launch_fwht<512>(src_d, dst_d, rows, scale, stream);
            return true;
        case 384:
            launch_kronecker<384, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 768:
            launch_kronecker<768, 12>(src_d, dst_d, rows, scale, stream);
            return true;
        case 640:
            launch_kronecker<640, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        case 1280:
            launch_kronecker<1280, 20>(src_d, dst_d, rows, scale, stream);
            return true;
        default:
            return false;
    }
}

namespace {

bool fwht_is_cast(const ggml_tensor * t) {
    return t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_CONT);
}

bool fwht_is_had(const ggml_tensor * mm) {
    return mm && mm->op == GGML_OP_MUL_MAT &&
           ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           mm->type == GGML_TYPE_F32 && mm->src[1] && mm->src[1]->type == GGML_TYPE_F32;
}

bool fwht_row_vec(const ggml_tensor * v, int64_t ne0) {
    return v && v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) &&
           v->ne[0] == ne0 && v->ne[1] == 1 && v->ne[2] == 1 && v->ne[3] == 1;
}

bool fwht_pick_scale(const ggml_tensor * mul, const ggml_tensor ** act, const ggml_tensor ** scale) {
    if (!mul || mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || !mul->src[0] || !mul->src[1]) {
        return false;
    }
    const ggml_tensor * a = mul->src[0];
    const ggml_tensor * b = mul->src[1];
    if (ggml_are_same_shape(a, mul) && fwht_row_vec(b, mul->ne[0]) && ggml_is_contiguous(a)) {
        *act = a;
        *scale = b;
        return true;
    }
    if (ggml_are_same_shape(b, mul) && fwht_row_vec(a, mul->ne[0]) && ggml_is_contiguous(b)) {
        *act = b;
        *scale = a;
        return true;
    }
    return false;
}

int fwht_follow_casts(const ggml_cgraph * cgraph, int i, const ggml_tensor * cur, const ggml_tensor ** viewed) {
    *viewed = cur;
    while (i < cgraph->n_nodes && fwht_is_cast(cgraph->nodes[i]) && cgraph->nodes[i]->src[0] == *viewed) {
        *viewed = cgraph->nodes[i];
        ++i;
    }
    return i;
}

bool fwht_uses_ok(const ggml_cgraph * cgraph, int begin, int end) {
    for (int k = begin; k < end; ++k) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, k) != 1) {
            return false;
        }
    }
    return true;
}

} // namespace

int ggml_sycl_try_fwht_scale_fuse(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph, int node_idx) {
    if (ggml_sycl_get_env("GGML_SYCL_FWHT_FUSE", 1) <= 0 || node_idx + 1 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * n0 = cgraph->nodes[node_idx];
    const ggml_tensor * act = nullptr;
    const ggml_tensor * scale = nullptr;
    const ggml_tensor * viewed = n0;
    const int after = fwht_follow_casts(cgraph, node_idx + 1, n0, &viewed);
    if (after >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * n_last = cgraph->nodes[after];
    const ggml_tensor * src = nullptr;
    const void * src_data = nullptr;
    ggml_tensor * dst = nullptr;
    const ggml_tensor * sin = nullptr;
    const ggml_tensor * sout = nullptr;
    if (n0->op == GGML_OP_MUL && fwht_is_had(n_last)) {
        if (!fwht_pick_scale(n0, &act, &scale)) {
            return 0;
        }
        const ggml_tensor * fwht_src = n_last->src[1];
        if (fwht_src != viewed && (!fwht_is_cast(fwht_src) || fwht_src->src[0] != n0)) {
            return 0;
        }
        if (fwht_src->ne[0] != 128 || ggml_nelements(act) != ggml_nelements(fwht_src)) {
            return 0;
        }
        if ((n0->flags & GGML_TENSOR_FLAG_OUTPUT) || !fwht_uses_ok(cgraph, node_idx, after)) {
            return 0;
        }
        src = fwht_src;
        src_data = act->data;
        dst = n_last;
        sin = scale;
    } else if (fwht_is_had(n0) && n_last->op == GGML_OP_MUL) {
        if (!fwht_pick_scale(n_last, &act, &scale)) {
            return 0;
        }
        if (act != viewed && (!fwht_is_cast(act) || act->src[0] != n0)) {
            return 0;
        }
        if (n0->src[1]->ne[0] != 128) {
            return 0;
        }
        if ((n0->flags & GGML_TENSOR_FLAG_OUTPUT) || !fwht_uses_ok(cgraph, node_idx, after)) {
            return 0;
        }
        src = n0->src[1];
        src_data = n0->src[1]->data;
        dst = n_last;
        sout = scale;
    } else {
        return 0;
    }
    if (!ggml_sycl_op_fwht(ctx, src, dst, sin, sout, src_data)) {
        return 0;
    }
    return after - node_idx + 1;
}
